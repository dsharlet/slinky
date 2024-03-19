#include "builder/pipeline.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <list>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "builder/infer_bounds.h"
#include "builder/node_mutator.h"
#include "builder/optimizations.h"
#include "builder/simplify.h"
#include "builder/substitute.h"
#include "runtime/evaluate.h"
#include "runtime/expr.h"
#include "runtime/pipeline.h"
#include "runtime/print.h"

namespace slinky {

buffer_expr::buffer_expr(symbol_id sym, std::size_t rank, index_t elem_size)
    : sym_(sym), elem_size_(elem_size), producer_(nullptr), constant_(nullptr) {
  dims_.reserve(rank);
  auto var = variable::make(sym);
  for (index_t i = 0; i < static_cast<index_t>(rank); ++i) {
    interval_expr bounds = buffer_bounds(var, i);
    expr stride = buffer_stride(var, i);
    expr fold_factor = buffer_fold_factor(var, i);
    dims_.push_back({bounds, stride, fold_factor});
  }
}

buffer_expr::buffer_expr(symbol_id sym, const_raw_buffer_ptr constant_buffer)
    : sym_(sym), elem_size_(constant_buffer->elem_size), producer_(nullptr), constant_(std::move(constant_buffer)) {
  assert(constant_ != nullptr);
  dims_.reserve(constant_->rank);

  for (index_t d = 0; d < static_cast<index_t>(constant_->rank); ++d) {
    expr min = constant_->dims[d].min();
    expr max = constant_->dims[d].max();
    expr stride = constant_->dims[d].stride();
    expr fold_factor = constant_->dims[d].fold_factor();
    dims_.push_back({slinky::bounds(min, max), stride, fold_factor});
  }
}

buffer_expr_ptr buffer_expr::make(symbol_id sym, std::size_t rank, index_t elem_size) {
  return buffer_expr_ptr(new buffer_expr(sym, rank, elem_size));
}

buffer_expr_ptr buffer_expr::make(node_context& ctx, const std::string& sym, std::size_t rank, index_t elem_size) {
  return buffer_expr_ptr(new buffer_expr(ctx.insert_unique(sym), rank, elem_size));
}

buffer_expr_ptr buffer_expr::make_constant(symbol_id sym, const_raw_buffer_ptr constant_buffer) {
  return buffer_expr_ptr(new buffer_expr(sym, std::move(constant_buffer)));
}
buffer_expr_ptr buffer_expr::make_constant(
    node_context& ctx, const std::string& sym, const_raw_buffer_ptr constant_buffer) {
  return buffer_expr_ptr(new buffer_expr(ctx.insert_unique(sym), std::move(constant_buffer)));
}

void buffer_expr::set_producer(func* f) {
  assert(producer_ == nullptr || f == nullptr);
  producer_ = f;
}

box_expr buffer_expr::bounds() const {
  box_expr result(rank());
  for (std::size_t d = 0; d < rank(); ++d) {
    result[d] = dim(d).bounds;
  }
  return result;
}

func::func(
    call_stmt::callable impl, std::vector<input> inputs, std::vector<output> outputs, call_stmt::callable_attrs attrs)
    : impl_(std::move(impl)), attrs_(attrs), inputs_(std::move(inputs)), outputs_(std::move(outputs)) {
  add_this_to_buffers();
}

func::func(input input, output out, std::optional<std::vector<char>> padding)
    : func(nullptr, {std::move(input)}, {std::move(out)}) {
  padding_ = std::move(padding);
}

func::func(std::vector<input> inputs, output out) : func(nullptr, std::move(inputs), {std::move(out)}) {}

func::func(func&& m) noexcept { *this = std::move(m); }
func& func::operator=(func&& m) noexcept {
  if (this == &m) return *this;
  m.remove_this_from_buffers();
  impl_ = std::move(m.impl_);
  inputs_ = std::move(m.inputs_);
  outputs_ = std::move(m.outputs_);
  loops_ = std::move(m.loops_);
  compute_at_ = std::move(m.compute_at_);
  padding_ = std::move(m.padding_);
  add_this_to_buffers();
  return *this;
}

func::~func() { remove_this_from_buffers(); }

void func::add_this_to_buffers() {
  for (auto& i : outputs_) {
    i.buffer->set_producer(this);
  }
}
void func::remove_this_from_buffers() {
  for (auto& i : outputs_) {
    i.buffer->set_producer(nullptr);
  }
}

stmt func::make_call() const {
  if (impl_) {
    call_stmt::symbol_list inputs;
    call_stmt::symbol_list outputs;
    for (const func::input& i : inputs_) {
      inputs.push_back(i.sym());
    }
    for (const func::output& i : outputs_) {
      outputs.push_back(i.sym());
    }
    return call_stmt::make(impl_, std::move(inputs), std::move(outputs), attrs_);
  } else {
    std::vector<stmt> copies;
    for (const func::input& input : inputs_) {
      assert(outputs_.size() == 1);
      std::vector<expr> src_x;
      std::vector<symbol_id> dst_x;
      for (const interval_expr& i : input.bounds) {
        assert(match(i.min, i.max));
        src_x.push_back(i.min);
      }
      for (const var& i : outputs_[0].dims) {
        dst_x.push_back(i.sym());
      }
      stmt copy = copy_stmt::make(input.sym(), src_x, outputs_[0].sym(), dst_x, padding_);
      if (!input.output_slice.empty()) {
        copy = slice_buffer::make(outputs_[0].sym(), input.output_slice, copy);
      }
      copies.push_back(copy);
    }
    return block::make(std::move(copies));
  }
}

func func::make_concat(std::vector<buffer_expr_ptr> in, output out, std::size_t dim, std::vector<expr> bounds) {
  assert(in.size() + 1 == bounds.size());
  std::size_t rank = out.buffer->rank();

  std::vector<box_expr> crops;
  std::vector<func::input> inputs;
  for (std::size_t i = 0; i < in.size(); ++i) {
    // Prepare the input.
    assert(in[i]->rank() == rank);
    func::input input;

    input.buffer = in[i];
    input.bounds.resize(rank);
    for (std::size_t d = 0; d < rank; ++d) {
      input.bounds[d] = point(out.dims[d]);
    }
    // We translate the input by the bounds to make concatenation a bit more natural (the concatenated buffers will
    // start at index 0 in the concatenated dimension).
    input.bounds[dim] -= bounds[i];

    // Translate the bounds into the crop needed by make_copy.
    // We leave the dimensions not concatenated undefined so infer_bounds will require each input to provide the full
    // output in those dimensions.
    input.output_crop.resize(dim + 1);
    input.output_crop[dim] = range(bounds[i], bounds[i + 1]);

    inputs.push_back(std::move(input));
  }
  return make_copy(std::move(inputs), std::move(out));
}

func func::make_stack(std::vector<buffer_expr_ptr> in, output out, std::size_t dim) {
  std::size_t rank = out.buffer->rank();
  assert(out.dims.size() == rank);
  assert(rank > 0);
  dim = std::min(rank - 1, dim);

  std::vector<box_expr> crops;
  std::vector<func::input> inputs;
  for (std::size_t i = 0; i < in.size(); ++i) {
    // Prepare the input.
    assert(in[i]->rank() + 1 == rank);
    func::input input;
    input.buffer = in[i];
    input.bounds.resize(rank);
    for (std::size_t d = 0; d < rank; ++d) {
      input.bounds[d] = point(out.dims[d]);
    }

    // Remove the stack dimension of the output from the input bounds, and slice the output at this point.
    input.bounds.erase(input.bounds.begin() + dim);
    input.output_slice.resize(dim + 1);
    input.output_slice[dim] = static_cast<index_t>(i);

    inputs.push_back(std::move(input));
  }
  // Also apply the slice to the output dimensions.
  out.dims.erase(out.dims.begin() + dim);
  return make_copy(std::move(inputs), std::move(out));
}

namespace {

void get_output_bounds(
    const std::vector<func::output>& outputs, symbol_map<expr>& output_mins, symbol_map<expr>& output_maxs) {
  for (const func::output& o : outputs) {
    for (std::size_t d = 0; d < o.dims.size(); ++d) {
      expr dim_min = o.buffer->dim(d).min();
      expr dim_max = o.buffer->dim(d).max();
      std::optional<expr>& min = output_mins[o.dims[d]];
      std::optional<expr>& max = output_maxs[o.dims[d]];
      min = min ? slinky::min(*min, dim_min) : dim_min;
      max = max ? slinky::max(*max, dim_max) : dim_max;
    }
  }
}

box_expr compute_input_bounds(
    const func* f, const func::input& i, const symbol_map<expr>& output_mins, const symbol_map<expr>& output_maxs) {
  symbol_map<expr> output_mins_i = output_mins;
  symbol_map<expr> output_maxs_i = output_maxs;
  if (!i.output_crop.empty()) {
    const box_expr& crop = i.output_crop;
    assert(f->outputs().size() == 1);
    const func::output& o = f->outputs()[0];
    // We have an output crop for this input. Apply it to our bounds.
    // TODO: It would be nice if this were simply a crop_buffer inserted in the right place. However, that is
    // difficult to do because it could be used in several places, each with a different output crop to apply.
    for (std::size_t d = 0; d < std::min(crop.size(), o.dims.size()); ++d) {
      std::optional<expr>& min = output_mins_i[o.dims[d]];
      std::optional<expr>& max = output_maxs_i[o.dims[d]];
      assert(min);
      assert(max);
      if (crop[d].min.defined()) min = slinky::max(*min, crop[d].min);
      if (crop[d].max.defined()) max = slinky::min(*max, crop[d].max);
    }
  }

  box_expr crop(i.buffer->rank());
  for (int d = 0; d < static_cast<int>(crop.size()); ++d) {
    expr min = substitute(i.bounds[d].min, output_mins_i);
    expr max = substitute(i.bounds[d].max, output_maxs_i);
    // The bounds may have been negated.
    crop[d] = simplify(slinky::bounds(min, max) | slinky::bounds(max, min));
  }
  return crop;
}

class unionize_crop : public node_mutator {
  symbol_id target;
  const box_expr& crop;

public:
  bool found = false;

  unionize_crop(symbol_id target, const box_expr& crop) : target(target), crop(crop) {}

  void visit(const crop_buffer* op) override {
    if (op->sym != target) {
      node_mutator::visit(op);
      return;
    }

    // Don't recursively mutate, once we crop the buffer here, it doesn't need to be cropped again.
    slinky::box_expr new_crop = crop | op->bounds;
    for (int d = 0; d < static_cast<int>(new_crop.size()); d++) {
      new_crop[d] = simplify(new_crop[d]);
    }
    set_result(crop_buffer::make(target, new_crop, op->body));
    found = true;
  }
};

// Expand an existing crop for `target` to include `crop`, or add a new crop if there was no existing crop.
stmt add_crop_union(stmt s, symbol_id target, const box_expr& crop) {
  unionize_crop m(target, crop);
  s = m.mutate(s);
  if (!m.found) {
    s = crop_buffer::make(target, crop, s);
  }
  return s;
}

bool operator==(const loop_id& a, const loop_id& b) {
  if (!a.func) {
    return !b.func;
  } else if (a.func == b.func) {
    assert(a.var.defined());
    assert(b.var.defined());
    return a.var.sym() == b.var.sym();
  } else {
    return false;
  }
}

void topological_sort_impl(const func* f, std::set<const func*>& visited, std::vector<const func*>& order,
    std::map<const func*, std::vector<const func*>>& deps, std::set<buffer_expr_ptr>& constants) {
  for (const auto& i : f->inputs()) {
    const auto& input = i.buffer;
    if (input->constant()) {
      constants.insert(input);
      continue;
    }
    if (!input->producer()) {
      continue;
    }
    // Record that f is consumer of input->producer.
    deps[input->producer()].push_back(f);

    if (visited.count(input->producer()) > 0) {
      continue;
    }
    topological_sort_impl(input->producer(), visited, order, deps, constants);
  }
  visited.insert(f);
  order.push_back(f);
}

void topological_sort(const std::vector<buffer_expr_ptr>& outputs, std::vector<const func*>& order,
    std::map<const func*, std::vector<const func*>>& deps, std::set<buffer_expr_ptr>& constants) {
  std::set<const func*> visited;
  for (const auto& i : outputs) {
    topological_sort_impl(i->producer(), visited, order, deps, constants);
  }

  // Reverse the order, so outputs go first.
  std::reverse(order.begin(), order.end());
}

// A simple structure to hold the node of the loop tree.
struct loop_tree_node {
  // Index of the parent node.
  int parent_index = -1;
  loop_id loop;
};

// Find a path from the node to the root of the tree.
void find_path_from_root(const std::vector<loop_tree_node>& loop_tree, int node_id, std::vector<int>& path_from_root) {
  path_from_root.push_back(node_id);
  while (node_id > 0) {
    node_id = loop_tree[node_id].parent_index;
    path_from_root.push_back(node_id);
  }
  std::reverse(path_from_root.begin(), path_from_root.end());
}

// Compare two paths and return the last point where they match.
int compare_paths_up_to(
    const std::vector<int>& base_path_from_root, const std::vector<int>& other_path_from_root, int max_match) {
  max_match = std::min(max_match, static_cast<int>(other_path_from_root.size()) - 1);
  for (int iy = 0; iy <= max_match; iy++) {
    if (other_path_from_root[iy] != base_path_from_root[iy]) {
      max_match = iy - 1;
      break;
    }
  }
  return max_match;
}

// Compute the least common ancestor of multiple nodes in the tree.
int lca(const std::vector<loop_tree_node>& loop_tree, const std::vector<int>& parent_ids) {
  // This is not the most optimal algorithm and likely can be improved
  // if we see it as a bottleneck later.

  // For each of the nodes find the path to the root of the tree.
  std::vector<std::vector<int>> paths_from_root(parent_ids.size());
  for (int ix = 0; ix < static_cast<int>(parent_ids.size()); ix++) {
    find_path_from_root(loop_tree, parent_ids[ix], paths_from_root[ix]);
  }

  // Compare paths to the root node until the diverge. The last node before
  // the diversion point is the least common ancestor.
  int max_match = paths_from_root[0].size() - 1;
  for (int ix = 1; ix < static_cast<int>(paths_from_root.size()); ix++) {
    max_match = compare_paths_up_to(paths_from_root[0], paths_from_root[ix], max_match);
  }

  return paths_from_root[0][max_match];
}

void compute_innermost_locations(const std::vector<const func*>& order,
    const std::map<const func*, std::vector<const func*>> deps, std::map<const func*, loop_id>& compute_at_levels) {
  // A tree which stores loop nest.
  std::vector<loop_tree_node> loop_tree;
  // Mapping between function and it's most innermost location.
  std::map<const func*, int> func_to_loop_tree;
  // Push the root loop.
  loop_tree.push_back({-1, loop_id()});

  // Iterate over functions in topological order starting from the output and build a loop nest tree.
  for (const auto& f : order) {
    int parent_id = -1;

    const auto& p = deps.find(f);
    if (p != deps.end()) {
      assert(!p->second.empty());

      // If we have an explicitly set compute_at location we should use that.
      if (f->compute_at()) {
        // TODO(vksnk): check if compute_at is a valid location based on computed
        // innermost location.
        for (int ix = 0; ix < static_cast<int>(loop_tree.size()); ix++) {
          if (loop_tree[ix].loop == *f->compute_at()) {
            parent_id = ix;
          }
        }
        compute_at_levels[f] = *f->compute_at();
      } else {
        // Check all of the consumers and find their innermost locations.
        std::vector<int> parent_ids;
        for (const auto& d : p->second) {
          const auto& node = func_to_loop_tree.find(d);
          assert(node != func_to_loop_tree.end());
          parent_ids.push_back(node->second);
        }

        if (parent_ids.size() == 1) {
          // We have just one consumer, so use its innermost location.
          parent_id = parent_ids[0];
        } else {
          // There are multiple consumers, so we need to find the least common ancestor
          // of their innermost locations.
          parent_id = lca(loop_tree, parent_ids);
        }

        compute_at_levels[f] = loop_tree[parent_id].loop;
      }
    } else {
      // This is an output so should be computed at root.
      parent_id = 0;
      compute_at_levels[f] = loop_id();
    }

    assert(parent_id != -1);

    // Add loops for this function to the loop nest. The loops are defined
    // from innermost to outermost, so iterate in reverse order.
    for (int i = f->loops().size() - 1; i >= 0; i--) {
      const auto& l = f->loops()[i];
      loop_tree.push_back({parent_id, {f, l.var}});
      parent_id = loop_tree.size() - 1;
    }
    func_to_loop_tree[f] = parent_id;
  }
}

void compute_allocation_bounds(const std::vector<const func*>& order, symbol_map<box_expr>& allocation_bounds) {
  ;
  for (const func* f : order) {
    symbol_map<expr> output_mins, output_maxs;
    get_output_bounds(f->outputs(), output_mins, output_maxs);

    for (const auto& i : f->inputs()) {
      box_expr crop = compute_input_bounds(f, i, output_mins, output_maxs);
      auto& bound = allocation_bounds[i.sym()];
      if (bound) {
        *bound = *bound | crop;
        std::cout << "Updating bound to - " << i.sym() << " - " << *bound << std::endl;
      } else {
        allocation_bounds[i.sym()] = crop;
        std::cout << "Setting bound to - " << i.sym() << " - " << *allocation_bounds[i.sym()] << std::endl;
      }
    }
  }
}

//
std::vector<dim_expr> substitute_from_map(std::vector<dim_expr> dims, span<const std::pair<expr, expr>> substitutions) {
  for (dim_expr& dim : dims) {
    dim_expr new_dim = dim;
    for (const std::pair<expr, expr>& j : substitutions) {
      new_dim.bounds.min = substitute(new_dim.bounds.min, j.first, j.second);
      new_dim.bounds.max = substitute(new_dim.bounds.max, j.first, j.second);
      new_dim.stride = substitute(new_dim.stride, j.first, j.second);
      new_dim.fold_factor = substitute(new_dim.fold_factor, j.first, j.second);
    }
    dim = new_dim;
  }
  return dims;
}

class pipeline_builder {
  std::set<buffer_expr_ptr> allocated;

  // Topologically sorted functions.
  std::vector<const func*> order_;
  // A mapping between func's and their compute_at locations.
  std::map<const func*, loop_id> compute_at_levels_;

  symbol_map<box_expr> allocation_bounds_;
  symbol_map<std::vector<dim_expr>> inferred_bounds_;

  std::vector<symbol_id> input_syms_;

  // Add crops to the inputs of f, using buffer intrinsics to get the bounds of the output.
  stmt add_input_crops(stmt result, const func* f) {
    // Find the bounds of the outputs required in each dimension. This is the union of the all the intervals from each
    // output associated with a particular dimension.
    assert(!f->outputs().empty());
    symbol_map<expr> output_mins, output_maxs;
    get_output_bounds(f->outputs(), output_mins, output_maxs);

    // Use the output bounds, and the bounds expressions of the inputs, to determine the bounds required of the input.
    for (const func::input& i : f->inputs()) {
      // We want to take the union of this crop with any existing crop of this buffer.
      box_expr crop = compute_input_bounds(f, i, output_mins, output_maxs);
      result = add_crop_union(result, i.sym(), crop);
    }
    return result;
  }

  stmt crop_for_loop(stmt body, const func* f, const func::loop_info& loop) {
    // Crop all the outputs of this buffer for this loop.
    for (const func::output& o : f->outputs()) {
      for (int d = 0; d < static_cast<int>(o.dims.size()); ++d) {
        if (o.dims[d].sym() == loop.sym()) {
          expr loop_max = buffer_max(var(o.sym()), d);
          interval_expr bounds = slinky::bounds(loop.var, min(simplify(loop.var + loop.step - 1), loop_max));
          body = crop_dim::make(o.sym(), d, bounds, body);
        }
      }
    }
    return body;
  }

  interval_expr get_loop_bounds(const func* f, const func::loop_info& loop) {
    interval_expr bounds = interval_expr::union_identity();
    for (const func::output& o : f->outputs()) {
      for (int d = 0; d < static_cast<int>(o.dims.size()); ++d) {
        if (o.dims[d].sym() == loop.sym()) {
          // This output uses this loop. Add it to the bounds.
          bounds |= o.buffer->dim(d).bounds;
        }
      }
    }
    return simplify(bounds);
  }

  stmt make_loop(stmt body, const func* f, const func::loop_info& loop = func::loop_info()) {
    loop_id here = {f, loop.var};

    body = build(body, f, here);

    if (loop.defined()) {
      // The loop body is done, and we have an actual loop to make here. Crop the body.
      body = crop_for_loop(body, f, loop);
      // And make the actual loop.
      body = loop::make(loop.sym(), loop.mode, get_loop_bounds(f, loop), loop.step, body);
    }
    return body;
  }

  stmt produce(const func* f) {
    stmt result = f->make_call();
    if (f->loops().empty()) {
      result = add_input_crops(result, f);
    }

    // Generate the loops that we want to be explicit.
    for (const auto& loop : f->loops()) {
      result = make_loop(result, f, loop);
    }

    return result;
  }

public:
  pipeline_builder(const std::vector<buffer_expr_ptr>& inputs, const std::vector<buffer_expr_ptr>& outputs,
      std::set<buffer_expr_ptr>& constants) {
    // To start with, we need to produce the outputs.
    for (auto& i : outputs) {
      allocated.insert(i);
    }

    for (const buffer_expr_ptr& i : inputs) {
      input_syms_.push_back(i->sym());
    }
    for (const buffer_expr_ptr& i : constants) {
      input_syms_.push_back(i->sym());
    }

    // Dependencies between the functions.
    std::map<const func*, std::vector<const func*>> deps;
    topological_sort(outputs, order_, deps, constants);

    // Build a loop nest tree and computes compute_at locations when neccessary.
    compute_innermost_locations(order_, deps, compute_at_levels_);

    // Compute allocation bounds.
    compute_allocation_bounds(order_, allocation_bounds_);
  }

  // This function works together with the produce() function to
  // build an initial IR. The high-level approach is the following:
  // * the `build()` function looks through the list of func's
  //   to find funcs which need to be produced or allocated at given
  //   loop level `at`. If func need to be produced it calls the
  //   `produce()` function which actually produces the body of the
  //   func.
  // * the `produce()` for a given func produces it's body along
  //   with the necessary loops defined for this function. For each
  //   of the new loops, the `build()` is called for the case when there
  //   are func which need to be produced in that new loop.
  stmt build(stmt body, const func* base_f, const loop_id& at) {
    stmt result;
    std::vector<stmt> produced_stmt;
    // Build the functions computed at this loop level.
    for (int ix = order_.size() - 1; ix >= 0; ix--) {
      const auto& f = order_[ix];
      const auto& compute_at = compute_at_levels_.find(f);
      assert(compute_at != compute_at_levels_.end());
      if (compute_at->second == at) {
        if (result.defined()) {
          result = add_input_crops(result, f);
        }
        result = block::make({result, produce(f)});
      }
    }

    result = block::make({result, body});

    if (base_f) {
      result = add_input_crops(result, base_f);
    }

    // Add all allocations at this loop level.
    for (int ix = order_.size() - 1; ix >= 0; ix--) {
      const auto& f = order_[ix];
      for (const auto& o : f->outputs()) {
        const auto& b = o.buffer;
        if (allocated.count(b)) continue;

        if ((b->store_at() && *b->store_at() == at) || (!b->store_at() && at.root())) {
          // // b->dim()
          // // const auto& bounds = allocation_bounds_[b->sym()];
          // // assert(bounds);

          std::vector<std::pair<expr, expr>> substitutions;

          expr alloc_var = variable::make(b->sym());

          box_expr& bounds = *allocation_bounds_[b->sym()];
          expr stride = static_cast<index_t>(b->elem_size());
          for (index_t d = 0; d < static_cast<index_t>(bounds.size()); ++d) {
            const interval_expr& bounds_d = bounds[d];

            substitutions.emplace_back(buffer_min(alloc_var, d), bounds_d.min);
            substitutions.emplace_back(buffer_max(alloc_var, d), bounds_d.max);
            substitutions.emplace_back(buffer_stride(alloc_var, d), stride);

            // We didn't initially set up the buffer with an extent, but the user might have used it.
            expr extent = bounds_d.extent();
            substitutions.emplace_back(buffer_extent(alloc_var, d), extent);
            stride *= min(extent, buffer_fold_factor(alloc_var, d));
          }
          std::vector<dim_expr> dims = substitute_from_map(b->dims(), substitutions);

          std::vector<dim_expr> shape;
          for (const auto& d : dims) {
            shape.push_back({d.bounds});
          }

          // Record the inferred dims.
          inferred_bounds_[b->sym()] = shape;

          result = allocate::make(b->sym(), b->storage(), b->elem_size(), dims, result);
          // result = allocate::make(b->sym(), b->storage(), b->elem_size(), b->dims(), result);
        }
      }
    }

    return result;
  }

  stmt add_input_checks(stmt body) {
    // Now we should know the bounds required of the inputs. Add checks that the inputs are sufficient.
    std::vector<stmt> checks;
    for (symbol_id i : input_syms_) {
      expr buf_var = variable::make(i);
      const std::optional<box_expr>& bounds = allocation_bounds_[i];
      assert(bounds);
      for (int d = 0; d < static_cast<int>(bounds->size()); ++d) {
        checks.push_back(check::make(buffer_min(buf_var, d) <= (*bounds)[d].min));
        checks.push_back(check::make(buffer_max(buf_var, d) >= (*bounds)[d].max));
        checks.push_back(check::make((*bounds)[d].extent() <= buffer_fold_factor(buf_var, d)));
      }
    }
    return block::make(std::move(checks), std::move(body));
  }

  stmt make_buffers(stmt body) {
    for (int ix = order_.size() - 1; ix >= 0; ix--) {
      const auto& f = order_[ix];

      for (const auto& o : f->outputs()) {
        const auto& b = o.buffer;

        std::optional<std::vector<dim_expr>> maybe_dims = inferred_bounds_[b->sym()];
        if (maybe_dims) {
          std::cout << "Found dims for " - b->sym() << "\n";
          body = make_buffer::make(b->sym(), expr(), expr(), *maybe_dims, body);
        } else {
          std::cout << "Didn't found dims for " - b->sym() << "\n";
        }
      }
    }
    return body;
  }

  const std::vector<symbol_id>& input_syms() { return input_syms_; }
};

void add_buffer_checks(const buffer_expr_ptr& b, bool output, std::vector<stmt>& checks) {
  int rank = static_cast<int>(b->rank());
  expr buf_var = variable::make(b->sym());
  checks.push_back(check::make(buf_var != 0));
  checks.push_back(check::make(buffer_rank(buf_var) == rank));
  checks.push_back(check::make(buffer_at(buf_var) != 0));
  checks.push_back(check::make(buffer_elem_size(buf_var) == b->elem_size()));
  for (int d = 0; d < rank; ++d) {
    expr fold_factor = buffer_fold_factor(buf_var, d);
    checks.push_back(check::make(b->dim(d).min() == buffer_min(buf_var, d)));
    checks.push_back(check::make(b->dim(d).max() == buffer_max(buf_var, d)));
    checks.push_back(check::make(b->dim(d).stride == buffer_stride(buf_var, d)));
    checks.push_back(check::make(b->dim(d).fold_factor == fold_factor));
    if (output) {
      checks.push_back(check::make(b->dim(d).extent() <= fold_factor));
    }
  }
}

bool is_verbose() {
  auto* s = std::getenv("SLINKY_VERBOSE");
  return (s && std::atoi(s) == 1);
}

stmt build_pipeline(node_context& ctx, const std::vector<buffer_expr_ptr>& inputs,
    const std::vector<buffer_expr_ptr>& outputs, std::set<buffer_expr_ptr>& constants, const build_options& options) {
  pipeline_builder builder(inputs, outputs, constants);

  stmt result;
  result = builder.build(result, nullptr, loop_id());
  // This inserts input checks around the statement, but the bounds
  // can be defined in the terms of the inner allocations.
  result = builder.add_input_checks(result);
  result = builder.make_buffers(result);

  std::cout << "Initial IR:\n" << std::tie(result, ctx) << std::endl;

  result = infer_bounds(result, ctx, builder.input_syms());

  // Add checks that the buffer constraints the user set are satisfied.
  std::vector<stmt> checks;
  for (const buffer_expr_ptr& i : inputs) {
    add_buffer_checks(i, /*output=*/false, checks);
  }
  for (const buffer_expr_ptr& i : outputs) {
    add_buffer_checks(i, /*output=*/true, checks);
  }
  result = block::make(std::move(checks), std::move(result));

  result = simplify(result);
  std::cout << "After simplification: \n" << std::tie(result, ctx) << std::endl;
  // Try to reuse buffers and eliminate copies where possible.
  if (!options.no_alias_buffers) {
    result = alias_buffers(result);
  }

  // `evaluate` currently can't handle `copy_stmt`, so this is required.
  result = implement_copies(result, ctx);

  result = simplify(result);

  result = fix_buffer_races(result);

  if (options.no_checks) {
    result = recursive_mutate<check>(result, [](const check* op) { return stmt(); });
  }

  if (is_verbose()) {
    std::cout << std::tie(result, ctx) << std::endl;
  }

  return result;
}

std::vector<var> vars(const std::vector<buffer_expr_ptr>& bufs) {
  std::vector<var> result;
  result.reserve(bufs.size());
  for (const buffer_expr_ptr& i : bufs) {
    result.push_back(i->sym());
  }
  return result;
}

std::vector<std::pair<symbol_id, const_raw_buffer_ptr>> constant_map(const std::set<buffer_expr_ptr>& constants) {
  std::vector<std::pair<symbol_id, const_raw_buffer_ptr>> result;
  result.reserve(constants.size());
  for (const buffer_expr_ptr& i : constants) {
    result.push_back({i->sym(), i->constant()});
  }
  return result;
}

}  // namespace

pipeline build_pipeline(node_context& ctx, std::vector<var> args, const std::vector<buffer_expr_ptr>& inputs,
    const std::vector<buffer_expr_ptr>& outputs, const build_options& options) {
  std::set<buffer_expr_ptr> constants;
  stmt body = build_pipeline(ctx, inputs, outputs, constants, options);
  pipeline p;
  p.args = std::move(args);
  p.inputs = vars(inputs);
  p.outputs = vars(outputs);
  p.constants = constant_map(constants);
  p.body = std::move(body);
  return p;
}

pipeline build_pipeline(node_context& ctx, const std::vector<buffer_expr_ptr>& inputs,
    const std::vector<buffer_expr_ptr>& outputs, const build_options& options) {
  return build_pipeline(ctx, {}, inputs, outputs, options);
}

}  // namespace slinky