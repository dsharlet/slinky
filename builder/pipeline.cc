#include "builder/pipeline.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "base/chrome_trace.h"
#include "builder/node_mutator.h"
#include "builder/optimizations.h"
#include "builder/simplify.h"
#include "builder/slide_and_fold_storage.h"
#include "builder/substitute.h"
#include "runtime/depends_on.h"
#include "runtime/evaluate.h"
#include "runtime/expr.h"
#include "runtime/pipeline.h"
#include "runtime/print.h"

namespace slinky {

// in print.cc, don't want to expose this.
const node_context* set_default_print_context(const node_context* ctx);

buffer_expr::buffer_expr(var sym, std::size_t rank, expr elem_size)
    : sym_(sym), elem_size_(std::move(elem_size)), producer_(nullptr), constant_(nullptr) {
  dims_.reserve(rank);
  for (index_t i = 0; i < static_cast<index_t>(rank); ++i) {
    interval_expr bounds = buffer_bounds(sym, i);
    expr stride = buffer_stride(sym, i);
    expr fold_factor = buffer_fold_factor(sym, i);
    dims_.push_back({bounds, stride, fold_factor});
  }
}

buffer_expr::buffer_expr(var sym, const_raw_buffer_ptr constant_buffer)
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

buffer_expr_ptr buffer_expr::make(var sym, std::size_t rank, expr elem_size) {
  return buffer_expr_ptr(new buffer_expr(sym, rank, std::move(elem_size)));
}

buffer_expr_ptr buffer_expr::make(node_context& ctx, const std::string& sym, std::size_t rank, expr elem_size) {
  return buffer_expr_ptr(new buffer_expr(ctx.insert_unique(sym), rank, std::move(elem_size)));
}

buffer_expr_ptr buffer_expr::make(var sym, const_raw_buffer_ptr constant_buffer) {
  return buffer_expr_ptr(new buffer_expr(sym, std::move(constant_buffer)));
}
buffer_expr_ptr buffer_expr::make(node_context& ctx, const std::string& sym, const_raw_buffer_ptr constant_buffer) {
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
    call_stmt::callable impl, std::vector<input> inputs, std::vector<output> outputs, call_stmt::attributes attrs)
    : impl_(std::move(impl)), attrs_(std::move(attrs)), inputs_(std::move(inputs)), outputs_(std::move(outputs)) {
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
  attrs_ = std::move(m.attrs_);
  user_data_ = m.user_data_;
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
      std::vector<var> dst_x;
      for (const interval_expr& i : input.bounds) {
        assert(match(i.min, i.max));
        src_x.push_back(i.min);
      }
      for (const var& i : outputs_[0].dims) {
        dst_x.push_back(i);
      }
      stmt copy = copy_stmt::make(input.sym(), src_x, outputs_[0].sym(), dst_x, padding_);
      if (!input.input_crop.empty()) {
        copy = crop_buffer::make(inputs_[0].sym(), inputs_[0].sym(), input.input_crop, copy);
      }
      if (!input.output_crop.empty()) {
        copy = crop_buffer::make(outputs_[0].sym(), outputs_[0].sym(), input.output_crop, copy);
      }
      if (!input.output_slice.empty()) {
        copy = slice_buffer::make(outputs_[0].sym(), outputs_[0].sym(), input.output_slice, copy);
      }
      copies.push_back(copy);
    }
    return block::make(std::move(copies));
  }
}

func func::make_concat(std::vector<buffer_expr_ptr> in, output out, std::size_t dim, std::vector<expr> bounds) {
  assert(in.size() + 1 == bounds.size());
  std::size_t rank = out.buffer->rank();

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
    input.input_crop.resize(dim + 1);
    input.input_crop[dim] = range(0, bounds[i + 1] - bounds[i]);
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

// This mutator replaces uses of buffer metadata with variables, and later defines those symbols.
// This is used to avoid a massive footgun for users of slinky: when defining bounds expressions, crops, and so on, it
// is very tempting to use buffer metadata. Doing this is problematic because the buffer metadata will be affected by
// crops and slices generated by slinky, which change the meaning of the expressions. This avoids this problem by
// replacing buffer metadata expressions with variables, and remembers the values to define these replacements outside
// of the relevant crops.
class sanitize_user_exprs : public node_mutator {
public:
  node_context& ctx;
  std::map<expr, var, node_less> replacements;
  std::vector<var> external;

  sanitize_user_exprs(node_context& ctx) : ctx(ctx) {}

  stmt define_replacements(stmt s) const {
    std::vector<std::pair<var, expr>> lets;
    lets.reserve(replacements.size());
    for (const auto& i : replacements) {
      lets.push_back({i.second, i.first});
    }
    return let_stmt::make(std::move(lets), std::move(s));
  }

  void visit(const variable* op) override {
    if (op->field == buffer_field::none) {
      node_mutator::visit(op);
      return;
    }

    // Don't lift internally allocated buffer metadata expressions.
    // TODO: This should be a proper API error.
    assert(std::binary_search(external.begin(), external.end(), op->sym));

    auto i = replacements.insert(std::pair<const expr, var>(op, 0));
    if (i.second) {
      i.first->second = ctx.insert_unique("g");
    }
    set_result(variable::make(i.first->second));
  }

  void visit(const call* op) override {
    if (!is_buffer_intrinsic(op->intrinsic)) {
      node_mutator::visit(op);
      return;
    }

    // Don't lift internally allocated buffer metadata expressions.
    assert(op->args.size() >= 1 && as_variable(op->args[0]));
    // TODO: This should be a proper API error.
    assert(std::binary_search(external.begin(), external.end(), *as_variable(op->args[0])));

    auto i = replacements.insert(std::pair<const expr, var>(op, 0));
    if (i.second) {
      i.first->second = ctx.insert_unique("g");
    }
    set_result(variable::make(i.first->second));
  }
};

bounds_map get_output_bounds(const std::vector<func::output>& outputs) {
  bounds_map output_bounds;
  for (const func::output& o : outputs) {
    for (index_t d = 0; d < static_cast<index_t>(o.dims.size()); ++d) {
      std::optional<interval_expr>& output_bounds_d = output_bounds[o.dims[d]];
      if (!output_bounds_d) {
        output_bounds_d = buffer_bounds(o.sym(), d);
      } else {
        *output_bounds_d |= buffer_bounds(o.sym(), d);
      }
    }
  }
  return output_bounds;
}

box_expr compute_input_bounds(
    const func* f, const func::input& i, const bounds_map& output_bounds, sanitize_user_exprs& sanitizer) {
  box_expr crop(i.buffer->rank());
  for (std::size_t d = 0; d < std::min(crop.size(), i.bounds.size()); ++d) {
    crop[d] = bounds_of(sanitizer.mutate(i.bounds[d]), output_bounds);

    if (d < i.input_crop.size()) {
      // We have an output crop for this input, intersect with the crop we have.
      // TODO: It would be nice if this were simply a crop_buffer inserted in the right place. However, that is
      // difficult to do because it could be used in several places, each with a different output crop to apply.
      crop[d] &= sanitizer.mutate(i.input_crop[d]);
    }
  }

  return crop;
}

bool operator==(const loop_id& a, const loop_id& b) {
  if (!a.func) {
    return !b.func;
  } else if (a.func == b.func) {
    assert(a.var.defined());
    assert(b.var.defined());
    return a.var == b.var;
  } else {
    return false;
  }
}

void topological_sort_impl(const func* f, std::set<const func*>& processing, std::set<const func*>& visited,
    std::vector<const func*>& order, std::map<const func*, std::vector<const func*>>& deps) {
  if (visited.count(f) > 0) {
    return;
  }

  assert(processing.count(f) == 0);
  processing.insert(f);
  for (const auto& i : f->inputs()) {
    const auto& input = i.buffer;
    if (!input->producer()) {
      continue;
    }
    // Record that f is consumer of input->producer.
    deps[input->producer()].push_back(f);

    topological_sort_impl(input->producer(), processing, visited, order, deps);
  }
  processing.erase(f);
  visited.insert(f);
  order.push_back(f);
}

void topological_sort(const std::vector<buffer_expr_ptr>& outputs, std::vector<const func*>& order,
    std::map<const func*, std::vector<const func*>>& deps) {
  std::set<const func*> processing;
  std::set<const func*> visited;
  for (const auto& i : outputs) {
    topological_sort_impl(i->producer(), processing, visited, order, deps);
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
std::vector<int> find_path_from_root(const std::vector<loop_tree_node>& loop_tree, int node_id) {
  std::vector<int> path_from_root = {node_id};
  while (node_id > 0) {
    node_id = loop_tree[node_id].parent_index;
    path_from_root.push_back(node_id);
  }
  std::reverse(path_from_root.begin(), path_from_root.end());
  return path_from_root;
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
  std::vector<std::vector<int>> paths_from_root;
  paths_from_root.reserve(parent_ids.size());
  for (std::size_t parent_id : parent_ids) {
    paths_from_root.push_back(find_path_from_root(loop_tree, parent_id));
  }

  // Compare paths to the root node until the diverge. The last node before
  // the diversion point is the least common ancestor.
  int max_match = static_cast<int>(paths_from_root[0].size()) - 1;
  for (const std::vector<int>& path : paths_from_root) {
    max_match = compare_paths_up_to(paths_from_root[0], path, max_match);
  }

  return paths_from_root[0][max_match];
}

void compute_innermost_locations(const std::vector<const func*>& order,
    const std::map<const func*, std::vector<const func*>> deps, std::map<const func*, loop_id>& compute_at_levels,
    std::map<const func*, loop_id>& realization_levels) {
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
    for (auto l = f->loops().rbegin(); l != f->loops().rend(); ++l) {
      loop_tree.push_back({parent_id, {f, l->var}});
      parent_id = static_cast<int>(loop_tree.size()) - 1;
    }
    if (f->loops().empty()) {
      realization_levels[f] = compute_at_levels[f];
    } else {
      realization_levels[f] = {f, f->loops().front().var};
    }
    func_to_loop_tree[f] = parent_id;
  }
}

// Update dims vector by substittuting expression from the map.
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

// Perform a substitute limited to call or copy inputs only.
stmt substitute_inputs(const stmt& s, const symbol_map<var>& subs) {
  class m : public stmt_mutator {
    const symbol_map<var>& subs;

  public:
    m(const symbol_map<var>& subs) : subs(subs) {}

    void visit(const call_stmt* op) override {
      bool changed = false;
      call_stmt::symbol_list inputs;
      for (const auto& i : op->inputs) {
        if (subs[i]) {
          changed = true;
          inputs.push_back(*subs[i]);
        } else {
          inputs.push_back(i);
        }
      }

      if (changed) {
        set_result(call_stmt::make(op->target, std::move(inputs), op->outputs, op->attrs));
      } else {
        set_result(op);
      }
    }

    void visit(const copy_stmt* op) override {
      if (subs[op->src]) {
        set_result(copy_stmt::make(*subs[op->src], op->src_x, op->dst, op->dst_x, op->padding));
      } else {
        set_result(op);
      }
    }
  };
  return m(subs).mutate(s);
}

class pipeline_builder {
  node_context& ctx;

  struct allocation_candidate {
    buffer_expr_ptr buffer;
    int deps_count = 0;
    int consumers_produced = 0;
    int lifetime_start = -1;
    int lifetime_end = -1;

    explicit allocation_candidate(buffer_expr_ptr b) : buffer(b) {}
  };

  struct statement_with_range {
    stmt body;
    int start = 0;
    int end = 0;
    // Stores every allocation inside of the range.
    std::set<var> allocations;

    static statement_with_range merge(const statement_with_range& a, const statement_with_range& b) {
      assert(a.end + 1 == b.start);
      statement_with_range r;
      r.body = block::make({a.body, b.body});
      r.start = a.start;
      r.end = b.end;
      std::set_union(a.allocations.begin(), a.allocations.end(), b.allocations.begin(), b.allocations.end(),
          std::inserter(r.allocations, r.allocations.begin()));
      return r;
    }
  };

  struct loop_id_less {
    bool operator()(const loop_id& a, const loop_id& b) const {
      if (a.root() && b.root()) return false;
      if (a.root()) return true;
      if (b.root()) return false;
      if (a.func == b.func) return a.var < b.var;
      return a.func < b.func;
    }
  };
  std::map<loop_id, std::set<var>, loop_id_less> candidates_for_allocation_;
  // Information tracking the lifetimes of the buffers.
  symbol_map<allocation_candidate> allocation_info_;
  std::set<var> copy_inputs_;
  std::map<var, std::vector<var>> copy_deps_;
  // Direct buffer dependencies wrt to bounds for each buffer.
  std::map<var, std::set<var>> bounds_deps_;

  // Set of loops we've generated so far, this is used solely for the correctness
  // checks.
  std::set<loop_id, loop_id_less> loops_;

  int functions_produced_ = 0;

  // Topologically sorted functions.
  std::vector<const func*> order_;
  // A mapping between func's and their compute_at locations.
  std::map<const func*, loop_id> compute_at_levels_;
  // A mapping between func's and the place where their actual call statement
  // will be generated. This is different from the compute_at_levels_ map
  // for the case when the func has loops. In this case compute_at will
  // point to the loop_id at which loops should be placed and this structure
  // will point to the func's own innermost loop.
  std::map<const func*, loop_id> realization_levels_;

  symbol_map<box_expr> allocation_bounds_;
  symbol_map<std::vector<dim_expr>> inferred_dims_;
  symbol_map<std::vector<dim_expr>> inferred_shapes_;
  symbol_map<std::vector<interval_expr>> inferred_bounds_;

  std::map<var, buffer_expr_ptr> input_syms_;
  std::map<var, buffer_expr_ptr> output_syms_;
  std::map<var, buffer_expr_ptr> constants_;

  sanitize_user_exprs sanitizer_;

  void substitute_buffer_dims() {
    for (auto i = order_.rbegin(); i != order_.rend(); ++i) {
      const func* f = *i;
      for (const func::output& o : f->outputs()) {
        const buffer_expr_ptr& b = o.buffer;
        if (output_syms_.count(b->sym())) continue;

        // First substitute the bounds.
        std::vector<std::pair<expr, expr>> substitutions;
        assert(allocation_bounds_[b->sym()]);
        const box_expr& bounds = *allocation_bounds_[b->sym()];
        for (index_t d = 0; d < static_cast<index_t>(bounds.size()); ++d) {
          const interval_expr& bounds_d = bounds[d];
          substitutions.emplace_back(buffer_min(b->sym(), d), bounds_d.min);
          substitutions.emplace_back(buffer_max(b->sym(), d), bounds_d.max);
        }
        std::vector<dim_expr> dims = substitute_from_map(b->dims(), substitutions);

        substitutions.clear();
        for (index_t d = 0; d < static_cast<index_t>(bounds.size()); ++d) {
          substitutions.emplace_back(buffer_stride(b->sym(), d), expr());
        }
        dims = substitute_from_map(dims, substitutions);

        std::vector<dim_expr> shape;
        std::vector<interval_expr> tmp;
        for (const auto& d : dims) {
          shape.push_back({d.bounds});
          tmp.push_back({d.bounds});
        }

        // Record the inferred dims.
        inferred_dims_[b->sym()] = dims;
        inferred_shapes_[b->sym()] = shape;
        inferred_bounds_[b->sym()] = tmp;
      }
    }

    for (const auto& i : input_syms_) {
      if (!allocation_bounds_[i.first]) continue;
      inferred_bounds_[i.first] = allocation_bounds_[i.first];
    }
  }

  stmt crop_for_loop(stmt body, const func* f, const func::loop_info& loop) {
    // Crop all the outputs of this func for this loop.
    for (const func::output& o : f->outputs()) {
      for (int d = 0; d < static_cast<int>(o.dims.size()); ++d) {
        if (o.dims[d] == loop.sym()) {
          expr loop_step = sanitizer_.mutate(loop.step);
          interval_expr bounds = slinky::bounds(loop.var, simplify(loop.var + loop_step - 1));
          body = crop_dim::make(o.sym(), o.sym(), d, bounds, body);
        }
      }
    }
    return body;
  }

  interval_expr get_loop_bounds(const func* f, const func::loop_info& loop) {
    interval_expr bounds = interval_expr::union_identity();
    for (const func::output& o : f->outputs()) {
      for (int d = 0; d < static_cast<int>(o.dims.size()); ++d) {
        if (o.dims[d] == loop.sym()) {
          // This output uses this loop. Add it to the bounds.
          bounds |= buffer_bounds(o.sym(), d);
        }
      }
    }
    return simplify(bounds);
  }

  void compute_allocation_bounds() {
    for (const func* f : order_) {
      bounds_map output_bounds = get_output_bounds(f->outputs());

      for (const auto& i : f->inputs()) {
        box_expr crop = compute_input_bounds(f, i, output_bounds, sanitizer_);
        auto& bound = allocation_bounds_[i.sym()];
        if (bound) {
          *bound = *bound | crop;
        } else {
          allocation_bounds_[i.sym()] = crop;
        }
        for (const auto& o : f->outputs()) {
          bounds_deps_[i.sym()].insert(o.sym());
        }
      }
    }

    // Check to see if there are any *intermediate* outputs that don't
    // have allocation bounds; if there are, create an empty allocation
    // bounds for them.
    for (const func* f : order_) {
      for (const auto& o : f->outputs()) {
        if (output_syms_.count(o.sym())) continue;
        if (allocation_bounds_[o.sym()]) continue;
        box_expr crop(o.buffer->rank());
        for (std::size_t d = 0; d < crop.size(); ++d) {
          crop[d] = {std::numeric_limits<index_t>::max(), std::numeric_limits<index_t>::min()};
        }
        allocation_bounds_[o.sym()] = crop;
      }
    }
  }

  // Returns generated statement for this function, as well as the
  // lifetime range covered by it.
  statement_with_range produce(const func* f) {
    stmt result = sanitizer_.mutate(f->make_call());

    for (const func::output& o : f->outputs()) {
      const buffer_expr_ptr& b = o.buffer;
      if (output_syms_.count(b->sym())) continue;

      if (b->store_at()) {
        assert(loops_.count(*b->store_at()) > 0);
        candidates_for_allocation_[*b->store_at()].insert(b->sym());
      } else {
        candidates_for_allocation_[loop_id()].insert(b->sym());
      }
      std::optional<allocation_candidate>& info = allocation_info_[b->sym()];
      info->buffer = b;
      info->lifetime_start = functions_produced_;

      if (info->consumers_produced == info->deps_count) {
        info->lifetime_end = functions_produced_;
      }
    }

    for (const auto& i : f->inputs()) {
      const auto& input = i.buffer;
      if (input->constant()) {
        constants_[input->sym()] = input;
      }
      if (!input->producer()) {
        continue;
      }

      std::optional<allocation_candidate>& info = allocation_info_[input->sym()];
      if (!info) continue;
      info->consumers_produced++;

      if (info->consumers_produced == info->deps_count) {
        info->lifetime_end = functions_produced_;
      }
    }

    functions_produced_++;

    return {result, functions_produced_ - 1, functions_produced_ - 1};
  }

  // Wraps provided body statement with the allocation node for a given buffer.
  statement_with_range produce_allocation(
      const buffer_expr_ptr& b, statement_with_range s, symbol_map<var>& uncropped_subs) {
    var uncropped = ctx.insert_unique(ctx.name(b->sym()) + ".uncropped");
    uncropped_subs[b->sym()] = uncropped;
    s.body = clone_buffer::make(uncropped, b->sym(), s.body);

    const std::vector<dim_expr>& dims = *inferred_dims_[b->sym()];
    assert(allocation_bounds_[b->sym()]);
    const box_expr& bounds = *allocation_bounds_[b->sym()];
    s.body = allocate::make(b->sym(), b->storage(), b->elem_size(), dims, s.body);

    std::vector<stmt> checks;
    for (std::size_t d = 0; d < std::min(dims.size(), bounds.size()); ++d) {
      checks.push_back(check::make(dims[d].min() <= bounds[d].min));
      checks.push_back(check::make(dims[d].max() >= bounds[d].max));
    }

    s.body = block::make(std::move(checks), s.body);
    s.allocations.insert(b->sym());
    return s;
  }

  // Computes number of consumers for each of the buffers.
  void compute_deps_count() {
    for (const func* f : order_) {
      for (const func::output& o : f->outputs()) {
        const buffer_expr_ptr& b = o.buffer;
        if (output_syms_.count(b->sym())) continue;

        if (!allocation_info_[b->sym()]) {
          allocation_info_[b->sym()].emplace(b);
        }

        if (!f->impl() && f->padding()) {
          // Collect all buffers which are outputs of the copy
          // and the inputs of the copy as their dependencies.
          copy_inputs_.insert(b->sym());
          for (const auto& i : f->inputs()) {
            const auto& input = i.buffer;

            if (!input->producer()) {
              continue;
            }

            copy_deps_[b->sym()].push_back(input->sym());
          }
        }
      }
    }

    for (const func* f : order_) {
      for (const auto& i : f->inputs()) {
        const auto& input = i.buffer;

        if ((!f->impl() && !f->padding()) || (input->constant())) {
          // Collect all buffers which are inputs to the copy
          // and the outputs of the copy as their dependencies.
          copy_inputs_.insert(input->sym());
          for (const func::output& o : f->outputs()) {
            const buffer_expr_ptr& b = o.buffer;
            if (!input->constant() && output_syms_.count(b->sym())) continue;
            copy_deps_[input->sym()].push_back(b->sym());
          }
        }

        if (allocation_info_[input->sym()]) {
          allocation_info_[input->sym()]->deps_count++;
        }
      }
    }
  }

  bool has_all_allocations(buffer_expr_ptr candidate, std::set<var>& allocations) {
    for (const auto& b : copy_deps_[candidate->sym()]) {
      if (allocations.count(b) == 0) return false;
    }

    return true;
  }

  void place_constrained_buffers(std::vector<statement_with_range>& results, std::set<var>& candidates,
      const std::vector<allocation_candidate>& special, std::vector<allocation_candidate>& new_special,
      symbol_map<var>& uncropped_subs) {
    new_special.reserve(special.size());
    for (std::size_t iy = 0; iy < special.size(); iy++) {
      bool found = false;
      for (std::size_t ix = 0; ix < results.size(); ix++) {
        const buffer_expr_ptr& candidate = special[iy].buffer;

        if (!has_all_allocations(candidate, results[ix].allocations)) continue;

        // The block range must fully cover the allocation range.
        if (results[ix].start <= special[iy].lifetime_start && special[iy].lifetime_end <= results[ix].end) {
          results[ix] = produce_allocation(candidate, results[ix], uncropped_subs);
          candidates.erase(candidate->sym());
          found = true;
          break;
        }
      }

      if (found) continue;
      new_special.push_back(special[iy]);
    }
  }

  void place_constant_buffers(statement_with_range* results, std::size_t num_results) {
    for (std::size_t ix = 0; ix < num_results; ix++) {
      std::vector<var> constants_to_remove;
      constants_to_remove.reserve(constants_.size());

      for (const auto& i : constants_) {
        const buffer_expr_ptr& candidate = i.second;

        if (!has_all_allocations(candidate, results[ix].allocations)) continue;

        results[ix].body = constant_buffer::make(candidate->sym(), candidate->constant(), results[ix].body);
        constants_to_remove.push_back(i.first);
      }

      for (var b : constants_to_remove) {
        constants_.erase(b);
      }
    }
  }

  // This attempts to lay out allocation nodes such that the nesting
  // is minimized. The general idea is to iteratively build up a tree of
  // allocations starting from the allocations with the allocations with the
  // shortest life time as the lowest level of the tree. This is not always possible
  // in general to do and there are corner cases where nesting of allocations will
  // have depth N regardless of the approach, but in most practical situations this
  // will produce a structure close to the tree (for example, for the linear pipeline it
  // should build a perfect tree of depth ~log(N)). Similarly, the complexity of this
  // algorithm is O(N^2) for the worst case, but for the most practical pipelines it's
  // should be O(N*log(N)).
  statement_with_range lay_out_allocations(
      const loop_id& at, std::vector<statement_with_range> results, symbol_map<var>& uncropped_subs) {
    // The vector of allocations at this loop level.
    std::vector<allocation_candidate> lifetimes;
    // The same as above, but also has a set of special dependencies to
    // satisfy.
    std::vector<allocation_candidate> special;
    for (const auto& b : candidates_for_allocation_[at]) {
      if (output_syms_.count(b)) continue;
      if (copy_inputs_.count(b) > 0) {
        special.push_back(*allocation_info_[b]);
      } else {
        lifetimes.push_back(*allocation_info_[b]);
      }
    }

    // Sort vector by (end - start) and then start and then buffer sym.
    auto lifetime_less = [](allocation_candidate a, allocation_candidate b) {
      if ((a.lifetime_end - a.lifetime_start) == (b.lifetime_end - b.lifetime_start)) {
        if (a.lifetime_start == b.lifetime_start) {
          return a.buffer->sym() < b.buffer->sym();
        }
        return a.lifetime_start < b.lifetime_start;
      }

      return (a.lifetime_end - a.lifetime_start) < (b.lifetime_end - b.lifetime_start);
    };

    std::sort(lifetimes.begin(), lifetimes.end(), lifetime_less);
    std::sort(special.begin(), special.end(), lifetime_less);

    int iteration_count = 0;
    while (true) {
      std::vector<allocation_candidate> new_lifetimes;
      std::vector<statement_with_range> new_results;
      std::vector<allocation_candidate> new_special;

      new_lifetimes.reserve(lifetimes.size());
      new_results.reserve(results.size());

      std::size_t result_index = 0;
      for (std::size_t ix = 0; ix < lifetimes.size() && result_index < results.size();) {
        // Skip function bodies which go before the current buffer.
        while (result_index < results.size() && results[result_index].end < lifetimes[ix].lifetime_start) {
          new_results.push_back(results[result_index]);
          result_index++;
        }

        int new_min = std::numeric_limits<int>::max();
        int new_max = std::numeric_limits<int>::min();

        // Find which function bodies overlap with the lifetime of the buffer.
        std::vector<stmt> new_block;
        std::set<var> combined_allocs;
        while (result_index < results.size() && results[result_index].start <= lifetimes[ix].lifetime_end &&
               lifetimes[ix].lifetime_start <= results[result_index].end) {
          new_min = std::min(new_min, results[result_index].start);
          new_max = std::max(new_max, results[result_index].end);
          new_block.push_back(results[result_index].body);
          combined_allocs.insert(results[result_index].allocations.begin(), results[result_index].allocations.end());
          result_index++;
        }

        // Combine overlapping function bodies and wrap them into current buffer allocation.
        if (!new_block.empty()) {
          stmt new_body = block::make(new_block);

          buffer_expr_ptr b = lifetimes[ix].buffer;

          statement_with_range new_result = {new_body, new_min, new_max, std::move(combined_allocs)};
          new_result = produce_allocation(b, new_result, uncropped_subs);
          new_results.push_back(new_result);

          candidates_for_allocation_[at].erase(b->sym());
        }

        // Move to the next buffer.
        ix++;

        // Skip buffers which go before the next statement range/.
        while (ix < lifetimes.size() && lifetimes[ix].lifetime_start <= new_max) {
          new_lifetimes.push_back(lifetimes[ix]);
          ix++;
        }
      }

      new_results.insert(new_results.end(), results.begin() + result_index, results.end());

      // See if any of the blocks can be wrapped in the allocations which are inputs to the copy.
      // This only can happen if all of it's dependencies are inside of the block.
      place_constrained_buffers(new_results, candidates_for_allocation_[at], special, new_special, uncropped_subs);

      // Attempt to place constant buffers as close to their usage location as possible.
      place_constant_buffers(new_results.data(), new_results.size());

      // No changes, so leave the loop.
      bool no_changes = (lifetimes.size() == new_lifetimes.size() && special.size() == new_special.size());

      lifetimes = std::move(new_lifetimes);
      special = std::move(new_special);
      results = std::move(new_results);

      if (no_changes) break;

      iteration_count++;
    }

    if (results.empty()) return {};

    statement_with_range result;
    result = results.front();
    for (std::size_t ix = 1; ix < results.size(); ix++) {
      result = statement_with_range::merge(result, results[ix]);
    }

    // Add all remaining allocations at this loop level. The allocations can be added in any order. This order enables
    // aliasing copy dsts to srcs, which is more flexible than aliasing srcs to dsts.
    for (const func* f : order_) {
      for (const func::output& o : f->outputs()) {
        const buffer_expr_ptr& b = o.buffer;
        if (output_syms_.count(b->sym())) continue;
        if (candidates_for_allocation_[at].count(b->sym()) == 0) continue;

        result = produce_allocation(b, result, uncropped_subs);
        candidates_for_allocation_[at].erase(b->sym());
      }
    }

    // Try again for the combined statement.
    place_constant_buffers(&result, 1);

    return result;
  }

  // This function works together with the produce() and make_loop() functions
  // to build an initial IR. The high-level approach is the following:
  // * the `build()` function looks through the list of func's
  //   to find funcs which need to be produced or allocated at given
  //   loop level `at`. If func need to be produced it calls the
  //   `produce()` function which actually produces the body of the
  //   func. If func has loops it calls the 'make_loop()' func to produce
  //   corresponding loops.
  // * the `produce()` for a given func produces it's body.
  // * the `make_loop()` will produce the necessary loops defined for the function.
  //   For each of the new loops, the `build()` is called for the case when there
  //   are func which need to be produced in that new loop.
  std::vector<statement_with_range> build(const func* base_f, const loop_id& at, symbol_map<var>& uncropped_subs) {
    std::vector<statement_with_range> results;
    // Build the functions computed at this loop level.
    for (auto i = order_.rbegin(); i != order_.rend(); ++i) {
      const func* f = *i;
      const auto& compute_at = compute_at_levels_.find(f);
      assert(compute_at != compute_at_levels_.end());
      std::set<var> old_candidates = candidates_for_allocation_[at];

      const auto& realize_at = realization_levels_.find(f);
      assert(realize_at != realization_levels_.end());

      if (compute_at->second == at && !f->loops().empty()) {
        // Generate the loops that we want to be explicit by recursively calling make_loop starting
        // from the outer loop.
        statement_with_range f_body = make_loop(f, f->loops().size() - 1);

        // This is a special case for the buffers which are produced and consumed inside
        // of this loop. In this case we simply wrap loop body with corresponding allocations.
        if (candidates_for_allocation_[at].size() > old_candidates.size() + 1) {
          std::vector<var> to_remove;
          for (auto b : candidates_for_allocation_[at]) {
            // We only want candidates which are not in the old_candidates list.
            if (old_candidates.count(b) > 0) continue;
            if (copy_inputs_.count(b) > 0) continue;
            std::optional<allocation_candidate>& info = allocation_info_[b];
            if (info->consumers_produced != info->deps_count) continue;

            f_body = produce_allocation(info->buffer, f_body, uncropped_subs);
            to_remove.push_back(b);
          }
          for (auto b : to_remove) {
            candidates_for_allocation_[at].erase(b);
          }
        }

        results.push_back(f_body);
      } else if (realize_at->second == at) {
        results.push_back(produce(f));
      }
    }

    return results;
  }

  void find_transitive_deps_impl(const var& v, std::set<var>& result) {
    result.insert(v);
    for (const var& p : bounds_deps_[v]) {
      // No point in visiting the node we've already seen.
      // NOTE(vksnk): this check is O(lg N) and we can make it O(1) by using
      // an additional `visited` array, but I don't think it's critical for
      // performance at all.
      if (result.count(p) > 0) continue;
      find_transitive_deps_impl(p, result);
    }
  }

  // Find transitive dependencies for a set of buffers wrt their bounds dependencies using DFS.
  void find_transitive_deps(const std::vector<var>& buffers, std::set<var>& result) {
    for (const var& v : buffers) {
      find_transitive_deps_impl(v, result);
    }
  }

public:
  pipeline_builder(
      node_context& ctx, const std::vector<buffer_expr_ptr>& inputs, const std::vector<buffer_expr_ptr>& outputs)
      : ctx(ctx), sanitizer_(ctx) {
    // Dependencies between the functions.
    std::map<const func*, std::vector<const func*>> deps;
    topological_sort(outputs, order_, deps);

    sanitizer_.external.reserve(outputs.size() + inputs.size());
    for (auto& i : outputs) {
      output_syms_[i->sym()] = i;
      sanitizer_.external.push_back(i->sym());
    }
    for (const buffer_expr_ptr& i : inputs) {
      input_syms_[i->sym()] = i;
      sanitizer_.external.push_back(i->sym());
    }
    std::sort(sanitizer_.external.begin(), sanitizer_.external.end());

    // Build a loop nest tree and computes compute_at locations when neccessary.
    compute_innermost_locations(order_, deps, compute_at_levels_, realization_levels_);

    // Compute allocation bounds.
    compute_allocation_bounds();

    // Substitute inferred bounds into user provided dims.
    substitute_buffer_dims();

    // Compute number of consumers for each of the buffers.
    compute_deps_count();
  }

  const std::vector<var>& external_symbols() const { return sanitizer_.external; }

  // Creates a loop body for a given function including all function bodies computed inside of the loops.
  // It may recursively call itself if there are nested loops, it's assumed that loops are produced
  // starting from the outer one. If base_f function is nullptr, the assumption is that we need to
  // create a "root" loop which  only will have body.
  statement_with_range make_loop(const func* base_f, int loop_index) {
    func::loop_info loop;
    loop_id here;
    if (base_f != nullptr) {
      loop = base_f->loops()[loop_index];
      here = {base_f, loop.var};
    }

    symbol_map<var> uncropped_subs;
    std::vector<statement_with_range> results = build(base_f, here, uncropped_subs);

    if (loop_index > 0) {
      statement_with_range inner_loop = make_loop(base_f, loop_index - 1);
      results.push_back(inner_loop);
    }

    statement_with_range body = lay_out_allocations(here, std::move(results), uncropped_subs);

    // Substitute references to the intermediate buffers with the 'name.uncropped' when they
    // are used as an input arguments. This does a batch substitution by replacing multiple
    // buffer names at once and relies on the fact that the same var can't be written
    // by two different funcs.
    body.body = substitute_inputs(body.body, uncropped_subs);

    if (here.root()) return body;

    // Find which buffers are used inside of the body.
    std::vector<var> buffers_used = find_buffer_dependencies(body.body);
    std::set<var> transitive_deps;
    // NOTE(vksnk): we could be more clever here and stop once we reach this loop's parent buffer(s)
    // which will avoid adding unnecessary crops which are not affected by this loop's crop_dim.
    find_transitive_deps(buffers_used, transitive_deps);

    // Add crops for the used buffers using previously inferred bounds.
    // Input syms should be the innermost.
    for (const auto& i : input_syms_) {
      var sym = i.first;
      if (!allocation_bounds_[sym]) continue;
      if (transitive_deps.count(sym) == 0) continue;
      body.body = crop_buffer::make(sym, sym, *allocation_bounds_[sym], body.body);
    }

    // Followed by intermediate buffers in the reverse topological order
    // (i.e. the outermost buffers are closer to the outputs of the pipeline).
    for (auto i = order_.rbegin(); i != order_.rend(); ++i) {
      const func* f = *i;

      if (f == base_f) {
        // Don't really need to emit buffer_crop for base_f, because they will
        // have crop_dim anyway.
        continue;
      }
      for (const func::output& o : f->outputs()) {
        const buffer_expr_ptr& b = o.buffer;
        if (!inferred_bounds_[b->sym()]) continue;
        if (transitive_deps.count(b->sym()) == 0) continue;
        body.body = crop_buffer::make(b->sym(), b->sym(), *inferred_bounds_[b->sym()], body.body);
      }
    }

    // The loop body is done, and we have an actual loop to make here. Crop the body.
    body.body = crop_for_loop(body.body, base_f, loop);
    // And make the actual loop.
    expr loop_step = sanitizer_.mutate(loop.step);
    interval_expr loop_bounds = get_loop_bounds(base_f, loop);
    // Make sure that a loop variable is unique.
    std::string loop_var_name;
    if (base_f->outputs().size() == 1) {
      loop_var_name = ctx.name(base_f->outputs()[0].sym()) + ".";
    }
    loop_var_name += ctx.name(loop.sym());
    var loop_var = ctx.insert_unique(loop_var_name);
    body.body = substitute(body.body, loop.sym(), loop_var);
    body.body = loop::make(loop_var, loop.max_workers, loop_bounds, loop_step, body.body);

    return body;
  }

  stmt define_sanitized_replacements(const stmt& body) { return sanitizer_.define_replacements(body); }

  // Add checks that the inputs are sufficient based on inferred bounds.
  stmt add_input_checks(const stmt& body) {
    std::vector<stmt> checks;
    for (const auto& i : input_syms_) {
      const std::optional<box_expr>& bounds = allocation_bounds_[i.first];
      if (!bounds) {
        // This input must have been unused, ignore it.
        continue;
      }
      for (int d = 0; d < static_cast<int>(bounds->size()); ++d) {
        checks.push_back(check::make(i.second->dim(d).min() <= (*bounds)[d].min));
        checks.push_back(check::make(i.second->dim(d).max() >= (*bounds)[d].max));
      }
    }
    return block::make(std::move(checks), std::move(body));
  }

  // Wrap the statement into make_buffer-s to define the bounds of allocations.
  stmt make_buffers(stmt body) {
    // Place all remaining constant_buffer-s.
    for (std::pair<var, buffer_expr_ptr> i : constants_) {
      body = constant_buffer::make(i.first, i.second->constant(), std::move(body));
    }
    for (auto i = order_.rbegin(); i != order_.rend(); ++i) {
      const func* f = *i;
      for (const func::output& o : f->outputs()) {
        const buffer_expr_ptr& b = o.buffer;
        const std::optional<std::vector<dim_expr>>& maybe_dims = inferred_shapes_[b->sym()];
        if (!maybe_dims) continue;
        body = make_buffer::make(b->sym(), expr(), expr(), *maybe_dims, std::move(body));
      }
    }
    return body;
  }
};

void check_buffer(const buffer_expr_ptr& b, std::vector<stmt>& checks) {
  int rank = static_cast<int>(b->rank());
  checks.push_back(check::make(expr(b->sym()) != 0));
  checks.push_back(check::make(buffer_rank(b->sym()) == rank));
}

void check_buffer_constraints(const buffer_expr_ptr& b, bool output, std::vector<stmt>& checks) {
  int rank = static_cast<int>(b->rank());
  checks.push_back(check::make(buffer_elem_size(b->sym()) == b->elem_size()));
  for (int d = 0; d < rank; ++d) {
    expr fold_factor = buffer_fold_factor(b->sym(), d);
    checks.push_back(check::make(b->dim(d).min() == buffer_min(b->sym(), d)));
    checks.push_back(check::make(b->dim(d).max() == buffer_max(b->sym(), d)));
    checks.push_back(check::make(b->dim(d).stride == buffer_stride(b->sym(), d)));
    checks.push_back(check::make(b->dim(d).fold_factor == fold_factor));
    if (output) {
      checks.push_back(check::make(or_else(fold_factor == dim::unfolded, b->dim(d).extent() <= fold_factor)));
    }
  }
}

bool is_verbose() {
  auto* s = std::getenv("SLINKY_VERBOSE");
  return (s && std::atoi(s) == 1);
}

// This function inserts calls to trace_begin and trace_end around calls and loops.
stmt inject_traces(const stmt& s, node_context& ctx) {
  class injector : public stmt_mutator {
  public:
    node_context& ctx;
    var token;
    var names_buf;
    std::vector<char> names;
    std::map<std::string, index_t> name_to_offset;

    injector(node_context& ctx) : ctx(ctx), token(ctx, "__trace_token"), names_buf(ctx, "__trace_names") {}

    // Returns a pointer to the names buffer for the argument.
    expr get_trace_arg(const std::string& arg) {
      // If we already have this name, use it. Otherwise, add it to the names buffer.
      auto i = name_to_offset.insert(std::pair<std::string, index_t>(arg, names.size()));
      if (i.second) {
        names.insert(names.end(), arg.begin(), arg.end());
        names.push_back(0);
      }
      expr args[] = {i.first->second};
      return buffer_at(names_buf, args);
    }

    expr get_trace_arg(const call_stmt* op) {
      if (!op->attrs.name.empty()) {
        return get_trace_arg(op->attrs.name);
      } else {
        return get_trace_arg("call");
      }
    }

    stmt add_trace(stmt s, expr trace_arg) {
      expr trace_begin = call::make(intrinsic::trace_begin, {trace_arg});
      expr trace_end = call::make(intrinsic::trace_end, {token});
      return let_stmt::make({{token, trace_begin}}, block::make({std::move(s), check::make(trace_end)}));
    }

    void visit(const call_stmt* op) override { set_result(add_trace(stmt(op), get_trace_arg(op))); }
    void visit(const loop* op) override {
      expr iter_name = get_trace_arg("loop " + ctx.name(op->sym) + " iteration");
      expr loop_name = get_trace_arg("loop " + ctx.name(op->sym));
      stmt body = add_trace(mutate(op->body), iter_name);
      stmt result = clone_with(op, std::move(body));
      set_result(add_trace(std::move(result), loop_name));
    }
  };

  injector m(ctx);
  stmt result = m.mutate(s);
  result = m.add_trace(result, m.get_trace_arg("pipeline"));
  buffer<char, 1> names_const_buf(m.names.data(), m.names.size());
  result = constant_buffer::make(m.names_buf, raw_buffer::make_copy(names_const_buf), result);
  return result;
}

stmt build_pipeline(node_context& ctx, const std::vector<buffer_expr_ptr>& inputs,
    const std::vector<buffer_expr_ptr>& outputs, std::vector<std::pair<var, expr>> lets, const build_options& options) {
  scoped_trace trace("build_pipeline");
  const node_context* old_context = set_default_print_context(&ctx);

  pipeline_builder builder(ctx, inputs, outputs);

  stmt result;
  result = builder.make_loop(nullptr, 0).body;
  result = builder.add_input_checks(result);
  result = builder.make_buffers(result);
  result = builder.define_sanitized_replacements(result);

  std::vector<stmt> constraint_checks;
  for (const buffer_expr_ptr& i : inputs) {
    check_buffer_constraints(i, /*output=*/false, constraint_checks);
  }
  for (const buffer_expr_ptr& i : outputs) {
    check_buffer_constraints(i, /*output=*/true, constraint_checks);
  }
  result = block::make(std::move(constraint_checks), std::move(result));

  // Add user provided lets after checking buffers are non-null and of expected rank, but before checking constraints,
  // which may use these lets.
  result = let_stmt::make(std::move(lets), result);

  std::vector<stmt> buffer_checks;
  for (const buffer_expr_ptr& i : inputs) {
    check_buffer(i, buffer_checks);
  }
  for (const buffer_expr_ptr& i : outputs) {
    check_buffer(i, buffer_checks);
  }
  result = block::make(std::move(buffer_checks), std::move(result));

  result = slide_and_fold_storage(result, ctx);
  result = deshadow(result, builder.external_symbols(), ctx);
  result = simplify(result);

  // Try to reuse buffers and eliminate copies where possible.
  if (!options.no_alias_buffers) {
    result = alias_copies(result, ctx, inputs, outputs);
    result = alias_in_place(result, outputs);
  }

  // `evaluate` currently can't handle `copy_stmt`, so this is required.
  result = implement_copies(result, ctx);

  // `implement_copies` adds shadowed declarations, remove them before simplifying.
  result = deshadow(result, builder.external_symbols(), ctx);
  result = simplify(result);

  result = fuse_siblings(result);

  if (options.no_checks) {
    result = recursive_mutate<check>(result, [](const check* op) { return stmt(); });
    // Simplify again, in case there are lets that the checks used that are now dead.
    result = simplify(result);
  }

  result = insert_early_free(result);

  if (options.trace) {
    result = inject_traces(result, ctx);
  }

  // This pass adds closures around parallel loop bodies, any following passes need to maintain this closure.
  result = optimize_symbols(result, ctx);

  result = canonicalize_nodes(result);

  if (is_verbose()) {
    std::cout << result << std::endl;
  }

  set_default_print_context(old_context);

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

}  // namespace

pipeline build_pipeline(node_context& ctx, std::vector<var> args, const std::vector<buffer_expr_ptr>& inputs,
    const std::vector<buffer_expr_ptr>& outputs, std::vector<std::pair<var, expr>> lets, const build_options& options) {
  stmt body = build_pipeline(ctx, inputs, outputs, lets, options);
  pipeline p;
  p.args = args;
  p.inputs = vars(inputs);
  p.outputs = vars(outputs);
  p.body = std::move(body);
  return p;
}

pipeline build_pipeline(node_context& ctx, const std::vector<buffer_expr_ptr>& inputs,
    const std::vector<buffer_expr_ptr>& outputs, const build_options& options) {
  return build_pipeline(ctx, {}, inputs, outputs, {}, options);
}

}  // namespace slinky