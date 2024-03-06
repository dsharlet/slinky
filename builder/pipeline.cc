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

buffer_expr::buffer_expr(symbol_id sym, index_t elem_size, std::size_t rank)
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

buffer_expr::buffer_expr(symbol_id sym, const raw_buffer* buffer)
    : sym_(sym), elem_size_(buffer->elem_size), producer_(nullptr), constant_(buffer) {
  dims_.reserve(buffer->rank);

  for (index_t d = 0; d < static_cast<index_t>(buffer->rank); ++d) {
    expr min = buffer->dims[d].min();
    expr max = buffer->dims[d].max();
    expr stride = buffer->dims[d].stride();
    expr fold_factor = buffer->dims[d].fold_factor();
    dims_.push_back({slinky::bounds(min, max), stride, fold_factor});
  }
}

buffer_expr_ptr buffer_expr::make(symbol_id sym, index_t elem_size, std::size_t rank) {
  return buffer_expr_ptr(new buffer_expr(sym, elem_size, rank));
}

buffer_expr_ptr buffer_expr::make(node_context& ctx, const std::string& sym, index_t elem_size, std::size_t rank) {
  return buffer_expr_ptr(new buffer_expr(ctx.insert_unique(sym), elem_size, rank));
}

buffer_expr_ptr buffer_expr::make(symbol_id sym, const raw_buffer* buffer) {
  return buffer_expr_ptr(new buffer_expr(sym, buffer));
}
buffer_expr_ptr buffer_expr::make(node_context& ctx, const std::string& sym, const raw_buffer* buffer) {
  return buffer_expr_ptr(new buffer_expr(ctx.insert_unique(sym), buffer));
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
    for (int d = 0; d < new_crop.size(); d++) {
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

class pipeline_builder {
  // We're going to incrementally build the body, starting at the end of the pipeline and adding
  // producers as necessary.
  std::set<buffer_expr_ptr> to_produce;
  std::list<buffer_expr_ptr> to_allocate;
  std::set<buffer_expr_ptr> produced, consumed;
  std::set<buffer_expr_ptr> allocated;

  stmt result;

  const std::vector<const func*>& order_;
  const std::map<const func*, loop_id>& compute_at_levels_;

public:
  pipeline_builder(const std::vector<buffer_expr_ptr>& inputs,
                    const std::vector<buffer_expr_ptr>& outputs,
                    std::set<buffer_expr_ptr>& constants,
                    const std::vector<const func*>& order,
                    const std::map<const func*, loop_id>& compute_at_levels) :
                    order_(order), compute_at_levels_(compute_at_levels) {
    // To start with, we need to produce the outputs.
    for (auto& i : outputs) {
      to_produce.insert(i);
      allocated.insert(i);
    }
    for (auto& i : inputs) {
      produced.insert(i);
    }

    // Find all the buffers we need to produce.
    while (true) {
      std::set<buffer_expr_ptr> produce_next;
      for (const buffer_expr_ptr& i : to_produce) {
        if (!i->producer()) {
          // Must be an input.
          continue;
        }

        for (const func::input& j : i->producer()->inputs()) {
          if (!to_produce.count(j.buffer)) {
            if (j.buffer->constant()) {
              constants.insert(j.buffer);
            } else {
              produce_next.insert(j.buffer);
            }
          }
        }
      }
      if (produce_next.empty()) break;

      to_produce.insert(produce_next.begin(), produce_next.end());
    }
  }

  // f can be called if it doesn't have an output that is consumed by a not yet produced buffer's producer.
  bool can_produce(const func* f) const {
    for (const buffer_expr_ptr& p : to_produce) {
      if (produced.count(p)) {
        // This buffer is already produced.
        continue;
      }
      if (!p->producer()) {
        // Must be an input.
        continue;
      }
      if (p->producer() == f) {
        // This is the producer we are considering now.
        continue;
      }
      for (const func::output& o : f->outputs()) {
        for (const func::input& i : p->producer()->inputs()) {
          if (i.buffer == o.buffer) {
            // f produces a buffer that one of the other yet to be produced buffers needs as an
            // input.
            return false;
          }
        }
      }
    }
    return true;
  }

  // Find the func f to run next. This is the func that produces a buffer we need that we have not
  // yet produced, and all the buffers produced by f are ready to be consumed.
  const func* find_next_producer(const loop_id& at = loop_id()) const {
    for (const buffer_expr_ptr& i : to_produce) {
      if (produced.count(i)) continue;

      if (!i->producer()) {
        // This is probably an input.
        continue;
      }

      if (!can_produce(i->producer())) {
        // This isn't ready to be produced yet.
        continue;
      }

      if (i->producer()->compute_at()) {
        if (!(*i->producer()->compute_at() == at)) {
          // This shouldn't be computed here.
          continue;
        }
      } else if (!at.root()) {
        // By default, we want to compute everything as soon as it is consumed, so it ends up in the innermost loop
        // possible. But if we do this blindly, then unrelated pipelines that aren't consumed inside these loops at
        // all will get generated here.
        if (!consumed.count(i)) {
          continue;
        }
      }

      // We're in the right place, and the func is ready to be computed!
      return i->producer();
    }
    return nullptr;
  }

  bool complete() const { return produced.size() == to_produce.size(); }

  // Add crops to the inputs of f, using buffer intrinsics to get the bounds of the output.
  stmt add_input_crops(stmt result, const func* f) {
    // Find the bounds of the outputs required in each dimension. This is the union of the all the intervals from each
    // output associated with a particular dimension.
    std::cout << "add_input_crops" << std::endl;
    assert(!f->outputs().empty());
    symbol_map<expr> output_mins, output_maxs;
    for (const func::output& o : f->outputs()) {
      for (std::size_t d = 0; d < o.dims.size(); ++d) {
        expr dim_min = o.buffer->dim(d).min();
        expr dim_max = o.buffer->dim(d).max();
        std::optional<expr>& min = output_mins[o.dims[d]];
        std::optional<expr>& max = output_maxs[o.dims[d]];
        min = min ? slinky::min(*min, dim_min) : dim_min;
        max = max ? slinky::max(*max, dim_max) : dim_max;
      }
    }
    // Use the output bounds, and the bounds expressions of the inputs, to determine the bounds required of the input.
    for (const func::input& i : f->inputs()) {
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
        std::cout << min << " " << max << std::endl;
        // The bounds may have been negated.
        crop[d] = simplify(slinky::bounds(min, max) | slinky::bounds(max, min));
      }
      // We want to take the union of this crop with any existing crop of this buffer.
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

  stmt make_allocations(stmt body, const loop_id& at = loop_id()) {
    for (const buffer_expr_ptr& i : to_allocate) {
      if (allocated.count(i)) continue;
      // TODO: I think this check is technically OK, but it is sloppy and allows incorrect explicit schedules (e.g. if
      // i->store_at() was set, but we didn't find the storage location).
      if (at.root() || (i->store_at() && *i->store_at() == at)) {
        body = allocate::make(i->sym(), i->storage(), i->elem_size(), i->dims(), body);
        allocated.insert(i);
      }
    }
    for (const buffer_expr_ptr& i : allocated) {
      to_allocate.remove(i);
    }
    return body;
  }

  stmt make_producers(const loop_id& at, const func* f) {
    if (const func* next = find_next_producer(at)) {
      stmt result = produce(next, at);
      result = add_input_crops(result, f);
      result = block::make({make_producers(at, next), std::move(result)});
      return result;
    } else {
      return {};
    }
  }

  stmt make_loop(stmt body, const func* f, const func::loop_info& loop = func::loop_info()) {
    loop_id here = {f, loop.var};
    // Before making the loop, we need to produce any funcs that should be produced here.
    body = block::make({make_producers(here, f), body});

    // Make any allocations that should be here.
    body = make_allocations(body, here);

    if (loop.defined()) {
      // The loop body is done, and we have an actual loop to make here. Crop the body.
      body = crop_for_loop(body, f, loop);
      // And make the actual loop.
      body = loop::make(loop.sym(), loop.mode, get_loop_bounds(f, loop), loop.step, body);
    }
    return body;
  }

  // Producing a func means:
  // - Generating a call to the function f
  // - Wrapping f with the loops it wanted to be explicit
  // - Producing all the buffers that f consumes (recursively).
  stmt produce(const func* f, const loop_id& current_at = loop_id()) {
    stmt result = f->make_call();
    result = add_input_crops(result, f);
    for (const func::output& i : f->outputs()) {
      produced.insert(i.buffer);
      if (!allocated.count(i.buffer)) {
        to_allocate.push_front(i.buffer);
      }
    }
    for (const func::input& i : f->inputs()) {
      consumed.insert(i.buffer);
    }

    // Generate the loops that we want to be explicit.
    for (const auto& loop : f->loops()) {
      result = make_loop(result, f, loop);
    }

    // Try to make any other producers needed here.
    result = block::make({make_producers(current_at, f), result});
    // std::cout << "end of produce() :" << result << std::endl;
    return result;
  }

  stmt make_loop_new(stmt body, const func* f, const func::loop_info& loop = func::loop_info(), bool is_inner_loop = false) {
    loop_id here = {f, loop.var};
    // // Before making the loop, we need to produce any funcs that should be produced here.
    // body = block::make({make_producers(here, f), body});

    // // Make any allocations that should be here.
    // body = make_allocations(body, here);
    // body = block::make(build_new(here);

    body = build_new(body, f, here);
  
    std::cout << "make_loop_new" << std::endl;
    if (loop.defined()) {
      // if (is_inner_loop) {
      //   body = add_input_crops(body, f);
      // }
      // The loop body is done, and we have an actual loop to make here. Crop the body.
      body = crop_for_loop(body, f, loop);
      // And make the actual loop.
      body = loop::make(loop.sym(), loop.mode, get_loop_bounds(f, loop), loop.step, body);
    }
    return body;
  }
  
  stmt produce_new(const func* f) {
    stmt result = f->make_call();
    if (f->loops().empty()) {
      result = add_input_crops(result, f);
    }
    // for (const func::output& i : f->outputs()) {
    //   produced.insert(i.buffer);
    //   if (!allocated.count(i.buffer)) {
    //     to_allocate.push_front(i.buffer);
    //   }
    // }
    // for (const func::input& i : f->inputs()) {
    //   consumed.insert(i.buffer);
    // }

    // Generate the loops that we want to be explicit.
    bool is_inner_loop = true;
    for (const auto& loop : f->loops()) {
      result = make_loop_new(result, f, loop, is_inner_loop);
      is_inner_loop = false;
    }

    // // Try to make any other producers needed here.
    // result = block::make({make_producers(current_at, f), result});

    return result;
  }

  stmt build_new(stmt body, const func* base_f, const loop_id& at) {
    stmt result;
    std::vector<stmt> produced_stmt;
    for (int ix = order_.size() - 1; ix >= 0; ix--) {
      const auto& f = order_[ix];
      const auto& compute_at = compute_at_levels_.find(f);
      assert(compute_at != compute_at_levels_.end());
      if (compute_at->second == at) {
        std::cout << "Producing " << f->name() << std::endl;
        // produced_stmt.push_back(produce_new(f));
        if (result.defined()) {
          result = add_input_crops(result, f);
        }
        result = block::make({result, produce_new(f)});
      }
    }
    
    result = block::make({result, body});

    // for (int ix = 0; ix < order_.size(); ix++) {
    //   const auto& f = order_[ix];
    //   const auto& compute_at = compute_at_levels_.find(f);
    //   assert(compute_at != compute_at_levels_.end());
    //   if (compute_at->second == at) {
    //     result = add_input_crops(result, f);
    //   }
    // }

    if (base_f) {
      result = add_input_crops(result, base_f);
    }

    for (int ix = order_.size() - 1; ix >= 0; ix--) {
      const auto& f = order_[ix];
      for (const auto& o: f->outputs()) {
        const auto& b = o.buffer;
        if (allocated.count(b)) continue;

        if ((b->store_at() && *b->store_at() == at) || (!b->store_at() && at.root())) {
          std::cout << "Storing " << f->name() << std::endl;
          result = allocate::make(b->sym(), b->storage(), b->elem_size(), b->dims(), result);
          // allocated.insert(b);
        }
      }
    }

    return result;
  }
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

void topological_sort_impl(const func* f, std::set<const func *>& visited, 
                            std::vector<const func *>& order, 
                            std::map<const func*, std::vector<const func*>>& deps) {
  std::cout << ">> Visiting - " << f->name() << std::endl;
  for (const auto& i: f->inputs()) {
    const auto& input = i.buffer;
    if (input->constant()) {
      continue;
    }
    if (!input->producer()) {
      continue;
    }
    // Record that f is consumer of input->producer.
    deps[input->producer()].push_back(f);

    if(visited.count(input->producer()) > 0) {
      continue;
    }
    topological_sort_impl(input->producer(), visited, order, deps);
  }
  std::cout << ">> Pushing into visited " << f->name() << std::endl;
  visited.insert(f);
  order.push_back(f);
}

void topological_sort(const std::vector<buffer_expr_ptr>& outputs,
                      std::vector<const func *>& order,
                      std::map<const func*, std::vector<const func*>>& deps) {
  std::cout << ">>> topological_sort - " << std::endl;
  std::set<const func* > visited;
  for(const auto& i: outputs) {
    topological_sort_impl(i->producer(), visited, order, deps);
  }
  std::reverse(order.begin(), order.end());
  std::cout << "Final order: " << std::endl;
  for (const auto& f: order) {
    std::cout << "@@ " << f->name() << std::endl;
    if (f->compute_at()) {
      if (f->compute_at()->root()) {
         std::cout << "compute_at root" << std::endl;
      } else {
        std::cout << "compute_at " << f->compute_at()->func->name() << std::endl;
      }
    }
    
    for(const auto& l: f->loops()) {
      std::cout << "$" << l.sym() << std::endl;
    }
  }
  std::cout << "Final deps: " << std::endl;
  for (const auto& p: deps) {
    std::cout << "## " << p.first->name() << std::endl;
  }
}

struct loop_tree_node {
  int parent_index = -1;
  loop_id loop;
};

int lca(const std::vector<loop_tree_node>& loop_tree, const std::vector<int>& parent_ids) {
  std::cout << "!! LCA: " << std::endl;
  std::vector<std::vector<int>> pathes_to_root(parent_ids.size());
  for (int ix = 0; ix < (int)parent_ids.size(); ix++) {
    int parent_id = parent_ids[ix];
    pathes_to_root[ix].push_back(parent_id);
    std::cout << "!! start processing: " << parent_id << std::endl;
    while (parent_id > 0) {
      parent_id = loop_tree[parent_id].parent_index;
      pathes_to_root[ix].push_back(parent_id);
      std::cout << "!! processing: " << parent_id << std::endl;
    }
    std::reverse(pathes_to_root[ix].begin(), pathes_to_root[ix].end());
  }

  int max_match = pathes_to_root[0].size() - 1;
  std::cout << "Initial max_match - " << max_match << std::endl;
  for (int ix = 1; ix < (int)pathes_to_root.size(); ix++) {
    max_match = std::min(max_match, (int)pathes_to_root.size() - 1);
    for (int iy = 0; iy <= max_match; iy++) {
      std::cout << "Pathes: " << pathes_to_root[ix][iy] << " " << pathes_to_root[0][iy] << std::endl;
      if (pathes_to_root[ix][iy] != pathes_to_root[0][iy]) {
        max_match = iy - 1;
        std::cout << "Updated max_match - " << max_match << std::endl;
        break;
      }
    }
  }

  std::cout << "Max match up to " << max_match << std::endl;
  return pathes_to_root[0][max_match];
}

void compute_innermost_locations(const std::vector<const func*>& order, 
                                  const std::map<const func*, std::vector<const func*>> deps, 
                                  std::map<const func*, loop_id>& compute_at_levels) {
  std::cout << ">>> Build tree:" << std::endl;
  // A tree which stores loop nest.
  std::vector<loop_tree_node> loop_tree;
  // Mapping between function and it's most innermost location.
  std::map<const func*, int> func_to_loop_tree;
  // Push root loop.
  loop_tree.push_back({-1, loop_id()});
  
  // Iterate over functions in topological order starting from the output and build a loop nest tree.
  for (const auto& f: order) {
    std::cout << "@@ Processing " << f->name() << std::endl;
    if (f->compute_at()) {
      if (f->compute_at()->root()) {
         std::cout << "compute_at root" << std::endl;
      } else {
        std::cout << "compute_at " << f->compute_at()->func->name() << std::endl;
      }
    }
    const auto& p = deps.find(f);
    if (p != deps.end()) {
      assert(!p->second.empty());
      // Check all of the consumers and find their innermost locations.
      std::vector<int> parent_ids;
      for (const auto& d: p->second) {
        std::cout << "produces for " << d->name() << std::endl;
        const auto& node = func_to_loop_tree.find(d);
        assert(node != func_to_loop_tree.end());
        parent_ids.push_back(node->second);
        std::cout << "loop tree node " << node->second << std::endl;
      }
      int parent_id = -1;
      if (parent_ids.size() == 1) {
        // We have just one consumer, so use it innermost location.
        parent_id = parent_ids[0];
      } else {
        std::cout << "Multiple consumers, need to call LCA" << std::endl;
        // There are multiple consumers, so we need to find least common ancestor
        // of their innermost locations.
        parent_id = lca(loop_tree, parent_ids);
      }
      assert(parent_id != -1);
      
      compute_at_levels[f] = loop_tree[parent_id].loop;

      // Add loops for this function to the loop nest.
      // for (const auto& l: f->loops()) {
      for (int i = f->loops().size() - 1; i >= 0; i--) {
        const auto& l = f->loops()[i];
        std::cout << "Adding a loop " << l.sym() << std::endl;
        loop_tree.push_back({parent_id, {f, l.var}});
        parent_id = loop_tree.size() - 1;
      }
      func_to_loop_tree[f] = parent_id;
    } else {
      // This is an output so should be computed at root.
      std::cout << "This is a root node" << std::endl;
      int parent_id = 0;
      compute_at_levels[f] = loop_id();
      // Add loops for this function to the loop nest.
      // for (const auto& l: f->loops()) {
      for (int i = f->loops().size() - 1; i >= 0; i--) {
        const auto& l = f->loops()[i];
        std::cout << "Adding a loop " << l.sym() << std::endl;
        loop_tree.push_back({parent_id, {f, l.var}});
        parent_id = loop_tree.size() - 1;
      }
      func_to_loop_tree[f] = parent_id;
    }
  }
}

stmt build_pipeline(node_context& ctx, const std::vector<buffer_expr_ptr>& inputs,
    const std::vector<buffer_expr_ptr>& outputs, std::set<buffer_expr_ptr>& constants, const build_options& options) {
  
  // Topologically sorted functions.
  std::vector<const func*> order;
  // Dependencies between the functions.
  std::map<const func*, std::vector<const func*>> deps;
  topological_sort(outputs, order, deps);

  // Build a loop nest tree and computes compute_at locations when neccessary.
  std::map<const func*, loop_id> compute_at_levels;
  compute_innermost_locations(order, deps, compute_at_levels);

  std::cout << "Final compute_at locations: " << std::endl;
  for (const auto& p: compute_at_levels) {
    std::cout << p.first->name() << " -> ";
    if (p.second.root()) {
      std::cout << "<root>";
    } else {
      std::cout << "<" << p.second.func->name() << ", " << p.second.sym() << ">";
    }
    std::cout << std::endl;
  }

  pipeline_builder builder(inputs, outputs, constants, order, compute_at_levels);

  stmt new_result;
  new_result = builder.build_new(new_result, nullptr, loop_id());
  std::cout << "+++ New loop initital loop nest: \n" << std::tie(new_result, ctx) << std::endl;
  stmt result;

  while (!builder.complete()) {
    // Find a buffer to produce.
    const func* f = builder.find_next_producer();

    // Call the producer.
    if (!f) {
      // TODO: Make a better error here.
      std::cerr << "Problem in dependency graph" << std::endl;
      std::abort();
    }

    stmt produce_f = builder.produce(f);
    produce_f = builder.make_allocations(produce_f);
    result = block::make({std::move(result), std::move(produce_f)});
  }
  std::cout << "+++ Initital loop nest: \n" << std::tie(result, ctx) << std::endl;
  result = new_result;
  // Add checks that the buffer constraints the user set are satisfied.
  std::vector<stmt> checks;
  for (const buffer_expr_ptr& i : inputs) {
    add_buffer_checks(i, /*output=*/false, checks);
  }
  for (const buffer_expr_ptr& i : outputs) {
    add_buffer_checks(i, /*output=*/true, checks);
  }
  result = block::make(std::move(checks), std::move(result));

  std::vector<symbol_id> input_syms;
  input_syms.reserve(inputs.size());
  for (const buffer_expr_ptr& i : inputs) {
    input_syms.push_back(i->sym());
  }
  for (const buffer_expr_ptr& i : constants) {
    input_syms.push_back(i->sym());
  }
  result = infer_bounds(result, ctx, input_syms);
  // std::cout << "After infer_bounds\n" << result << std::endl;

  result = simplify(result);

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

  std::cout << std::tie(result, ctx) << std::endl;

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
    const std::vector<buffer_expr_ptr>& outputs, const build_options& options) {
  std::set<buffer_expr_ptr> constants;
  stmt body = build_pipeline(ctx, inputs, outputs, constants, options);
  return pipeline(std::move(args), vars(inputs), vars(outputs), std::move(body));
}

pipeline build_pipeline(node_context& ctx, const std::vector<buffer_expr_ptr>& inputs,
    const std::vector<buffer_expr_ptr>& outputs, const build_options& options) {
  return build_pipeline(ctx, {}, inputs, outputs, options);
}

}  // namespace slinky