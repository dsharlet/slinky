#include "builder/func.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <list>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "runtime/evaluate.h"
#include "runtime/expr.h"
#include "builder/infer_bounds.h"
#include "builder/node_mutator.h"
#include "builder/optimizations.h"
#include "runtime/pipeline.h"
#include "runtime/print.h"
#include "builder/simplify.h"
#include "builder/substitute.h"

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
    dims_.push_back({bounds(min, max), stride, fold_factor});
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

func::func(callable impl, std::vector<input> inputs, std::vector<output> outputs)
    : impl_(std::move(impl)), inputs_(std::move(inputs)), outputs_(std::move(outputs)) {
  add_this_to_buffers();
}

func::func(input input, output out, std::vector<char> padding) : func(nullptr, {std::move(input)}, {std::move(out)}) {
  padding_ = std::move(padding);
}

func::func(std::vector<input> inputs, output out) : func(nullptr, std::move(inputs), {std::move(out)}) {}

func::func(func&& m) { *this = std::move(m); }
func& func::operator=(func&& m) {
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
    return call_stmt::make(impl_, std::move(inputs), std::move(outputs));
  } else {
    assert(padding_.empty() || inputs_.size() == 1);
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
      copies.push_back(copy_stmt::make(input.sym(), src_x, outputs_[0].sym(), dst_x, padding_));
    }
    return block::make(copies);
  }
}

namespace {

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

public:
  pipeline_builder(const std::vector<buffer_expr_ptr>& inputs, const std::vector<buffer_expr_ptr>& outputs,
      std::set<buffer_expr_ptr>& constants) {
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
      box_expr crop(i.buffer->rank());
      for (int d = 0; d < static_cast<int>(crop.size()); ++d) {
        // TODO (https://github.com/dsharlet/slinky/issues/21): We may have been given bounds on the input that are
        // smaller than the bounds implied by the output, e.g. in the case of copy with padding.
        expr min = substitute(i.bounds[d].min, output_mins);
        expr max = substitute(i.bounds[d].max, output_maxs);
        // The bounds may have been negated.
        crop[d] = simplify(slinky::bounds(min, max) | slinky::bounds(max, min));
      }
      result = crop_buffer::make(i.sym(), crop, result);
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
      result = block::make({make_producers(at, next), result});
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
    return result;
  }
};

void add_buffer_checks(const buffer_expr_ptr& b, bool output, std::vector<stmt>& checks) {
  int rank = static_cast<int>(b->rank());
  expr buf_var = variable::make(b->sym());
  checks.push_back(check::make(buf_var != 0));
  // TODO: Maybe this check is overzealous (https://github.com/dsharlet/slinky/issues/17).
  checks.push_back(check::make(buffer_rank(buf_var) == rank));
  checks.push_back(check::make(buffer_base(buf_var) != 0));
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

stmt build_pipeline(node_context& ctx, const std::vector<buffer_expr_ptr>& inputs,
    const std::vector<buffer_expr_ptr>& outputs, std::set<buffer_expr_ptr>& constants,
    const build_options& options) {
  pipeline_builder builder(inputs, outputs, constants);

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
    result = block::make({result, produce_f});
  }
  // Add checks that the buffer constraints the user set are satisfied.
  std::vector<stmt> checks;
  for (const buffer_expr_ptr& i : inputs) {
    add_buffer_checks(i, /*output=*/false, checks);
  }
  for (const buffer_expr_ptr& i : outputs) {
    add_buffer_checks(i, /*output=*/true, checks);
  }
  result = block::make(block::make(checks), result);

  std::vector<symbol_id> input_syms;
  input_syms.reserve(inputs.size());
  for (const buffer_expr_ptr& i : inputs) {
    input_syms.push_back(i->sym());
  }
  for (const buffer_expr_ptr& i : constants) {
    input_syms.push_back(i->sym());
  }
  result = infer_bounds(result, ctx, input_syms);

  result = fix_buffer_races(result);

  result = simplify(result);

  if (options.no_checks) {
    class remove_checks : public node_mutator {
    public:
      void visit(const check* op) override { set_result(stmt()); }
    };

    result = remove_checks().mutate(result);
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