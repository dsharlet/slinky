#include "pipeline.h"

#include <cassert>
#include <iostream>
#include <list>
#include <map>
#include <set>

#include "evaluate.h"
#include "infer_bounds.h"
#include "node_mutator.h"
#include "optimizations.h"
#include "print.h"
#include "simplify.h"
#include "substitute.h"

namespace slinky {

buffer_expr::buffer_expr(symbol_id sym, index_t elem_size, std::size_t rank)
    : sym_(sym), elem_size_(elem_size), producer_(nullptr) {
  dims_.reserve(rank);
  auto var = variable::make(sym);
  for (index_t i = 0; i < static_cast<index_t>(rank); ++i) {
    interval_expr bounds = buffer_bounds(var, i);
    expr stride = buffer_stride(var, i);
    expr fold_factor = buffer_fold_factor(var, i);
    dims_.emplace_back(bounds, stride, fold_factor);
  }
}

buffer_expr::buffer_expr(const raw_buffer& c) { 
  dims_.reserve(c.rank);

  for (index_t d = 0; d < static_cast<index_t>(c.rank); ++d) {
    expr min = c.dims[d].min();
    expr max = c.dims[d].max();
    expr stride = c.dims[d].stride();
    expr fold_factor = c.dims[d].fold_factor();
    dims_.emplace_back(bounds(min, max), stride, fold_factor);
  }
}

buffer_expr_ptr buffer_expr::make(symbol_id sym, index_t elem_size, std::size_t rank) {
  return buffer_expr_ptr(new buffer_expr(sym, elem_size, rank));
}

buffer_expr_ptr buffer_expr::make(node_context& ctx, const std::string& sym, index_t elem_size, std::size_t rank) {
  return buffer_expr_ptr(new buffer_expr(ctx.insert(sym), elem_size, rank));
}

buffer_expr_ptr buffer_expr::make(const raw_buffer& buffer) { return buffer_expr_ptr(new buffer_expr(buffer)); }

void buffer_expr::add_producer(func* f) {
  assert(producer_ == nullptr);
  producer_ = f;
}

void buffer_expr::add_consumer(func* f) {
  assert(std::find(consumers_.begin(), consumers_.end(), f) == consumers_.end());
  consumers_.push_back(f);
}

func::func(callable impl, std::vector<input> inputs, std::vector<output> outputs)
    : impl_(std::move(impl)), inputs_(std::move(inputs)), outputs_(std::move(outputs)) {
  for (auto& i : inputs_) {
    i.buffer->add_consumer(this);
  }
  for (auto& i : outputs_) {
    i.buffer->add_producer(this);
  }
}

func::func(std::vector<input> inputs, output out, std::vector<char> padding)
    : func(nullptr, std::move(inputs), {std::move(out)}) {
  padding_ = std::move(padding);
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
    // TODO: We should be able to handle copies from multiple inputs.
    assert(inputs_.size() == 1);
    assert(outputs_.size() == 1);
    std::vector<expr> src_x;
    std::vector<symbol_id> dst_x;
    for (const interval_expr& i : inputs_[0].bounds) {
      assert(match(i.min, i.max));
      src_x.push_back(i.min);
    }
    for (const var& i : outputs_[0].dims) {
      dst_x.push_back(i.sym());
    }
    return copy_stmt::make(inputs_[0].sym(), src_x, outputs_[0].sym(), dst_x, padding_);
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
  std::set<buffer_expr_ptr> produced;
  std::set<buffer_expr_ptr> allocated;

  struct crop_info {
    int dim;
    interval_expr bounds;
  };

  stmt result;

public:
  pipeline_builder(const std::vector<buffer_expr_ptr>& inputs, const std::vector<buffer_expr_ptr>& outputs) {
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
            produce_next.insert(j.buffer);
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

      if (!(i->producer()->compute_at() == at)) {
        // This shouldn't be computed here.
        continue;
      }

      // We're in the right place, and the func is ready to be computed!
      return i->producer();
    }
    return nullptr;
  }

  bool complete() const { return produced.size() == to_produce.size(); }

  // Add crops to the inputs of f, using buffer intrinsics to get the bounds of the output.
  stmt add_input_crops(stmt result, const func* f) {
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
    for (const func::input& i : f->inputs()) {
      box_expr crop(i.buffer->rank());
      for (int d = 0; d < static_cast<int>(crop.size()); ++d) {
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
          interval_expr bounds = slinky::bounds(loop.var, min(loop.var + loop.step - 1, loop_max));
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
      if (i->store_at() == at && !allocated.count(i)) {
        body = allocate::make(i->storage(), i->sym(), i->elem_size(), i->dims(), body);
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
      body = loop::make(loop.sym(), get_loop_bounds(f, loop), loop.step, body);
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

    // Generate the loops that we want to be explicit.
    for (const auto& loop : f->loops()) {
      result = make_loop(result, f, loop);
    }

    // Try to make any other producers needed here.
    result = block::make({make_producers(current_at, f), result});
    return result;
  }
};

void add_buffer_checks(const buffer_expr_ptr& b, std::vector<stmt>& checks) {
  int rank = static_cast<int>(b->rank());
  expr buf_var = variable::make(b->sym());
  checks.push_back(check::make(buf_var != 0));
  // TODO: Maybe this check is overzealous (https://github.com/dsharlet/slinky/issues/17).
  checks.push_back(check::make(buffer_rank(buf_var) == rank));
  checks.push_back(check::make(buffer_base(buf_var) != 0));
  for (int d = 0; d < rank; ++d) {
    checks.push_back(check::make(b->dim(d).min() == buffer_min(buf_var, d)));
    checks.push_back(check::make(b->dim(d).max() == buffer_max(buf_var, d)));
    checks.push_back(check::make(b->dim(d).stride == buffer_stride(buf_var, d)));
    checks.push_back(check::make(b->dim(d).fold_factor == buffer_fold_factor(buf_var, d)));
  }
}

stmt build_pipeline(node_context& ctx, const std::vector<buffer_expr_ptr>& inputs,
    const std::vector<buffer_expr_ptr>& outputs, const build_options& options) {
  pipeline_builder builder(inputs, outputs);

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

    result = builder.produce(f);
    result = builder.make_allocations(result);
  }

  std::vector<symbol_id> input_syms;
  input_syms.reserve(inputs.size());
  for (const buffer_expr_ptr& i : inputs) {
    input_syms.push_back(i->sym());
  }
  result = infer_bounds(result, ctx, input_syms);

  // Add checks that the buffer constraints the user set are satisfied.
  std::vector<stmt> checks;
  for (const buffer_expr_ptr& i : inputs) {
    add_buffer_checks(i, checks);
  }
  for (const buffer_expr_ptr& i : outputs) {
    add_buffer_checks(i, checks);
  }
  result = block::make(block::make(checks), result);

  result = simplify(result);

  result = implement_copies(result, ctx);

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

}  // namespace

pipeline::pipeline(node_context& ctx, std::vector<var> args, std::vector<buffer_expr_ptr> inputs,
    std::vector<buffer_expr_ptr> outputs, const build_options& options)
    : inputs_(std::move(inputs)), outputs_(std::move(outputs)) {
  for (const var& i : args) {
    args_.push_back(i.sym());
  }
  body = build_pipeline(ctx, inputs_, outputs_, options);
}

pipeline::pipeline(node_context& ctx, std::vector<buffer_expr_ptr> inputs, std::vector<buffer_expr_ptr> outputs,
    const build_options& options)
    : pipeline(ctx, {}, std::move(inputs), std::move(outputs), options) {}

index_t pipeline::evaluate(scalars args, buffers inputs, buffers outputs, eval_context& ctx) const {
  assert(args.size() == args_.size());
  assert(inputs.size() == inputs_.size());
  assert(outputs.size() == outputs_.size());

  for (std::size_t i = 0; i < args.size(); ++i) {
    ctx[args_[i]] = args[i];
  }
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    ctx[inputs_[i]->sym()] = reinterpret_cast<index_t>(inputs[i]);
  }
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    ctx[outputs_[i]->sym()] = reinterpret_cast<index_t>(outputs[i]);
  }

  return slinky::evaluate(body, ctx);
}

index_t pipeline::evaluate(buffers inputs, buffers outputs, eval_context& ctx) const {
  return evaluate({}, inputs, outputs, ctx);
}

index_t pipeline::evaluate(scalars args, buffers inputs, buffers outputs) const {
  eval_context ctx;
  return evaluate(args, inputs, outputs, ctx);
}

index_t pipeline::evaluate(buffers inputs, buffers outputs) const {
  eval_context ctx;
  return evaluate(scalars(), inputs, outputs, ctx);
}

}  // namespace slinky