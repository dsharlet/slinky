#include "pipeline.h"

#include <cassert>
#include <iostream>
#include <set>
#include <map>

#include "evaluate.h"
#include "infer_bounds.h"
#include "node_mutator.h"
#include "print.h"
#include "simplify.h"
#include "substitute.h"
#include "optimizations.h"

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

buffer_expr_ptr buffer_expr::make(symbol_id sym, index_t elem_size, std::size_t rank) {
  return buffer_expr_ptr(new buffer_expr(sym, elem_size, rank));
}

buffer_expr_ptr buffer_expr::make(node_context& ctx, const std::string& sym, index_t elem_size, std::size_t rank) {
  return buffer_expr_ptr(new buffer_expr(ctx.insert(sym), elem_size, rank));
}

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

namespace {

class pipeline_builder {
  // We're going to incrementally build the body, starting at the end of the pipeline and adding
  // producers as necessary.
  std::set<buffer_expr_ptr> to_produce;
  std::set<buffer_expr_ptr> to_allocate;
  std::set<buffer_expr_ptr> produced;
  std::set<buffer_expr_ptr> allocated;

  struct crop_info {
    int dim;
    interval_expr bounds;
  };

  using scope_crops = std::map<symbol_id, std::vector<crop_info>>;

  std::vector<scope_crops> crops;

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
  const func* find_next_producer(const func* in = nullptr, const var& loop = var()) const {
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

      if (!in) {
        assert(i->producer()->compute_at().f == nullptr);
        return i->producer();
      }

      const loop_id& at = i->producer()->compute_at();
      if (at.f == in && at.loop.sym() == loop.sym()) return i->producer();
    }
    return nullptr;
  }

  bool complete() const { return produced.size() == to_produce.size(); }

  stmt add_crops(stmt result, const func* f) {
    for (const auto& s : crops) {
      for (const func::output& o : f->outputs()) {
        // Find the crops for this buffer in this scope level s.
        auto b = s.find(o.buffer->sym());
        if (b == s.end()) continue;

        // Add all the crops for this buffer.
        for (const crop_info& c : b->second) {
          result = crop_dim::make(b->first, c.dim, c.bounds, result);
        }
      }
    }
    return result;
  }

  stmt make_loop(stmt body, const func* f, const var& loop) {
    // Find the bounds of this loop.
    interval_expr bounds = interval_expr::union_identity();
    // Crop all the outputs of this buffer for this loop.
    crops.emplace_back();
    scope_crops& to_crop = crops.back();
    for (const func::output& o : f->outputs()) {
      for (int d = 0; d < static_cast<int>(o.dims.size()); ++d) {
        if (o.dims[d].sym() == loop.sym()) {
          to_crop[o.buffer->sym()].emplace_back(d, point(loop));
          // This output uses this loop. Add it to the bounds.
          bounds |= o.buffer->dim(d).bounds;
        }
      }
    }

    body = add_crops(body, f);

    // Before making this loop, see if there are any producers we need to insert here.
    while (const func* next = find_next_producer(f, loop)) {
      produce(body, next);
    }

    for (const buffer_expr_ptr& i : to_allocate) {
      const loop_id& at = i->store_at();
      if (at.f == f && at.loop.sym() == loop.sym()) {
        body = allocate::make(i->storage(), i->sym(), i->elem_size(), i->dims(), body);
        allocated.insert(i);
      }
    }
    for (const buffer_expr_ptr& i : allocated) {
      to_allocate.erase(i);
    }

    bounds.min = simplify(bounds.min);
    bounds.max = simplify(bounds.max);

    stmt result = loop::make(loop.sym(), bounds, 1, body);
    crops.pop_back();
    return result;
  }

  void produce(stmt& result, const func* f, bool root = false) {
    for (const func::output& i : f->outputs()) {
      if (!allocated.count(i.buffer)) {
        to_allocate.insert(i.buffer);
      }
    }
    stmt call_f = call_func::make(f->impl(), f);

    for (const func::output& i : f->outputs()) {
      produced.insert(i.buffer);
    }

    // Generate the loops that we want to be explicit.
    for (const auto& loop : f->loops()) {
      call_f = make_loop(call_f, f, loop);
    }
    result = block::make({call_f, result});
    if (root) {
      for (const auto& i : to_allocate) {
        result = allocate::make(i->storage(), i->sym(), i->elem_size(), i->dims(), result);
        allocated.insert(i);
      }
      to_allocate.clear();
    }
  }
};

void add_buffer_checks(const buffer_expr_ptr& b, std::vector<stmt>& checks) {
  int rank = static_cast<int>(b->rank());
  expr buf_var = variable::make(b->sym());
  checks.push_back(check::make(buf_var != 0));
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

    builder.produce(result, f, /*root=*/true);
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