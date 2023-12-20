#include "pipeline.h"

#include <cassert>
#include <set>
#include <iostream>

#include "evaluate.h"
#include "print.h"

namespace slinky {

buffer_expr::buffer_expr(node_context& ctx, const std::string& name, std::size_t rank) : producer_(nullptr) {
  name_ = ctx.insert(name);
  dims_.reserve(rank);
  for (std::size_t i = 0; i < rank; ++i) {
    std::string dim_name = name + "." + std::to_string(i);
    expr min = make_variable(ctx, dim_name + ".min");
    expr extent = make_variable(ctx, dim_name + ".extent");
    expr stride = make_variable(ctx, dim_name + ".stride");
    expr fold_factor = make_variable(ctx, dim_name + ".fold_factor");
    dims_.emplace_back(min, extent, stride, fold_factor);
  }
}

buffer_expr_ptr buffer_expr::make(node_context& ctx, const std::string& name, std::size_t rank) {
  return buffer_expr_ptr(new buffer_expr(ctx, name, rank));
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

namespace {

stmt build_pipeline(node_context& ctx, const std::vector<buffer_expr_ptr>& inputs, const std::vector<buffer_expr_ptr>& outputs) {
  // We're going to incrementally build the body, starting at the end of the pipeline and adding producers as necessary.
  std::set<buffer_expr_ptr> to_produce;
  std::set<buffer_expr_ptr> produced;
  stmt result;

  std::set<buffer_expr_ptr> defined;

  // To start with, we need to produce the outputs.
  for (auto& i : outputs) {
    to_produce.insert(i);
    defined.insert(i);
  }
  // And we've already "produced" the inputs.
  for (auto& i : inputs) {
    produced.insert(i);
    defined.insert(i);
  }

  while (!to_produce.empty()) {
    // Find a buffer to produce.
    const func* f = nullptr;
    for (auto i = to_produce.begin(); !f && i != to_produce.end(); ++i) {
      f = (*i)->producer();

      for (const func* j : (*i)->consumers()) {
        for (auto& k : j->outputs()) {
          if (k.buffer == *i) continue;
          if (to_produce.count(k.buffer)) {
            // j produces a buffer that is needed by another func that has not yet run.
            f = nullptr;
            break;
          }
        }
      }
    }

    // Call the producer.
    if (!f) {
      // TODO: Make a better error here.
      std::cerr << "Problem in dependency graph" << std::endl;
      std::abort();
    }

    // TODO: We shouldn't need this wrapper, it might add measureable overhead.
    // All it does is split a span of buffers into two spans of buffers.
    std::size_t input_count = f->inputs().size();
    std::size_t output_count = f->outputs().size();
    auto wrapper = [impl = f->impl(), input_count, output_count](std::span<const index_t>, std::span<buffer_base*> buffers) -> index_t {
      assert(buffers.size() == input_count + output_count);
      return impl(buffers.subspan(0, input_count), buffers.subspan(input_count, output_count));
    };
    std::vector<expr> buffer_args;
    buffer_args.reserve(input_count + output_count);
    std::vector<buffer_expr_ptr> allocations;
    allocations.reserve(output_count);
    for (const func::input& i : f->inputs()) {
      buffer_args.push_back(variable::make(i.buffer->name()));
    }
    for (const func::output& i : f->outputs()) {
      buffer_args.push_back(variable::make(i.buffer->name()));
      if (!defined.count(i.buffer)) {
        allocations.push_back(i.buffer);
      }
    }
    stmt call_f = call::make(wrapper, {}, std::move(buffer_args));
    result = result.defined() ? block::make(call_f, result) : call_f;
    for (const auto& i : allocations) {
      result = allocate::make(memory_type::heap, i->name(), i->dims(), result);
    }

    // We've just run f, which produced its outputs.
    for (auto& i : f->outputs()) {
      produced.insert(i.buffer);
      to_produce.erase(i.buffer);
    }
    // Now make sure its inputs get produced.
    for (auto& i : f->inputs()) {
      if (!produced.count(i.buffer)) {
        to_produce.insert(i.buffer);
      }
    }
  }

  return result;
}

}  // namespace

pipeline::pipeline(node_context& ctx, std::vector<buffer_expr_ptr> inputs, std::vector<buffer_expr_ptr> outputs)
  : inputs_(std::move(inputs)), outputs_(std::move(outputs)) {
  body = build_pipeline(ctx, inputs_, outputs_);
}

namespace {

void set_or_assert(eval_context& ctx, const expr& e, index_t value) {
  if (const variable* v = e.as<variable>()) {
    ctx.set(v->name, value);
  } else {
    assert(evaluate(e) == value);
  }
}

void set_buffer(eval_context& ctx, const buffer_expr_ptr& buf_expr, const buffer_base* buf) {
  assert(buf_expr->rank() == buf->rank);

  ctx.set(buf_expr->name(), reinterpret_cast<index_t>(buf));

  // TODO: It might be useful/necessary to first set all the variable fields, and then
  // assert all the non-variable fields, to support doing something like this
  // (e is buffer_expr_ptr):
  //
  // e->dim(1).stride_bytes = e->dim(0).extent * e->dim(0).stride_bytes
  //
  // i.e. require that the buffer has no padding between each instance of dim(1).
  for (std::size_t i = 0; i < buf->rank; ++i) {
    const auto& dim_expr = buf_expr->dim(i);
    const auto& dim = buf->dims[i];

    set_or_assert(ctx, dim_expr.min, dim.min);
    set_or_assert(ctx, dim_expr.extent, dim.extent);
    set_or_assert(ctx, dim_expr.stride_bytes, dim.stride_bytes);
    set_or_assert(ctx, dim_expr.fold_factor, dim.fold_factor);
  }
}

}  // namespace

index_t pipeline::evaluate(std::span<buffer_base*> inputs, std::span<buffer_base*> outputs) {
  assert(inputs.size() == inputs_.size());
  assert(outputs.size() == outputs_.size());

  eval_context ctx;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    set_buffer(ctx, inputs_[i], inputs[i]);
  }
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    set_buffer(ctx, outputs_[i], outputs[i]);
  }

  return slinky::evaluate(body, ctx);
}

}  // namespace slinky