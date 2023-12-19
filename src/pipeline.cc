#include "pipeline.h"

#include <cassert>
#include <iostream>

#include "print.h"

namespace slinky {

buffer_expr::buffer_expr(node_context& ctx, const std::string& name, std::size_t rank) {
  base_ = make_variable(ctx, name + ".base");
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
  assert(std::find(producers_.begin(), producers_.end(), f) == producers_.end());
  producers_.push_back(f);
}

void buffer_expr::add_consumer(func* f) {
  assert(std::find(consumers_.begin(), consumers_.end(), f) == consumers_.end());
  consumers_.push_back(f);
}

func::func(callable impl, std::vector<input> inputs, std::vector<output> outputs)
  : impl(std::move(impl)), inputs(std::move(inputs)), outputs(std::move(outputs)) {
  for (auto& i : inputs) {
    i.buffer->add_consumer(this);
  }
  for (auto& i : outputs) {
    i.buffer->add_producer(this);
  }
}

index_t func::evaluate(eval_context& ctx) {
  return 0;
}

pipeline::pipeline(std::vector<buffer_expr_ptr> inputs, std::vector<buffer_expr_ptr> outputs)
  : inputs_(std::move(inputs)), outputs_(std::move(outputs)) {

}

namespace {

void set_or_assert(eval_context& ctx, const expr& e, index_t value) {
  if (const variable* v = e.as<variable>()) {
    ctx.set(v->name, value);
  }
  else {
    assert(evaluate(e) == value);
  }
}

void set_buffer(eval_context& ctx, const buffer_expr_ptr& buf_expr, const buffer_base* buf) {
  assert(buf_expr->rank() == buf->rank);

  // set_or_assert is probably overkill here.
  set_or_assert(ctx, buf_expr->base(), reinterpret_cast<index_t>(buf->base));

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