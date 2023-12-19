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

index_t pipeline::evaluate(eval_context& ctx) {
  return 0;
}

}  // namespace slinky