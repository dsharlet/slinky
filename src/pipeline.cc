#include "pipeline.h"

#include <cassert>
#include <set>
#include <iostream>

#include "evaluate.h"
#include "print.h"
#include "infer_allocate_bounds.h"
#include "simplify.h"

namespace slinky {
  
buffer_expr::buffer_expr(symbol_id name, index_t elem_size, std::size_t rank) : name_(name), elem_size_(elem_size), producer_(nullptr) {
  dims_.reserve(rank);
  auto var = variable::make(name);
  for (index_t i = 0; i < static_cast<index_t>(rank); ++i) {
    expr min = load_buffer_meta::make(var, buffer_meta::min, i);
    expr extent = load_buffer_meta::make(var, buffer_meta::extent, i);
    expr stride_bytes = load_buffer_meta::make(var, buffer_meta::stride_bytes, i);
    expr fold_factor = load_buffer_meta::make(var, buffer_meta::fold_factor, i);
    dims_.emplace_back(min, extent, stride_bytes, fold_factor);
  }
}

buffer_expr_ptr buffer_expr::make(symbol_id name, index_t elem_size, std::size_t rank) {
  return buffer_expr_ptr(new buffer_expr(name, elem_size, rank));
}

buffer_expr_ptr buffer_expr::make(node_context& ctx, const std::string& name, index_t elem_size, std::size_t rank) {
  return buffer_expr_ptr(new buffer_expr(ctx.insert(name), elem_size, rank));
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
    std::vector<symbol_id> buffer_args;
    buffer_args.reserve(input_count + output_count);
    std::vector<buffer_expr_ptr> allocations;
    allocations.reserve(output_count);
    for (const func::input& i : f->inputs()) {
      buffer_args.push_back(i.buffer->name());
    }
    for (const func::output& i : f->outputs()) {
      buffer_args.push_back(i.buffer->name());
      if (!defined.count(i.buffer)) {
        allocations.push_back(i.buffer);
      }
    }
    stmt call_f = call::make(std::move(wrapper), {}, std::move(buffer_args), f);
    result = result.defined() ? block::make(call_f, result) : call_f;
    for (const auto& i : allocations) {
      result = allocate::make(memory_type::heap, i->name(), i->elem_size(), i->dims(), result);
      defined.insert(i);
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

  print(std::cout, result, &ctx);
  symbol_map<std::vector<dim_expr>> bounds;
  for (const auto& i : inputs) {
    bounds.set(i->name(), i->dims());
  }
  for (const auto& i : outputs) {
    bounds.set(i->name(), i->dims());
  }
  result = infer_allocate_bounds(result, bounds);
  print(std::cerr, result, &ctx);

  result = simplify(result);
  print(std::cerr, result, &ctx);

  return result;
}

}  // namespace

pipeline::pipeline(node_context& ctx, std::vector<buffer_expr_ptr> inputs, std::vector<buffer_expr_ptr> outputs)
  : inputs_(std::move(inputs)), outputs_(std::move(outputs)) {
  body = build_pipeline(ctx, inputs_, outputs_);
}

namespace {

void set_buffer(eval_context& ctx, const buffer_expr_ptr& buf_expr, const buffer_base* buf) {
  assert(buf_expr->rank() == buf->rank);

  ctx.set(buf_expr->name(), reinterpret_cast<index_t>(buf));

  for (std::size_t i = 0; i < buf->rank; ++i) {
    // If these asserts fail, it's because the user has added constraints to the buffer_expr,
    // e.g. buf.dim[0].stride_bytes = 4, and the buffer passed in does not satisfy that
    // constraint.
    assert(evaluate(buf_expr->dim(i).min, ctx) == buf->dims[i].min);
    assert(evaluate(buf_expr->dim(i).extent, ctx) == buf->dims[i].extent);
    assert(evaluate(buf_expr->dim(i).stride_bytes, ctx) == buf->dims[i].stride_bytes);
    assert(evaluate(buf_expr->dim(i).fold_factor, ctx) == buf->dims[i].fold_factor);
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