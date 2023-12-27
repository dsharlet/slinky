#include "pipeline.h"

#include <cassert>
#include <iostream>
#include <set>

#include "evaluate.h"
#include "infer_bounds.h"
#include "print.h"
#include "simplify.h"
#include "substitute.h"

namespace slinky {

buffer_expr::buffer_expr(symbol_id name, index_t elem_size, std::size_t rank)
    : name_(name), elem_size_(elem_size), producer_(nullptr) {
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

class pipeline_builder {
  // We're going to incrementally build the body, starting at the end of the pipeline and adding
  // producers as necessary.
  std::set<buffer_expr_ptr> to_produce;
  std::set<buffer_expr_ptr> to_allocate;
  std::set<buffer_expr_ptr> produced;
  std::set<buffer_expr_ptr> allocated;

  struct crop_info {
    int dim;
    expr min;
    expr extent;
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
  const func* find_next_producer(const func* in = nullptr, const expr& loop = expr()) const {
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

      assert(loop.defined());
      const loop_id& at = i->producer()->compute_at();
      if (at.f == in && *as_variable(at.loop) == *as_variable(loop)) return i->producer();
    }
    return nullptr;
  }

  bool complete() const { return produced.size() == to_produce.size(); }

  stmt add_crops(stmt result, const func* f) {
    for (const auto& s : crops) {
      for (const func::output& o : f->outputs()) {
        // Find the crops for this buffer in this scope level s.
        auto b = s.find(o.buffer->name());
        if (b == s.end()) continue;

        // Add all the crops for this buffer.
        for (const crop_info& c : b->second) {
          result = crop_dim::make(b->first, c.dim, c.min, c.extent, result);
        }
      }
    }
    return result;
  }

  stmt make_loop(stmt body, const func* f, const expr& loop) {
    // Find the bounds of this loop.
    interval bounds = interval::union_identity();
    // Crop all the outputs of this buffer for this loop.
    crops.emplace_back();
    scope_crops& to_crop = crops.back();
    for (const func::output& o : f->outputs()) {
      for (int d = 0; d < o.dims.size(); ++d) {
        if (*as_variable(o.dims[d]) == *as_variable(loop)) {
          to_crop[o.buffer->name()].emplace_back(d, loop, 1);
          // This output uses this loop. Add it to the bounds.
          bounds |= interval(o.buffer->dim(d).min, o.buffer->dim(d).max());
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
      if (at.f == f && *as_variable(at.loop) == *as_variable(loop)) {
        body = allocate::make(i->storage(), i->name(), i->elem_size(), i->dims(), body);
        allocated.insert(i);
      }
    }
    for (const buffer_expr_ptr& i : allocated) {
      to_allocate.erase(i);
    }

    stmt result = loop::make(*as_variable(loop), simplify(bounds.min), simplify(bounds.max + 1), body);
    crops.pop_back();
    return result;
  }

  void produce(stmt& result, const func* f, bool root = false) {
    // TODO(https://github.com/dsharlet/slinky/issues/7): We shouldn't need this wrapper,
    // it might add measureable overhead. All it does is split a span of buffers into two
    // spans of buffers.
    std::size_t input_count = f->inputs().size();
    std::size_t output_count = f->outputs().size();
    auto wrapper = [impl = f->impl(), input_count, output_count](
                       std::span<const index_t>, std::span<buffer_base*> buffers) -> index_t {
      assert(buffers.size() == input_count + output_count);
      return impl(buffers.subspan(0, input_count), buffers.subspan(input_count, output_count));
    };
    std::vector<symbol_id> buffer_args;
    buffer_args.reserve(input_count + output_count);
    for (const func::input& i : f->inputs()) {
      buffer_args.push_back(i.buffer->name());
    }
    for (const func::output& i : f->outputs()) {
      buffer_args.push_back(i.buffer->name());
      if (!allocated.count(i.buffer)) {
        to_allocate.insert(i.buffer);
      }
    }
    stmt call_f = call::make(std::move(wrapper), {}, std::move(buffer_args), f);

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
        result = allocate::make(i->storage(), i->name(), i->elem_size(), i->dims(), result);
        allocated.insert(i);
      }
      to_allocate.clear();
    }
  }
};

stmt build_pipeline(
    node_context& ctx, const std::vector<buffer_expr_ptr>& inputs, const std::vector<buffer_expr_ptr>& outputs) {
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

  print(std::cerr, result, &ctx);

  std::vector<symbol_id> input_names;
  input_names.reserve(inputs.size());
  for (const buffer_expr_ptr& i : inputs) {
    input_names.push_back(i->name());
  }
  result = infer_bounds(result, ctx, input_names);

  // Add checks that the buffer constraints the user set are satisfied.
  std::vector<stmt> checks;
  for (const buffer_expr_ptr& i : inputs) {
    expr buf_var = variable::make(i->name());
    for (int d = 0; d < static_cast<int>(i->rank()); ++d) {
      checks.push_back(check::make(i->dim(d).min == load_buffer_meta::make(buf_var, buffer_meta::min, d)));
      checks.push_back(check::make(i->dim(d).extent == load_buffer_meta::make(buf_var, buffer_meta::extent, d)));
      checks.push_back(
          check::make(i->dim(d).stride_bytes == load_buffer_meta::make(buf_var, buffer_meta::stride_bytes, d)));
      checks.push_back(
          check::make(i->dim(d).fold_factor == load_buffer_meta::make(buf_var, buffer_meta::fold_factor, d)));
    }
  }
  result = block::make(block::make(checks), result);

  result = sliding_window(result, ctx);

  result = simplify(result);
  print(std::cerr, result, &ctx);

  return result;
}

}  // namespace

pipeline::pipeline(node_context& ctx, std::vector<buffer_expr_ptr> inputs, std::vector<buffer_expr_ptr> outputs)
    : inputs_(std::move(inputs)), outputs_(std::move(outputs)) {
  body = build_pipeline(ctx, inputs_, outputs_);
}

index_t pipeline::evaluate(std::span<const buffer_base*> inputs, std::span<const buffer_base*> outputs) const {
  assert(inputs.size() == inputs_.size());
  assert(outputs.size() == outputs_.size());

  eval_context ctx;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    ctx[inputs_[i]->name()] = reinterpret_cast<index_t>(inputs[i]);
  }
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    ctx[outputs_[i]->name()] = reinterpret_cast<index_t>(outputs[i]);
  }

  return slinky::evaluate(body, ctx);
}

}  // namespace slinky