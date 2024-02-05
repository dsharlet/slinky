#include "runtime/evaluate.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "runtime/buffer.h"
#include "runtime/depends_on.h"
#include "runtime/expr.h"
#include "runtime/print.h"
#include "runtime/util.h"

namespace slinky {

bool can_evaluate(intrinsic fn) {
  switch (fn) {
  case intrinsic::abs: return true;
  default: return false;
  }
}

void dump_context_for_expr(
    std::ostream& s, const symbol_map<index_t>& ctx, const expr& deps_of, const node_context* symbols = nullptr) {
  for (symbol_id i = 0; i < ctx.size(); ++i) {
    std::string sym = symbols ? symbols->name(i) : "<" + std::to_string(i) + ">";
    auto deps = depends_on(deps_of, i);
    if (!deps_of.defined() || deps.var) {
      if (ctx.contains(i)) {
        s << "  " << sym << " = " << *ctx.lookup(i) << std::endl;
      } else {
        s << "  " << sym << " = <>" << std::endl;
      }
    } else if (!deps_of.defined() || deps.buffer) {
      if (ctx.contains(i)) {
        const raw_buffer* buf = reinterpret_cast<const raw_buffer*>(*ctx.lookup(i));
        s << "  " << sym << " = {base=" << buf->base << ", elem_size=" << buf->elem_size << ", dims={";
        for (std::size_t d = 0; d < buf->rank; ++d) {
          const dim& dim = buf->dims[d];
          s << "{min=" << dim.min() << ", max=" << dim.max() << ", extent=" << dim.extent()
            << ", stride=" << dim.stride();
          if (dim.fold_factor() != dim::unfolded) {
            s << ", fold_factor=" << dim.fold_factor();
          }
          s << "}";
          if (d + 1 < buf->rank) {
            s << ",";
          }
        }
        s << "}" << std::endl;
      }
    }
  }
}

namespace {

// This is a very slow implementation of copy_stmt. The expectation is that copies will have been lowered to aliases or
// calls to `copy` in buffer.h/cc instead of relying on this implementation.
void copy_stmt_impl(
    eval_context& ctx, const raw_buffer& src, const dim* dst_dims, void* dst_base, const copy_stmt& c, int dim) {
  const class dim& dst_dim = dst_dims[dim];
  index_t dst_stride = dst_dim.stride();
  for (index_t dst_x = dst_dim.begin(); dst_x < dst_dim.end(); ++dst_x) {
    auto s = set_value_in_scope(ctx, c.dst_x[dim], dst_x);
    if (dim == 0) {
      const void* src_base = src.base;
      for (std::size_t d = 0; d < src.rank; ++d) {
        const class dim& src_dim = src.dims[d];

        index_t src_x = evaluate(c.src_x[d], ctx);
        if (src_dim.contains(src_x)) {
          src_base = offset_bytes(src_base, src_dim.flat_offset_bytes(src_x));
        } else {
          src_base = nullptr;
          break;
        }
      }
      if (src_base) {
        memcpy(dst_base, src_base, src.elem_size);
      } else if (!c.padding.empty()) {
        memcpy(dst_base, c.padding.data(), src.elem_size);
      } else {
        // Leave unmodified.
      }
    } else {
      copy_stmt_impl(ctx, src, dst_dims, dst_base, c, dim - 1);
    }
    dst_base = offset_bytes(dst_base, dst_stride);
  }
}

void copy_stmt_impl(eval_context& ctx, const raw_buffer& src, const raw_buffer& dst, const copy_stmt& c) {
  assert(c.src_x.size() == src.rank);
  assert(c.dst_x.size() == dst.rank);
  assert(dst.elem_size == src.elem_size);
  assert(c.padding.empty() || dst.elem_size == c.padding.size());
  if (dst.rank == 0) {
    // The buffer is scalar.
    assert(src.rank == 0);
    memcpy(dst.base, src.base, dst.elem_size);
  } else {
    copy_stmt_impl(ctx, src, dst.dims, dst.base, c, dst.rank - 1);
  }
}

// TODO(https://github.com/dsharlet/slinky/issues/2): I think the T::accept/node_visitor::visit
// overhead (two virtual function calls per node) might be significant. This could be implemented
// as a switch statement instead.
class evaluator : public node_visitor {
public:
  index_t result = 0;
  eval_context& context;

  evaluator(eval_context& context) : context(context) {}

  // Skip the visitor pattern (two virtual function calls) for some frequently used node types.
  void visit(const expr& op) {
    switch (op.type()) {
    case node_type::variable: visit(reinterpret_cast<const variable*>(op.get())); return;
    case node_type::constant: visit(reinterpret_cast<const constant*>(op.get())); return;
    default: op.accept(this);
    }
  }

  void visit(const stmt& op) {
    switch (op.type()) {
    // case node_type::call_stmt: visit(reinterpret_cast<const call_stmt*>(op.get())); return;
    // case node_type::crop_dim: visit(reinterpret_cast<const crop_dim*>(op.get())); return;
    // case node_type::slice_dim: visit(reinterpret_cast<const slice_dim*>(op.get())); return;
    // case node_type::block: visit(reinterpret_cast<const block*>(op.get())); return;
    default: op.accept(this);
    }
  }

  // Assume `e` is defined, evaluate it and return the result.
  index_t eval_expr(const expr& e) {
    visit(e);
    index_t r = result;
    result = 0;
    return r;
  }

  // If `e` is defined, evaluate it and return the result. Otherwise, return default `def`.
  index_t eval_expr(const expr& e, index_t def) {
    if (e.defined()) {
      return eval_expr(e);
    } else {
      return def;
    }
  }

  void visit(const variable* op) override {
    auto value = context.lookup(op->sym);
    assert(value);
    result = *value;
  }

  void visit(const wildcard* op) override {
    // Maybe evaluating this should just be an error.
    auto value = context.lookup(op->sym);
    assert(value);
    result = *value;
  }

  void visit(const constant* op) override { result = op->value; }

  template <typename T>
  void visit_let(const T* op) {
    // This is a bit ugly but we really want to avoid heap allocations here.
    const size_t size = op->lets.size();
    using sv_type = std::pair<symbol_id, std::optional<index_t>>;
    sv_type* old_values = SLINKY_ALLOCA(sv_type, size);
    (void) new (old_values) std::optional<index_t>[size];

    for (size_t i = 0; i < size; ++i) {
      const auto& let = op->lets[i];
      old_values[i] = {let.first, context[let.first]};
      context[let.first] = eval_expr(let.second);
    }
    visit(op->body);
    for (size_t i = 0; i < size; ++i) {
      context[old_values[i].first] = old_values[i].second;
    }
  }

  void visit(const let* op) override { visit_let(op); }
  void visit(const let_stmt* op) override { visit_let(op); }

  void visit(const add* op) override { result = eval_expr(op->a) + eval_expr(op->b); }
  void visit(const sub* op) override { result = eval_expr(op->a) - eval_expr(op->b); }
  void visit(const mul* op) override { result = eval_expr(op->a) * eval_expr(op->b); }
  void visit(const div* op) override { result = euclidean_div(eval_expr(op->a), eval_expr(op->b)); }
  void visit(const mod* op) override { result = euclidean_mod(eval_expr(op->a), eval_expr(op->b)); }
  void visit(const class min* op) override { result = std::min(eval_expr(op->a), eval_expr(op->b)); }
  void visit(const class max* op) override { result = std::max(eval_expr(op->a), eval_expr(op->b)); }
  void visit(const equal* op) override { result = eval_expr(op->a) == eval_expr(op->b); }
  void visit(const not_equal* op) override { result = eval_expr(op->a) != eval_expr(op->b); }
  void visit(const less* op) override { result = eval_expr(op->a) < eval_expr(op->b); }
  void visit(const less_equal* op) override { result = eval_expr(op->a) <= eval_expr(op->b); }
  void visit(const logical_and* op) override { result = eval_expr(op->a) != 0 && eval_expr(op->b) != 0; }
  void visit(const logical_or* op) override { result = eval_expr(op->a) != 0 || eval_expr(op->b) != 0; }
  void visit(const logical_not* op) override { result = eval_expr(op->a) == 0; }

  void visit(const class select* op) override {
    if (eval_expr(op->condition)) {
      result = eval_expr(op->true_value);
    } else {
      result = eval_expr(op->false_value);
    }
  }

  index_t eval_buffer_metadata(const call* op) {
    assert(op->args.size() == 1);
    raw_buffer* buf = reinterpret_cast<raw_buffer*>(eval_expr(op->args[0]));
    assert(buf);
    switch (op->intrinsic) {
    case intrinsic::buffer_rank: return buf->rank;
    case intrinsic::buffer_elem_size: return buf->elem_size;
    case intrinsic::buffer_base: return reinterpret_cast<index_t>(buf->base);
    case intrinsic::buffer_size_bytes: return buf->size_bytes();
    default: std::abort();
    }
  }

  index_t eval_dim_metadata(const call* op) {
    assert(op->args.size() == 2);
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(eval_expr(op->args[0]));
    assert(buffer);
    index_t d = eval_expr(op->args[1]);
    assert(d < static_cast<index_t>(buffer->rank));
    const slinky::dim& dim = buffer->dim(d);
    switch (op->intrinsic) {
    case intrinsic::buffer_min: return dim.min();
    case intrinsic::buffer_max: return dim.max();
    case intrinsic::buffer_extent: return dim.extent();
    case intrinsic::buffer_stride: return dim.stride();
    case intrinsic::buffer_fold_factor: return dim.fold_factor();
    default: std::abort();
    }
  }

  void* eval_buffer_at(const call* op) {
    assert(op->args.size() >= 1);
    raw_buffer* buf = reinterpret_cast<raw_buffer*>(eval_expr(op->args[0]));
    void* result = buf->base;
    assert(op->args.size() <= buf->rank + 1);
    for (std::size_t d = 0; d < op->args.size() - 1; ++d) {
      if (op->args[d + 1].defined()) {
        result = offset_bytes(result, buf->dims[d].flat_offset_bytes(eval_expr(op->args[d + 1])));
      }
    }
    return result;
  }

  void visit(const call* op) override {
    switch (op->intrinsic) {
    case intrinsic::positive_infinity: std::cerr << "Cannot evaluate positive_infinity" << std::endl; std::abort();
    case intrinsic::negative_infinity: std::cerr << "Cannot evaluate negative_infinity" << std::endl; std::abort();
    case intrinsic::indeterminate: std::cerr << "Cannot evaluate indeterminate" << std::endl; std::abort();

    case intrinsic::abs:
      assert(op->args.size() == 1);
      result = std::abs(eval_expr(op->args[0]));
      return;

    case intrinsic::buffer_rank:
    case intrinsic::buffer_elem_size:
    case intrinsic::buffer_base:
    case intrinsic::buffer_size_bytes: result = eval_buffer_metadata(op); return;

    case intrinsic::buffer_min:
    case intrinsic::buffer_max:
    case intrinsic::buffer_extent:
    case intrinsic::buffer_stride:
    case intrinsic::buffer_fold_factor: result = eval_dim_metadata(op); return;

    case intrinsic::buffer_at: result = reinterpret_cast<index_t>(eval_buffer_at(op)); return;
    default: std::cerr << "Unknown intrinsic: " << op->intrinsic << std::endl; std::abort();
    }
  }

  void visit(const block* op) override {
    for (const auto& s : op->stmts) {
      if (result != 0) break;
      visit(s);
    }
  }

  void visit(const loop* op) override {
    index_t min = eval_expr(op->bounds.min);
    index_t max = eval_expr(op->bounds.max);
    index_t step = eval_expr(op->step, 1);
    if (op->mode == loop_mode::parallel) {
      assert(context.enqueue_many);
      assert(context.wait_for);
      struct shared_state {
        // We track the loop progress with two variables: `i` is the next iteration to run, and `done` is the number of
        // iterations completed. This allows us to check if the loop is done without relying on the workers actually
        // running. If the thread pool is busy, then we might enqueue workers that never run until after the loop is
        // done. Waiting for these to return (after doing nothing) would risk deadlock.
        std::atomic<index_t> i, done;

        // We want copies of these in the shared state so we can allow the worker to run after returning from this
        // scope.
        index_t min, max, step;

        // The first non-zero result is stored here.
        std::atomic<index_t> result;

        shared_state(index_t min, index_t max, index_t step)
            : i(min), done(min), min(min), max(max), step(step), result(0) {}
      };
      auto state = std::make_shared<shared_state>(min, max, step);
      // It is safe to capture op even though it's a pointer, because we only access it after we know that we're still
      // in this scope.
      // TODO: Can we do this without capturing context by value?
      auto worker = [state, context = this->context, op]() mutable {
        while (state->result == 0) {
          index_t i = state->i.fetch_add(state->step);
          if (!(state->min <= i && i <= state->max)) break;

          context[op->sym] = i;
          // Evaluate the parallel loop body with our copy of the context.
          index_t result = evaluate(op->body, context);
          if (result != 0) {
            state->result = result;
          }
          state->done += state->step;
        }
      };
      // TODO: It's wasteful to enqueue a worker per thread if we have fewer tasks than workers.
      context.enqueue_many(worker);
      worker();
      // While the loop still isn't done, work on other tasks.
      context.wait_for([&]() { return state->result != 0 || !(min <= state->done && state->done <= max); });
      result = state->result;
    } else {
      assert(op->mode == loop_mode::serial);
      // TODO(https://github.com/dsharlet/slinky/issues/3): We don't get a reference to context[op->sym] here
      // because the context could grow and invalidate the reference. This could be fixed by having evaluate
      // fully traverse the expression to find the max symbol_id, and pre-allocate the context up front. It's
      // not clear this optimization is necessary yet.
      std::optional<index_t> old_value = context[op->sym];
      for (index_t i = min; result == 0 && min <= i && i <= max; i += step) {
        context[op->sym] = i;
        visit(op->body);
      }
      context[op->sym] = old_value;
    }
  }

  void visit(const if_then_else* op) override {
    if (eval_expr(op->condition)) {
      if (op->true_body.defined()) {
        visit(op->true_body);
      }
    } else {
      if (op->false_body.defined()) {
        visit(op->false_body);
      }
    }
  }

  void visit(const call_stmt* op) override {
    result = op->target(context);
    if (result) {
      if (context.call_failed) {
        context.call_failed(op);
      } else {
        std::cerr << "call_stmt failed: " << stmt(op) << "->" << result << std::endl;
        std::abort();
      }
    }
  }

  void visit(const copy_stmt* op) override {
    const raw_buffer* src = reinterpret_cast<raw_buffer*>(context.lookup(op->src, 0));
    const raw_buffer* dst = reinterpret_cast<raw_buffer*>(context.lookup(op->dst, 0));

    copy_stmt_impl(context, *src, *dst, *op);
  }

  void visit(const allocate* op) override {
    std::size_t rank = op->dims.size();
    raw_buffer* buffer = SLINKY_ALLOCA(raw_buffer, 1);
    buffer->elem_size = op->elem_size;
    buffer->rank = rank;
    buffer->dims = SLINKY_ALLOCA(dim, rank);

    for (std::size_t i = 0; i < rank; ++i) {
      slinky::dim& dim = buffer->dim(i);
      dim.set_bounds(eval_expr(op->dims[i].min()), eval_expr(op->dims[i].max()));
      dim.set_stride(eval_expr(op->dims[i].stride));
      dim.set_fold_factor(eval_expr(op->dims[i].fold_factor, dim::unfolded));
    }

    if (op->storage == memory_type::stack) {
      buffer->base = alloca(buffer->size_bytes());
    } else {
      assert(op->storage == memory_type::heap);
      buffer->allocation = nullptr;
      if (context.allocate) {
        assert(context.free);
        context.allocate(op->sym, buffer);
      } else {
        buffer->allocate();
      }
    }

    auto set_buffer = set_value_in_scope(context, op->sym, reinterpret_cast<index_t>(buffer));
    visit(op->body);

    if (op->storage == memory_type::heap) {
      if (context.free) {
        assert(context.allocate);
        context.free(op->sym, buffer);
      } else {
        buffer->free();
      }
    }
  }

  void visit(const make_buffer* op) override {
    std::size_t rank = op->dims.size();
    raw_buffer* buffer = SLINKY_ALLOCA(raw_buffer, 1);
    buffer->elem_size = eval_expr(op->elem_size);
    buffer->base = reinterpret_cast<void*>(eval_expr(op->base));
    buffer->rank = rank;
    buffer->dims = SLINKY_ALLOCA(dim, rank);

    for (std::size_t i = 0; i < rank; ++i) {
      slinky::dim& dim = buffer->dim(i);
      dim.set_bounds(eval_expr(op->dims[i].min()), eval_expr(op->dims[i].max()));
      dim.set_stride(eval_expr(op->dims[i].stride));
      dim.set_fold_factor(eval_expr(op->dims[i].fold_factor, dim::unfolded));
    }

    auto set_buffer = set_value_in_scope(context, op->sym, reinterpret_cast<index_t>(buffer));
    visit(op->body);
  }

  void visit(const clone_buffer* op) override {
    raw_buffer* src = reinterpret_cast<raw_buffer*>(*context.lookup(op->sym));

    raw_buffer* buffer = SLINKY_ALLOCA(raw_buffer, 1);
    buffer->dims = SLINKY_ALLOCA(dim, src->rank);
    buffer->elem_size = src->elem_size;
    buffer->base = src->base;
    buffer->rank = src->rank;
    memcpy(buffer->dims, src->dims, sizeof(dim) * src->rank);

    auto set_buffer = set_value_in_scope(context, op->sym, reinterpret_cast<index_t>(buffer));
    visit(op->body);
  }

  void visit(const crop_buffer* op) override {
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(*context.lookup(op->sym));
    assert(buffer);

    struct interval {
      index_t min;
      index_t max;
    };

    std::size_t crop_rank = op->bounds.size();
    interval* old_bounds = SLINKY_ALLOCA(interval, crop_rank);

    void* old_base = buffer->base;
    for (std::size_t d = 0; d < crop_rank; ++d) {
      slinky::dim& dim = buffer->dims[d];
      index_t old_min = dim.min();
      index_t old_max = dim.max();
      old_bounds[d].min = old_min;
      old_bounds[d].max = old_max;

      // Allow these expressions to be undefined, and if so, they default to their existing values.
      index_t min = std::max(old_min, eval_expr(op->bounds[d].min, old_min));
      index_t max = std::min(old_max, eval_expr(op->bounds[d].max, old_max));

      if (max >= min) {
        index_t offset = dim.flat_offset_bytes(min);
        // Crops can't span a folding boundary if they move the base pointer.
        assert(offset == 0 || (max - old_min) / dim.fold_factor() == (min - old_min) / dim.fold_factor());
        buffer->base = offset_bytes(buffer->base, offset);
      }

      dim.set_bounds(min, max);
    }

    visit(op->body);

    buffer->base = old_base;
    for (std::size_t d = 0; d < crop_rank; ++d) {
      slinky::dim& dim = buffer->dims[d];
      dim.set_bounds(old_bounds[d].min, old_bounds[d].max);
    }
  }

  void visit(const crop_dim* op) override {
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(*context.lookup(op->sym));
    assert(buffer);
    slinky::dim& dim = buffer->dims[op->dim];
    index_t old_min = dim.min();
    index_t old_max = dim.max();

    index_t min = std::max(old_min, eval_expr(op->bounds.min, old_min));
    index_t max = std::min(old_max, eval_expr(op->bounds.max, old_max));

    void* old_base = buffer->base;
    if (max >= min) {
      buffer->base = offset_bytes(buffer->base, dim.flat_offset_bytes(min));
      // Crops can't span a folding boundary if they move the base pointer.
      assert(buffer->base == old_base || (max - old_min) / dim.fold_factor() == (min - old_min) / dim.fold_factor());
    }

    dim.set_bounds(min, max);

    visit(op->body);

    buffer->base = old_base;
    dim.set_bounds(old_min, old_max);
  }

  void visit(const slice_buffer* op) override {
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(*context.lookup(op->sym));
    assert(buffer);

    // TODO: If we really care about stack usage here, we could find the number of dimensions we actually need first.
    dim* dims = SLINKY_ALLOCA(dim, buffer->rank);

    std::size_t rank = 0;
    index_t offset = 0;
    for (std::size_t d = 0; d < buffer->rank; ++d) {
      if (d < op->at.size() && op->at[d].defined()) {
        offset += buffer->dims[d].flat_offset_bytes(eval_expr(op->at[d]));
      } else {
        dims[rank++] = buffer->dims[d];
      }
    }

    void* old_base = buffer->base;
    buffer->base = offset_bytes(buffer->base, offset);
    std::swap(buffer->rank, rank);
    std::swap(buffer->dims, dims);

    visit(op->body);

    buffer->base = old_base;
    buffer->rank = rank;
    buffer->dims = dims;
  }

  void visit(const slice_dim* op) override {
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(*context.lookup(op->sym));
    assert(buffer);

    // The rank of the result is equal to the current rank, less any sliced dimensions.
    dim* old_dims = buffer->dims;

    buffer->dims = SLINKY_ALLOCA(dim, buffer->rank - 1);

    index_t at = eval_expr(op->at);
    index_t offset = old_dims[op->dim].flat_offset_bytes(at);
    void* old_base = buffer->base;
    buffer->base = offset_bytes(buffer->base, offset);

    for (int d = 0; d < op->dim; ++d) {
      buffer->dims[d] = old_dims[d];
    }
    for (int d = op->dim + 1; d < static_cast<int>(buffer->rank); ++d) {
      buffer->dims[d - 1] = old_dims[d];
    }
    buffer->rank -= 1;

    visit(op->body);

    buffer->base = old_base;
    buffer->rank += 1;
    buffer->dims = old_dims;
  }

  void visit(const truncate_rank* op) override {
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(*context.lookup(op->sym));
    assert(buffer);

    std::size_t old_rank = buffer->rank;
    buffer->rank = op->rank;

    visit(op->body);

    buffer->rank = old_rank;
  }

  void visit(const check* op) override {
    result = eval_expr(op->condition, 0) != 0 ? 0 : 1;
    if (result) {
      if (context.check_failed) {
        context.check_failed(op->condition);
      } else {
        std::cerr << "Check failed: " << op->condition << std::endl;
        std::cerr << "Context: " << std::endl;
        dump_context_for_expr(std::cerr, context, op->condition);
        std::abort();
      }
    }
  }
};

}  // namespace

index_t evaluate(const expr& e, eval_context& context) {
  evaluator eval(context);
  e.accept(&eval);
  return eval.result;
}

index_t evaluate(const stmt& s, eval_context& context) {
  evaluator eval(context);
  s.accept(&eval);
  return eval.result;
}

index_t evaluate(const expr& e) {
  eval_context ctx;
  return evaluate(e, ctx);
}

index_t evaluate(const stmt& s) {
  eval_context ctx;
  return evaluate(s, ctx);
}

}  // namespace slinky
