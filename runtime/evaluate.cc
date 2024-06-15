#include "runtime/evaluate.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <utility>

#include "base/chrome_trace.h"
#include "runtime/buffer.h"
#include "runtime/depends_on.h"
#include "runtime/expr.h"
#include "runtime/print.h"
#include "runtime/stmt.h"

namespace slinky {

bool can_evaluate(intrinsic fn) {
  switch (fn) {
  case intrinsic::abs: return true;
  default: return false;
  }
}

void dump_context_for_expr(
    std::ostream& s, const symbol_map<index_t>& ctx, const expr& deps_of, const node_context* symbols = nullptr) {
  for (std::size_t i = 0; i < ctx.size(); ++i) {
    std::string sym = symbols ? symbols->name(var(i)) : "<" + std::to_string(i) + ">";
    auto deps = depends_on(deps_of, var(i));
    if (!deps_of.defined() || deps.var) {
      if (ctx[i]) {
        s << "  " << sym << " = " << *ctx[i] << std::endl;
      } else {
        s << "  " << sym << " = <>" << std::endl;
      }
    } else if (!deps_of.defined() || deps.buffer_meta) {
      if (ctx[i]) {
        const raw_buffer* buf = reinterpret_cast<const raw_buffer*>(*ctx[i]);
        s << "  " << sym << " = " << *buf << std::endl;
      }
    }
  }
}

void eval_context::parallel_for(
    index_t begin, index_t end, index_t step, std::function<void(index_t)> body, int max_workers) {
  scoped_trace trace("parallel_for");
  if (begin >= end) {
    return;
  } else if (begin + step >= end) {
    body(begin);
    return;
  }

  assert(enqueue_many);
  assert(enqueue);
  assert(wait_for);
  struct shared_state {
    // We track the loop progress with two variables: `i` is the next iteration to run, and `done` is the number of
    // iterations completed. This allows us to check if the loop is done without relying on the workers actually
    // running. If the thread pool is busy, then we might enqueue workers that never run until after the loop is
    // done. Waiting for these to return (after doing nothing) would risk deadlock.
    std::atomic<index_t> i, done;

    // We want copies of these in the shared state so we can allow the worker to run after returning from this
    // scope.
    index_t begin, end;
    index_t step;

    // Which threads are working on this loop.
    std::set<std::thread::id> working_threads;
    std::mutex m;

    // This should be called when entering a worker. If it returns false, we are already in the call stack of a
    // worker on this loop, and should return to work on other tasks instead.
    bool begin_work() {
      std::unique_lock l(m);
      std::thread::id tid = std::this_thread::get_id();
      return working_threads.emplace(tid).second;
    }

    void end_work() {
      std::unique_lock l(m);
      auto i = working_threads.find(std::this_thread::get_id());
      assert(i != working_threads.end());
      working_threads.erase(i);
    }

    shared_state(index_t begin, index_t end, index_t step) : i(begin), done(begin), begin(begin), end(end), step(step) {}
  };
  auto state = std::make_shared<shared_state>(begin, end, step);
  // It is safe to capture op even though it's a pointer, because we only access it after we know that we're still
  // in this scope.
  // TODO: Can we do this without capturing context by value?
  auto worker = [state, body]() mutable {
    if (!state->begin_work()) return;
    scoped_trace trace("worker");

    while (true) {
      index_t i = state->i.fetch_add(state->step);
      if (!(state->begin <= i && i < state->end)) break;
      body(i);
      state->done += state->step;
    }

    state->end_work();
  };
  if (max_workers == std::numeric_limits<int>::max()) {
    // TODO: It's wasteful to enqueue a worker per thread if we have fewer tasks than workers.
    enqueue_many(worker);
  } else {
    enqueue(max_workers - 1, worker);
  }
  worker();
  scoped_trace trace_wait("wait_for");
  // While the loop still isn't done, work on other tasks.
  wait_for([&]() { return !(begin <= state->done && state->done < end); });
}

namespace {

struct allocated_buffer : public raw_buffer {
  void* allocation;
};

struct interval {
  index_t min, max;
};

class evaluator {
public:
  eval_context& context;

  // We want to propagate undefined values when we hit them.
  bool undef;

  evaluator(eval_context& context) : context(context) {}

  // Assume `e` is defined, evaluate it and return the result.
  SLINKY_ALWAYS_INLINE index_t eval(const expr& e) {
    // It helps a lot to inline this for common node types, but we don't want to do that for every node everywhere. So
    // we handle common node types here, and call a non-inlined handler for the less common nodes below.
    switch (e.type()) {
    case expr_node_type::variable: return eval(static_cast<const variable*>(e.get()));
    case expr_node_type::constant: return eval(static_cast<const constant*>(e.get()));
    case expr_node_type::call: return eval(static_cast<const call*>(e.get()));
    default: return eval_non_inlined(e);
    }
  }

  SLINKY_NO_INLINE index_t eval_non_inlined(const expr& e) {
    switch (e.type()) {
    case expr_node_type::let: return eval(static_cast<const let*>(e.get()));
    case expr_node_type::logical_not: return eval(static_cast<const logical_not*>(e.get()));
    case expr_node_type::select: return eval(static_cast<const class select*>(e.get()));
    default: return eval_binary(e);
    }
  }

  index_t eval_binary(const expr& e) {
    const binary_op* op = static_cast<const binary_op*>(e.get());
    index_t a = eval(op->a);
    index_t b = eval(op->b);
    switch (op->type) {
    case expr_node_type::add: return make_binary<add>(a, b);
    case expr_node_type::sub: return make_binary<sub>(a, b);
    case expr_node_type::mul: return make_binary<mul>(a, b);
    case expr_node_type::div: return make_binary<div>(a, b);
    case expr_node_type::mod: return make_binary<mod>(a, b);
    case expr_node_type::min: return make_binary<class min>(a, b);
    case expr_node_type::max: return make_binary<class max>(a, b);
    case expr_node_type::equal: return make_binary<equal>(a, b);
    case expr_node_type::not_equal: return make_binary<not_equal>(a, b);
    case expr_node_type::less: return make_binary<less>(a, b);
    case expr_node_type::less_equal: return make_binary<less_equal>(a, b);
    case expr_node_type::logical_and: return make_binary<logical_and>(a, b);
    case expr_node_type::logical_or: return make_binary<logical_or>(a, b);
    default: std::abort();
    }
  }

  // If `e` is defined, evaluate it and return the result. Otherwise, return default `def`.
  index_t eval(const expr& e, index_t def) {
    undef = false;
    if (e.defined()) {
      index_t result = eval(e);
      return undef ? def : result;
    } else {
      return def;
    }
  }

  interval eval(const interval_expr& x) {
    index_t min = eval(x.min);
    if (x.is_point()) {
      return {min, min};
    } else {
      return {min, eval(x.max)};
    }
  }
  interval eval(const interval_expr& x, interval def) {
    if (x.is_point() && x.min.defined()) {
      index_t result = eval(x.min);
      return {result, result};
    } else {
      return {eval(x.min, def.min), eval(x.max, def.max)};
    }
  }

  dim eval(const dim_expr& x) {
    dim result;
    interval bounds = eval(x.bounds);
    result.set_bounds(bounds.min, bounds.max);
    result.set_stride(eval(x.stride, dim::auto_stride));
    result.set_fold_factor(eval(x.fold_factor, dim::unfolded));
    return result;
  }

  index_t eval(const variable* op) {
    auto value = context.lookup(op->sym);
    assert(value);
    return *value;
  }

  static index_t eval(const constant* op) { return op->value; }

  SLINKY_NO_STACK_PROTECTOR index_t eval(const let* op) {
    // This is a bit ugly but we really want to avoid heap allocations here.
    const size_t size = op->lets.size();
    std::optional<index_t>* old_values = SLINKY_ALLOCA(std::optional<index_t>, size);
    (void)new (old_values) std::optional<index_t>[ size ];

    for (size_t i = 0; i < size; ++i) {
      const auto& let = op->lets[i];
      old_values[i] = context[let.first];
      context[let.first] = eval(let.second);
    }
    index_t result = eval(op->body);
    for (size_t i = 0; i < size; ++i) {
      const auto& let = op->lets[i];
      context[let.first] = old_values[i];
    }
    return result;
  }

  index_t eval(const logical_not* op) { return eval(op->a) == 0; }

  index_t eval(const class select* op) {
    if (eval(op->condition)) {
      if (op->true_value.defined()) {
        return eval(op->true_value);
      } else {
        undef = true;
        return 0;
      }
    } else {
      if (op->false_value.defined()) {
        return eval(op->false_value);
      } else {
        undef = true;
        return 0;
      }
    }
  }

  bool eval_short_circuit_op(const call* op) {
    for (const expr& i : op->args) {
      index_t x = eval(i);
      if (!x && op->intrinsic == intrinsic::and_then) {
        return false;
      } else if (x && op->intrinsic == intrinsic::or_else) {
        return true;
      }
    }
    return op->intrinsic == intrinsic::and_then;
  }

  index_t eval_define_undef(const call* op) {
    assert(op->args.size() == 2);
    index_t def = eval(op->args[1]);
    return eval(op->args[0], def);
  }

  index_t eval_buffer_metadata(const call* op) {
    assert(op->args.size() == 1);
    raw_buffer* buf = reinterpret_cast<raw_buffer*>(eval(op->args[0]));
    assert(buf);
    switch (op->intrinsic) {
    case intrinsic::buffer_rank: return buf->rank;
    case intrinsic::buffer_elem_size: return buf->elem_size;
    case intrinsic::buffer_size_bytes: return buf->size_bytes();
    default: std::abort();
    }
  }

  index_t eval_dim_metadata(const call* op) {
    assert(op->args.size() == 2);
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(eval(op->args[0]));
    assert(buffer);
    index_t d = eval(op->args[1]);
    assert(d < static_cast<index_t>(buffer->rank));
    const slinky::dim& dim = buffer->dim(d);
    switch (op->intrinsic) {
    case intrinsic::buffer_min: return dim.min();
    case intrinsic::buffer_max: return dim.max();
    case intrinsic::buffer_stride: return dim.stride();
    case intrinsic::buffer_fold_factor: return dim.fold_factor();
    default: std::abort();
    }
  }

  void* eval_buffer_at(const call* op) {
    assert(op->args.size() >= 1);
    raw_buffer* buf = reinterpret_cast<raw_buffer*>(eval(op->args[0]));
    void* result = buf->base;
    assert(op->args.size() <= buf->rank + 1);
    for (std::size_t d = 0; d < op->args.size() - 1; ++d) {
      if (op->args[d + 1].defined()) {
        index_t at = eval(op->args[d + 1]);
        if (result && buf->dims[d].contains(at)) {
          result = offset_bytes_non_null(result, buf->dims[d].flat_offset_bytes(at));
        } else {
          result = nullptr;
        }
      }
    }
    return result;
  }

  index_t eval_semaphore_init(const call* op) {
    assert(op->args.size() == 2);
    index_t* sem = reinterpret_cast<index_t*>(eval(op->args[0]));
    index_t count = eval(op->args[1], 0);
    context.atomic_call([=]() { *sem = count; });
    return 1;
  }

  SLINKY_NO_STACK_PROTECTOR index_t eval_semaphore_signal(const call* op) {
    assert(op->args.size() % 2 == 0);
    std::size_t sem_count = op->args.size() / 2;
    index_t** sems = SLINKY_ALLOCA(index_t*, sem_count);
    index_t* counts = SLINKY_ALLOCA(index_t, sem_count);
    for (std::size_t i = 0; i < sem_count; ++i) {
      sems[i] = reinterpret_cast<index_t*>(eval(op->args[i * 2 + 0]));
      counts[i] = eval(op->args[i * 2 + 1], 1);
    }
    context.atomic_call([=]() {
      for (std::size_t i = 0; i < sem_count; ++i) {
        *sems[i] += counts[i];
      }
    });
    return 1;
  }

  SLINKY_NO_STACK_PROTECTOR index_t eval_semaphore_wait(const call* op) {
    assert(op->args.size() % 2 == 0);
    std::size_t sem_count = op->args.size() / 2;
    index_t** sems = SLINKY_ALLOCA(index_t*, sem_count);
    index_t* counts = SLINKY_ALLOCA(index_t, sem_count);
    for (std::size_t i = 0; i < sem_count; ++i) {
      sems[i] = reinterpret_cast<index_t*>(eval(op->args[i * 2 + 0]));
      counts[i] = eval(op->args[i * 2 + 1], 1);
    }
    context.wait_for([=]() {
      // Check we can acquire all of the semaphores before acquiring any of them.
      for (std::size_t i = 0; i < sem_count; ++i) {
        if (*sems[i] < counts[i]) return false;
      }
      // Acquire them all.
      for (std::size_t i = 0; i < sem_count; ++i) {
        *sems[i] -= counts[i];
      }
      return true;
    });
    return 1;
  }

  index_t eval_trace_begin(const call* op) {
    assert(op->args.size() == 1);
    const char* name = reinterpret_cast<const char*>(eval(op->args[0]));
    return context.trace_begin ? context.trace_begin(name) : 0;
  }

  index_t eval_trace_end(const call* op) {
    assert(op->args.size() == 1);
    if (context.trace_end) {
      context.trace_end(eval(op->args[0]));
    }
    return 1;
  }

  index_t eval_free(const call* op) {
    assert(op->args.size() == 1);
    var sym = *as_variable(op->args[0]);
    allocated_buffer* buf = reinterpret_cast<allocated_buffer*>(*context.lookup(sym));
    context.free(sym, buf, buf->allocation);
    buf->allocation = nullptr;
    return 1;
  }

  SLINKY_NO_INLINE index_t eval(const call* op) {
    switch (op->intrinsic) {
    case intrinsic::positive_infinity: std::cerr << "Cannot evaluate positive_infinity" << std::endl; std::abort();
    case intrinsic::negative_infinity: std::cerr << "Cannot evaluate negative_infinity" << std::endl; std::abort();
    case intrinsic::indeterminate: std::cerr << "Cannot evaluate indeterminate" << std::endl; std::abort();

    case intrinsic::abs: assert(op->args.size() == 1); return std::abs(eval(op->args[0]));

    case intrinsic::and_then:
    case intrinsic::or_else: return eval_short_circuit_op(op);

    case intrinsic::define_undef: return eval_define_undef(op);

    case intrinsic::buffer_rank:
    case intrinsic::buffer_elem_size:
    case intrinsic::buffer_size_bytes: return eval_buffer_metadata(op);

    case intrinsic::buffer_min:
    case intrinsic::buffer_max:
    case intrinsic::buffer_stride:
    case intrinsic::buffer_fold_factor: return eval_dim_metadata(op);

    case intrinsic::buffer_at: return reinterpret_cast<index_t>(eval_buffer_at(op));

    case intrinsic::semaphore_init: return eval_semaphore_init(op);
    case intrinsic::semaphore_signal: return eval_semaphore_signal(op);
    case intrinsic::semaphore_wait: return eval_semaphore_wait(op);

    case intrinsic::trace_begin: return eval_trace_begin(op);
    case intrinsic::trace_end: return eval_trace_end(op);

    case intrinsic::free: return eval_free(op);

    default: std::cerr << "Unknown intrinsic: " << op->intrinsic << std::endl; std::abort();
    }
  }

  SLINKY_ALWAYS_INLINE index_t eval(const stmt& op) {
    // It helps a lot to inline this for common node types, but we don't want to do that for every node everywhere. So
    // we handle common node types here, and call a non-inlined handler for the less common nodes below.
    switch (op.type()) {
    case stmt_node_type::call_stmt: return eval(reinterpret_cast<const call_stmt*>(op.get()));
    case stmt_node_type::copy_stmt: return eval(reinterpret_cast<const copy_stmt*>(op.get()));
    case stmt_node_type::crop_dim: return eval(reinterpret_cast<const crop_dim*>(op.get()));
    case stmt_node_type::slice_dim: return eval(reinterpret_cast<const slice_dim*>(op.get()));
    default: return eval_non_inlined(op);
    }
  }

  SLINKY_NO_INLINE index_t eval_non_inlined(const stmt& op) {
    switch (op.type()) {
    case stmt_node_type::let_stmt: return eval(reinterpret_cast<const let_stmt*>(op.get()));
    case stmt_node_type::block: return eval(reinterpret_cast<const block*>(op.get()));
    case stmt_node_type::loop: return eval(reinterpret_cast<const loop*>(op.get()));
    case stmt_node_type::allocate: return eval(reinterpret_cast<const allocate*>(op.get()));
    case stmt_node_type::make_buffer: return eval(reinterpret_cast<const make_buffer*>(op.get()));
    case stmt_node_type::clone_buffer: return eval(reinterpret_cast<const clone_buffer*>(op.get()));
    case stmt_node_type::crop_buffer: return eval(reinterpret_cast<const crop_buffer*>(op.get()));
    case stmt_node_type::slice_buffer: return eval(reinterpret_cast<const slice_buffer*>(op.get()));
    case stmt_node_type::transpose: return eval(reinterpret_cast<const transpose*>(op.get()));
    case stmt_node_type::check: return eval(reinterpret_cast<const check*>(op.get()));
    default: std::abort();
    }
  }

  SLINKY_NO_STACK_PROTECTOR index_t eval(const let_stmt* op) {
    // This is a bit ugly but we really want to avoid heap allocations here.
    const size_t size = op->lets.size();
    std::optional<index_t>* old_values = SLINKY_ALLOCA(std::optional<index_t>, size);
    (void)new (old_values) std::optional<index_t>[ size ];

    for (size_t i = 0; i < size; ++i) {
      const auto& let = op->lets[i];
      old_values[i] = context[let.first];
      context[let.first] = eval(let.second);
    }
    index_t result = eval(op->body);
    for (size_t i = 0; i < size; ++i) {
      const auto& let = op->lets[i];
      context[let.first] = old_values[i];
    }
    return result;
  }

  index_t eval(const block* op) {
    for (const auto& s : op->stmts) {
      index_t result = eval(s);
      if (result) return result;
    }
    return 0;
  }

  index_t eval(const loop* op) {
    interval bounds = eval(op->bounds);
    index_t step = eval(op->step, 1);
    if (op->max_workers > 1) {
      std::atomic<index_t> result = 0;
      context.parallel_for(bounds.min, bounds.max + 1, step, [context = this->context, op, &result](index_t i) mutable {
        context[op->sym] = i;
        // Evaluate the parallel loop body with our copy of the context.
        index_t result_i = evaluate(op->body, context);
        if (result_i != 0) {
          index_t zero = 0;
          result.compare_exchange_strong(zero, result_i);
        }
      }, op->max_workers);
      return result;
    } else {
      // TODO(https://github.com/dsharlet/slinky/issues/3): We don't get a reference to context[op->sym] here
      // because the context could grow and invalidate the reference. This could be fixed by having evaluate
      // fully traverse the expression to find the max var, and pre-allocate the context up front. It's
      // not clear this optimization is necessary yet.
      std::optional<index_t> old_value = context[op->sym];
      index_t result = 0;
      for (index_t i = bounds.min; result == 0 && bounds.min <= i && i <= bounds.max; i += step) {
        context[op->sym] = i;
        result = eval(op->body);
      }
      context[op->sym] = old_value;
      return result;
    }
  }

  index_t eval(const call_stmt* op) {
    index_t result = op->target(op, context);
    if (result) {
      if (context.call_failed) {
        context.call_failed(op);
      } else {
        std::cerr << "call_stmt failed: " << stmt(op) << "->" << result << std::endl;
        std::abort();
      }
    }
    return result;
  }

  index_t eval(const copy_stmt* op) {
    std::cerr << "copy_stmt should have been implemented by calls to copy/pad." << std::endl;
    std::abort();
  }

  // Not using SLINKY_NO_STACK_PROTECTOR here because this actually could allocate a lot of memory on the stack.
  index_t eval(const allocate* op) {
    std::size_t rank = op->dims.size();
    allocated_buffer buffer;
    buffer.elem_size = eval(op->elem_size);
    buffer.rank = rank;
    buffer.dims = SLINKY_ALLOCA(dim, rank);

    for (std::size_t d = 0; d < rank; ++d) {
      buffer.dim(d) = eval(op->dims[d]);
    }

    if (op->storage == memory_type::stack) {
      buffer.init_strides();
      buffer.base = alloca(buffer.size_bytes());
      buffer.allocation = nullptr;
    } else {
      assert(op->storage == memory_type::heap);
      buffer.allocation = context.allocate(op->sym, &buffer);
    }

    auto set_buffer = set_value_in_scope(context, op->sym, reinterpret_cast<index_t>(&buffer));
    index_t result = eval(op->body);

    if (op->storage == memory_type::heap) {
      context.free(op->sym, &buffer, buffer.allocation);
    }

    return result;
  }

  SLINKY_NO_STACK_PROTECTOR index_t eval(const make_buffer* op) {
    std::size_t rank = op->dims.size();
    raw_buffer buffer;
    buffer.elem_size = eval(op->elem_size, 0);
    buffer.base = reinterpret_cast<void*>(eval(op->base, 0));
    buffer.rank = rank;
    buffer.dims = SLINKY_ALLOCA(dim, rank);

    for (std::size_t d = 0; d < rank; ++d) {
      buffer.dim(d) = eval(op->dims[d]);
    }

    auto set_buffer = set_value_in_scope(context, op->sym, reinterpret_cast<index_t>(&buffer));
    return eval(op->body);
  }

  index_t eval(const clone_buffer* op) {
    raw_buffer* src_buf = reinterpret_cast<raw_buffer*>(*context.lookup(op->src));
    assert(src_buf);

    raw_buffer clone = *src_buf;
    clone.dims = SLINKY_ALLOCA(dim, src_buf->rank);
    memcpy(clone.dims, src_buf->dims, sizeof(dim) * src_buf->rank);
    auto set_buffer = set_value_in_scope(context, op->sym, reinterpret_cast<index_t>(&clone));
    return eval(op->body);
  }

  SLINKY_NO_STACK_PROTECTOR index_t eval_shadowed(const crop_buffer* op) {
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(*context.lookup(op->sym));
    assert(buffer);

    std::size_t crop_rank = op->bounds.size();
    interval* old_bounds = SLINKY_ALLOCA(interval, crop_rank);

    void* old_base = buffer->base;
    for (std::size_t d = 0; d < crop_rank; ++d) {
      slinky::dim& dim = buffer->dims[d];
      index_t old_min = dim.min();
      index_t old_max = dim.max();
      old_bounds[d].min = old_min;
      old_bounds[d].max = old_max;

      interval bounds = eval(op->bounds[d], {old_min, old_max});
      buffer->crop(d, bounds.min, bounds.max);
    }

    index_t result = eval(op->body);

    buffer->base = old_base;
    for (std::size_t d = 0; d < crop_rank; ++d) {
      slinky::dim& dim = buffer->dims[d];
      dim.set_bounds(old_bounds[d].min, old_bounds[d].max);
    }
    return result;
  }

  index_t eval_shadowed(const crop_dim* op) {
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(*context.lookup(op->sym));
    assert(buffer);
    slinky::dim& dim = buffer->dims[op->dim];
    index_t old_min = dim.min();
    index_t old_max = dim.max();
    void* old_base = buffer->base;

    interval bounds = eval(op->bounds, {old_min, old_max});
    buffer->crop(op->dim, bounds.min, bounds.max);
    index_t result = eval(op->body);

    buffer->base = old_base;
    dim.set_bounds(old_min, old_max);

    return result;
  }

  SLINKY_NO_STACK_PROTECTOR index_t eval_shadowed(const slice_buffer* op) {
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(*context.lookup(op->sym));
    assert(buffer);
    assert(op->at.size() <= buffer->rank);

    // TODO: If we really care about stack usage here, we could find the number of dimensions we actually need first.
    dim* dims = SLINKY_ALLOCA(dim, buffer->rank);

    std::size_t rank = 0;
    void* old_base = buffer->base;

    for (std::size_t d = 0; d < buffer->rank; ++d) {
      if (d < op->at.size() && op->at[d].defined()) {
        if (buffer->base) {
          index_t at_d = eval(op->at[d]);
          if (buffer->dims[d].contains(at_d)) {
            buffer->base = offset_bytes_non_null(buffer->base, buffer->dims[d].flat_offset_bytes(at_d));
          } else {
            buffer->base = nullptr;
          }
        }
      } else {
        dims[rank++] = buffer->dims[d];
      }
    }

    std::swap(buffer->rank, rank);
    std::swap(buffer->dims, dims);

    index_t result = eval(op->body);

    buffer->base = old_base;
    buffer->rank = rank;
    buffer->dims = dims;

    return result;
  }

  SLINKY_NO_STACK_PROTECTOR index_t eval_shadowed(const slice_dim* op) {
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(*context.lookup(op->sym));
    assert(buffer);
    assert(op->dim < static_cast<int>(buffer->rank));

    // The rank of the result is equal to the current rank, less any sliced dimensions.
    dim* old_dims = buffer->dims;

    buffer->dims = SLINKY_ALLOCA(dim, buffer->rank - 1);

    void* old_base = buffer->base;
    if (buffer->base) {
      index_t at = eval(op->at);
      if (old_dims[op->dim].contains(at)) {
        buffer->base = offset_bytes_non_null(buffer->base, old_dims[op->dim].flat_offset_bytes(at));
      } else {
        buffer->base = nullptr;
      }
    }

    for (int d = 0; d < op->dim; ++d) {
      buffer->dims[d] = old_dims[d];
    }
    for (int d = op->dim + 1; d < static_cast<int>(buffer->rank); ++d) {
      buffer->dims[d - 1] = old_dims[d];
    }
    buffer->rank -= 1;

    index_t result = eval(op->body);

    buffer->base = old_base;
    buffer->rank += 1;
    buffer->dims = old_dims;

    return result;
  }

  index_t eval_shadowed(const transpose* op) {
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(*context.lookup(op->sym));
    assert(buffer);

    std::size_t old_rank = buffer->rank;
    buffer->rank = op->dims.size();

    index_t result;
    if (op->is_truncate()) {
      result = eval(op->body);
    } else {
      dim* dims = buffer->dims;
      buffer->dims = SLINKY_ALLOCA(dim, buffer->rank);

      for (std::size_t i = 0; i < op->dims.size(); ++i) {
        buffer->dims[i] = dims[op->dims[i]];
      }

      result = eval(op->body);

      buffer->dims = dims;
    }
    buffer->rank = old_rank;
    return result;
  }

  template <typename T>
  index_t eval_maybe_shadowed(const T* op) {
    if (op->sym == op->src) {
      // The operation is shadowed, we can use eval_shadowed.
      return eval_shadowed(op);
    } else {
      // The operation is not shadowed. Make a clone and use eval_shadowed on the clone. This is not as efficient as it
      // could be, but the shadowed case should be faster, so we'll optimize for that case, and prefer that case when
      // constructing programs.
      raw_buffer* src_buf = reinterpret_cast<raw_buffer*>(*context.lookup(op->src));
      assert(src_buf);

      raw_buffer clone = *src_buf;
      clone.dims = SLINKY_ALLOCA(dim, src_buf->rank);
      memcpy(clone.dims, src_buf->dims, sizeof(dim) * src_buf->rank);
      auto set_buffer = set_value_in_scope(context, op->sym, reinterpret_cast<index_t>(&clone));
      return eval_shadowed(op);
    }
  }

  index_t eval(const crop_buffer* op) { return eval_maybe_shadowed(op); }
  index_t eval(const crop_dim* op) { return eval_maybe_shadowed(op); }
  index_t eval(const slice_buffer* op) { return eval_maybe_shadowed(op); }
  index_t eval(const slice_dim* op) { return eval_maybe_shadowed(op); }
  index_t eval(const transpose* op) { return eval_maybe_shadowed(op); }

  index_t eval(const check* op) {
    if (!eval(op->condition, 0)) {
      if (context.check_failed) {
        context.check_failed(op->condition);
      } else {
        std::cerr << "Check failed: " << op->condition << std::endl;
        std::cerr << "Context: " << std::endl;
        dump_context_for_expr(std::cerr, context, op->condition);
        std::abort();
      }
      return 1;
    } else {
      return 0;
    }
  }
};

}  // namespace

index_t evaluate(const expr& e, eval_context& context) {
  evaluator eval(context);
  return eval.eval(e);
}

index_t evaluate(const stmt& s, eval_context& context) {
  evaluator eval(context);
  return eval.eval(s);
}

index_t evaluate(const expr& e) {
  eval_context ctx;
  return evaluate(e, ctx);
}

index_t evaluate(const stmt& s) {
  eval_context ctx;
  return evaluate(s, ctx);
}

namespace {

class constant_evaluator : public expr_visitor {
public:
  std::optional<index_t> result;

  std::optional<index_t> eval(const expr& e) {
    if (e.defined()) {
      e.accept(this);
      return result;
    } else {
      return std::nullopt;
    }
  }

  void visit(const variable* op) override { result = std::nullopt; }
  void visit(const constant* op) override { result = op->value; }

  void visit(const let* op) override { result = std::nullopt; }

  template <typename T>
  void visit_binary(const T* op) {
    std::optional<index_t> a = eval(op->a);
    std::optional<index_t> b = eval(op->b);
    if (a && b) {
      result = make_binary<T>(*a, *b);
    } else {
      result = std::nullopt;
    }
  }

  void visit(const add* op) override { visit_binary(op); }
  void visit(const sub* op) override { visit_binary(op); }
  void visit(const mul* op) override { visit_binary(op); }
  void visit(const div* op) override { visit_binary(op); }
  void visit(const mod* op) override { visit_binary(op); }
  void visit(const class min* op) override { visit_binary(op); }
  void visit(const class max* op) override { visit_binary(op); }
  void visit(const equal* op) override { visit_binary(op); }
  void visit(const not_equal* op) override { visit_binary(op); }
  void visit(const less* op) override { visit_binary(op); }
  void visit(const less_equal* op) override { visit_binary(op); }
  void visit(const logical_and* op) override { visit_binary(op); }
  void visit(const logical_or* op) override { visit_binary(op); }
  void visit(const logical_not* op) override {
    std::optional<index_t> a = eval(op->a);
    if (a) {
      result = *a == 0;
    } else {
      result = std::nullopt;
    }
  }

  void visit(const class select* op) override {
    std::optional<index_t> c = eval(op->condition);
    std::optional<index_t> t = eval(op->true_value);
    std::optional<index_t> f = eval(op->false_value);
    if (c && *c && t) {
      result = *t;
    } else if (c && !*c && f) {
      result = *f;
    } else {
      result = std::nullopt;
    }
  }

  void visit(const call* op) override {
    switch (op->intrinsic) {
    case intrinsic::abs: {
      assert(op->args.size() == 1);
      std::optional<index_t> x = eval(op->args[0]);
      if (x) {
        result = std::abs(*x);
      } else {
        result = std::nullopt;
      }
      return;
    }
    case intrinsic::and_then:
    case intrinsic::or_else: {
      for (const expr& i : op->args) {
        std::optional<index_t> x = eval(i);
        if (x) {
          if (x && !*x && op->intrinsic == intrinsic::and_then) {
            result = false;
            return;
          } else if (x && *x && op->intrinsic == intrinsic::or_else) {
            result = true;
            return;
          }
        } else {
          result = std::nullopt;
          return;
        }
      }
      result = op->intrinsic == intrinsic::and_then;
      return;
    }

    default: result = std::nullopt; return;
    }
  }
};

}  // namespace

std::optional<index_t> evaluate_constant(const expr& e) { return constant_evaluator().eval(e); }

}  // namespace slinky
