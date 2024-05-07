#include <benchmark/benchmark.h>

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "runtime/evaluate.h"
#include "runtime/expr.h"
#include "base/thread_pool.h"

namespace slinky {

node_context ctx;
var buf(ctx, "buf");
var src(ctx, "src");
var dst(ctx, "dst");
var buf2(ctx, "buf2");
var x(ctx, "x");
var y(ctx, "y");
var z(ctx, "z");
var w(ctx, "w");

constexpr index_t iterations = 100000;

// These benchmarks mostly work by generating nodes around a call counter, and wrapping that node with a loop.
stmt make_call_counter(std::atomic<int>& calls) {
  return call_stmt::make(
      [&](const call_stmt*, eval_context& ctx) -> index_t {
        ++calls;
        return 0;
      },
      {}, {}, {});
}

stmt make_loop(stmt body) { return loop::make(x, loop::serial, range(0, iterations), 1, body); }

// For nodes that need a buffer, we can add a buffer outside that loop, the cost of constructing it will be negligible.
stmt make_buf(var buf, int rank, stmt body) {
  std::vector<dim_expr> dims;
  for (int i = 0; i < rank; ++i) {
    dims.push_back({{0, 100}, i, dim::unfolded});
  }
  return make_buffer::make(buf, 0, 1, dims, body);
}

void BM_call(benchmark::State& state) {
  std::atomic<int> calls = 0;
  stmt body = make_loop(make_call_counter(calls));

  for (auto _ : state) {
    evaluate(body);
  }

  state.SetItemsProcessed(calls);
}

BENCHMARK(BM_call);

void BM_let(benchmark::State& state) {
  std::atomic<int> calls = 0;
  std::vector<std::pair<var, expr>> values = {{y, x}, {z, y}, {w, z}};
  values.resize(state.range(0));
  stmt body = make_loop(let_stmt::make(values, make_call_counter(calls)));

  for (auto _ : state) {
    evaluate(body);
  }

  state.SetItemsProcessed(calls);
}

BENCHMARK(BM_let)->DenseRange(1, 3);

void BM_block(benchmark::State& state) {
  std::atomic<int> calls = 0;
  std::vector<stmt> call_counters(state.range(0), make_call_counter(calls));
  stmt body = make_loop(block::make(call_counters));

  for (auto _ : state) {
    evaluate(body);
  }

  state.SetItemsProcessed(calls);
}

BENCHMARK(BM_block)->RangeMultiplier(2)->Range(2, 16);

void BM_crop_dim(benchmark::State& state) {
  std::atomic<int> calls = 0;
  stmt c = crop_dim::make(dst, state.range(0) ? src : dst, 0, {1, 10}, make_call_counter(calls));
  stmt l = make_loop(c);
  stmt body = make_buf(src, 3, make_buf(dst, 3, l));

  for (auto _ : state) {
    evaluate(body);
  }

  state.SetItemsProcessed(calls);
}

BENCHMARK(BM_crop_dim)->DenseRange(0, 1);

void BM_crop_buffer(benchmark::State& state) {
  std::atomic<int> calls = 0;
  stmt c = crop_buffer::make(dst, state.range(0) ? src : dst, {{1, 10}, {}, {2, 20}}, make_call_counter(calls));
  stmt l = make_loop(c);
  stmt body = make_buf(src, 3, make_buf(dst, 3, l));

  for (auto _ : state) {
    evaluate(body);
  }

  state.SetItemsProcessed(calls);
}

BENCHMARK(BM_crop_buffer)->DenseRange(0, 1);

void BM_slice_dim(benchmark::State& state) {
  std::atomic<int> calls = 0;
  stmt c = slice_dim::make(dst, state.range(0) ? src : dst, 1, 10, make_call_counter(calls));
  stmt l = make_loop(c);
  stmt body = make_buf(src, 3, make_buf(dst, 3, l));

  for (auto _ : state) {
    evaluate(body);
  }

  state.SetItemsProcessed(calls);
}

BENCHMARK(BM_slice_dim)->DenseRange(0, 1);

void BM_slice_buffer(benchmark::State& state) {
  std::atomic<int> calls = 0;
  stmt c = slice_buffer::make(dst, state.range(0) ? src : dst, {10, {}, 20}, make_call_counter(calls));
  stmt l = make_loop(c);
  stmt body = make_buf(src, 3, make_buf(dst, 3, l));

  for (auto _ : state) {
    evaluate(body);
  }

  state.SetItemsProcessed(calls);
}

BENCHMARK(BM_slice_buffer)->DenseRange(0, 1);

void BM_allocate(benchmark::State& state) {
  std::atomic<int> calls = 0;
  stmt c = allocate::make(buf, memory_type::stack, 1, {{{0, 100}, 1, dim::unfolded}}, make_call_counter(calls));
  stmt body = make_loop(c);

  for (auto _ : state) {
    evaluate(body);
  }

  state.SetItemsProcessed(calls);
}

BENCHMARK(BM_allocate);

void BM_make_buffer(benchmark::State& state) {
  std::atomic<int> calls = 0;
  stmt body = make_loop(make_buf(buf, 3, make_call_counter(calls)));

  for (auto _ : state) {
    evaluate(body);
  }

  state.SetItemsProcessed(calls);
}

BENCHMARK(BM_make_buffer);

void BM_buffer_metadata(benchmark::State& state) {
  std::atomic<int> calls = 0;
  std::vector<dim_expr> dims = {buffer_dim(buf, 0), buffer_dim(buf, 1), buffer_dim(buf, 2)};
  stmt clone = make_buffer::make(buf2, buffer_at(buf), buffer_elem_size(buf), dims, make_call_counter(calls));
  stmt body = make_buf(buf, 3, make_loop(clone));

  for (auto _ : state) {
    evaluate(body);
  }

  state.SetItemsProcessed(calls);
}

BENCHMARK(BM_buffer_metadata);

void benchmark_parallel_loop(benchmark::State& state, bool synchronize) {
  const int workers = state.range(0);

  std::atomic<int> calls = 0;
  stmt body = make_call_counter(calls);

  index_t sem = workers;
  if (synchronize) {
    body = block::make({check::make(semaphore_wait(reinterpret_cast<index_t>(&sem))), body,
        check::make(semaphore_signal(reinterpret_cast<index_t>(&sem)))});
  }
  body = loop::make(x, workers, range(0, iterations), 1, body);

  thread_pool t(workers);

  eval_context eval_ctx;
  eval_ctx.enqueue_many = [&](thread_pool::task f) { t.enqueue(t.thread_count(), std::move(f)); };
  eval_ctx.enqueue = [&](int n, thread_pool::task f) { t.enqueue(n, std::move(f)); };
  eval_ctx.wait_for = [&](const std::function<bool()>& f) { t.wait_for(f); };
  eval_ctx.atomic_call = [&](const thread_pool::task& f) { t.atomic_call(f); };

  for (auto _ : state) {
    evaluate(body, eval_ctx);
  }

  state.SetItemsProcessed(calls);
}

void BM_parallel_loop(benchmark::State& state) { benchmark_parallel_loop(state, /*synchronize=*/false); }
void BM_semaphores(benchmark::State& state) { benchmark_parallel_loop(state, /*synchronize=*/true); }

BENCHMARK(BM_parallel_loop)->RangeMultiplier(2)->Range(1, 16);
BENCHMARK(BM_semaphores)->RangeMultiplier(2)->Range(1, 16);

}  // namespace slinky
