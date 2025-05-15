#include <benchmark/benchmark.h>

#include <atomic>
#include <random>
#include <vector>

#include "base/thread_pool.h"

namespace slinky {

struct unshared {
  alignas(cache_line_size) int value;
};

void BM_parallel_for_overhead(benchmark::State& state) {
  const int workers = state.range(0);
  thread_pool_impl t(workers - 1);

  std::vector<unshared> values(workers);
  while (state.KeepRunningBatch(workers)) {
    t.parallel_for(workers, [&](int i) { values[i].value++; });
  }
}

BENCHMARK(BM_parallel_for_overhead)->RangeMultiplier(2)->Range(1, 32);

void BM_parallel_for(benchmark::State& state) {
  const int workers = state.range(0);
  thread_pool_impl t(workers - 1);

  const int n = 1000000;

  std::vector<unshared> values(workers);
  while (state.KeepRunningBatch(values.size())) {
    t.parallel_for(n, [&](int i) { values[i % workers].value++; });
  }
}

BENCHMARK(BM_parallel_for)->RangeMultiplier(2)->Range(1, 32);

void BM_parallel_for_nested(benchmark::State& state) {
  const int workers = state.range(0);
  thread_pool_impl t(workers - 1);

  const int n = 1000;

  std::vector<unshared> values(workers * workers);
  while (state.KeepRunningBatch(values.size())) {
    t.parallel_for(
        n, [&](int i) { t.parallel_for(n, [&](int j) { values[(i % workers) * workers + j % workers].value++; }); });
  }
}

BENCHMARK(BM_parallel_for_nested)->RangeMultiplier(2)->Range(1, 32);

int fibonacci(thread_pool_impl& p, int n, int parallel_threshold = 0) {
  if (n <= 1) {
    return 1;
  } else if (n < parallel_threshold) {
    return fibonacci(p, n - 1, parallel_threshold) + fibonacci(p, n - 2, parallel_threshold);
  } else {
    std::atomic<int> result = 0;
    p.parallel_for(
        2, [n, parallel_threshold, &p, &result](int i) { result += fibonacci(p, n - i - 1, parallel_threshold); });
    return result;
  }
}

void BM_fibonacci(benchmark::State& state, int n, int g) {
  const int workers = state.range(0);
  thread_pool_impl t(workers - 1);

  for (auto i : state) {
    fibonacci(t, n, g);
  }
}

void BM_fibonacci_fine(benchmark::State& state) { BM_fibonacci(state, 25, 8); }
void BM_fibonacci_coarse(benchmark::State& state) { BM_fibonacci(state, 32, 16); }
  
BENCHMARK(BM_fibonacci_fine)->RangeMultiplier(2)->Range(1, 32);
BENCHMARK(BM_fibonacci_coarse)->RangeMultiplier(2)->Range(1, 32);

}  // namespace slinky
