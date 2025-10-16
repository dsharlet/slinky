#include <benchmark/benchmark.h>

#include <atomic>
#include <random>
#include <vector>

#include "slinky/base/thread_pool_impl.h"

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

}  // namespace slinky
