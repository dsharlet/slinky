#include <benchmark/benchmark.h>

#include "base/arithmetic.h"

namespace slinky {

template <typename Fn>
void BM_binary(benchmark::State& state, Fn&& fn) {
  std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> values(1000);
  std::generate(values.begin(), values.end(), []() { return std::make_pair(rand(), rand()); });
  while (state.KeepRunningBatch(values.size())) {
    for (const auto& i : values) {
      benchmark::DoNotOptimize(fn(i.first, i.second));
    }
  }
}

void BM_euclidean_div(benchmark::State& state) { BM_binary(state, euclidean_div<std::ptrdiff_t>); }
void BM_euclidean_mod(benchmark::State& state) { BM_binary(state, euclidean_mod<std::ptrdiff_t>); }
void BM_euclidean_mod_positive_modulus(benchmark::State& state) {
  BM_binary(state, euclidean_mod_positive_modulus<std::ptrdiff_t>);
}
void BM_saturate_add(benchmark::State& state) { BM_binary(state, saturate_add<std::ptrdiff_t>); }
void BM_saturate_sub(benchmark::State& state) { BM_binary(state, saturate_sub<std::ptrdiff_t>); }
void BM_saturate_mul(benchmark::State& state) { BM_binary(state, saturate_mul<std::ptrdiff_t>); }
void BM_saturate_div(benchmark::State& state) { BM_binary(state, saturate_div<std::ptrdiff_t>); }
void BM_saturate_mod(benchmark::State& state) { BM_binary(state, saturate_mod<std::ptrdiff_t>); }
void BM_gcd(benchmark::State& state) { BM_binary(state, gcd<std::ptrdiff_t>); }
void BM_lcm(benchmark::State& state) { BM_binary(state, lcm<std::ptrdiff_t>); }

BENCHMARK(BM_euclidean_div);
BENCHMARK(BM_euclidean_mod);
BENCHMARK(BM_euclidean_mod_positive_modulus);
BENCHMARK(BM_saturate_add);
BENCHMARK(BM_saturate_sub);
BENCHMARK(BM_saturate_mul);
BENCHMARK(BM_saturate_div);
BENCHMARK(BM_saturate_mod);
BENCHMARK(BM_gcd);
BENCHMARK(BM_lcm);

}  // namespace slinky
