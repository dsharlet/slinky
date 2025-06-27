#include <benchmark/benchmark.h>
#include <cstdlib>

#include "base/arithmetic.h"

namespace slinky {

template <typename Fn>
void BM_binary(benchmark::State& state, Fn&& fn) {
  std::vector<std::ptrdiff_t> values(1000);
  std::generate(values.begin(), values.end(), []() { return rand(); });
  while (state.KeepRunningBatch(values.size() - 1)) {
    for (size_t i = 1; i < values.size(); ++i) {
      benchmark::DoNotOptimize(fn(values[i - 1], values[i]));
    }
  }
}

void BM_euclidean_div(benchmark::State& state) {
  BM_binary(state, [](std::ptrdiff_t a, std::ptrdiff_t b) { return euclidean_div<std::ptrdiff_t>(a, b); });
}
void BM_euclidean_mod(benchmark::State& state) {
  BM_binary(state, [](std::ptrdiff_t a, std::ptrdiff_t b) { return euclidean_mod<std::ptrdiff_t>(a, b); });
}
void BM_euclidean_div_positive_divisor(benchmark::State& state) {
  BM_binary(state, euclidean_div_positive_divisor<std::ptrdiff_t>);
}
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
BENCHMARK(BM_euclidean_div_positive_divisor);
BENCHMARK(BM_euclidean_mod_positive_modulus);
BENCHMARK(BM_saturate_add);
BENCHMARK(BM_saturate_sub);
BENCHMARK(BM_saturate_mul);
BENCHMARK(BM_saturate_div);
BENCHMARK(BM_saturate_mod);
BENCHMARK(BM_gcd);
BENCHMARK(BM_lcm);

}  // namespace slinky
