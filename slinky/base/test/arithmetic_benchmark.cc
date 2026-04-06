#include <benchmark/benchmark.h>
#include <cstdlib>

#include "slinky/base/arithmetic.h"

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
void BM_add_sat(benchmark::State& state) { BM_binary(state, add_sat<std::ptrdiff_t>); }
void BM_sub_sat(benchmark::State& state) { BM_binary(state, sub_sat<std::ptrdiff_t>); }
void BM_mul_sat(benchmark::State& state) { BM_binary(state, mul_sat<std::ptrdiff_t>); }
void BM_div_sat(benchmark::State& state) { BM_binary(state, div_sat<std::ptrdiff_t>); }
void BM_gcd(benchmark::State& state) { BM_binary(state, gcd<std::ptrdiff_t>); }
void BM_lcm(benchmark::State& state) { BM_binary(state, lcm<std::ptrdiff_t>); }

BENCHMARK(BM_euclidean_div);
BENCHMARK(BM_euclidean_mod);
BENCHMARK(BM_euclidean_div_positive_divisor);
BENCHMARK(BM_euclidean_mod_positive_modulus);
BENCHMARK(BM_add_sat);
BENCHMARK(BM_sub_sat);
BENCHMARK(BM_mul_sat);
BENCHMARK(BM_div_sat);
BENCHMARK(BM_gcd);
BENCHMARK(BM_lcm);

}  // namespace slinky
