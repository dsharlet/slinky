#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>

#include "runtime/buffer.h"

namespace slinky {

void memset_slice(void* base, index_t extent) { memset(base, 0, extent); }

template <typename Fn>
void BM_for_each_contiguous_slice(benchmark::State& state, Fn fn) {
  std::vector<index_t> extents = {state.range(0) + 16, state.range(1), state.range(2)};
  while (extents.back() == 1) {
    extents.pop_back();
  }
  buffer<char, 3> buf(extents);
  buf.allocate();
  buf.dim(0).set_extent(state.range(0));

  for (auto _ : state) {
    for_each_contiguous_slice(buf, fn);
  }
}

template <typename Fn>
void BM_for_each_slice_hardcoded(benchmark::State& state, Fn fn) {
  std::vector<index_t> extents = {state.range(0) + 16, state.range(1), state.range(2)};
  buffer<char, 3> buf(extents);
  buf.allocate();
  buf.dim(0).set_extent(state.range(0));

  for (auto _ : state) {
    char* base_i = buf.base();
    for (index_t i = 0; i < buf.dim(2).extent(); ++i, base_i += buf.dim(2).stride()) {
      char* base_j = base_i;
      for (index_t j = 0; j < buf.dim(1).extent(); ++j, base_j += buf.dim(1).stride()) {
        fn(base_j, buf.dim(0).extent());
      }
    }
  }
}

// The difference between these two benchmarks on the same size buffer gives an indication of how much time is spent in
// overhead inside for_each_contiguous_slice.
void BM_for_each_contiguous_slice(benchmark::State& state) { BM_for_each_contiguous_slice(state, memset_slice); }
void BM_for_each_slice_hardcoded(benchmark::State& state) { BM_for_each_slice_hardcoded(state, memset_slice); }

BENCHMARK(BM_for_each_contiguous_slice)->Args({1024, 16, 1});
BENCHMARK(BM_for_each_slice_hardcoded)->Args({1024, 16, 1});
BENCHMARK(BM_for_each_contiguous_slice)->Args({1024, 4, 4});
BENCHMARK(BM_for_each_slice_hardcoded)->Args({1024, 4, 4});

}  // namespace slinky
