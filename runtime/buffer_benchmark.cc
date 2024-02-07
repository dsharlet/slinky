#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>

#include "runtime/buffer.h"

namespace slinky {

__attribute__((noinline)) void no_op_slice(void*, index_t) {}
__attribute__((noinline)) void memset_slice(void* base, index_t extent) { memset(base, 0, extent); }

template <typename Fn>
void BM_for_each_contiguous_slice(benchmark::State& state, Fn fn) {
  std::vector<index_t> extents = {state.range(0) + 16, state.range(1), state.range(2)};
  while (extents.back() == 0) {
    extents.pop_back();
  }
  buffer<char, 3> buf(extents);
  buf.allocate();
  buf.dim(0).set_extent(state.range(0));

  for (auto _ : state) {
    for_each_contiguous_slice(buf, fn);
  }
}

// The difference between these two benchmarks on the same size buffer gives an indication of how much time is spent in
// overhead inside for_each_contiguous_slice.
void BM_for_each_contiguous_slice_no_op(benchmark::State& state) { BM_for_each_contiguous_slice(state, no_op_slice); }
void BM_for_each_contiguous_slice_memset(benchmark::State& state) { BM_for_each_contiguous_slice(state, memset_slice); }

BENCHMARK(BM_for_each_contiguous_slice_no_op)->Args({1024, 16, 0});
BENCHMARK(BM_for_each_contiguous_slice_memset)->Args({1024, 16, 0});
BENCHMARK(BM_for_each_contiguous_slice_no_op)->Args({1024, 4, 4});
BENCHMARK(BM_for_each_contiguous_slice_memset)->Args({1024, 4, 4});

}  // namespace slinky
