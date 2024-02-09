#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>

#include "runtime/buffer.h"

namespace slinky {

std::vector<index_t> state_to_vector(std::size_t max_size, const benchmark::State& state) {
  std::vector<index_t> vec(max_size);
  for (std::size_t i = 0; i < max_size; ++i) {
    vec[i] = state.range(i);
  }
  while (vec.back() == -1) {
    vec.pop_back();
  }
  return vec;
}

void BM_memcpy(benchmark::State& state) {
  std::size_t size = state.range(0);
  char* src = new char[size];
  char* dst = new char[size];

  memset(src, 0, size);

  for (auto _ : state) {
    memcpy(dst, src, size);
  }

  delete[] src;
  delete[] dst;
}

BENCHMARK(BM_memcpy)->Arg(1024 * 1024);

void BM_copy(benchmark::State& state) {
  std::vector<index_t> extents = state_to_vector(4, state);
  buffer<char, 4> src(extents);
  buffer<char, 4> dst(extents);
  src.allocate();
  dst.allocate();

  for (auto _ : state) {
    copy(src, dst);
  }
}

BENCHMARK(BM_copy)->Args({1024, 256, 4, -1});
BENCHMARK(BM_copy)->Args({32, 32, 256, 4});

void BM_copy_padded(benchmark::State& state) {
  std::vector<index_t> extents = state_to_vector(4, state);
  buffer<char, 4> src(extents);
  buffer<char, 4> dst(extents);
  dst.dim(0).set_min_extent(0, extents[0] + 16);
  src.allocate();
  dst.allocate();

  for (auto _ : state) {
    copy(src, dst);
  }
}

BENCHMARK(BM_copy_padded)->Args({1024, 256, 4, -1});
BENCHMARK(BM_copy_padded)->Args({32, 32, 256, 4});

void BM_memset(benchmark::State& state) {
  std::size_t size = state.range(0);
  char* dst = new char[size];

  for (auto _ : state) {
    memset(dst, 0, size);
  }

  delete[] dst;
}

BENCHMARK(BM_memset)->Arg(1024 * 1024);

void BM_fill(benchmark::State& state) {
  std::vector<index_t> extents = state_to_vector(4, state);
  buffer<char, 4> dst(extents);
  dst.allocate();

  char five = 0;

  for (auto _ : state) {
    fill(dst, &five);
  }
}

BENCHMARK(BM_fill)->Args({1024, 256, 4, -1});
BENCHMARK(BM_fill)->Args({32, 32, 256, 4});

void BM_fill_padded(benchmark::State& state) {
  std::vector<index_t> extents = state_to_vector(4, state);
  buffer<char, 4> dst(extents);
  dst.dim(0).set_stride(dst.dim(0).stride() + 16);
  dst.allocate();

  char five = 0;

  for (auto _ : state) {
    fill(dst, &five);
  }
}

BENCHMARK(BM_fill_padded)->Args({1024, 256, 4, -1});
BENCHMARK(BM_fill_padded)->Args({32, 32, 256, 4});

void BM_pad(benchmark::State& state) {
  std::vector<index_t> extents = state_to_vector(4, state);
  buffer<char, 4> dst(extents);
  dst.allocate();

  buffer<char, 4> src(extents);
  for (std::size_t d = 0; d < src.rank; ++d) {
    src.dim(d).set_bounds(1, extents[d] - 1);
  }

  char five = 0;

  for (auto _ : state) {
    pad(src.dims, dst, &five);
  }
}

BENCHMARK(BM_pad)->Args({1024, 256, 4, -1});
BENCHMARK(BM_pad)->Args({32, 32, 256, 4});

void memset_slice(void* base, index_t extent) { memset(base, 0, extent); }

template <typename Fn>
void BM_for_each_contiguous_slice(benchmark::State& state, Fn fn) {
  std::vector<index_t> extents = state_to_vector(3, state);
  extents[0] += 64;  // Insert padding after the first dimension.
  buffer<char, 3> buf(extents);
  buf.allocate();
  buf.dim(0).set_extent(state.range(0));

  for (auto _ : state) {
    for_each_contiguous_slice(buf, fn);
  }
}

template <typename Fn>
void BM_for_each_slice_hardcoded(benchmark::State& state, Fn fn) {
  std::vector<index_t> extents = state_to_vector(3, state);
  extents[0] += 64;  // Insert padding after the first dimension.
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

BENCHMARK(BM_for_each_contiguous_slice)->Args({64, 16, 1});
BENCHMARK(BM_for_each_slice_hardcoded)->Args({64, 16, 1});
BENCHMARK(BM_for_each_contiguous_slice)->Args({64, 4, 4});
BENCHMARK(BM_for_each_slice_hardcoded)->Args({64, 4, 4});

}  // namespace slinky
