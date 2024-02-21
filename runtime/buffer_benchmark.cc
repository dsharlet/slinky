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

template <typename Fn>
__attribute__((noinline)) void no_inline(Fn&& fn) {
  fn();
}

void BM_memset(benchmark::State& state) {
  std::size_t size = state.range(0);
  char* dst = new char[size];

  for (auto _ : state) {
    no_inline([=]() { memset(dst, 0, size); });
  }

  benchmark::DoNotOptimize(dst);

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
  dst.dim(0).set_min_extent(0, extents[0] + 16);
  dst.allocate();
  dst.dim(0).set_min_extent(0, extents[0]);

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

void BM_memcpy(benchmark::State& state) {
  std::size_t size = state.range(0);
  char* src = new char[size];
  char* dst = new char[size];

  memset(src, 0, size);
  memset(dst, 0, size);

  for (auto _ : state) {
    no_inline([=]() { memcpy(dst, src, size); });
  }

  benchmark::DoNotOptimize(src);
  benchmark::DoNotOptimize(dst);

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
  dst.dim(0).set_min_extent(0, extents[0]);

  for (auto _ : state) {
    copy(src, dst);
  }
}

BENCHMARK(BM_copy_padded)->Args({1024, 256, 4, -1});
BENCHMARK(BM_copy_padded)->Args({32, 32, 256, 4});

constexpr index_t slice_extent = 64;

void memset_slice(index_t, void* base) { memset(base, 0, slice_extent); }

template <typename Fn>
void BM_for_each_slice_1x(benchmark::State& state, Fn fn) {
  std::vector<index_t> extents = state_to_vector(3, state);
  extents[0] += 64;  // Insert padding after the first dimension.
  buffer<char, 3> buf(extents);
  buf.allocate();
  buf.dim(0).set_extent(state.range(0));

  auto fn_wrapper = [fn = std::move(fn)](const raw_buffer& buf) { fn(slice_extent, buf.base); };

  for (auto _ : state) {
    for_each_slice(1, buf, fn_wrapper);
  }
}

template <typename Fn>
void BM_for_each_contiguous_slice_1x(benchmark::State& state, Fn fn) {
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
void BM_for_each_slice_hardcoded_1x(benchmark::State& state, Fn fn) {
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
        fn(buf.dim(0).extent(), base_j);
      }
    }
  }
}

// The difference between these two benchmarks on the same size buffer gives an indication of how much time is spent in
// overhead inside for_each_contiguous_slice.
void BM_fill_for_each_slice(benchmark::State& state) { BM_for_each_slice_1x(state, memset_slice); }
void BM_fill_for_each_contiguous_slice(benchmark::State& state) { BM_for_each_contiguous_slice_1x(state, memset_slice); }
void BM_fill_for_each_slice_hardcoded(benchmark::State& state) { BM_for_each_slice_hardcoded_1x(state, memset_slice); }

BENCHMARK(BM_fill_for_each_slice)->Args({slice_extent, 16, 1});
BENCHMARK(BM_fill_for_each_contiguous_slice)->Args({slice_extent, 16, 1});
BENCHMARK(BM_fill_for_each_slice_hardcoded)->Args({slice_extent, 16, 1});
BENCHMARK(BM_fill_for_each_slice)->Args({slice_extent, 4, 4});
BENCHMARK(BM_fill_for_each_contiguous_slice)->Args({slice_extent, 4, 4});
BENCHMARK(BM_fill_for_each_slice_hardcoded)->Args({slice_extent, 4, 4});

void memcpy_slices(index_t extent, void* dst, const void* src) { memcpy(dst, src, extent); }

template <typename Fn>
void BM_for_each_slice_2x(benchmark::State& state, Fn fn) {
  std::vector<index_t> extents = state_to_vector(3, state);
  extents[0] += 64;  // Insert padding after the first dimension.
  buffer<char, 3> dst(extents);
  dst.allocate();
  dst.dim(0).set_extent(state.range(0));

  buffer<char, 3> src(extents);
  src.allocate();
  src.dim(0).set_extent(state.range(0));

  char x = 42;
  fill(src, &x);

  auto fn_wrapper = [fn = std::move(fn)](const raw_buffer& dst, const raw_buffer& src) { fn(slice_extent, dst.base, src.base); };

  for (auto _ : state) {
    for_each_slice(1, dst, fn_wrapper, src);
  }
}

template <typename Fn>
void BM_for_each_contiguous_slice_2x(benchmark::State& state, Fn fn) {
  std::vector<index_t> extents = state_to_vector(3, state);
  extents[0] += 64;  // Insert padding after the first dimension.
  buffer<char, 3> dst(extents);
  dst.allocate();
  dst.dim(0).set_extent(state.range(0));

  buffer<char, 3> src(extents);
  src.allocate();
  src.dim(0).set_extent(state.range(0));

  char x = 42;
  fill(src, &x);

  for (auto _ : state) {
    for_each_contiguous_slice(dst, fn, src);
  }
}

template <typename Fn>
void BM_for_each_slice_hardcoded_2x(benchmark::State& state, Fn fn) {
  std::vector<index_t> extents = state_to_vector(3, state);
  extents[0] += 64;  // Insert padding after the first dimension.
  buffer<char, 3> dst(extents);
  dst.allocate();
  dst.dim(0).set_extent(state.range(0));

  buffer<char, 3> src(extents);
  src.allocate();
  src.dim(0).set_extent(state.range(0));

  char x = 42;
  fill(src, &x);

  for (auto _ : state) {
    char* dst_i = dst.base();
    const char* src_i = src.base();
    for (index_t i = 0; i < dst.dim(2).extent(); ++i, dst_i += dst.dim(2).stride(), src_i += src.dim(2).stride()) {
      char* dst_j = dst_i;
      const char* src_j = src_i;
      for (index_t j = 0; j < dst.dim(1).extent(); ++j, dst_j += dst.dim(1).stride(), src_j += src.dim(1).stride()) {
        fn(dst.dim(0).extent(), dst_j, src_j);
      }
    }
  }
}
void BM_copy_for_each_slice(benchmark::State& state) {
  BM_for_each_slice_2x(state, memcpy_slices);
}
void BM_copy_for_each_contiguous_slice(benchmark::State& state) {
  BM_for_each_contiguous_slice_2x(state, memcpy_slices);
}
void BM_copy_for_each_slice_hardcoded(benchmark::State& state) {
  BM_for_each_slice_hardcoded_2x(state, memcpy_slices);
}

BENCHMARK(BM_copy_for_each_slice)->Args({slice_extent, 16, 1});
BENCHMARK(BM_copy_for_each_contiguous_slice)->Args({slice_extent, 16, 1});
BENCHMARK(BM_copy_for_each_slice_hardcoded)->Args({slice_extent, 16, 1});
BENCHMARK(BM_copy_for_each_slice)->Args({slice_extent, 4, 4});
BENCHMARK(BM_copy_for_each_contiguous_slice)->Args({slice_extent, 4, 4});
BENCHMARK(BM_copy_for_each_slice_hardcoded)->Args({slice_extent, 4, 4});

}  // namespace slinky
