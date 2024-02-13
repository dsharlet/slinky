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

void BM_memcpy(benchmark::State& state) {
  std::size_t size = state.range(0);
  char* src = new char[size];
  char* dst = new char[size];

  memset(src, 0, size);
  memset(dst, 0, size);

  for (auto _ : state) {
    no_inline([=]() { memcpy(dst, src, size); });
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
    no_inline([=]() { memset(dst, 0, size); });
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

constexpr index_t slice_extent = 64;

void memset_slice(void* base, index_t extent) { memset(base, 0, slice_extent); }

template <typename Fn>
void BM_for_each_slice(benchmark::State& state, Fn fn) {
  std::vector<index_t> extents = state_to_vector(3, state);
  extents[0] += 64;  // Insert padding after the first dimension.
  buffer<char, 3> buf(extents);
  buf.allocate();
  buf.dim(0).set_extent(state.range(0));

  auto fn_wrapper = [fn = std::move(fn)](const raw_buffer& buf) { fn(buf.base, slice_extent); };

  for (auto _ : state) {
    for_each_slice(1, buf, fn_wrapper);
  }
}

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
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * buf.size_bytes());
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
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * buf.size_bytes());
}

// The difference between these two benchmarks on the same size buffer gives an indication of how much time is spent in
// overhead inside for_each_contiguous_slice.
void BM_for_each_slice(benchmark::State& state) { BM_for_each_slice(state, memset_slice); }
void BM_for_each_contiguous_slice(benchmark::State& state) { BM_for_each_contiguous_slice(state, memset_slice); }
void BM_for_each_slice_hardcoded(benchmark::State& state) { BM_for_each_slice_hardcoded(state, memset_slice); }

BENCHMARK(BM_for_each_slice)->Args({slice_extent, 16, 1});
BENCHMARK(BM_for_each_contiguous_slice)->Args({slice_extent, 16, 1});
BENCHMARK(BM_for_each_slice_hardcoded)->Args({slice_extent, 16, 1});
BENCHMARK(BM_for_each_slice)->Args({slice_extent, 4, 4});
BENCHMARK(BM_for_each_contiguous_slice)->Args({slice_extent, 4, 4});
BENCHMARK(BM_for_each_slice_hardcoded)->Args({slice_extent, 4, 4});

void memcpy_slices(void* dst, index_t extent, void* src) { memcpy(dst, src, extent); }

template <typename Fn>
void BM_for_each_contiguous_slice_multi(benchmark::State& state, Fn fn) {
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
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * (dst.size_bytes() + src.size_bytes()));
}

void BM_for_each_contiguous_slice_multi(benchmark::State& state) {
  BM_for_each_contiguous_slice_multi(state, memcpy_slices);
}

BENCHMARK(BM_for_each_contiguous_slice_multi)->Args({slice_extent, 16, 1});
BENCHMARK(BM_for_each_contiguous_slice_multi)->Args({slice_extent, 4, 4});

void add_slices(void* dst, index_t extent, void* src1, void* src2) {
  const char* s1 = reinterpret_cast<const char*>(src1);
  const char* s2 = reinterpret_cast<const char*>(src2);
  char* d = reinterpret_cast<char*>(dst);
  for (index_t i = 0; i < extent; i++) {
    d[i] = s1[i] + s2[i];
  }
}

template <typename Fn>
void BM_for_each_contiguous_slice_multi3(benchmark::State& state, Fn fn) {
  std::vector<index_t> extents = state_to_vector(3, state);
  extents[0] += 64;  // Insert padding after the first dimension.

  buffer<char, 3> dst(extents);
  dst.allocate();
  dst.dim(0).set_extent(state.range(0));

  buffer<char, 3> src1(extents);
  src1.allocate();
  src1.dim(0).set_extent(state.range(0));
  buffer<char, 3> src2(extents);
  src2.allocate();
  src2.dim(0).set_extent(state.range(0));

  char x = 42;
  fill(src1, &x);
  fill(src2, &x);

  for (auto _ : state) {
    for_each_contiguous_slice(dst, fn, src1, src2);
  }
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations()) * (dst.size_bytes() + src1.size_bytes() + src2.size_bytes()));
}

void BM_for_each_contiguous_slice_multi3(benchmark::State& state) {
  BM_for_each_contiguous_slice_multi3(state, add_slices);
}

BENCHMARK(BM_for_each_contiguous_slice_multi3)->Args({slice_extent, 16, 1});
BENCHMARK(BM_for_each_contiguous_slice_multi3)->Args({slice_extent, 4, 4});

}  // namespace slinky
