#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>

#include "runtime/buffer.h"

namespace slinky {

constexpr int padding_size = 64;  // one cache line, barely matters

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

template <typename T, std::size_t N>
void allocate_buffer(buffer<T, N>& buf, const std::vector<index_t>& extents, index_t dim_0_padding = 0) {
  assert(extents.size() <= N);
  index_t stride = buf.elem_size;
  buf.rank = extents.size();
  for (std::size_t d = 0; d < extents.size(); ++d) {
    buf.dim(d).set_min_extent(0, extents[d]);
    buf.dim(d).set_stride(stride);
    stride *= buf.dim(d).extent() + (d == 0 ? dim_0_padding : 0);
  }
  buf.allocate();
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
  buffer<char, 4> dst;
  allocate_buffer(dst, state_to_vector(4, state));

  char five = 0;

  for (auto _ : state) {
    fill(dst, &five);
  }
}

BENCHMARK(BM_fill)->Args({1024, 256, 4, -1});
BENCHMARK(BM_fill)->Args({32, 32, 256, 4});

void BM_fill_padded(benchmark::State& state) {
  buffer<char, 4> dst;
  allocate_buffer(dst, state_to_vector(4, state), padding_size);

  char five = 0;

  for (auto _ : state) {
    fill(dst, &five);
  }
}

BENCHMARK(BM_fill_padded)->Args({1024, 256, 4, -1});
BENCHMARK(BM_fill_padded)->Args({32, 32, 256, 4});

void BM_pad(benchmark::State& state) {
  std::vector<index_t> extents = state_to_vector(4, state);
  buffer<char, 4> dst;
  allocate_buffer(dst, extents);

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
  buffer<char, 4> src;
  buffer<char, 4> dst;
  allocate_buffer(src, extents);
  allocate_buffer(dst, extents);

  for (auto _ : state) {
    copy(src, dst);
  }
}

BENCHMARK(BM_copy)->Args({1024, 256, 4, -1});
BENCHMARK(BM_copy)->Args({32, 32, 256, 4});

void BM_copy_padded(benchmark::State& state) {
  std::vector<index_t> extents = state_to_vector(4, state);
  buffer<char, 4> src;
  buffer<char, 4> dst;
  allocate_buffer(src, extents);
  allocate_buffer(dst, extents, padding_size);

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
  buffer<char, 3> buf;
  allocate_buffer(buf, extents, padding_size);

  auto fn_wrapper = [fn = std::move(fn)](const raw_buffer& buf) { fn(slice_extent, buf.base); };

  for (auto _ : state) {
    for_each_slice(1, buf, fn_wrapper);
  }
}

template <typename Fn>
void BM_for_each_slice_fused_1x(benchmark::State& state, Fn fn) {
  std::vector<index_t> extents = state_to_vector(3, state);
  buffer<char, 3> buf;
  allocate_buffer(buf, extents, padding_size);

  auto fn_wrapper = [fn = std::move(fn)](const raw_buffer& buf) { fn(slice_extent, buf.base); };

  slinky::dim buf_fused_dims[3];
  for (auto _ : state) {
    raw_buffer buf_fused = buf;
    buf_fused.dims = &buf_fused_dims[0];
    memcpy(buf_fused.dims, buf.dims, buf.rank * sizeof(slinky::dim));
    // TODO: If this can be made as fast as `for_each_contiguous_slice`, maybe we should just get rid of that helper in
    // favor of this combination.
    optimize_dims(buf_fused);
    for_each_slice(1, buf, fn_wrapper);
  }
}

template <typename Fn>
void BM_for_each_contiguous_slice_1x(benchmark::State& state, Fn fn) {
  std::vector<index_t> extents = state_to_vector(3, state);
  buffer<char, 3> buf;
  allocate_buffer(buf, extents, padding_size);

  for (auto _ : state) {
    for_each_contiguous_slice(buf, fn);
  }
}

template <typename Fn>
void BM_for_each_slice_hardcoded_1x(benchmark::State& state, Fn fn) {
  std::vector<index_t> extents = state_to_vector(3, state);
  buffer<char, 3> buf;
  allocate_buffer(buf, extents, padding_size);

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
void BM_fill_for_each_slice_fused(benchmark::State& state) { BM_for_each_slice_fused_1x(state, memset_slice); }
void BM_fill_for_each_contiguous_slice(benchmark::State& state) {
  BM_for_each_contiguous_slice_1x(state, memset_slice);
}
void BM_fill_for_each_slice_hardcoded(benchmark::State& state) { BM_for_each_slice_hardcoded_1x(state, memset_slice); }

BENCHMARK(BM_fill_for_each_slice)->Args({slice_extent, 16, 1});
BENCHMARK(BM_fill_for_each_slice_fused)->Args({slice_extent, 16, 1});
BENCHMARK(BM_fill_for_each_contiguous_slice)->Args({slice_extent, 16, 1});
BENCHMARK(BM_fill_for_each_slice_hardcoded)->Args({slice_extent, 16, 1});
BENCHMARK(BM_fill_for_each_slice)->Args({slice_extent, 4, 4});
BENCHMARK(BM_fill_for_each_slice_fused)->Args({slice_extent, 4, 4});
BENCHMARK(BM_fill_for_each_contiguous_slice)->Args({slice_extent, 4, 4});
BENCHMARK(BM_fill_for_each_slice_hardcoded)->Args({slice_extent, 4, 4});

void memcpy_slices(index_t extent, void* dst, const void* src) { memcpy(dst, src, extent); }

template <typename Fn>
void BM_for_each_slice_2x(benchmark::State& state, Fn fn) {
  std::vector<index_t> extents = state_to_vector(3, state);
  buffer<char, 3> dst;
  allocate_buffer(dst, extents, padding_size);

  buffer<char, 3> src;
  allocate_buffer(src, extents);

  char x = 42;
  fill(src, &x);

  auto fn_wrapper = [fn = std::move(fn)](
                        const raw_buffer& dst, const raw_buffer& src) { fn(slice_extent, dst.base, src.base); };

  for (auto _ : state) {
    for_each_slice(1, dst, fn_wrapper, src);
  }
}

template <typename Fn>
void BM_for_each_slice_fused_2x(benchmark::State& state, Fn fn) {
  std::vector<index_t> extents = state_to_vector(3, state);
  buffer<char, 3> dst;
  allocate_buffer(dst, extents, padding_size);

  buffer<char, 3> src;
  allocate_buffer(src, extents);

  char x = 42;
  fill(src, &x);

  auto fn_wrapper = [fn = std::move(fn)](
                        const raw_buffer& dst, const raw_buffer& src) { fn(slice_extent, dst.base, src.base); };

  slinky::dim dst_fused_dims[3];
  slinky::dim src_fused_dims[3];

  for (auto _ : state) {
    raw_buffer dst_fused = dst;
    raw_buffer src_fused = src;
    dst_fused.dims = &dst_fused_dims[0];
    src_fused.dims = &src_fused_dims[0];
    memcpy(dst_fused.dims, dst.dims, dst.rank * sizeof(slinky::dim));
    memcpy(src_fused.dims, src.dims, src.rank * sizeof(slinky::dim));
    // TODO: If this can be made as fast as `for_each_contiguous_slice`, maybe we should just get rid of that helper in
    // favor of this combination.
    optimize_dims(dst_fused, src_fused);
    for_each_slice(1, dst_fused, fn_wrapper, src_fused);
  }
}

template <typename Fn>
void BM_for_each_contiguous_slice_2x(benchmark::State& state, Fn fn) {
  std::vector<index_t> extents = state_to_vector(3, state);
  buffer<char, 3> dst;
  allocate_buffer(dst, extents, padding_size);

  buffer<char, 3> src;
  allocate_buffer(src, extents);

  char x = 42;
  fill(src, &x);

  for (auto _ : state) {
    for_each_contiguous_slice(dst, fn, src);
  }
}

template <typename Fn>
void BM_for_each_slice_hardcoded_2x(benchmark::State& state, Fn fn) {
  std::vector<index_t> extents = state_to_vector(3, state);
  buffer<char, 3> dst;
  allocate_buffer(dst, extents, padding_size);

  buffer<char, 3> src;
  allocate_buffer(src, extents);

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
void BM_copy_for_each_slice(benchmark::State& state) { BM_for_each_slice_2x(state, memcpy_slices); }
void BM_copy_for_each_slice_fused(benchmark::State& state) { BM_for_each_slice_fused_2x(state, memcpy_slices); }
void BM_copy_for_each_contiguous_slice(benchmark::State& state) {
  BM_for_each_contiguous_slice_2x(state, memcpy_slices);
}
void BM_copy_for_each_slice_hardcoded(benchmark::State& state) { BM_for_each_slice_hardcoded_2x(state, memcpy_slices); }

BENCHMARK(BM_copy_for_each_slice)->Args({slice_extent, 16, 1});
BENCHMARK(BM_copy_for_each_slice_fused)->Args({slice_extent, 16, 1});
BENCHMARK(BM_copy_for_each_contiguous_slice)->Args({slice_extent, 16, 1});
BENCHMARK(BM_copy_for_each_slice_hardcoded)->Args({slice_extent, 16, 1});
BENCHMARK(BM_copy_for_each_slice)->Args({slice_extent, 4, 4});
BENCHMARK(BM_copy_for_each_slice_fused)->Args({slice_extent, 4, 4});
BENCHMARK(BM_copy_for_each_contiguous_slice)->Args({slice_extent, 4, 4});
BENCHMARK(BM_copy_for_each_slice_hardcoded)->Args({slice_extent, 4, 4});

}  // namespace slinky
