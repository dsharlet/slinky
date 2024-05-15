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

BENCHMARK(BM_memset)->Arg(1024);

void BM_fill(benchmark::State& state) {
  std::vector<index_t> extents = state_to_vector(4, state);
  buffer<void, 3> dst(3, extents[0]);
  extents.erase(extents.begin());
  allocate_buffer(dst, extents);

  int five = 5;

  for (auto _ : state) {
    fill(dst, &five);
  }
}

BENCHMARK(BM_fill)->Args({1, 256, 4, -1});
BENCHMARK(BM_fill)->Args({2, 128, 4, -1});
BENCHMARK(BM_fill)->Args({4, 64, 4, -1});
BENCHMARK(BM_fill)->Args({1, 64, 4, 4});
BENCHMARK(BM_fill)->Args({2, 32, 4, 4});
BENCHMARK(BM_fill)->Args({4, 16, 4, 4});

void BM_fill_padded(benchmark::State& state) {
  buffer<char, 3> dst;
  allocate_buffer(dst, state_to_vector(3, state), padding_size);

  char five = 5;

  for (auto _ : state) {
    fill(dst, &five);
  }
}

BENCHMARK(BM_fill_padded)->Args({256, 4, -1});
BENCHMARK(BM_fill_padded)->Args({64, 4, 4});

void BM_pad(benchmark::State& state) {
  std::vector<index_t> extents = state_to_vector(3, state);
  buffer<char, 3> dst;
  allocate_buffer(dst, extents);

  buffer<char, 3> src(extents);
  for (std::size_t d = 0; d < src.rank; ++d) {
    src.dim(d).set_bounds(1, extents[d] - 2);
  }

  char five = 0;

  for (auto _ : state) {
    pad(src.dims, dst, &five);
  }
}

BENCHMARK(BM_pad)->Args({256, 4, -1});
BENCHMARK(BM_pad)->Args({64, 4, 4});

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

BENCHMARK(BM_memcpy)->Arg(1024);

void BM_copy(benchmark::State& state) {
  std::vector<index_t> extents = state_to_vector(3, state);
  buffer<char, 3> src;
  buffer<char, 3> dst;
  allocate_buffer(src, extents);
  allocate_buffer(dst, extents);

  for (auto _ : state) {
    copy(src, dst);
  }
}

BENCHMARK(BM_copy)->Args({256, 4, -1});
BENCHMARK(BM_copy)->Args({64, 4, 4});

void BM_copy_padded(benchmark::State& state) {
  std::vector<index_t> extents = state_to_vector(3, state);
  buffer<char, 3> src;
  buffer<char, 3> dst;
  allocate_buffer(src, extents);
  allocate_buffer(dst, extents, padding_size);

  for (auto _ : state) {
    copy(src, dst);
  }
}

BENCHMARK(BM_copy_padded)->Args({256, 4, -1});
BENCHMARK(BM_copy_padded)->Args({64, 4, 4});

constexpr index_t slice_extent = 64;

void memset_slice(index_t extent, void* base) { memset(base, 0, extent); }

template <typename Fn>
void BM_for_each_element_1x(benchmark::State& state, Fn fn) {
  std::vector<index_t> extents = state_to_vector(3, state);
  buffer<char, 3> buf;
  allocate_buffer(buf, extents, padding_size);

  assert(buf.dim(0).extent() * buf.elem_size == slice_extent);
  auto fn_wrapper = [fn = std::move(fn)](void* a) { fn(slice_extent, a); };

  buf.slice(0);
  for (auto _ : state) {
    for_each_element(fn_wrapper, buf);
  }
}

template <typename Fn>
void BM_for_each_element_fused_1x(benchmark::State& state, Fn fn) {
  std::vector<index_t> extents = state_to_vector(3, state);
  buffer<char, 3> buf;
  allocate_buffer(buf, extents, padding_size);

  slinky::dim buf_fused_dims[3];
  for (auto _ : state) {
    raw_buffer buf_fused = buf;
    buf_fused.dims = &buf_fused_dims[0];
    std::copy_n(buf.dims, buf.rank, buf_fused.dims);
    // TODO: If this can be made as fast as `for_each_contiguous_slice`, maybe we should just get rid of that helper in
    // favor of this combination.
    optimize_dims(buf_fused);
    index_t dim0_size = buf_fused.dim(0).extent() * buf.elem_size;
    buf_fused.slice(0);
    for_each_element([=](void* x) { fn(dim0_size, x); }, buf_fused);
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
void BM_for_each_element_hardcoded_1x(benchmark::State& state, Fn fn) {
  std::vector<index_t> extents = state_to_vector(3, state);
  buffer<char, 3> buf;
  allocate_buffer(buf, extents, padding_size);

  for (auto _ : state) {
    index_t dim0_size = buf.dim(0).extent() * buf.elem_size;
    char* base_i = buf.base();
    for (index_t i = 0; i < buf.dim(2).extent(); ++i, base_i += buf.dim(2).stride()) {
      char* base_j = base_i;
      for (index_t j = 0; j < buf.dim(1).extent(); ++j, base_j += buf.dim(1).stride()) {
        fn(dim0_size, base_j);
      }
    }
  }
}

// The difference between these two benchmarks on the same size buffer gives an indication of how much time is spent in
// overhead inside for_each_contiguous_slice.
void BM_fill_for_each_element(benchmark::State& state) { BM_for_each_element_1x(state, memset_slice); }
void BM_fill_for_each_element_fused(benchmark::State& state) { BM_for_each_element_fused_1x(state, memset_slice); }
void BM_fill_for_each_contiguous_slice(benchmark::State& state) {
  BM_for_each_contiguous_slice_1x(state, memset_slice);
}
void BM_fill_for_each_element_hardcoded(benchmark::State& state) {
  BM_for_each_element_hardcoded_1x(state, memset_slice);
}

BENCHMARK(BM_fill_for_each_element)->Args({slice_extent, 16, 1});
BENCHMARK(BM_fill_for_each_element_fused)->Args({slice_extent, 16, 1});
BENCHMARK(BM_fill_for_each_contiguous_slice)->Args({slice_extent, 16, 1});
BENCHMARK(BM_fill_for_each_element_hardcoded)->Args({slice_extent, 16, 1});
BENCHMARK(BM_fill_for_each_element)->Args({slice_extent, 4, 4});
BENCHMARK(BM_fill_for_each_element_fused)->Args({slice_extent, 4, 4});
BENCHMARK(BM_fill_for_each_contiguous_slice)->Args({slice_extent, 4, 4});
BENCHMARK(BM_fill_for_each_element_hardcoded)->Args({slice_extent, 4, 4});

void memcpy_slice(index_t extent, void* dst, const void* src) { memcpy(dst, src, extent); }

template <typename Fn>
void BM_for_each_element_2x(benchmark::State& state, Fn fn) {
  std::vector<index_t> extents = state_to_vector(3, state);
  buffer<char, 3> dst;
  allocate_buffer(dst, extents, padding_size);

  buffer<char, 3> src;
  allocate_buffer(src, extents);

  char x = 42;
  fill(src, &x);

  assert(dst.dim(0).extent() == src.dim(0).extent());
  assert(dst.dim(0).extent() * dst.elem_size == slice_extent);
  auto fn_wrapper = [fn = std::move(fn)](void* a, const void* b) { fn(slice_extent, a, b); };

  dst.slice(0);
  src.slice(0);
  for (auto _ : state) {
    for_each_element(fn_wrapper, dst, src);
  }
}

template <typename Fn>
void BM_for_each_element_fused_2x(benchmark::State& state, Fn fn) {
  std::vector<index_t> extents = state_to_vector(3, state);
  buffer<char, 3> dst;
  allocate_buffer(dst, extents, padding_size);

  buffer<char, 3> src;
  allocate_buffer(src, extents);

  char x = 42;
  fill(src, &x);

  slinky::dim dst_fused_dims[3];
  slinky::dim src_fused_dims[3];

  for (auto _ : state) {
    raw_buffer dst_fused = dst;
    raw_buffer src_fused = src;
    dst_fused.dims = &dst_fused_dims[0];
    src_fused.dims = &src_fused_dims[0];
    std::copy_n(dst.dims, dst.rank, dst_fused.dims);
    std::copy_n(src.dims, src.rank, src_fused.dims);
    // TODO: If this can be made as fast as `for_each_contiguous_slice`, maybe we should just get rid of that helper in
    // favor of this combination.
    optimize_dims(dst_fused, src_fused);
    assert(dst_fused.dim(0).extent() == src_fused.dim(0).extent());
    index_t dim0_size = dst_fused.dim(0).extent() * dst_fused.elem_size;
    dst_fused.slice(0);
    src_fused.slice(0);
    for_each_element([=](void* a, const void* b) { fn(dim0_size, a, b); }, dst_fused, src_fused);
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
void BM_for_each_element_hardcoded_2x(benchmark::State& state, Fn fn) {
  std::vector<index_t> extents = state_to_vector(3, state);
  buffer<char, 3> dst;
  allocate_buffer(dst, extents, padding_size);

  buffer<char, 3> src;
  allocate_buffer(src, extents);

  char x = 42;
  fill(src, &x);

  for (auto _ : state) {
    index_t dim0_size = dst.dim(0).extent() * dst.elem_size;
    char* dst_i = dst.base();
    const char* src_i = src.base();
    for (index_t i = 0; i < dst.dim(2).extent(); ++i, dst_i += dst.dim(2).stride(), src_i += src.dim(2).stride()) {
      char* dst_j = dst_i;
      const char* src_j = src_i;
      for (index_t j = 0; j < dst.dim(1).extent(); ++j, dst_j += dst.dim(1).stride(), src_j += src.dim(1).stride()) {
        fn(dim0_size, dst_j, src_j);
      }
    }
  }
}
void BM_copy_for_each_element(benchmark::State& state) { BM_for_each_element_2x(state, memcpy_slice); }
void BM_copy_for_each_element_fused(benchmark::State& state) { BM_for_each_element_fused_2x(state, memcpy_slice); }
void BM_copy_for_each_contiguous_slice(benchmark::State& state) {
  BM_for_each_contiguous_slice_2x(state, memcpy_slice);
}
void BM_copy_for_each_element_hardcoded(benchmark::State& state) {
  BM_for_each_element_hardcoded_2x(state, memcpy_slice);
}

BENCHMARK(BM_copy_for_each_element)->Args({slice_extent, 16, 1});
BENCHMARK(BM_copy_for_each_element_fused)->Args({slice_extent, 16, 1});
BENCHMARK(BM_copy_for_each_contiguous_slice)->Args({slice_extent, 16, 1});
BENCHMARK(BM_copy_for_each_element_hardcoded)->Args({slice_extent, 16, 1});
BENCHMARK(BM_copy_for_each_element)->Args({slice_extent, 4, 4});
BENCHMARK(BM_copy_for_each_element_fused)->Args({slice_extent, 4, 4});
BENCHMARK(BM_copy_for_each_contiguous_slice)->Args({slice_extent, 4, 4});
BENCHMARK(BM_copy_for_each_element_hardcoded)->Args({slice_extent, 4, 4});

void BM_fill_batch_dims(benchmark::State& state) {
  buffer<char, 8> dst;
  allocate_buffer(dst, {slice_extent, 1, 1, 1, 1, 1, 1, 1});

  char five = 5;

  for (auto _ : state) {
    fill(dst, &five);
  }
}

BENCHMARK(BM_fill_batch_dims);

void BM_for_each_element_batch_dims(benchmark::State& state) {
  buffer<char, 8> dst;
  allocate_buffer(dst, {slice_extent, 1, 1, 1, 1, 1, 1, 1});

  for (auto _ : state) {
    raw_buffer dst_sliced = dst;
    dst_sliced.slice(0);
    for_each_element([](void* x) { memset_slice(slice_extent, x); }, dst_sliced);
  }
}

BENCHMARK(BM_for_each_element_batch_dims);

}  // namespace slinky
