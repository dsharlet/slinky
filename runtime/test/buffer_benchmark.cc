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

void BM_for_each_element_1x(benchmark::State& state) {
  std::vector<index_t> extents = state_to_vector(3, state);
  buffer<char, 3> buf;
  allocate_buffer(buf, extents, padding_size);

  buf.slice(0);
  for (auto _ : state) {
    for_each_element([&](const void*) {}, buf);
  }
}

void BM_for_each_contiguous_slice_1x(benchmark::State& state) {
  std::vector<index_t> extents = state_to_vector(3, state);
  buffer<char, 3> buf;
  allocate_buffer(buf, extents, padding_size);

  for (auto _ : state) {
    for_each_contiguous_slice(buf, [&](index_t, const void*) {});
  }
}

// The difference between these two benchmarks on the same size buffer gives an indication of how much time is spent in
// overhead inside for_each_contiguous_slice.
void BM_fill_for_each_element(benchmark::State& state) { BM_for_each_element_1x(state); }
void BM_fill_for_each_contiguous_slice(benchmark::State& state) { BM_for_each_contiguous_slice_1x(state); }

BENCHMARK(BM_fill_for_each_element)->Args({64, 16, 1});
BENCHMARK(BM_fill_for_each_contiguous_slice)->Args({64, 16, 1});
BENCHMARK(BM_fill_for_each_element)->Args({64, 4, 4});
BENCHMARK(BM_fill_for_each_contiguous_slice)->Args({64, 4, 4});

void memcpy_slice(index_t extent, void* dst, const void* src) { memcpy(dst, src, extent); }

void BM_for_each_element_2x(benchmark::State& state) {
  std::vector<index_t> extents = state_to_vector(3, state);
  buffer<char, 3> dst;
  allocate_buffer(dst, extents, padding_size);

  buffer<char, 3> src;
  allocate_buffer(src, extents);

  char x = 42;
  fill(src, &x);

  dst.slice(0);
  src.slice(0);
  for (auto _ : state) {
    for_each_element([&](const void*, const void*) {}, dst, src);
  }
}

void BM_for_each_contiguous_slice_2x(benchmark::State& state) {
  std::vector<index_t> extents = state_to_vector(3, state);
  buffer<char, 3> dst;
  allocate_buffer(dst, extents, padding_size);

  buffer<char, 3> src;
  allocate_buffer(src, extents);

  char x = 42;
  fill(src, &x);

  for (auto _ : state) {
    for_each_contiguous_slice(
        dst, [&](index_t, const void*, const void*) {}, src);
  }
}

void BM_copy_for_each_element(benchmark::State& state) { BM_for_each_element_2x(state); }
void BM_copy_for_each_contiguous_slice(benchmark::State& state) { BM_for_each_contiguous_slice_2x(state); }

BENCHMARK(BM_copy_for_each_element)->Args({64, 16, 1});
BENCHMARK(BM_copy_for_each_contiguous_slice)->Args({64, 16, 1});
BENCHMARK(BM_copy_for_each_element)->Args({64, 4, 4});
BENCHMARK(BM_copy_for_each_contiguous_slice)->Args({64, 4, 4});

void BM_fill_batch_dims(benchmark::State& state) {
  buffer<char, 8> dst;
  allocate_buffer(dst, {64, 1, 1, 1, 1, 1, 1, 1});

  char five = 5;

  for (auto _ : state) {
    fill(dst, &five);
  }
}

BENCHMARK(BM_fill_batch_dims);

void BM_for_each_element_batch_dims(benchmark::State& state) {
  buffer<char, 8> dst;
  allocate_buffer(dst, {64, 1, 1, 1, 1, 1, 1, 1});

  for (auto _ : state) {
    raw_buffer dst_sliced = dst;
    dst_sliced.slice(0);
    for_each_element([&](const void* x) {}, dst_sliced);
  }
}

BENCHMARK(BM_for_each_element_batch_dims);

void BM_init_strides(benchmark::State& state) {
  int extent0 = state.range(0);
  int extent1 = state.range(1);
  int extent2 = state.range(2);
  int extent3 = state.range(3);

  for (auto _ : state) {
    buffer<int, 4> buf;
    buf.dim(0).set_min_extent(0, extent0);
    buf.dim(1).set_min_extent(0, extent1);
    buf.dim(2).set_min_extent(0, extent2);
    buf.dim(3).set_min_extent(0, extent3);

    buf.init_strides();
  }
}

BENCHMARK(BM_init_strides)->Args({5, 4, 3, 2});
BENCHMARK(BM_init_strides)->Args({4, 3, 2, 1});
BENCHMARK(BM_init_strides)->Args({3, 2, 1, 1});
BENCHMARK(BM_init_strides)->Args({2, 1, 1, 1});
BENCHMARK(BM_init_strides)->Args({1, 1, 1, 1});

void BM_optimize_dims_1x(benchmark::State& state) {
  std::vector<index_t> extents = state_to_vector(3, state);
  buffer<char, 3> buf;
  allocate_buffer(buf, extents, padding_size);

  for (auto _ : state) {
    buffer<char, 3> buf_fused(buf);
    optimize_dims(buf_fused);
  }
}

void BM_optimize_dims_2x(benchmark::State& state) {
  std::vector<index_t> extents = state_to_vector(3, state);
  buffer<char, 3> dst;
  allocate_buffer(dst, extents, padding_size);

  buffer<char, 3> src;
  allocate_buffer(src, extents);

  for (auto _ : state) {
    buffer<char, 3> src_fused(src);
    buffer<char, 3> dst_fused(dst);
    optimize_dims(dst_fused, src_fused);
  }
}

BENCHMARK(BM_optimize_dims_1x)->Args({64, 16, 1});
BENCHMARK(BM_optimize_dims_1x)->Args({64, 4, 4});
BENCHMARK(BM_optimize_dims_2x)->Args({64, 16, 1});
BENCHMARK(BM_optimize_dims_2x)->Args({64, 4, 4});

}  // namespace slinky
