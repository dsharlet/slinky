#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>

#include "base/test/seeded_test.h"
#include "runtime/buffer.h"

namespace slinky {

bool operator==(const dim& a, const dim& b) { return memcmp(&a, &b, sizeof(dim)) == 0; }

template <typename Rng>
int random(Rng& rng, int min, int max) {
  return rng() % (max - min + 1) + min;
}

template <typename T, std::size_t N, typename Rng>
void init_random(Rng& rng, buffer<T, N>& buf) {
  buf.allocate();
  std::size_t flat_size = buf.size_bytes();
  std::size_t i = 0;
  for (; i + 3 < flat_size; i += 4) {
    reinterpret_cast<int*>(buf.base())[i >> 2] = rng();
  }
  for (; i < flat_size; ++i) {
    reinterpret_cast<char*>(buf.base())[i] = rng();
  }
}

template <typename F>
void for_each_index(span<const dim> dims, int d, index_t* is, const F& f) {
  if (d == 0) {
    for (index_t i = dims[0].begin(); i < dims[0].end(); ++i) {
      is[0] = i;
      f(span<const index_t>(is, is + dims.size()));
    }
  } else {
    for (index_t i = dims[d].begin(); i < dims[d].end(); ++i) {
      is[d] = i;
      for_each_index(dims, d - 1, is, f);
    }
  }
}

template <typename F>
SLINKY_NO_STACK_PROTECTOR void for_each_index(span<const dim> dims, const F& f) {
  if (dims.empty()) {
    f(span<const index_t>{});
  } else {
    index_t* i = SLINKY_ALLOCA(index_t, dims.size());
    for_each_index(dims, dims.size() - 1, i, f);
  }
}
template <typename F>
void for_each_index(const raw_buffer& buf, const F& f) {
  for_each_index(span<const dim>{buf.dims, buf.rank}, f);
}

template <typename T, std::size_t N, typename Value>
bool is_filled_buffer(const buffer<T, N>& buf, Value value) {
  int errors = 0;
  for_each_element([value, &errors](const T* x) { errors += *x != value; }, buf);
  return errors == 0;
}

struct randomize_options {
  int padding_min = 0;
  int padding_max = 3;
  bool allow_broadcast = false;
  bool allow_fold = false;
  bool randomize_rank = false;
};

template <typename T, std::size_t N, typename Rng>
void randomize_strides_and_padding(Rng& rng, buffer<T, N>& buf, const randomize_options& options) {
  std::vector<int> permutation(buf.rank);
  std::iota(permutation.begin(), permutation.end(), 0);
  if (random(rng, 0, 3) == 0) {
    // Randomize the strides ordering.
    std::shuffle(permutation.begin(), permutation.end(), rng);
  }

  index_t stride = buf.elem_size;
  for (std::size_t d : permutation) {
    slinky::dim& dim = buf.dim(d);
    // Expand the bounds randomly.
    dim.set_bounds(dim.min() - random(rng, options.padding_min, options.padding_max),
        dim.max() + random(rng, options.padding_min, options.padding_max));
    if (dim.extent() <= 0) {
      dim.set_extent(1);
    }
    if (options.allow_broadcast && random(rng, 0, 9) == 0) {
      dim = slinky::dim::broadcast();
    } else {
      dim.set_stride(stride);
      // Add some extra random padding.
      stride *= dim.extent() + random(rng, 0, 3) * buf.elem_size;
    }
    if (options.allow_fold && random(rng, 0, 9) == 0) {
      dim.set_fold_factor(random(rng, 1, 7));
    }
  }

  if (options.randomize_rank) {
    buf.rank = random(rng, 0, buf.rank);
  }
}

TEST(raw_buffer, make_copy) {
  gtest_seeded_mt19937 rng;

  buffer<int, 2> src({10, 20});
  init_random(rng, src);

  auto dst = raw_buffer::make_copy(src);
  ASSERT_EQ(src.rank, dst->rank);
  ASSERT_EQ(src.dim(0).min(), dst->dim(0).min());
  ASSERT_EQ(src.dim(0).extent(), dst->dim(0).extent());
  ASSERT_EQ(src.dim(1).min(), dst->dim(1).min());
  ASSERT_EQ(src.dim(1).extent(), dst->dim(1).extent());
  ASSERT_EQ(src.size_bytes(), dst->size_bytes());
  ASSERT_NE(src.base(), dst->base);

  for (int i = 0; i < dst->dim(1).extent(); ++i) {
    for (int j = 0; j < dst->dim(0).extent(); ++j) {
      ASSERT_EQ(src(j, i), *reinterpret_cast<int*>(dst->address_at(j, i)));
    }
  }
}

TEST(buffer, buffer) {
  buffer<int, 2> buf({10, 20});

  ASSERT_EQ(buf.rank, 2);

  ASSERT_EQ(buf.dim(0).min(), 0);
  ASSERT_EQ(buf.dim(0).extent(), 10);
  ASSERT_EQ(buf.dim(0).stride(), sizeof(int));
  ASSERT_EQ(buf.dim(0).fold_factor(), dim::unfolded);

  ASSERT_EQ(buf.dim(1).min(), 0);
  ASSERT_EQ(buf.dim(1).extent(), 20);
  ASSERT_EQ(buf.dim(1).stride(), buf.dim(0).stride() * buf.dim(0).extent());
  ASSERT_EQ(buf.dim(1).fold_factor(), dim::unfolded);

  // buf should not have memory yet.
  ASSERT_EQ(buf.base(), nullptr);

  buf.allocate();

  for (int i = 0; i < buf.dim(1).extent(); ++i) {
    for (int j = 0; j < buf.dim(0).extent(); ++j) {
      buf(j, i) = i * 10 + j;
    }
  }

  for (int i = 0; i < 10 * 20; ++i) {
    ASSERT_EQ(i, buf.base()[i]);
  }
}

TEST(buffer, address_at_slice) {
  buffer<int, 2> buf;

  ASSERT_EQ(buf.rank, 2);

  buf.dim(0) = {4, 14};
  buf.dim(1) = {5, 20};
  buf.allocate();

  ASSERT_EQ(&buf(), buf.base());
  ASSERT_EQ(&buf(4), buf.base());
  ASSERT_EQ(&buf(4, 5), buf.base());
  ASSERT_EQ(&buf(slice, 5), buf.base());
}

TEST(buffer, folded_address_at_slice) {
  buffer<char, 2> buf;

  ASSERT_EQ(buf.rank, 2);

  buf.dim(0) = {4, 14, dim::auto_stride, 3};
  buf.dim(1) = {5, 20, dim::auto_stride, 6};
  buf.allocate();

  ASSERT_EQ(&buf(), buf.base());
  ASSERT_EQ(&buf(4), buf.base() + buf.dim(0).flat_offset_bytes(4));
  ASSERT_EQ(&buf(slice, 5), buf.base() + buf.dim(1).flat_offset_bytes(5));
}

TEST(buffer, empty_buffer) {
  buffer<int, 3> buf({1, 0, 2});

  ASSERT_EQ(buf.rank, 3);

  buf.allocate();
}

bool test_copy_broadcast(int elem_size, int size) {
  buffer<void, 1> buf({size}, elem_size);
  buf.allocate();
  std::vector<uint8_t> value(elem_size);
  std::iota(value.begin(), value.end(), 0);
  copy(*raw_buffer::make_scalar(value.size(), value.data()), buf);
  const uint8_t* data = reinterpret_cast<const uint8_t*>(buf.base());
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < elem_size; ++j) {
      if (*data++ != j) {
        return false;
      }
    }
  }
  return true;
}

TEST(buffer, copy_broadcast) {
  for (int elem_size : {1, 2, 3, 4, 8, 12, 16, 63, 64, 65}) {
    for (int size : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1024 / elem_size, 1024 * 1024 / elem_size}) {
      ASSERT_TRUE(test_copy_broadcast(elem_size, size)) << elem_size << " " << size;
    }
  }
}

TEST(buffer, pad_scalar) {
  scalar<int> dst(1);
  scalar<int> padding(2);
  pad(nullptr, dst, padding);
  ASSERT_EQ(dst.value, 1);
}

TEST(buffer, pad_1d) {
  buffer<int, 1> dst({10});
  dst.allocate();
  std::iota(dst.base(), dst.base() + dst.elem_count(), 0);
  scalar<int> padding(0);
  dim src(3, 6);
  pad(&src, dst, padding);
  ASSERT_THAT(span<int>(dst.base(), dst.elem_count()), testing::ElementsAre(0, 0, 0, 3, 4, 5, 6, 0, 0, 0));
}

TEST(buffer, pad_2d) {
  buffer<int, 2> dst({4, 4});
  dst.allocate();
  std::iota(dst.base(), dst.base() + dst.elem_count(), 0);
  scalar<int> padding(0);
  dim src[] = {{1, 2}, {1, 2}};
  pad(src, dst, padding);
  ASSERT_THAT(
      span<int>(dst.base(), dst.elem_count()), testing::ElementsAre(0, 0, 0, 0, 0, 5, 6, 0, 0, 9, 10, 0, 0, 0, 0, 0));
}

TEST(buffer, shallow_copy) {
  gtest_seeded_mt19937 rng;

  buffer<int, 2> buf({10, 20});
  init_random(rng, buf);
  buffer<int, 2> buf2 = buf;
  ASSERT_EQ(buf.base(), buf2.base());
  ASSERT_EQ(buf.elem_size, buf2.elem_size);
  ASSERT_EQ(buf.rank, buf2.rank);
  ASSERT_EQ(buf.dim(0), buf2.dim(0));
  ASSERT_EQ(buf.dim(1), buf2.dim(1));

  ASSERT_NE(buf.dims, buf2.dims);
}

TEST(buffer, shallow_copy_different_capacity) {
  gtest_seeded_mt19937 rng;

  buffer<int, 2> buf({10, 20});
  init_random(rng, buf);
  buffer<int, 3> buf2 = buf;
  ASSERT_EQ(buf.base(), buf2.base());
  ASSERT_EQ(buf.elem_size, buf2.elem_size);
  ASSERT_EQ(buf.rank, buf2.rank);
  ASSERT_EQ(buf.dim(0), buf2.dim(0));
  ASSERT_EQ(buf.dim(1), buf2.dim(1));

  ASSERT_NE(buf.dims, buf2.dims);
}

TEST(buffer, folded) {
  buffer<char, 2> buf({10, 20});
  ASSERT_EQ(buf.size_bytes(), 10 * 20);
  buf.dim(1).set_fold_factor(2);
  ASSERT_EQ(buf.size_bytes(), 10 * 2);
  buf.allocate();

  ASSERT_EQ(&buf(0, 0), &buf(0, 2));
}

TEST(buffer, rank0) {
  buffer<int> buf;
  ASSERT_EQ(buf.rank, 0);
  ASSERT_EQ(buf.size_bytes(), 4);

  // buf should not have memory yet.
  ASSERT_EQ(buf.base(), nullptr);

  buf.allocate();

  buf() = 3;
  ASSERT_EQ(buf(), 3);
}

TEST(buffer, slice_leading) {
  buffer<int, 5> buf({1, 2, 3, 4, 5});
  raw_buffer sliced = buf;

  sliced.slice(0);
  ASSERT_EQ(sliced.rank, 4);
  ASSERT_EQ(sliced.dims, buf.dims + 1);
  ASSERT_EQ(sliced.dim(0), buf.dim(1));
  ASSERT_EQ(sliced.dim(1), buf.dim(2));
  ASSERT_EQ(sliced.dim(2), buf.dim(3));
  ASSERT_EQ(sliced.dim(3), buf.dim(4));

  sliced.slice({0, 1});
  ASSERT_EQ(sliced.rank, 2);
  ASSERT_EQ(sliced.dims, buf.dims + 3);
  ASSERT_EQ(sliced.dim(0), buf.dim(3));
  ASSERT_EQ(sliced.dim(1), buf.dim(4));
}

TEST(buffer, slice_non_leading) {
  buffer<int, 3> buf({1, 2, 3});
  raw_buffer sliced = buf;

  sliced.slice(1);
  ASSERT_EQ(sliced.rank, 2);
  ASSERT_EQ(sliced.dims, buf.dims);
  ASSERT_EQ(sliced.dim(0), buf.dim(0));
  ASSERT_EQ(sliced.dim(1), buf.dim(2));
}

TEST(buffer, slice_leading_and_trailing) {
  buffer<int, 3> buf({1, 2, 3});
  raw_buffer sliced = buf;

  sliced.slice({0, 2});
  ASSERT_EQ(sliced.rank, 1);
  ASSERT_EQ(sliced.dims, buf.dims + 1);
  ASSERT_EQ(sliced.dim(0), buf.dim(1));
}

TEST(buffer, slice_0_2) {
  buffer<int, 5> buf({1, 2, 3, 4, 5});
  buffer<int, 5> sliced = buf;

  sliced.slice({0, 2});
  ASSERT_EQ(sliced.rank, 3);
  ASSERT_EQ(sliced.dim(0), buf.dim(1));
  ASSERT_EQ(sliced.dim(1), buf.dim(3));
  ASSERT_EQ(sliced.dim(2), buf.dim(4));
}

TEST(buffer, slice_0_2_4) {
  buffer<int, 6> buf({1, 2, 3, 4, 5, 6});
  buffer<int, 6> sliced = buf;

  sliced.slice({0, 2, 4});
  ASSERT_EQ(sliced.rank, 3);
  ASSERT_EQ(sliced.dim(0), buf.dim(1));
  ASSERT_EQ(sliced.dim(1), buf.dim(3));
  ASSERT_EQ(sliced.dim(2), buf.dim(5));
}

TEST(buffer, slice_1_3_5) {
  buffer<int, 6> buf({1, 2, 3, 4, 5, 6});
  buffer<int, 6> sliced = buf;

  sliced.slice({1, 3, 5});
  ASSERT_EQ(sliced.rank, 3);
  ASSERT_EQ(sliced.dim(0), buf.dim(0));
  ASSERT_EQ(sliced.dim(1), buf.dim(2));
  ASSERT_EQ(sliced.dim(2), buf.dim(4));
}

TEST(buffer, for_each_element_folded) {
  buffer<char, 1> buf({10});
  buf.dim(0).set_fold_factor(4);
  buf.allocate();
  int count = 0;
  for_each_element(
      [&](char* i) {
        *i = 7;
        count++;
      },
      buf);
  ASSERT_EQ(count, 10);
  ASSERT_TRUE(is_filled_buffer(buf, 7));
}

TEST(buffer, for_each_element_cropped) {
  buffer<char, 1> src({10});
  buffer<char, 1> dst({10});
  src.crop(0, 2, 6);
  dst.allocate();
  src.allocate();
  copy(scalar<char>(7), src);
  int total = 0;
  int in_bounds = 0;
  for_each_element(
      [&](char* o, const char* i) {
        *o = i ? *i : 0;
        in_bounds += i ? 1 : 0;
        ++total;
      },
      dst, src);
  ASSERT_EQ(total, 10);
  ASSERT_EQ(in_bounds, src.dim(0).extent());
}

TEST(buffer, for_each_contiguous_slice) {
  buffer<char, 3> buf({10, 20, 30});
  buf.allocate();
  int slices = 0;
  for_each_contiguous_slice(buf, [&](index_t slice_extent, char* slice) {
    std::fill_n(slice, slice_extent, 7);
    slices++;
  });
  ASSERT_EQ(slices, 1);
  ASSERT_TRUE(is_filled_buffer(buf, 7));
}

TEST(buffer, for_each_contiguous_slice_non_zero_min) {
  buffer<char, 3> buf({10, 20, 30});
  buf.allocate();
  buf.translate(1, 2, 3);
  int slices = 0;
  for_each_contiguous_slice(buf, [&](index_t slice_extent, char* slice) {
    std::fill_n(slice, slice_extent, 7);
    slices++;
  });
  ASSERT_EQ(slices, 1);
  ASSERT_TRUE(is_filled_buffer(buf, 7));
}

TEST(buffer, for_each_contiguous_folded) {
  buffer<char, 3> buf({10, 20, 30});
  buf.dim(1).set_fold_factor(4);
  buf.allocate();
  for (int crop_extent : {1, 2, 3, 4}) {
    buf.dim(1).set_min_extent(8, crop_extent);
    int slices = 0;
    for_each_contiguous_slice(buf, [&](index_t slice_extent, char* slice) {
      std::fill_n(slice, slice_extent, 7);
      slices++;
    });
    ASSERT_EQ(slices, 30);
    ASSERT_TRUE(is_filled_buffer(buf, 7));
  }
  // Also check an unaligned crop with the fold.
  buf.dim(1).set_min_extent(6, 4);
  int slices = 0;
  for_each_contiguous_slice(buf, [&](index_t slice_extent, char* slice) {
    std::fill_n(slice, slice_extent, 7);
    slices++;
  });
  ASSERT_EQ(slices, 120);
  ASSERT_TRUE(is_filled_buffer(buf, 7));
}

TEST(buffer, for_each_contiguous_folded_innermost) {
  buffer<char, 3> buf({10});
  buf.dim(0).set_fold_factor(4);
  buf.allocate();
  int slices = 0;
  for_each_contiguous_slice(buf, [&](index_t slice_extent, char* slice) {
    std::fill_n(slice, slice_extent, 7);
    slices++;
  });
  ASSERT_EQ(slices, 3);
  ASSERT_TRUE(is_filled_buffer(buf, 7));
}

TEST(buffer, for_each_contiguous_folded_innermost_dim_1) {
  buffer<char, 3> buf({10, 20});
  buf.dim(0).set_fold_factor(4);
  buf.init_strides();
  std::swap(buf.dim(0), buf.dim(1));
  buf.allocate();
  int slices = 0;
  for_each_contiguous_slice(buf, [&](index_t slice_extent, char* slice) {
    std::fill_n(slice, slice_extent, 7);
    slices++;
  });
  ASSERT_EQ(slices, 60);
  ASSERT_TRUE(is_filled_buffer(buf, 7));
}

TEST(buffer, for_each_contiguous_cropped) {
  buffer<char, 1> src({10});
  buffer<char, 1> dst({10});
  src.crop(0, 2, 6);
  dst.allocate();
  src.allocate();
  copy(scalar<char>(7), src);
  int slices = 0;
  int total = 0;
  int in_bounds = 0;
  for_each_contiguous_slice(
      dst,
      [&](index_t slice_extent, char* o, const char* i) {
        if (i) {
          std::copy_n(i, slice_extent, o);
        } else {
          std::fill_n(o, slice_extent, 0);
        }
        ++slices;
        in_bounds += i ? slice_extent : 0;
        total += slice_extent;
      },
      src);
  ASSERT_EQ(total, 10);
  ASSERT_EQ(in_bounds, src.dim(0).extent());
  ASSERT_EQ(slices, 3);
}

TEST(buffer, for_each_contiguous_slice_padded) {
  for (int padded_dim = 0; padded_dim < 2; ++padded_dim) {
    buffer<char, 3> buf({10, 20, 30});
    buf.allocate();
    buf.dim(padded_dim).set_bounds(0, 8);
    for_each_contiguous_slice(buf, [&](index_t slice_extent, char* slice) { std::fill_n(slice, slice_extent, 7); });
    ASSERT_TRUE(is_filled_buffer(buf, 7));
  }
}

TEST(buffer, for_each_contiguous_slice_non_innermost) {
  buffer<int, 3> buf({10, 20, 30});
  buf.allocate();
  std::swap(buf.dim(0), buf.dim(1));
  int slices = 0;
  for_each_contiguous_slice(buf, [&](index_t slice_extent, int* slice) {
    ASSERT_EQ(slice_extent, 10);
    slices++;
  });
  ASSERT_EQ(slices, buf.dim(0).extent() * buf.dim(2).extent());
}

template <typename T, typename Rng>
void test_for_each_contiguous_slice_fill(Rng& rng) {
  buffer<T, 4> dst;
  for (std::size_t d = 0; d < dst.rank; ++d) {
    dst.dim(d).set_min_extent(0, 5);
  }
  randomize_strides_and_padding(rng, dst, {-1, 1, false, true});
  dst.allocate();

  for_each_contiguous_slice(dst, [&](index_t slice_extent, T* dst) { std::fill_n(dst, slice_extent, 7); });

  ASSERT_TRUE(is_filled_buffer(dst, 7));
}

TEST(buffer, for_each_contiguous_slice_fill) {
  gtest_seeded_mt19937 rng;
  for (auto _ : fuzz_test(std::chrono::seconds(1))) {
    test_for_each_contiguous_slice_fill<char>(rng);
    test_for_each_contiguous_slice_fill<int>(rng);
  }
}

template <typename Src, typename Dst, typename Rng>
void test_for_each_contiguous_slice_copy(Rng& rng) {
  buffer<Src, 4> src;
  buffer<Dst, 4> dst;
  for (std::size_t d = 0; d < src.rank; ++d) {
    src.dim(d).set_min_extent(0, 3);
    dst.dim(d).set_min_extent(0, 3);
  }
  randomize_strides_and_padding(rng, src, {-1, 1, true, true, true});
  randomize_strides_and_padding(rng, dst, {-1, 1, false, false});
  init_random(rng, src);
  dst.allocate();

  for_each_contiguous_slice(
      dst,
      [&](index_t slice_extent, Dst* dst, const Src* src) {
        if (src) {
          std::copy_n(src, slice_extent, dst);
        } else {
          std::fill_n(dst, slice_extent, 0);
        }
      },
      src);

  for_each_index(dst, [&](const auto i) {
    auto src_i = i.subspan(0, src.rank);
    if (src.contains(src_i)) {
      ASSERT_EQ(dst(i), src(src_i));
    } else {
      ASSERT_EQ(dst(i), 0);
    }
  });
}

TEST(buffer, for_each_contiguous_slice_copy) {
  gtest_seeded_mt19937 rng;
  for (auto _ : fuzz_test(std::chrono::seconds(1))) {
    test_for_each_contiguous_slice_copy<char, char>(rng);
    test_for_each_contiguous_slice_copy<short, int>(rng);
    test_for_each_contiguous_slice_copy<int, int>(rng);
  }
}

template <typename Src, typename Dst, typename Rng>
void test_for_each_element_copy(Rng& rng) {
  buffer<Src, 4> src;
  buffer<Dst, 4> dst;
  for (std::size_t d = 0; d < src.rank; ++d) {
    src.dim(d).set_min_extent(0, 3);
    dst.dim(d).set_min_extent(0, 3);
  }
  randomize_strides_and_padding(rng, src, {-1, 1, true, true, true});
  randomize_strides_and_padding(rng, dst, {-1, 1, false, false});
  init_random(rng, src);
  dst.allocate();

  for_each_element([&](Dst* dst, const Src* src) { *dst = src ? *src : 0; }, dst, src);

  for_each_index(dst, [&](const auto i) {
    auto src_i = i.subspan(0, src.rank);
    if (src.contains(src_i)) {
      ASSERT_EQ(dst(i), src(src_i));
    } else {
      ASSERT_EQ(dst(i), 0);
    }
  });
}

TEST(buffer, for_each_element_copy) {
  gtest_seeded_mt19937 rng;
  for (auto _ : fuzz_test(std::chrono::seconds(1))) {
    test_for_each_element_copy<char, char>(rng);
    test_for_each_element_copy<short, int>(rng);
    test_for_each_element_copy<int, int>(rng);
  }
}

template <typename A, typename B, typename Dst, typename Rng>
void test_for_each_contiguous_slice_add(Rng& rng) {
  buffer<A, 4> a;
  buffer<B, 4> b;
  for (std::size_t d = 0; d < a.rank; ++d) {
    a.dim(d).set_min_extent(0, 5);
    b.dim(d).set_min_extent(0, 5);
  }

  buffer<Dst, 4> dst;
  for (std::size_t d = 0; d < a.rank; ++d) {
    dst.dim(d) = a.dim(d);
  }

  randomize_strides_and_padding(rng, a, {0, 1, true, true, true});
  randomize_strides_and_padding(rng, b, {0, 1, true, true, true});
  init_random(rng, a);
  init_random(rng, b);

  randomize_strides_and_padding(rng, dst, {-1, 0, false, false});
  dst.allocate();

  for_each_contiguous_slice(
      dst,
      [&](index_t slice_extent, Dst* dst, const A* a, const B* b) {
        for (index_t i = 0; i < slice_extent; ++i) {
          dst[i] = saturate_add<Dst>(a[i], b[i]);
        }
      },
      a, b);
  for_each_index(dst,
      [&](const auto i) { ASSERT_EQ(dst(i), saturate_add<Dst>(a(i.subspan(0, a.rank)), b(i.subspan(0, b.rank)))); });
}

TEST(buffer, for_each_contiguous_slice_add) {
  gtest_seeded_mt19937 rng;
  for (auto _ : fuzz_test(std::chrono::seconds(1))) {
    test_for_each_contiguous_slice_add<int, int, int>(rng);
    test_for_each_contiguous_slice_add<short, int, int>(rng);
    test_for_each_contiguous_slice_add<short, short, int>(rng);
  }
}

TEST(buffer, for_each_contiguous_slice_multi_fuse_lots) {
  // TODO: if/when buffer<> gets a move ctor, do this in a vector<>
  buffer<char, 3> buf1({10, 20, 30});
  buffer<char, 3> buf2({10, 20, 30});
  buffer<char, 3> buf3({10, 20, 30});
  buffer<char, 3> buf4({10, 20, 30});
  buffer<char, 3> buf5({10, 20, 30});
  buffer<char, 3> buf6({10, 20, 30});
  buffer<char, 3> buf7({10, 20, 30});
  buffer<char, 3> buf8({10, 20, 30});
  buffer<char, 3> buf9({10, 20, 30});
  buf1.allocate();
  buf2.allocate();
  buf3.allocate();
  buf4.allocate();
  buf5.allocate();
  buf6.allocate();
  buf7.allocate();
  buf8.allocate();
  buf9.allocate();
  int slices = 0;
  for_each_contiguous_slice(
      buf1,
      [&](index_t slice_extent, char* slice1, char* slice2, char* slice3, char* slice4, char* slice5, char* slice6,
          char* slice7, char* slice8, char* slice9) {
        std::fill_n(slice1, slice_extent, 1);
        std::fill_n(slice2, slice_extent, 2);
        std::fill_n(slice3, slice_extent, 3);
        std::fill_n(slice4, slice_extent, 4);
        std::fill_n(slice5, slice_extent, 5);
        std::fill_n(slice6, slice_extent, 6);
        std::fill_n(slice7, slice_extent, 7);
        std::fill_n(slice8, slice_extent, 8);
        std::fill_n(slice9, slice_extent, 9);
        slices++;
      },
      buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9);
  // These should fuse into a single slice
  ASSERT_EQ(slices, 1);
  ASSERT_TRUE(is_filled_buffer(buf1, 1));
  ASSERT_TRUE(is_filled_buffer(buf2, 2));
  ASSERT_TRUE(is_filled_buffer(buf3, 3));
  ASSERT_TRUE(is_filled_buffer(buf4, 4));
  ASSERT_TRUE(is_filled_buffer(buf5, 5));
  ASSERT_TRUE(is_filled_buffer(buf6, 6));
  ASSERT_TRUE(is_filled_buffer(buf7, 7));
  ASSERT_TRUE(is_filled_buffer(buf8, 8));
  ASSERT_TRUE(is_filled_buffer(buf9, 9));
}

TEST(buffer, for_each_element) {
  buffer<int, 2> buf({10, 20});
  buf.allocate();
  int elements = 0;
  for_each_element(
      [&](int* elt) {
        *elt = 7;
        elements++;
      },
      buf);
  int expected_elements = 1;
  for (std::size_t d = 0; d < buf.rank; ++d) {
    expected_elements *= buf.dim(d).extent();
  }
  ASSERT_EQ(elements, expected_elements);

  ASSERT_TRUE(is_filled_buffer(buf, 7));
}

TEST(buffer, for_each_element_rank_zero) {
  // Verify that zero-dimensional buffers work
  buffer<int, 0> buf({});
  buf.allocate();
  int elements = 0;
  for_each_element(
      [&](int* elt) {
        *elt = 1111;
        elements++;
      },
      buf);
  int expected_elements = 1;
  ASSERT_EQ(elements, expected_elements);
  ASSERT_EQ(buf(), 1111);
}

TEST(buffer, for_each_element_empty) {
  buffer<int, 2> buf({0, 20});
  buf.allocate();
  int elements = 0;
  for_each_element([&](int*) { elements++; }, buf);
  ASSERT_EQ(elements, 0);
}

template <typename T, std::size_t N>
void set_strides(buffer<T, N>& buf, int* permutation = nullptr, index_t* padding = nullptr, bool broadcast = false) {
  index_t stride = broadcast ? 0 : buf.elem_size;
  for (std::size_t i = 0; i < N; ++i) {
    dim& d = buf.dim(permutation ? permutation[i] : i);
    d.set_stride(stride);
    stride *= d.extent() + (padding ? padding[i] : 0);
    if (stride == 0) {
      stride = buf.elem_size;
    }
  }
}

TEST(buffer, for_each_element_fuzz) {
  gtest_seeded_mt19937 rng;

  for (auto _ : fuzz_test(std::chrono::seconds(1))) {
    constexpr int max_rank = 4;
    buffer<int, max_rank> bufs[3];
    for (buffer<int, max_rank>& buf : bufs) {
      buf.rank = random(rng, 0, max_rank);
      for (std::size_t d = 0; d < buf.rank; ++d) {
        buf.dim(d).set_bounds(random(rng, -4, 2), random(rng, -2, 4));
        buf.dim(d).set_stride(random(rng, 0, 4));
        buf.dim(d).set_fold_factor((rng() & 3) == 0 ? random(rng, 1, 4) : dim::unfolded);
      }
      buf.allocate();
    }
    for_each_element([](const void*, const void*, const void*) {}, bufs[0], bufs[1], bufs[2]);
    for_each_contiguous_slice(
        bufs[0], [](index_t, const void*, const void*, const void*) {}, bufs[1], bufs[2]);
  }
}

TEST(buffer, copy) {
  gtest_seeded_mt19937 rng;

  constexpr int max_rank = 4;
  for (auto _ : fuzz_test(std::chrono::seconds(1))) {
    int rank = random(rng, 0, max_rank);
    int elem_size = random(rng, 1, 12);

    buffer<void, max_rank> dst(rank, elem_size);
    for (int d = 0; d < rank; ++d) {
      dst.dim(d).set_min_extent(0, 5);
    }
    buffer<void, max_rank> src = dst;
    randomize_strides_and_padding(rng, src, {-1, 1, true, true});
    init_random(rng, src);

    // The padding can't be out of bounds, add one extra padding.
    buffer<void, max_rank> padding1 = dst;
    buffer<void, max_rank> padding2 = dst;
    randomize_strides_and_padding(rng, padding1, {1, 3, true, true});
    randomize_strides_and_padding(rng, padding2, {1, 3, true, true});
    init_random(rng, padding1);
    init_random(rng, padding2);

    randomize_strides_and_padding(rng, dst, {-1, 1, false, false});
    dst.allocate();

    slinky::copy(src, dst, padding1);
    for_each_index(dst, [&](auto i) {
      if (src.contains(i)) {
        ASSERT_EQ(memcmp(dst.address_at(i), src.address_at(i), elem_size), 0);
      } else {
        ASSERT_EQ(memcmp(dst.address_at(i), padding1.address_at(i), elem_size), 0);
      }
    });

    pad(src.dims, dst, padding2);
    for_each_index(dst, [&](auto i) {
      if (src.contains(i)) {
        // The src should not have been modified.
        ASSERT_EQ(memcmp(dst.address_at(i), src.address_at(i), elem_size), 0);
      } else {
        // But we should have new padding.
        ASSERT_EQ(memcmp(dst.address_at(i), padding2.address_at(i), elem_size), 0);
      }
    });
  }
}

TEST(buffer, copy_empty_src) {
  gtest_seeded_mt19937 rng;

  constexpr int rank = 4;
  constexpr int D = 5;

  for (int empty_dim = 0; empty_dim < rank; empty_dim++) {
    buffer<int, rank> src;
    for (int d = 0; d < rank; d++) {
      src.dim(0).set_min_extent(0, D);
    }
    src.dim(empty_dim).set_min_extent(std::numeric_limits<index_t>::max(), std::numeric_limits<index_t>::min());
    init_random(rng, src);

    buffer<int, rank> dst;
    for (int d = 0; d < rank; d++) {
      dst.dim(0).set_min_extent(0, D);
    }
    dst.allocate();
    copy(scalar<int>(7), dst);
    // The result of copying an empty buffer should be entirely padding.
    slinky::copy(src, dst, scalar<int>(3));
    ASSERT_TRUE(is_filled_buffer(dst, 3));
  }
}

TEST(fuse_contiguous_dims, same_rank) {
  buffer<int, 1> r1;
  buffer<int, 2> r2;
  buffer<int, 3> r3;

  ASSERT_TRUE(internal::same_rank(r1));
  ASSERT_TRUE(internal::same_rank(r2, r2));
  ASSERT_FALSE(internal::same_rank(r2, r1, r2));
  ASSERT_TRUE(internal::same_rank(r3, r3, r3));
}

TEST(fuse_contiguous_dims, fuse0) {
  buffer<int, 1> a({}), b({});
  fuse_contiguous_dims(a, b);
  ASSERT_EQ(a.rank, 0);
  ASSERT_EQ(b.rank, 0);
}

TEST(fuse_contiguous_dims, fuse1) {
  buffer<int, 1> a({3}), b({3});
  fuse_contiguous_dims(a, b);
  ASSERT_EQ(a.rank, 1);
  ASSERT_EQ(b.rank, 1);
  ASSERT_EQ(a.dim(0).extent(), 3);
  ASSERT_EQ(b.dim(0).extent(), 3);
}

TEST(fuse_contiguous_dims, fuse2) {
  buffer<int, 2> a({4, 5}), b({4, 5});
  fuse_contiguous_dims(a, b);
  ASSERT_EQ(a.rank, 1);
  ASSERT_EQ(b.rank, 1);
  ASSERT_EQ(a.dim(0).extent(), 4 * 5);
  ASSERT_EQ(b.dim(0).extent(), 4 * 5);
}

TEST(fuse_contiguous_dims, fuse2_with_broadcast) {
  buffer<int, 2> a({4, 5});
  buffer<int, 0> b;
  fuse_contiguous_dims(a, b);
  ASSERT_EQ(a.rank, 1);
  ASSERT_EQ(b.rank, 0);
  ASSERT_EQ(a.dim(0).extent(), 4 * 5);
}

TEST(fuse_contiguous_dims, fuse3) {
  buffer<int, 3> a({6, 7, 8}), b({6, 7, 8});
  fuse_contiguous_dims(a, b);
  ASSERT_EQ(a.rank, 1);
  ASSERT_EQ(b.rank, 1);
  ASSERT_EQ(a.dim(0).extent(), 6 * 7 * 8);
  ASSERT_EQ(b.dim(0).extent(), 6 * 7 * 8);
}

TEST(fuse_contiguous_dims, fuse_folded) {
  buffer<int, 3> a({6, 7, 8}), b({6, 7, 8});
  a.dim(2).set_fold_factor(3);
  fuse_contiguous_dims(a, b);
  ASSERT_EQ(a.rank, 1);
  ASSERT_EQ(b.rank, 1);
  ASSERT_EQ(a.dim(0).extent(), 6 * 7 * 8);
  ASSERT_EQ(a.dim(0).fold_factor(), 6 * 7 * 3);
  ASSERT_EQ(b.dim(0).extent(), 6 * 7 * 8);
  ASSERT_EQ(b.dim(0).fold_factor(), dim::unfolded);
}

TEST(fuse_contiguous_dims, fuse_broadcasted) {
  buffer<int, 3> a({6, 1, 1}), b({6, 1, 1});
  a.dim(1) = dim::broadcast();
  a.dim(2) = dim::broadcast();
  b.dim(1) = dim::broadcast();
  b.dim(2) = dim::broadcast();

  fuse_contiguous_dims(a, b);
  ASSERT_EQ(a.rank, 2);
  ASSERT_EQ(b.rank, 2);
  ASSERT_EQ(a.dim(0).extent(), 6);
  ASSERT_EQ(a.dim(0).stride(), 4);
  ASSERT_EQ(a.dim(1).stride(), 0);
  ASSERT_EQ(b.dim(0).extent(), 6);
  ASSERT_EQ(b.dim(0).stride(), 4);
  ASSERT_EQ(b.dim(1).stride(), 0);
}

TEST(fuse_contiguous_dims, fuse_implicit_broadcasted) {
  buffer<int, 3> a({6, 1, 1}), b({6});
  a.dim(1) = dim::broadcast();
  a.dim(2) = dim::broadcast();

  fuse_contiguous_dims(a, b);
  ASSERT_EQ(a.rank, 2);
  ASSERT_EQ(b.rank, 1);
  ASSERT_EQ(a.dim(0).extent(), 6);
  ASSERT_EQ(a.dim(0).stride(), 4);
  ASSERT_EQ(a.dim(1).stride(), 0);
  ASSERT_EQ(b.dim(0).extent(), 6);
  ASSERT_EQ(b.dim(0).stride(), 4);
}

TEST(fuse_contiguous_dims, fuse_extent1) {
  buffer<char, 3> a({1, 4, 3}), b({1, 3, 4});

  fuse_contiguous_dims(a, b);
  ASSERT_EQ(a.rank, 2);
  ASSERT_EQ(b.rank, 2);
  ASSERT_EQ(a.dim(0).extent(), 4);
  ASSERT_EQ(a.dim(0).stride(), 1);
  ASSERT_EQ(a.dim(1).extent(), 3);
  ASSERT_EQ(a.dim(1).stride(), 4);
  ASSERT_EQ(b.dim(0).extent(), 3);
  ASSERT_EQ(b.dim(0).stride(), 1);
  ASSERT_EQ(b.dim(1).extent(), 4);
  ASSERT_EQ(b.dim(1).stride(), 3);
}

TEST(fuse_contiguous_dims, cant_fuse_extent1) {
  buffer<char, 3> a({1, 4, 3}), b({1, 3, 4});
  a.dim(1).set_stride(0);

  fuse_contiguous_dims(a, b);
  ASSERT_EQ(a.rank, 3);
  ASSERT_EQ(b.rank, 3);
}

TEST(fuse_contiguous_dims, cant_fuse) {
  buffer<int, 4> a({2, 3, 4, 5}), b({2, 3, 4, 5});
  ASSERT_NE(a.dim(0).stride(), 0);
  ASSERT_NE(a.dim(0).stride(), a.dim(1).stride());
  std::swap(a.dim(2), a.dim(3));
  std::swap(b.dim(2), b.dim(3));
  fuse_contiguous_dims(a, b);
  ASSERT_EQ(a.rank, 3);
  ASSERT_EQ(b.rank, 3);
  ASSERT_EQ(a.dim(0).extent(), 6);
  ASSERT_EQ(a.dim(1).extent(), 5);
  ASSERT_EQ(a.dim(2).extent(), 4);
  ASSERT_EQ(b.dim(0).extent(), 6);
  ASSERT_EQ(b.dim(1).extent(), 5);
  ASSERT_EQ(b.dim(2).extent(), 4);
}

TEST(fuse_contiguous_dims, cant_fuse_broadcasted_inner) {
  buffer<int, 3> a({1, 7, 8}), b({6, 7, 8});
  a.dim(0) = dim::broadcast();
  fuse_contiguous_dims(a, b);
  ASSERT_EQ(a.rank, 2);
  ASSERT_EQ(b.rank, 2);
  ASSERT_EQ(a.dim(0).stride(), 0);
  ASSERT_EQ(a.dim(1).extent(), 7 * 8);
  ASSERT_EQ(a.dim(1).stride(), 4);
  ASSERT_EQ(b.dim(0).extent(), 6);
  ASSERT_EQ(b.dim(0).stride(), 4);
  ASSERT_EQ(b.dim(1).extent(), 7 * 8);
  ASSERT_EQ(b.dim(1).stride(), 24);
}

TEST(fuse_contiguous_dims, cant_fuse_broadcasted_outer) {
  buffer<int, 3> a({6, 7, 1}), b({6, 7, 8});
  a.dim(2) = dim::broadcast();
  fuse_contiguous_dims(a, b);
  ASSERT_EQ(a.rank, 2);
  ASSERT_EQ(b.rank, 2);
  ASSERT_EQ(a.dim(0).extent(), 6 * 7);
  ASSERT_EQ(a.dim(0).stride(), 4);
  ASSERT_EQ(a.dim(1).stride(), 0);
  ASSERT_EQ(b.dim(0).extent(), 6 * 7);
  ASSERT_EQ(b.dim(0).stride(), 4);
  ASSERT_EQ(b.dim(1).extent(), 8);
  ASSERT_EQ(b.dim(1).stride(), 6 * 7 * 4);
}

TEST(fuse_contiguous_dims, fuse_sets) {
  buffer<int, 4> a({2, 3, 4, 5}), b({2, 3, 4, 5});
  ASSERT_NE(a.dim(0).stride(), 0);
  ASSERT_NE(a.dim(0).stride(), a.dim(1).stride());
  const int dims_sets[] = {0, 0, 0, 1};
  fuse_contiguous_dims(dims_sets, a, b);
  ASSERT_EQ(a.rank, 2);
  ASSERT_EQ(b.rank, 2);
  ASSERT_EQ(a.dim(0).extent(), 24);
  ASSERT_EQ(a.dim(1).extent(), 5);
  ASSERT_EQ(b.dim(0).extent(), 24);
  ASSERT_EQ(b.dim(1).extent(), 5);
}

TEST(fuse_contiguous_dims, cant_fuse_sets) {
  buffer<int, 4> a({2, 3, 4, 5}), b({2, 3, 4, 5});
  ASSERT_NE(a.dim(0).stride(), 0);
  ASSERT_NE(a.dim(0).stride(), a.dim(1).stride());
  const int dims_sets[] = {0, 1, 0, 1};
  fuse_contiguous_dims(dims_sets, a, b);
  ASSERT_EQ(a.rank, 4);
  ASSERT_EQ(b.rank, 4);
}

}  // namespace slinky
