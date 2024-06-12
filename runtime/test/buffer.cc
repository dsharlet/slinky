#include <gtest/gtest.h>

#include <algorithm>
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
int random(Rng& rng, int min, int max) { return rng() % (max - min + 1) + min; }

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
  index_t* i = SLINKY_ALLOCA(index_t, dims.size());
  for_each_index(dims, dims.size() - 1, i, f);
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

bool test_fill(int elem_size, int size) {
  buffer<void, 1> buf({size}, elem_size);
  buf.allocate();
  std::vector<uint8_t> value(elem_size);
  std::iota(value.begin(), value.end(), 0);
  fill(buf, value.data());
  for (int i = 0; i < size * elem_size; ++i) {
    if (reinterpret_cast<const uint8_t*>(buf.base())[i] != i % elem_size) {
      return false;
    }
  }
  return true;
}

TEST(buffer, fill) {
  for (int size = 0; size < 100; ++size) {
    for (int elem_size : {1, 2, 3, 4, 8, 12, 16, 63, 64, 65}) {
      ASSERT_TRUE(test_fill(elem_size, size)) << elem_size << " " << size;
    }
  }
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

TEST(buffer, for_each_contiguous_slice) {
  buffer<char, 3> buf({10, 20, 30});
  buf.allocate();
  int slices = 0;
  for_each_contiguous_slice(buf, [&](index_t slice_extent, char* slice) {
    memset(slice, 7, slice_extent);
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
    memset(slice, 7, slice_extent);
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
      memset(slice, 7, slice_extent);
      slices++;
    });
    ASSERT_EQ(slices, crop_extent * 30);
    ASSERT_TRUE(is_filled_buffer(buf, 7));
  }
}

TEST(buffer, for_each_contiguous_slice_padded) {
  for (int padded_dim = 0; padded_dim < 2; ++padded_dim) {
    buffer<char, 3> buf({10, 20, 30});
    buf.allocate();
    buf.dim(padded_dim).set_bounds(0, 8);
    for_each_contiguous_slice(buf, [&](index_t slice_extent, char* slice) { memset(slice, 7, slice_extent); });
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

template <typename T>
void test_for_each_contiguous_slice_fill() {
  gtest_seeded_mt19937 rng;

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
  for (int cases = 0; cases < 1000; ++cases) {
    test_for_each_contiguous_slice_fill<char>();
    test_for_each_contiguous_slice_fill<int>();
  }
}

template <typename Src, typename Dst>
void test_for_each_contiguous_slice_copy() {
  gtest_seeded_mt19937 rng;

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
  for (int cases = 0; cases < 10000; ++cases) {
    test_for_each_contiguous_slice_copy<char, char>();
    test_for_each_contiguous_slice_copy<short, int>();
    test_for_each_contiguous_slice_copy<int, int>();
  }
}

template <typename Src, typename Dst>
void test_for_each_element_copy() {
  gtest_seeded_mt19937 rng;

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
  for (int cases = 0; cases < 10000; ++cases) {
    test_for_each_element_copy<char, char>();
    test_for_each_element_copy<short, int>();
    test_for_each_element_copy<int, int>();
  }
}

template <typename A, typename B, typename Dst>
void test_for_each_contiguous_slice_add() {
  gtest_seeded_mt19937 rng;

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
  for (int cases = 0; cases < 1000; ++cases) {
    test_for_each_contiguous_slice_add<int, int, int>();
    test_for_each_contiguous_slice_add<short, int, int>();
    test_for_each_contiguous_slice_add<short, short, int>();
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
        memset(slice1, 1, slice_extent);
        memset(slice2, 2, slice_extent);
        memset(slice3, 3, slice_extent);
        memset(slice4, 4, slice_extent);
        memset(slice5, 5, slice_extent);
        memset(slice6, 6, slice_extent);
        memset(slice7, 7, slice_extent);
        memset(slice8, 8, slice_extent);
        memset(slice9, 9, slice_extent);
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

TEST(buffer, for_each_tile_1x1) {
  buffer<int, 2> buf({10, 20});
  buf.allocate();

  int tiles = 0;
  const index_t all[] = {buf.dim(0).extent(), buf.dim(1).extent()};
  for_each_tile(all, buf, [&](const raw_buffer& i) {
    ASSERT_EQ(i.rank, 2);
    ASSERT_EQ(i.dim(0).extent(), all[0]);
    ASSERT_EQ(i.dim(1).extent(), all[1]);
    tiles++;
  });
  ASSERT_EQ(tiles, 1);
}

TEST(buffer, for_each_tile_uneven) {
  buffer<int, 2> buf({10, 20});
  buf.allocate();

  int tiles = 0;
  const index_t tile[] = {3, 6};
  for_each_tile(tile, buf, [&](const raw_buffer& i) {
    ASSERT_EQ(i.rank, 2);
    ASSERT_LE(i.dim(0).extent(), tile[0]);
    ASSERT_LE(i.dim(1).extent(), tile[1]);
    tiles++;
  });
  ASSERT_EQ(tiles, ceil_div<index_t>(buf.dim(0).extent(), tile[0]) * ceil_div<index_t>(buf.dim(1).extent(), tile[1]));
}

TEST(buffer, for_each_tile_all) {
  buffer<int, 2> buf({10, 20});
  buf.allocate();

  int tiles = 0;
  const index_t slice[] = {slinky::all, 5};
  for_each_tile(slice, buf, [&](const raw_buffer& i) {
    ASSERT_EQ(i.rank, 2);
    ASSERT_EQ(i.dim(0).extent(), buf.dim(0).extent());
    ASSERT_EQ(i.dim(1).extent(), slice[1]);
    tiles++;
  });
  ASSERT_EQ(tiles, ceil_div<index_t>(buf.dim(1).extent(), slice[1]));
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

TEST(buffer, for_each_element_empty) {
  buffer<int, 2> buf({0, 20});
  buf.allocate();
  int elements = 0;
  for_each_element([&](int*) { elements++; }, buf);
  ASSERT_EQ(elements, 0);
}

TEST(buffer, for_each_slice) {
  for (std::size_t slice_rank : {0, 1, 2}) {
    buffer<int, 2> buf({10, 20});
    buf.allocate();
    int slices = 0;
    int elements = 0;
    for_each_slice(slice_rank, buf, [&](const raw_buffer& slice) {
      ASSERT_EQ(slice.rank, slice_rank);
      const int seven = 7;
      fill(slice, &seven);
      slices++;
      int elements_slice = 1;
      for (std::size_t d = 0; d < slice.rank; ++d) {
        elements_slice *= slice.dim(d).extent();
      }
      elements += elements_slice;
    });
    int expected_slices = 1;
    int expected_elements = 1;
    for (std::size_t d = 0; d < buf.rank; ++d) {
      if (d >= slice_rank) {
        expected_slices *= buf.dim(d).extent();
      }
      expected_elements *= buf.dim(d).extent();
    }
    ASSERT_EQ(slices, expected_slices);
    ASSERT_EQ(elements, expected_elements);

    ASSERT_TRUE(is_filled_buffer(buf, 7));
  }
}

TEST(buffer, for_each_slice_copy_folded) {
  gtest_seeded_mt19937 rng;

  buffer<uint8_t, 2> src({20, 30});
  src.dim(1).set_fold_factor(2);
  init_random(rng, src);

  buffer<uint8_t, 2> dst({10, 20});
  dst.dim(1).set_min_extent(3, 20);
  dst.allocate();

  int slices = 0;
  for_each_slice(
      1, dst,
      [&](const raw_buffer& dst_slice, const raw_buffer& src_slice) {
        copy(src_slice, dst_slice);
        slices++;
      },
      src);
  int expected_slices = dst.dim(1).extent();
  ASSERT_EQ(slices, expected_slices);

  for (index_t y = dst.dim(1).begin(); y < dst.dim(1).end(); ++y) {
    for (index_t x = dst.dim(0).begin(); x < dst.dim(0).end(); ++x) {
      ASSERT_EQ(dst(x, y), src(x, y));
    }
  }
}

TEST(buffer, for_each_slice_sum) {
  gtest_seeded_mt19937 rng;

  buffer<short, 3> src({3, 10, 5});
  init_random(rng, src);

  buffer<int, 2> dst({10, 5});
  dst.allocate();

  for_each_slice(
      1, dst,
      [&](const raw_buffer& dst_slice, const raw_buffer& src_slice) {
        ASSERT_EQ(src_slice.rank, 2);
        ASSERT_EQ(dst_slice.rank, 1);
        auto& dst_t = dst_slice.cast<int>();
        auto& src_t = src_slice.cast<short>();
        for (index_t i = dst_t.dim(0).begin(); i < dst_t.dim(0).end(); ++i) {
          dst_t(i) = 0;
          for (index_t j = src_t.dim(0).begin(); j < src_t.dim(0).end(); ++j) {
            dst_t(i) += src_t(j, i);
          }
        }
      },
      src);

  for (index_t y = dst.dim(1).begin(); y < dst.dim(1).end(); ++y) {
    for (index_t x = dst.dim(0).begin(); x < dst.dim(0).end(); ++x) {
      int correct = src(0, x, y) + src(1, x, y) + src(2, x, y);
      ASSERT_EQ(dst(x, y), correct);
    }
  }
}

TEST(buffer, for_each_slice_broadcasted_slice) {
  gtest_seeded_mt19937 rng;

  buffer<int, 1> src({10});
  init_random(rng, src);

  buffer<int, 3> dst({10, 4, 3});
  dst.allocate();

  for_each_slice(
      2, dst,
      [&](const raw_buffer& dst_slice, const raw_buffer& src_slice) {
        ASSERT_EQ(src_slice.rank, 1);
        ASSERT_EQ(dst_slice.rank, 2);
        auto& dst_t = dst_slice.cast<int>();
        auto& src_t = src_slice.cast<int>();
        for (index_t i = dst_t.dim(1).begin(); i < dst_t.dim(1).end(); ++i) {
          for (index_t j = dst_t.dim(0).begin(); j < dst_t.dim(0).end(); ++j) {
            dst_t(j, i) = src_t(j);
          }
        }
      },
      src);

  for_each_index(dst, [&](auto i) { ASSERT_EQ(dst(i), src(i[0])); });
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

TEST(buffer, copy) {
  gtest_seeded_mt19937 rng;

  constexpr int max_rank = 4;
  for (int cases = 0; cases < 10000; ++cases) {
    int rank = random(rng, 0, max_rank);
    int elem_size = random(rng, 1, 12);

    std::vector<char> padding(elem_size);
    std::fill(padding.begin(), padding.end(), 7);

    buffer<void, max_rank> src(rank, elem_size);
    buffer<void, max_rank> dst(rank, elem_size);
    for (std::size_t d = 0; d < src.rank; ++d) {
      src.dim(d).set_min_extent(0, 5);
      dst.dim(d).set_min_extent(0, 5);
    }
    randomize_strides_and_padding(rng, src, {-1, 1, true, false});
    randomize_strides_and_padding(rng, dst, {-1, 1, false, false});
    init_random(rng, src);
    dst.allocate();

    slinky::copy(src, dst, padding.data());
    for_each_index(dst, [&](auto i) {
      if (src.contains(i)) {
        ASSERT_EQ(memcmp(dst.address_at(i), src.address_at(i), elem_size), 0);
      } else {
        ASSERT_EQ(memcmp(dst.address_at(i), padding.data(), elem_size), 0);
      }
    });

    std::vector<char> new_padding(elem_size);
    std::fill(new_padding.begin(), new_padding.end(), 3);
    pad(src.dims, dst, new_padding.data());
    for_each_index(dst, [&](auto i) {
      if (src.contains(i)) {
        // The src should not have been modified.
        ASSERT_EQ(memcmp(dst.address_at(i), src.address_at(i), elem_size), 0);
      } else {
        // But we should have new padding.
        ASSERT_EQ(memcmp(dst.address_at(i), new_padding.data(), elem_size), 0);
      }
    });
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
