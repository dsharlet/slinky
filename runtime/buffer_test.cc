#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>

#include "runtime/buffer.h"

std::mt19937& rng() {
  static std::mt19937 r{static_cast<uint32_t>(time(nullptr))};
  return r;
}

namespace slinky {

int random(int min, int max) { return rng()() % (max - min + 1) + min; }

template <typename T, std::size_t N>
void init_random(buffer<T, N>& buf) {
  buf.allocate();
  std::size_t flat_size = buf.size_bytes();
  for (std::size_t i = 0; i < flat_size; ++i) {
    reinterpret_cast<char*>(buf.base())[i] = random(0, 255);
  }
}

struct randomize_options {
  int padding_min = 0;
  int padding_max = 3;
  bool allow_broadcast = false;
  bool allow_fold = false;
};

template <typename T, std::size_t N>
void randomize_strides_and_padding(buffer<T, N>& buf, const randomize_options& options) {
  std::vector<int> permutation(buf.rank);
  std::iota(permutation.begin(), permutation.end(), 0);
  if (random(0, 3) == 0) {
    // Randomize the strides ordering.
    std::shuffle(permutation.begin(), permutation.end(), rng());
  }

  index_t stride = buf.elem_size;
  for (std::size_t d : permutation) {
    slinky::dim& dim = buf.dim(d);
    // Expand the bounds randomly.
    dim.set_bounds(dim.min() - random(options.padding_min, options.padding_max),
        dim.max() + random(options.padding_min, options.padding_max));
    assert(dim.extent() > 0);
    if (options.allow_broadcast && random(0, 9) == 0) {
      // Make this a broadcast.
      dim.set_stride(0);
    } else {
      dim.set_stride(stride);
      // Add some extra random padding.
      stride *= dim.extent() + random(0, 3) * buf.elem_size;
    }
    if (options.allow_fold && random(0, 9) == 0) {
      // Make sure the fold factor divides the min so the fold is valid.
      dim.set_fold_factor(std::max<index_t>(1, std::abs(dim.min())));
    }
  }
}

TEST(raw_buffer, make_copy) {
  buffer<int, 2> src({10, 20});
  init_random(src);

  auto dst = raw_buffer::make_copy(src);
  ASSERT_EQ(src.rank, dst->rank);
  ASSERT_EQ(src.dim(0).min(), dst->dim(0).min());
  ASSERT_EQ(src.dim(0).extent(), dst->dim(0).extent());
  ASSERT_EQ(src.dim(1).min(), dst->dim(1).min());
  ASSERT_EQ(src.dim(1).extent(), dst->dim(1).extent());
  ASSERT_EQ(src.size_bytes(), dst->size_bytes());
  ASSERT_NE(src.base(), dst->base);

  for_each_index(src, [&](auto i) { ASSERT_EQ(src(i), *reinterpret_cast<int*>(dst->address_at(i))); });
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

TEST(buffer, for_each_contiguous_slice) {
  buffer<char, 3> buf({10, 20, 30});
  buf.allocate();
  int slices = 0;
  for_each_contiguous_slice(buf, [&](index_t slice_extent, void* slice) {
    memset(slice, 7, slice_extent);
    slices++;
  });
  ASSERT_EQ(slices, 1);
  for_each_index(buf, [&](auto i) { ASSERT_EQ(buf(i), 7); });
}

TEST(buffer, for_each_contiguous_slice_non_zero_min) {
  buffer<char, 3> buf({10, 20, 30});
  buf.allocate();
  buf.translate(1, 2, 3);
  int slices = 0;
  for_each_contiguous_slice(buf, [&](index_t slice_extent, void* slice) {
    memset(slice, 7, slice_extent);
    slices++;
  });
  ASSERT_EQ(slices, 1);
  for_each_index(buf, [&](auto i) { ASSERT_EQ(buf(i), 7); });
}

TEST(buffer, for_each_contiguous_folded) {
  buffer<char, 3> buf({10, 20, 30});
  buf.dim(1).set_fold_factor(4);
  buf.allocate();
  for (int crop_extent : {1, 2, 3, 4}) {
    buf.dim(1).set_min_extent(8, crop_extent);
    int slices = 0;
    for_each_contiguous_slice(buf, [&](index_t slice_extent, void* slice) {
      memset(slice, 7, slice_extent);
      slices++;
    });
    ASSERT_EQ(slices, crop_extent * 30);
    for_each_index(buf, [&](auto i) { ASSERT_EQ(buf(i), 7); });
  }
}

TEST(buffer, for_each_contiguous_slice_padded) {
  for (int padded_dim = 0; padded_dim < 2; ++padded_dim) {
    buffer<char, 3> buf({10, 20, 30});
    buf.allocate();
    buf.dim(padded_dim).set_bounds(0, 8);
    for_each_contiguous_slice(buf, [&](index_t slice_extent, void* slice) { memset(slice, 7, slice_extent); });
    for_each_index(buf, [&](auto i) { ASSERT_EQ(buf(i), 7); });
  }
}

TEST(buffer, for_each_contiguous_slice_non_innermost) {
  buffer<int, 3> buf({10, 20, 30});
  buf.allocate();
  std::swap(buf.dim(0), buf.dim(1));
  int slices = 0;
  for_each_contiguous_slice(buf, [&](index_t slice_extent, void* slice) {
    ASSERT_EQ(slice_extent, 10);
    slices++;
  });
  ASSERT_EQ(slices, buf.dim(0).extent() * buf.dim(2).extent());
}

template <typename T>
void test_for_each_contiguous_slice_fill() {
  buffer<T, 4> dst;
  for (std::size_t d = 0; d < dst.rank; ++d) {
    dst.dim(d).set_min_extent(0, 5);
  }
  randomize_strides_and_padding(dst, {-1, 1, false, true});
  dst.allocate();

  for_each_contiguous_slice(
      dst, [&](index_t slice_extent, void* dst) { std::fill_n(reinterpret_cast<T*>(dst), slice_extent, 7); });

  for_each_index(dst, [&](const auto i) { ASSERT_EQ(dst(i), 7); });
}

TEST(buffer, for_each_contiguous_slice_fill) {
  for (int cases = 0; cases < 100; ++cases) {
    test_for_each_contiguous_slice_fill<char>();
    test_for_each_contiguous_slice_fill<int>();
  }
}

template <typename Src, typename Dst>
void test_for_each_contiguous_slice_copy() {
  buffer<Src, 4> src;
  buffer<Dst, 4> dst;
  for (std::size_t d = 0; d < src.rank; ++d) {
    src.dim(d).set_min_extent(0, 3);
    dst.dim(d).set_min_extent(0, 3);
  }
  randomize_strides_and_padding(src, {0, 1, true, true});
  randomize_strides_and_padding(dst, {-1, 0, false, false});
  init_random(src);
  dst.allocate();

  for_each_contiguous_slice(
      dst,
      [&](index_t slice_extent, void* dst, const void* src) {
        std::copy_n(reinterpret_cast<const Src*>(src), slice_extent, reinterpret_cast<Dst*>(dst));
      },
      src);
  for_each_index(dst, [&](const auto i) { ASSERT_EQ(src(i), dst(i)); });
}

TEST(buffer, for_each_contiguous_slice_copy) {
  for (int cases = 0; cases < 100; ++cases) {
    test_for_each_contiguous_slice_copy<char, char>();
    test_for_each_contiguous_slice_copy<short, int>();
    test_for_each_contiguous_slice_copy<int, int>();
  }
}

template <typename A, typename B, typename Dst>
void test_for_each_contiguous_slice_add() {
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

  randomize_strides_and_padding(a, {0, 1, true, true});
  randomize_strides_and_padding(b, {0, 1, true, true});
  init_random(a);
  init_random(b);

  randomize_strides_and_padding(dst, {-1, 0, false, false});
  dst.allocate();

  for_each_contiguous_slice(
      dst,
      [&](index_t slice_extent, void* dst_v, const void* a_v, const void* b_v) {
        Dst* dst = reinterpret_cast<int*>(dst_v);
        const A* a = reinterpret_cast<const A*>(a_v);
        const B* b = reinterpret_cast<const B*>(b_v);
        for (index_t i = 0; i < slice_extent; ++i) {
          dst[i] = saturate_add<Dst>(a[i], b[i]);
        }
      },
      a, b);
  for_each_index(dst, [&](const auto i) { ASSERT_EQ(dst(i), saturate_add<Dst>(a(i), b(i))); });
}

TEST(buffer, for_each_contiguous_slice_add) {
  for (int cases = 0; cases < 100; ++cases) {
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
      [&](index_t slice_extent, void* slice1, void* slice2, void* slice3, void* slice4, void* slice5, void* slice6,
          void* slice7, void* slice8, void* slice9) {
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
  for_each_index(buf1, [&](auto i) { ASSERT_EQ(buf1(i), 1); });
  for_each_index(buf2, [&](auto i) { ASSERT_EQ(buf2(i), 2); });
  for_each_index(buf3, [&](auto i) { ASSERT_EQ(buf3(i), 3); });
  for_each_index(buf4, [&](auto i) { ASSERT_EQ(buf4(i), 4); });
  for_each_index(buf5, [&](auto i) { ASSERT_EQ(buf5(i), 5); });
  for_each_index(buf6, [&](auto i) { ASSERT_EQ(buf6(i), 6); });
  for_each_index(buf7, [&](auto i) { ASSERT_EQ(buf7(i), 7); });
  for_each_index(buf8, [&](auto i) { ASSERT_EQ(buf8(i), 8); });
  for_each_index(buf9, [&](auto i) { ASSERT_EQ(buf9(i), 9); });
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

    for_each_index(buf, [&](auto i) { ASSERT_EQ(buf(i), 7); });
  }
}

TEST(buffer, for_each_slice_copy_folded) {
  buffer<uint8_t, 2> src({20, 30});
  src.dim(1).set_fold_factor(2);
  init_random(src);

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

  for_each_index(dst, [&](auto i) { ASSERT_EQ(dst(i), src(i)); });
}

TEST(buffer, for_each_slice_sum) {
  buffer<short, 3> src({3, 10, 5});
  init_random(src);

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

  for_each_index(dst, [&](auto i) {
    int correct = src(0, i[0], i[1]) + src(1, i[0], i[1]) + src(2, i[0], i[1]);
    ASSERT_EQ(dst(i), correct);
  });
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
  constexpr int max_rank = 4;
  for (int cases = 0; cases < 1000; ++cases) {
    int rank = random(0, max_rank);
    int elem_size = random(1, 12);

    std::vector<char> padding(elem_size);
    std::fill(padding.begin(), padding.end(), 7);

    buffer<void, max_rank> src(rank, elem_size);
    for (std::size_t d = 0; d < src.rank; ++d) {
      src.dim(d).set_min_extent(0, 5);
    }
    randomize_strides_and_padding(src, {-1, 1, true, false});
    init_random(src);

    buffer<void, max_rank> dst(rank, elem_size);
    for (std::size_t d = 0; d < src.rank; ++d) {
      dst.dim(d) = src.dim(d);
    }
    randomize_strides_and_padding(dst, {-1, 1, false, false});
    dst.allocate();

    slinky::copy(src, dst, padding.data());
    for_each_index(dst, [&](auto i) {
      if (src.contains(i)) {
        ASSERT_EQ(memcmp(dst.address_at(i), src.address_at(i), elem_size), 0);
      } else {
        ASSERT_EQ(memcmp(dst.address_at(i), padding.data(), elem_size), 0);
      }
    });

    for_each_contiguous_slice(src, [&](index_t extent, void* base) {
      for (index_t i = 0; i < extent * elem_size; ++i) {
        reinterpret_cast<char*>(base)[i] += 1;
      }
    });

    slinky::copy(src, dst, nullptr);
    for_each_index(dst, [&](auto i) {
      if (src.contains(i)) {
        // The copied area should have been copied.
        ASSERT_EQ(memcmp(dst.address_at(i), src.address_at(i), elem_size), 0);
      } else {
        // The padding should be unchanged.
        ASSERT_EQ(memcmp(dst.address_at(i), padding.data(), elem_size), 0);
      }
    });

    for_each_contiguous_slice(src, [&](index_t extent, void* base) {
      for (index_t i = 0; i < extent * elem_size; ++i) {
        reinterpret_cast<char*>(base)[i] += -1;
      }
    });

    std::vector<char> new_padding(elem_size);
    std::fill(new_padding.begin(), new_padding.end(), 3);
    pad(src.dims, dst, new_padding.data());
    for_each_index(dst, [&](auto i) {
      if (src.contains(i)) {
        // The src should not have been copied.
        ASSERT_NE(memcmp(dst.address_at(i), src.address_at(i), elem_size), 0);
      } else {
        // But we should have new padding.
        ASSERT_EQ(memcmp(dst.address_at(i), new_padding.data(), elem_size), 0);
      }
    });
  }
}

}  // namespace slinky
