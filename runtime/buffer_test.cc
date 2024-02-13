#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "runtime/buffer.h"

namespace slinky {

template <typename T, std::size_t N>
void init_random(buffer<T, N>& buf) {
  buf.allocate();
  std::size_t flat_size = buf.size_bytes() / sizeof(T);
  for (std::size_t i = 0; i < flat_size; ++i) {
    buf.base()[i] = rand();
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

// A non-standard size type that acts like an integer for testing.
struct big {
  uint64_t a, b;

  void assign(int i) {
    a = i;
    b = i / 2;
  }

  big() = default;
  big(int i) { assign(i); }
  big(const big&) = default;

  big& operator=(int i) {
    assign(i);
    return *this;
  }

  operator uint64_t() const { return a + b; }

  big& operator+=(int r) {
    a += r;
    return *this;
  }

  bool operator==(const big& r) { return a == r.a && b == r.b; }
  bool operator!=(const big& r) { return a != r.a || b != r.b; }
};

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

template <typename T, std::size_t Rank>
void test_copy() {
  int src_permutation[Rank];
  int dst_permutation[Rank];
  index_t src_padding[Rank] = {0};
  index_t dst_padding[Rank] = {0};
  for (bool broadcast : {false, true}) {
    for (bool reverse_src : {false, true}) {
      for (bool reverse_dst : {false, true}) {
        for (index_t pad_src : {0, 1}) {
          for (index_t pad_dst : {0, 1}) {
            for (std::size_t i = 0; i < Rank; ++i) {
              src_permutation[i] = reverse_src ? Rank - 1 - i : i;
              dst_permutation[i] = reverse_dst ? Rank - 1 - i : i;
              src_padding[i] = pad_src;
              dst_padding[i] = pad_dst;
            }

            T padding = 7;

            buffer<T, Rank> src;
            for (std::size_t d = 0; d < src.rank; ++d) {
              src.dim(d).set_min_extent(d - Rank / 2, d + 10);
            }
            set_strides(src, src_permutation, src_padding, broadcast);
            init_random(src);

            for (int dmin : {-1, 0, 1}) {
              for (int dmax : {-1, 0, 1}) {
                buffer<T, Rank> dst;
                for (std::size_t d = 0; d < dst.rank; ++d) {
                  dst.dim(d).set_bounds(src.dim(d).min() + dmin, src.dim(d).max() + dmax);
                }
                set_strides(dst, dst_permutation, dst_padding);
                dst.allocate();

                copy(src, dst, &padding);
                for_each_index(dst, [&](auto i) {
                  if (src.contains(i)) {
                    ASSERT_EQ(dst(i), src(i));
                  } else {
                    ASSERT_EQ(dst(i), padding);
                  }
                });

                for_each_contiguous_slice(src, [&](index_t extent, void* base) {
                  for (index_t i = 0; i < extent; ++i) {
                    reinterpret_cast<T*>(base)[i] += 1;
                  }
                });

                copy(src, dst, nullptr);
                for_each_index(dst, [&](auto i) {
                  if (src.contains(i)) {
                    // The copied area should have been copied.
                    ASSERT_EQ(dst(i), src(i));
                  } else {
                    // The padding should be unchanged.
                    ASSERT_EQ(dst(i), padding);
                  }
                });

                for_each_contiguous_slice(src, [&](index_t extent, void* base) {
                  for (index_t i = 0; i < extent; ++i) {
                    reinterpret_cast<T*>(base)[i] += -1;
                  }
                });

                T new_padding = 3;
                pad(src.dims, dst, &new_padding);
                for_each_index(dst, [&](auto i) {
                  if (src.contains(i)) {
                    // The src should not have been copied.
                    ASSERT_NE(dst(i), src(i));
                  } else {
                    // But we should have new padding.
                    ASSERT_EQ(dst(i), new_padding);
                  }
                });
              }
            }
          }
        }
      }
    }
  }
}

template <typename T>
void test_copy() {
  test_copy<T, 1>();
  test_copy<T, 2>();
  test_copy<T, 3>();
}

TEST(buffer, copy) {
  test_copy<uint8_t>();
  test_copy<uint16_t>();
  test_copy<uint32_t>();
  test_copy<uint64_t>();
  test_copy<big>();
}

TEST(buffer, for_each_contiguous_slice_multi) {
  buffer<char, 3> dst({10, 20, 30});
  buffer<char, 3> src({10, 20, 30});
  dst.allocate();
  src.allocate();
  char x = 42;
  fill(src, &x);
  int slices = 0;
  for_each_contiguous_slice(
      dst,
      [&](index_t slice_extent, void* dst, void* src) {
        const char* s = reinterpret_cast<const char*>(src);
        char* d = reinterpret_cast<char*>(dst);
        memcpy(d, s, slice_extent);
        slices++;
      },
      src);
  ASSERT_EQ(slices, 1);
  for_each_index(dst, [&](auto i) { ASSERT_EQ(dst(i), 42); });
  for_each_index(src, [&](auto i) { ASSERT_EQ(src(i), 42); });
}

TEST(buffer, for_each_contiguous_slice_multi_padded) {
  for (int padded_dim = 0; padded_dim < 2; ++padded_dim) {
    buffer<int, 3> buf({10, 20, 30});
    buf.allocate();
    buf.dim(padded_dim).set_min_extent(0, 8);
    buffer<int, 3> buf2({10, 20, 30});
    buf2.allocate();
    int value = 0;
    for_each_contiguous_slice(
        buf,
        [&](index_t slice_extent, void* slice, void* slice2) {
          int* s = reinterpret_cast<int*>(slice);
          int* s2 = reinterpret_cast<int*>(slice2);
          for (int i = 0; i < slice_extent; i++) {
            *s++ = value;
            *s2++ = value;
            value++;
          }
        },
        buf2);
    value = 0;
    for (int c = 0; c < (padded_dim == 2 ? 8 : 30); c++) {
      for (int y = 0; y < (padded_dim == 1 ? 8 : 20); y++) {
        for (int x = 0; x < (padded_dim == 0 ? 8 : 10); x++) {
          ASSERT_EQ(buf(x, y, c), value) << x << " " << y << " " << c;
          ASSERT_EQ(buf2(x, y, c), value) << x << " " << y << " " << c;
          value++;
        }
      }
    }
  }
}

TEST(buffer, for_each_contiguous_slice_multi_non_innermost) {
  buffer<int, 3> buf({10, 20, 30});
  buf.allocate();
  std::swap(buf.dim(0), buf.dim(1));
  buffer<int, 3> buf2({10, 20, 30});
  buf2.allocate();
  std::swap(buf2.dim(0), buf2.dim(1));
  int value = 0;
  for_each_contiguous_slice(
      buf,
      [&](index_t slice_extent, void* slice, void* slice2) {
        int* s = reinterpret_cast<int*>(slice);
        int* s2 = reinterpret_cast<int*>(slice2);
        for (int i = 0; i < slice_extent; i++) {
          *s++ = value;
          *s2++ = value;
          value++;
        }
      },
      buf2);
  value = 0;
  for (int c = 0; c < 30; c++) {
    for (int y = 0; y < 20; y++) {
      for (int x = 0; x < 10; x++) {
        ASSERT_EQ(buf(y, x, c), value) << x << " " << y << " " << c;
        ASSERT_EQ(buf2(y, x, c), value) << x << " " << y << " " << c;
        value++;
      }
    }
  }
}

TEST(buffer, for_each_contiguous_slice_multi_both_non_zero_min) {
  buffer<char, 3> buf({10, 20, 30});
  buffer<char, 3> buf2({10, 20, 30});
  buf.allocate();
  buf.translate(1, 2, 3);
  buf2.allocate();
  buf2.translate(1, 2, 3);
  int slices = 0;
  for_each_contiguous_slice(
      buf,
      [&](index_t slice_extent, void* slice, void* slice2) {
        memset(slice, 7, slice_extent);
        memset(slice2, 7, slice_extent);
        slices++;
      },
      buf2);
  // These should fuse into a single slice
  ASSERT_EQ(slices, 1);
  for_each_index(buf, [&](auto i) { ASSERT_EQ(buf(i), 7); });
  for_each_index(buf2, [&](auto i) { ASSERT_EQ(buf2(i), 7); });
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

TEST(buffer, for_each_contiguous_slice_multi_extra_buf_offset_negative) {
  buffer<int, 3> buf({10, 20, 30});
  buffer<int, 3> buf2({11, 21, 31});
  buf.allocate();
  buf2.allocate();
  buf2.translate(-1, -1, -1);
  int slices = 0;
  int value = 0;
  for_each_contiguous_slice(
      buf,
      [&](index_t slice_extent, void* slice, void* slice2) {
        int* s = reinterpret_cast<int*>(slice);
        int* s2 = reinterpret_cast<int*>(slice2);
        for (int i = 0; i < slice_extent; i++) {
          *s++ = value;
          *s2++ = value;
          value++;
        }
        slices++;
      },
      buf2);
  ASSERT_EQ(slices, 600);
  value = 0;
  for (int c = 0; c < 30; c++) {
    for (int y = 0; y < 20; y++) {
      for (int x = 0; x < 10; x++) {
        ASSERT_EQ(buf(x, y, c), value) << x << " " << y << " " << c;
        ASSERT_EQ(buf2(x, y, c), value) << x << " " << y << " " << c;
        value++;
      }
    }
  }
}

TEST(buffer, for_each_contiguous_slice_multi_folded_main_buffer) {
  buffer<char, 3> buf({10, 20, 30});
  buf.dim(1).set_fold_factor(4);
  buf.allocate();
  buffer<char, 3> buf2({10, 20, 30});
  buf2.allocate();
  char xx = 42;
  fill(buf2, &xx);
  for (int crop_extent : {1, 2, 3, 4}) {
    buf.dim(1).set_min_extent(8, crop_extent);
    int slices = 0;
    for_each_contiguous_slice(
        buf,
        [&](index_t slice_extent, void* slice, void* slice2) {
          memset(slice, 7, slice_extent);
          memset(slice2, 7, slice_extent);
          slices++;
        },
        buf2);
    ASSERT_EQ(slices, crop_extent * 30);
    for_each_index(buf, [&](auto i) { ASSERT_EQ(buf(i), 7); });
    for (int c = 0; c < 30; c++) {
      for (int y = 0; y < 20; y++) {
        for (int x = 0; x < 10; x++) {
          const char value = (y >= 8 && y < 8 + crop_extent) ? 7 : 42;
          ASSERT_EQ(buf2(x, y, c), value) << x << " " << y << " " << c;
        }
      }
    }
  }
}

TEST(buffer, for_each_contiguous_slice_multi_folded_other_buffer) {
  constexpr int W = 10, H = 20, C = 30;
  buffer<char, 3> buf({W, H, C});
  buf.allocate();
  buffer<char, 3> buf2({W, H, C});
  buf2.dim(1).set_fold_factor(4);
  buf2.allocate();
  int slices = 0;
  for_each_contiguous_slice(
      buf,
      [&](index_t slice_extent, void* slice, void* slice2) {
        memset(slice, 7, slice_extent);
        memset(slice2, 7, slice_extent);
        slices++;
      },
      buf2);
  ASSERT_EQ(slices, 600);
  for_each_index(buf, [&](auto i) { ASSERT_EQ(buf(i), 7); });
  for (int c = 0; c < C; c++) {
    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        ASSERT_EQ(buf(x, y, c), 7) << x << " " << y << " " << c;
        ASSERT_EQ(buf2(x, y, c), 7) << x << " " << y << " " << c;
      }
    }
  }
}

TEST(buffer, for_each_contiguous_slice_multi_folded_all) {
  constexpr int W = 30, H = 30, C = 30;
  buffer<char, 3> buf({W, H, C});
  buf.dim(1).set_fold_factor(2);
  buf.allocate();
  buffer<char, 3> buf2({W, H, C});
  buf2.dim(1).set_fold_factor(3);
  buf2.allocate();
  buffer<char, 3> buf3({W, H, C});
  buf3.dim(1).set_fold_factor(5);
  buf3.allocate();
  int slices = 0;
  for_each_contiguous_slice(
      buf,
      [&](index_t slice_extent, void* slice, void* slice2, void* slice3) {
        memset(slice, 7, slice_extent);
        memset(slice2, 8, slice_extent);
        memset(slice3, 9, slice_extent);
        slices++;
      },
      buf2, buf3);
  ASSERT_EQ(slices, 900);
  for_each_index(buf, [&](auto i) { ASSERT_EQ(buf(i), 7); });
  for (int c = 0; c < C; c++) {
    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        ASSERT_EQ(buf(x, y, c), 7) << x << " " << y << " " << c;
        ASSERT_EQ(buf2(x, y, c), 8) << x << " " << y << " " << c;
        ASSERT_EQ(buf3(x, y, c), 9) << x << " " << y << " " << c;
      }
    }
  }
}

TEST(buffer, for_each_contiguous_slice_multi_folded_all_with_offset) {
  const auto fill_slow = [](buffer<char, 3>& buf, char value) {
    for (int c = buf.dim(2).min(); c <= buf.dim(2).max(); c++) {
      for (int y = buf.dim(1).min(); y <= buf.dim(1).max(); y++) {
        for (int x = buf.dim(0).min(); x <= buf.dim(0).max(); x++) {
          buf(x, y, c) = value;
        }
      }
    }
  };

  constexpr int W = 30, H = 30, C = 30;
  buffer<char, 3> buf({W, H, C});
  buf.dim(1).set_fold_factor(2);
  buf.allocate();
  fill_slow(buf, 41);
  buffer<char, 3> buf2({W + 1, H + 1, C + 1});
  buf2.dim(1).set_fold_factor(3);
  buf2.allocate();
  buf2.translate(-1, -1, -1);
  fill_slow(buf2, 42);
  buffer<char, 3> buf3({W + 2, H + 2, C + 2});
  buf3.dim(1).set_fold_factor(5);
  buf3.allocate();
  buf3.translate(-1, -1, -1);
  fill_slow(buf3, 43);
  int slices = 0;
  for_each_contiguous_slice(
      buf,
      [&](index_t slice_extent, void* slice, void* slice2, void* slice3) {
        memset(slice, 7, slice_extent);
        memset(slice2, 8, slice_extent);
        memset(slice3, 9, slice_extent);
        slices++;
      },
      buf2, buf3);
  ASSERT_EQ(slices, 900);
  ASSERT_EQ(buf2(-1, -1, -1), 42);
  ASSERT_EQ(buf3(-1, -1, -1), 43);
  ASSERT_EQ(buf3(W, H, C), 43);
  for_each_index(buf, [&](auto i) { ASSERT_EQ(buf(i), 7); });
  for (int c = 0; c < C; c++) {
    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        ASSERT_EQ(buf(x, y, c), 7) << x << " " << y << " " << c;
        ASSERT_EQ(buf2(x, y, c), 8) << x << " " << y << " " << c;
        ASSERT_EQ(buf3(x, y, c), 9) << x << " " << y << " " << c;
      }
    }
  }
}

}  // namespace slinky
