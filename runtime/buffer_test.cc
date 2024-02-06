#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "runtime/buffer.h"

namespace slinky {

TEST(buffer, make) {
  auto buf = raw_buffer::make(2, 4);
  buf->dim(0).set_min_extent(0, 10);
  buf->dim(0).set_stride(4);
  buf->dim(1).set_min_extent(0, 20);
  buf->dim(1).set_stride(buf->dim(0).extent() * buf->dim(0).stride());

  ASSERT_EQ(buf->size_bytes(), buf->dim(0).extent() * buf->dim(1).extent() * buf->elem_size);
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
  buffer<int, 3> buf({10, 20, 30});
  buf.allocate();
  int slices = 0;
  for_each_contiguous_slice(buf, [&](void* slice, index_t slice_extent) {
    ASSERT_EQ(slice_extent, 10);
    slices++;
  });
  ASSERT_EQ(slices, buf.dim(1).extent() * buf.dim(2).extent());
}

TEST(buffer, for_each_contiguous_slice_non_innermost) {
  buffer<int, 3> buf({10, 20, 30});
  buf.allocate();
  std::swap(buf.dim(0), buf.dim(1));
  int slices = 0;
  for_each_contiguous_slice(buf, [&](void* slice, index_t slice_extent) {
    ASSERT_EQ(slice_extent, 10);
    slices++;
  });
  ASSERT_EQ(slices, buf.dim(0).extent() * buf.dim(2).extent());
}

TEST(buffer, for_each_tile_1x1) {
  buffer<int, 3> buf({10, 20, 5});
  buf.allocate();

  int tiles = 0;
  const auto all = std::make_tuple(buf.dim(0).extent(), buf.dim(1).extent());
  for_each_tile(all, buf, [&](const raw_buffer& i) {
    ASSERT_EQ(i.rank, 2);
    ASSERT_EQ(i.dim(0).extent(), std::get<0>(all));
    ASSERT_EQ(i.dim(1).extent(), std::get<1>(all));
    tiles++;
  });
  ASSERT_EQ(tiles, buf.dim(2).extent());
}

TEST(buffer, for_each_tile_uneven) {
  buffer<int, 3> buf({10, 20, 5});
  buf.allocate();

  int tiles = 0;
  const auto tile = std::make_tuple(3, 6);
  for_each_tile(tile, buf, [&](const raw_buffer& i) {
    ASSERT_EQ(i.rank, 2);
    ASSERT_LE(i.dim(0).extent(), std::get<0>(tile));
    ASSERT_LE(i.dim(1).extent(), std::get<1>(tile));
    tiles++;
  });
  ASSERT_EQ(tiles, ceil_div<index_t>(buf.dim(0).extent(), std::get<0>(tile)) *
                       ceil_div<index_t>(buf.dim(1).extent(), std::get<1>(tile)) * buf.dim(2).extent());
}

TEST(buffer, for_each_tile_all) {
  buffer<int, 3> buf({10, 20, 5});
  buf.allocate();

  int tiles = 0;
  const auto slice = std::tuple<slinky::all, std::integral_constant<int, 5>>();
  for_each_tile(slice, buf, [&](const raw_buffer& i) {
    ASSERT_EQ(i.rank, 2);
    ASSERT_EQ(i.dim(0).extent(), buf.dim(0).extent());
    ASSERT_EQ(i.dim(1).extent(), std::get<1>(slice));
    tiles++;
  });
  ASSERT_EQ(tiles, ceil_div<index_t>(buf.dim(1).extent(), std::get<1>(slice)) * buf.dim(2).extent());
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
            src.allocate();
            for_each_contiguous_slice(src, [&](void* base, index_t extent) {
              for (index_t i = 0; i < extent; ++i) {
                reinterpret_cast<T*>(base)[i] = rand();
              }
            });

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

                for_each_contiguous_slice(src, [&](void* base, index_t extent) {
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

                for_each_contiguous_slice(src, [&](void* base, index_t extent) {
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

}  // namespace slinky
