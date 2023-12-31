#include "copy.h"
#include "buffer.h"
#include "test.h"
#include "funcs.h"

#include <cstdint>

using namespace slinky;

// A non-standard size type that acts like an integer for testing.
struct big {
  uint64_t a, b;

  void assign(int i) {
    a = i * 2;
    b = i * 2 + 1;
  }

  big() = default;
  big(int i) { assign(i); }
  big(const big&) = default;

  void operator=(int i) { assign(i); }

  operator uint64_t() const { return a + b; }

  bool operator==(const big& r) { return a == r.a && b == r.b; }
  bool operator!=(const big& r) { return a != r.a || b != r.b; }
};

template <typename T, std::size_t N>
void set_strides(buffer<T, N>& buf, int* permutation = nullptr, index_t* padding = nullptr) {
  index_t stride = buf.elem_size;
  for (int i = 0; i < N; ++i) {
    dim& d = buf.dim(permutation ? permutation[i] : i);
    d.set_stride(stride);
    stride *= d.extent() + (padding ? padding[i] : 0);
  }
}

template <typename T, std::size_t Rank>
void test_copy() {
  int src_permutation[Rank];
  int dst_permutation[Rank];
  index_t src_padding[Rank] = {0};
  index_t dst_padding[Rank] = {0};
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
          set_strides(src, src_permutation, src_padding);
          src.allocate();
          for_each_index(src, [&](auto i) { src(i) = rand(); });

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

TEST(copy) {
  test_copy<uint8_t>();
  test_copy<uint16_t>();
  test_copy<uint32_t>();
  test_copy<uint64_t>();
  test_copy<big>();
}
