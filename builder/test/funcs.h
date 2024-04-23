#ifndef SLINKY_BUILDER_TEST_FUNCS_H
#define SLINKY_BUILDER_TEST_FUNCS_H

#include "runtime/buffer.h"

namespace slinky {

// This file provides a number of toy funcs for test pipelines.

// init_random() for raw_buffer requires allocation be done by caller
template <typename T>
void fill_random(raw_buffer& x) {
  assert(x.base != nullptr);
  for_each_index(x, [&](auto i) { *reinterpret_cast<T*>(x.address_at(i)) = (rand() % 20) - 10; });
}

template <typename T, std::size_t N>
void init_random(buffer<T, N>& x) {
  x.allocate();
  fill_random<T>(x);
}

// Copy from input to output.
// TODO: We should be able to just do this with raw_buffer and not make it a template.
template <typename T>
index_t copy_2d(const buffer<const T>& in, const buffer<T>& out) {
  copy(in, out, nullptr);
  return 0;
}

template <typename T>
index_t zero_padded_copy(const buffer<const T>& in, const buffer<T>& out) {
  assert(in.rank == out.rank);
  T zero = 0;
  slinky::copy(in, out, &zero);
  return 0;
}

// Copy rows, where the output y is -y in the input.
template <typename T>
index_t flip_y(const buffer<const T>& in, const buffer<T>& out) {
  assert(in.rank == 2);
  assert(out.rank == 2);
  std::size_t size = out.dim(0).extent() * out.elem_size;
  for (index_t y = out.dim(1).begin(); y < out.dim(1).end(); ++y) {
    const T* src = &in(out.dim(0).min(), -y);
    T* dst = &out(out.dim(0).min(), y);
    std::copy(src, src + size, dst);
  }
  return 0;
}

template <typename T>
index_t multiply_2(const buffer<const T>& in, const buffer<T>& out) {
  assert(in.rank == out.rank);
  for_each_index(out, [&](auto i) { out(i) = in(i) * 2; });
  return 0;
}

template <typename T>
index_t add_1(const buffer<const T>& in, const buffer<T>& out) {
  assert(in.rank == out.rank);
  for_each_index(out, [&](auto i) { out(i) = in(i) + 1; });
  return 0;
}

template <typename T>
index_t subtract(const buffer<const T>& a, const buffer<const T>& b, const buffer<T>& out) {
  assert(a.rank == out.rank);
  assert(b.rank == out.rank);
  for_each_index(out, [&](auto i) { out(i) = a(i) - b(i); });
  return 0;
}

// A 2D stencil, sums [x + dx0, x + dx1] x [y + dy0, y + dy]
template <typename T, int dx0, int dy0, int dx1, int dy1>
index_t sum_stencil(const buffer<const T>& in, const buffer<T>& out) {
  assert(in.rank == 2);
  assert(out.rank == 2);
  for (index_t y = out.dim(1).begin(); y < out.dim(1).end(); ++y) {
    for (index_t x = out.dim(0).begin(); x < out.dim(0).end(); ++x) {
      T sum = 0;
      for (index_t dy = dy0; dy <= dy1; ++dy) {
        for (index_t dx = dx0; dx <= dx1; ++dx) {
          sum += in(x + dx, y + dy);
        }
      }
      out(x, y) = sum;
    }
  }
  return 0;
}

// A centered 2D 3x3 stencil operation.
template <typename T>
index_t sum3x3(const buffer<const T>& in, const buffer<T>& out) {
  return sum_stencil<T, -1, -1, 1, 1>(in, out);
}

// A centered 2D 5x5 stencil operation.
template <typename T>
index_t sum5x5(const buffer<const T>& in, const buffer<T>& out) {
  return sum_stencil<T, -2, -2, 2, 2>(in, out);
}

}  // namespace slinky

#endif  // SLINKY_BUILDER_TEST_FUNCS_H