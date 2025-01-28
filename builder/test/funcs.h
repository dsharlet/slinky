#ifndef SLINKY_BUILDER_TEST_FUNCS_H
#define SLINKY_BUILDER_TEST_FUNCS_H

#include "runtime/buffer.h"

namespace slinky {

// This file provides a number of toy funcs for test pipelines.

// init_random() for raw_buffer requires allocation be done by caller
template <typename T, std::size_t N>
void fill_random(const buffer<T, N>& buf) {
  for_each_element([](T* v) { *v = (rand() & 15) - 8; }, buf);
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
  copy(in, out);
  return 0;
}

template <typename T>
index_t zero_padded_copy(const buffer<const T>& in, const buffer<T>& out) {
  assert(in.rank == out.rank);
  slinky::copy(in, out, static_cast<T>(0));
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
  for_each_element([&](T* out, const T* in) { *out = *in * 2; }, out, in);
  return 0;
}

template <typename T>
index_t add_1(const buffer<const T>& in, const buffer<T>& out) {
  assert(in.rank == out.rank);
  for_each_element([&](T* out, const T* in) { *out = *in + 1; }, out, in);
  return 0;
}

template <typename T>
index_t subtract(const buffer<const T>& a, const buffer<const T>& b, const buffer<T>& out) {
  for_each_element([&](T* out, const T* a, const T* b) { *out = (a ? *a : 0) - *b; }, out, a, b);
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

// Centered 1D separable stencil operations.
template <typename T>
index_t sum1x3(const buffer<const T>& in, const buffer<T>& out) {
  return sum_stencil<T, 0, -1, 0, 1>(in, out);
}
template <typename T>
index_t sum3x1(const buffer<const T>& in, const buffer<T>& out) {
  return sum_stencil<T, -1, 0, 1, 0>(in, out);
}

// A centered 2D 5x5 stencil operation.
template <typename T>
index_t sum5x5(const buffer<const T>& in, const buffer<T>& out) {
  return sum_stencil<T, -2, -2, 2, 2>(in, out);
}

template <typename T>
index_t upsample_nn_2x(const buffer<const T>& in, const buffer<T>& out) {
  for (index_t y = out.dim(1).begin(); y < out.dim(1).end(); ++y) {
    for (index_t x = out.dim(0).begin(); x < out.dim(0).end(); ++x) {
      out(x, y) = in((x + 0) >> 1, (y + 0) >> 1);
    }
  }
  return 0;
}

}  // namespace slinky

#endif  // SLINKY_BUILDER_TEST_FUNCS_H