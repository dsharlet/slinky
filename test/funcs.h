#ifndef SLINKY_TEST_FUNCS_H
#define SLINKY_TEST_FUNCS_H

#include "buffer.h"

namespace slinky {

// This file provides a number of toy funcs for test pipelines.

// Copy from input to output.
// TODO: We should be able to just do this with raw_buffer and not make it a template.
template <typename T>
index_t copy(const buffer<const T>& in, const buffer<T>& out) {
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
  for_each_index(out, [&](auto i) { out(i) = in(i)*2; });
  return 0;
}

template <typename T>
index_t add_1(const buffer<const T>& in, const buffer<T>& out) {
  assert(in.rank == out.rank);
  for_each_index(out, [&](auto i) { out(i) = in(i) + 1; });
  return 0;
}

template <typename T, std::size_t N>
void init_random(buffer<T, N>& x) {
  x.allocate();
  for_each_index(x, [&](auto i) { x(i) = (rand() % 20) - 10; });
}

// Matrix multiplication (not fast!)
template <typename T>
index_t matmul(const buffer<const T>& a, const buffer<const T>& b, const buffer<T>& c) {
  assert(a.rank == 2);
  assert(b.rank == 2);
  assert(c.rank == 2);
  assert(a.dim(1).begin() == b.dim(0).begin());
  assert(a.dim(1).end() == b.dim(0).end());
  assert(a.dim(1).stride() == sizeof(T));
  assert(b.dim(1).stride() == sizeof(T));
  assert(c.dim(1).stride() == sizeof(T));
  for (index_t i = c.dim(0).begin(); i < c.dim(0).end(); ++i) {
    for (index_t j = c.dim(1).begin(); j < c.dim(1).end(); ++j) {
      c(i, j) = 0;
      for (index_t k = a.dim(1).begin(); k < a.dim(1).end(); ++k) {
        c(i, j) += a(i, k) * b(k, j);
      }
    }
  }
  return 0;
}

// Matrix multiplication (not fast!)
template <typename T>
index_t outer_product(const buffer<const T>& a, const buffer<const T>& b, const buffer<T>& c) {
  assert(a.rank == 1);
  assert(b.rank == 1);
  assert(c.rank == 2);
  for (index_t j = c.dim(1).begin(); j < c.dim(1).end(); ++j) {
    for (index_t i = c.dim(0).begin(); i < c.dim(0).end(); ++i) {
      c(i, j) = a(i) * b(j);
    }
  }
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


}  // namespace slinky

#endif  // SLINKY_TEST_FUNCS_H
