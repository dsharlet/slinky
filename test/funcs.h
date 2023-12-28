#ifndef SLINKY_TEST_FUNCS_H
#define SLINKY_TEST_FUNCS_H

#include "buffer.h"

namespace slinky {

// This file provides a number of toy funcs for test pipelines.

template <typename F>
void for_each_index(std::span<const dim> dims, int d, std::span<index_t> is, F&& f) {
  if (d == 0) {
    for (index_t i = dims[0].begin(); i < dims[0].end(); ++i) {
      is[0] = i;
      f(is);
    }
  } else {
    for (index_t i = dims[d].begin(); i < dims[d].end(); ++i) {
      is[d] = i;
      for_each_index(dims, d - 1, is, f);
    }
  }
}

// Call `f(std::span<index_t>)` for each index in the range of `dims`.
template <typename F>
void for_each_index(std::span<const dim> dims, F&& f) {
  std::vector<index_t> i(dims.size());
  for_each_index(dims, dims.size() - 1, i, f);
}

// Call `f(std::span<index_t>)` for each index in the range of the dims of `b`.
template <typename F>
void for_each_index(const buffer_base& b, F&& f) {
  for_each_index({b.dims, b.rank}, f);
}

// Copy from input to output.
// TODO: We should be able to just do this with buffer_base and not make it a template.
template <typename T>
index_t copy(const buffer<const T>& in, const buffer<T>& out) {
  const T* src = &in(out.dim(0).min(), out.dim(1).min());
  T* dst = &out(out.dim(0).min(), out.dim(1).min());
  std::size_t size = out.dim(0).extent() * out.elem_size;
  for (int y = out.dim(1).begin(); y < out.dim(1).end(); ++y) {
    std::copy(src, src + size, dst);
    dst += out.dim(1).stride_bytes();
    src += in.dim(1).stride_bytes();
  }
  return 0;
}

// Like copy, but flips in the y dimension.
template <typename T>
index_t flip_y(const buffer<const T>& in, const buffer<T>& out) {
  assert(in.rank == 2);
  assert(out.rank == 2);
  std::size_t size = out.dim(0).extent() * out.elem_size;
  for (int y = out.dim(1).begin(); y < out.dim(1).end(); ++y) {
    const T* src = &in(out.dim(0).min(), -y);
    T* dst = &out(out.dim(0).min(), y);
    std::copy(src, src + size, dst);
  }
  return 0;
}

template <typename T>
index_t multiply_2(const buffer<const T>& in, const buffer<T>& out) {
  assert(in.rank == out.rank);
  for_each_index(out, [&](std::span<index_t> i) { out(i) = in(i)*2; });
  return 0;
}

template <typename T>
index_t add_1(const buffer<const T>& in, const buffer<T>& out) {
  assert(in.rank == out.rank);
  for_each_index(out, [&](std::span<index_t> i) { out(i) = in(i) + 1; });
  return 0;
}

template <typename T>
index_t add(const buffer<const T>& a, const buffer<const T>& b, const buffer<T>& out) {
  assert(a.rank == out.rank);
  assert(b.rank == out.rank);
  for_each_index(out, [&](std::span<index_t> i) { out(i) = a(i) + b(i); });
  return 0;
}

template <typename T>
index_t multiply(const buffer<const T>& a, const buffer<const T>& b, const buffer<T>& out) {
  assert(a.rank == out.rank);
  assert(b.rank == out.rank);
  for_each_index(out, [&](std::span<index_t> i) { out(i) = a(i) * b(i); });
  return 0;
}

template <typename T>
index_t max_0(const buffer<const T>& a, const buffer<T>& out) {
  assert(a.rank == out.rank);
  for_each_index(out, [&](std::span<index_t> i) { out(i) = std::max(a(i), 0); });
  return 0;
}

template <typename T, std::size_t N>
void init_random(buffer<T, N>& x) {
  x.allocate();
  for_each_index(x, [&](std::span<index_t> i) { x(i) = (rand() % 20) - 10; });
}

// Matrix multiplication (not fast!)
template <typename T>
index_t matmul(const buffer<const T>& a, const buffer<const T>& b, const buffer<T>& c) {
  assert(a.rank == 2);
  assert(b.rank == 2);
  assert(c.rank == 2);
  assert(a.dim(1).begin() == b.dim(0).begin());
  assert(a.dim(1).end() == b.dim(0).end());
  assert(a.dim(1).stride_bytes() == sizeof(T));
  assert(b.dim(1).stride_bytes() == sizeof(T));
  assert(c.dim(1).stride_bytes() == sizeof(T));
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

// A 2D 3x3 stencil operation.
template <typename T>
index_t sum3x3(const buffer<const T>& in, const buffer<T>& out) {
  assert(in.rank == 2);
  assert(out.rank == 2);
  for (index_t y = out.dim(1).begin(); y < out.dim(1).end(); ++y) {
    for (index_t x = out.dim(0).begin(); x < out.dim(0).end(); ++x) {
      T sum = 0;
      for (index_t dy = -1; dy <= 1; ++dy) {
        for (index_t dx = -1; dx <= 1; ++dx) {
          sum += in(x + dx, y + dy);
        }
      }
      out(x, y) = sum;
    }
  }
  return 0;
}

}  // namespace slinky

#endif  // SLINKY_TEST_FUNCS_H
