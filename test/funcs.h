#ifndef SLINKY_TEST_FUNCS_H
#define SLINKY_TEST_FUNCS_H

#include "buffer.h"

namespace slinky {

// This file provides a number of toy funcs for test pipelines.

// Copy from input to output.
// TODO: We should be able to just do this with buffer_base and not make it a template.
template <typename T>
index_t copy(const buffer<const T>& in, const buffer<T>& out) {
  const T* src = &in(out.dims[0].min, out.dims[1].min);
  T* dst = &out(out.dims[0].min, out.dims[1].min);
  std::size_t size = out.dims[0].extent * out.elem_size;
  for (int y = out.dims[1].begin(); y < out.dims[1].end(); ++y) {
    std::copy(src, src + size, dst);
    dst += out.dims[1].stride_bytes;
    src += in.dims[1].stride_bytes;
  }
  return 0;
}

// Like copy, but flips in the y dimension.
template <typename T>
index_t flip_y(const buffer<const T>& in, const buffer<T>& out) {
  std::size_t size = out.dims[0].extent * out.elem_size;
  for (int y = out.dims[1].begin(); y < out.dims[1].end(); ++y) {
    const T* src = &in(out.dims[0].min, -y);
    T* dst = &out(out.dims[0].min, y);
    std::copy(src, src + size, dst);
  }
  return 0;
}

// Matrix multiplication (not fast!)
template <typename T>
index_t matmul(const buffer<const T>& a, const buffer<const T>& b, const buffer<T>& c) {
  for (index_t i = c.dims[0].begin(); i < c.dims[0].end(); ++i) {
    for (index_t j = c.dims[1].begin(); j < c.dims[1].end(); ++j) {
      c(i, j) = 0;
      for (index_t k = a.dims[1].begin(); k < a.dims[1].end(); ++k) {
        c(i, j) += a(i, k) * b(k, j);
      }
    }
  }
  return 0;
}

// A 2D 3x3 stencil operation.
template <typename T>
index_t sum3x3(const buffer<const T>& in, const buffer<T>& out) {
  for (index_t y = out.dims[1].begin(); y < out.dims[1].end(); ++y) {
    for (index_t x = out.dims[0].begin(); x < out.dims[0].end(); ++x) {
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

#endif  // SLINKY_TEST_TEST_H
