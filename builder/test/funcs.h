#ifndef SLINKY_BUILDER_TEST_FUNCS_H
#define SLINKY_BUILDER_TEST_FUNCS_H

#include "runtime/buffer.h"

namespace slinky {

// This file provides a number of toy funcs for test pipelines.

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

}  // namespace slinky

#endif  // SLINKY_BUILDER_TEST_FUNCS_H