#ifndef SLINKY_RUNTIME_TEST_BUFFER_H
#define SLINKY_RUNTIME_TEST_BUFFER_H

#include <cassert>
#include <vector>

#include "runtime/buffer.h"

namespace slinky {

// Flatten a buffer into a vector. Not efficient, but useful for testing.
template <typename T>
std::vector<T> flatten(const raw_buffer& buf) {
  assert(buf.elem_size == sizeof(T));
  std::vector<T> flat(buf.elem_count());
  raw_buffer flat_buf;
  flat_buf.base = flat.data();
  flat_buf.elem_size = buf.elem_size;
  flat_buf.rank = buf.rank;
  flat_buf.dims = SLINKY_ALLOCA(dim, buf.rank);
  index_t stride = buf.elem_size;
  for (std::size_t d = 0; d < buf.rank; ++d) {
    flat_buf.dim(d).set_min_extent(buf.dim(d).min(), buf.dim(d).extent());
    flat_buf.dim(d).set_stride(stride);
    flat_buf.dim(d).set_fold_factor(dim::unfolded);
    stride *= flat_buf.dim(d).extent();
  }
  assert(stride == static_cast<index_t>(flat.size() * sizeof(T)));
  copy(buf, flat_buf);
  return flat;
}

template <typename T, std::size_t DimsSize>
std::vector<T> flatten(const buffer<T, DimsSize>& buf) {
  return flatten<T>(static_cast<const raw_buffer&>(buf));
}

}  // namespace slinky

#endif  // SLINKY_RUNTIME_TEST_BUFFER_H
