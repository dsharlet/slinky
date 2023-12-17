#ifndef LOCALITY_BUFFER_H
#define LOCALITY_BUFFER_H

#include <cstdint>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <bit>
#include <cmath>
#include <memory>

#include "euclidean_division.h"

namespace slinky {

// Helper to offset a pointer by a number of bytes.
template <typename T>
T* offset_bytes(T* x, std::ptrdiff_t bytes) {
  return reinterpret_cast<T*>(reinterpret_cast<char*>(x) + bytes);
}
template <typename T>
const T* offset_bytes(const T* x, std::ptrdiff_t bytes) {
  return reinterpret_cast<const T*>(reinterpret_cast<const char*>(x) + bytes);
}

struct dim {
  index_t min;
  index_t extent;
  index_t stride_bytes;
  index_t fold_factor;

  index_t begin() const { return min; }
  index_t end() const { return min + extent; }
  index_t max() const { return min + extent - 1; }

  constexpr std::ptrdiff_t flat_offset_bytes(index_t i) const {
    // If we use a mask instead of a fold factor, we can just make the mask -1 by default, and always
    // bitwise and to implement the fold factor.
    if (fold_factor <= 0) {
      return (i - min) * stride_bytes;
    } else {
      return euclidean_mod(i - min, fold_factor) * stride_bytes;
    }
  }
};

// We have some difficult requirements for this buffer object:
// 1. We want type safety in user code, but we also want to be able to treat buffers as generic.
// 2. We want to store metadata (dimensions) efficiently.
//   a. No extra storage for unused dimensions.
//   b. In the same allocation as the buffer object itself if needed.
//
// To this end, we have a class (struct) buffer_base:
// - Type erased.
// - Does not provide storage for dims (unless you use buffer_base::make).
//
// And a class buffer<T, DimsSize>:
// - Has a type, can be accessed via operator() and at.
// - Provides storage for DimsSize dims (default is 0). 
struct buffer_base : public std::enable_shared_from_this<buffer_base> {
  using dim = slinky::dim;

protected:
  static constexpr std::ptrdiff_t flat_offset_bytes_impl(const dim* dims, index_t i0) {
    return dims->flat_offset_bytes(i0);
  }

  template <typename... Indices>
  static constexpr std::ptrdiff_t flat_offset_bytes_impl(const dim* dims, index_t i0, Indices... indices) {
    return dims->flat_offset_bytes(i0) + flat_offset_bytes_impl(dims + 1, indices...);
  }

  static void destroy(buffer_base* buf) {
    buf->~buffer_base();
    delete[](char*)buf;
  }

  char* allocation;

public:
  void* base;
  std::size_t elem_size;
  std::size_t rank;
  dim* dims;

  buffer_base() = default;
  buffer_base(const buffer_base&) = delete;
  buffer_base(buffer_base&&) = delete;
  void operator=(const buffer_base&) = delete;
  void operator=(buffer_base&&) = delete;
  ~buffer_base() {
    free();
  }

  // Make a buffer and space for dims in the same object. Returns a unique_ptr, with the
  // understanding that unique_ptr can be converted to shared_ptr if needed.
  static std::unique_ptr<buffer_base, void(*)(buffer_base*)> make(std::size_t rank, std::size_t elem_size) {
    char* buf_and_dims = new char[sizeof(buffer_base) + sizeof(dim) * rank];
    buffer_base* buf = new (buf_and_dims) buffer_base();
    buf->base = nullptr;
    buf->allocation = nullptr;
    buf->rank = rank;
    buf->elem_size = elem_size;
    buf->dims = reinterpret_cast<dim*>(buf_and_dims + sizeof(buffer_base));
    memset(&buf->dims[0], 0, sizeof(dim) * rank);
    return { buf, destroy };
  }

  template <typename... Indices>
  constexpr std::ptrdiff_t flat_offset_bytes(Indices... indices) const {
    return flat_offset_bytes_impl(dims, indices...);
  }

  std::size_t size_bytes() const {
    index_t flat_min = 0;
    index_t flat_max = 0;
    for (std::size_t i = 0; i < rank; ++i) {
      flat_min += (dims[i].extent - 1) * std::min<index_t>(0, dims[i].stride_bytes);
      flat_max += (dims[i].extent - 1) * std::max<index_t>(0, dims[i].stride_bytes);
    }
    return flat_max - flat_min + elem_size;
  }

  // Does not call constructor or destructor of T!
  void allocate() {
    assert(base == nullptr);
    assert(allocation == nullptr);

    allocation = new char[size_bytes()];
    base = allocation;
  }

  void free() {
    delete[] allocation;
    allocation = nullptr;
    base = nullptr;
  }
};

template <typename T, std::size_t DimsSize = 0>
struct buffer : public buffer_base {
private:
  dim dims_storage[DimsSize];

public:
  using buffer_base::dims;
  using buffer_base::rank;
  using buffer_base::elem_size;
  using buffer_base::flat_offset_bytes;
  using buffer_base::allocate;
  using buffer_base::free;

  buffer() {
    buffer_base::base = nullptr;
    allocation = nullptr;
    rank = DimsSize;
    elem_size = sizeof(T);
    dims = DimsSize > 0 ? &dims_storage[0] : nullptr;
    memset(dims_storage, 0, sizeof(dims_storage));
  }

  buffer(std::initializer_list<index_t> extents) : buffer() {
    assert(extents.size() <= rank);
    index_t stride_bytes = elem_size;
    dim* d = dims;
    for (index_t extent : extents) {
      d->extent = extent;
      d->stride_bytes = stride_bytes;
      stride_bytes *= extent;
      ++d;
    }
  }

  T* base() const { return reinterpret_cast<T*>(buffer_base::base); }

  // Make a buffer and space for dims in the same object.
  static std::unique_ptr<buffer<T>, void(*)(buffer<T>*)> make(std::size_t rank) {
    auto buf = buffer_base::make(rank, sizeof(T));
    return { static_cast<buffer<T>*>(buf.release()), (void(*)(buffer<T>*))buffer_base::destroy};
  }

  template <typename... Indices>
  auto& at(Indices... indices) const {
    return *offset_bytes(base(), flat_offset_bytes(indices...));
  }

  template <typename... Indices>
  auto& operator() (Indices... indices) const {
    return at(indices...);
  }

  template <typename NewT>
  const buffer<NewT>& cast() const {
    assert(buffer_base::elem_size == sizeof(NewT));
    return *reinterpret_cast<const buffer<NewT>*>(this);
  }
};

}  // namespace slinky

#endif  // LOCALITY_BUFFER_H