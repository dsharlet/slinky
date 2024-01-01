#ifndef SLINKY_BUFFER_H
#define SLINKY_BUFFER_H

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>

#include "arithmetic.h"

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

// TODO(https://github.com/dsharlet/slinky/issues/1): This and buffer_expr in pipeline.h should have the same API
// (except for expr instead of index_t).
class dim {
  index_t min_;
  index_t extent_;
  index_t stride_;
  index_t fold_factor_;

public:
  dim() : min_(0), extent_(0), stride_(0), fold_factor_(0) {}

  index_t min() const { return min_; }
  index_t max() const { return min_ + extent_ - 1; }
  index_t begin() const { return min_; }
  index_t end() const { return min_ + extent_; }
  index_t extent() const { return extent_; }
  index_t stride() const { return stride_; }
  index_t fold_factor() const { return fold_factor_; }

  void set_extent(index_t extent) { extent_ = extent; }
  void set_point(index_t x) { min_ = x; extent_ = 1; }
  void set_bounds(index_t min, index_t max) { min_ = min; extent_ = max - min + 1; }
  void set_range(index_t begin, index_t end) {  min_ = begin; extent_ = end - begin; }
  void set_min_extent(index_t min, index_t extent) { min_ = min; extent_ = extent; }
  void set_stride(index_t stride) { stride_ = stride; }
  void set_fold_factor(index_t fold_factor) { fold_factor_ = fold_factor; }

  void translate(index_t offset) { min_ += offset; }

  bool contains(index_t x) const { return min() <= x && x <= max(); }

  std::ptrdiff_t flat_offset_bytes(index_t i) const {
    assert(i >= min_);
    assert(i <= max());
    // If we use a mask instead of a fold factor, we can just make the mask -1 by default, and
    // always bitwise and to implement the fold factor.
    if (fold_factor_ <= 0) {
      return (i - min_) * stride_;
    } else {
      return euclidean_mod(i - min_, fold_factor_) * stride_;
    }
  }
};

template <typename T, std::size_t DimsSize = 0>
class buffer;

class raw_buffer;

using raw_buffer_ptr = std::unique_ptr<raw_buffer, void (*)(raw_buffer*)>;

// We have some difficult requirements for this buffer object:
// 1. We want type safety in user code, but we also want to be able to treat buffers as generic.
// 2. We want to store metadata (dimensions) efficiently.
//   a. No extra storage for unused dimensions.
//   b. In the same allocation as the buffer object itself if needed.
//
// To this end, we have a class (struct) raw_buffer:
// - Type erased.
// - Does not provide storage for dims (unless you use raw_buffer::make).
//
// And a class buffer<T, DimsSize>:
// - Has a type, can be accessed via operator() and at.
// - Provides storage for DimsSize dims (default is 0).
class raw_buffer {
protected:
  static std::ptrdiff_t flat_offset_bytes_impl(const dim* dims, index_t i0) {
    return dims->flat_offset_bytes(i0);
  }

  template <typename... Indices>
  static std::ptrdiff_t flat_offset_bytes_impl(const dim* dims, index_t i0, Indices... indices) {
    return dims->flat_offset_bytes(i0) + flat_offset_bytes_impl(dims + 1, indices...);
  }

  template <typename... Indices>
  static bool contains_impl(const dim* dims, index_t i0, Indices... indices) {
    return dims->contains(i0) && contains_impl(dims + 1, indices...);
  }

  static void destroy(raw_buffer* buf) {
    buf->~raw_buffer();
    delete[] (char*)buf;
  }

public:
  char* allocation;
  void* base;
  std::size_t elem_size;
  std::size_t rank;
  slinky::dim* dims;

  raw_buffer() = default;
  raw_buffer(const raw_buffer&) = delete;
  raw_buffer(raw_buffer&&) = delete;
  void operator=(const raw_buffer&) = delete;
  void operator=(raw_buffer&&) = delete;
  ~raw_buffer() { free(); }

  slinky::dim& dim(std::size_t i) { return dims[i]; }
  const slinky::dim& dim(std::size_t i) const { return dims[i]; }

  template <typename... Indices>
  std::ptrdiff_t flat_offset_bytes(index_t i0, Indices... indices) const {
    return flat_offset_bytes_impl(dims, i0, indices...);
  }
  template <typename... Indices>
  void* address_at(index_t i0, Indices... indices) const {
    return offset_bytes(base, flat_offset_bytes(i0, indices...));
  }
  template <typename... Indices>
  bool contains(index_t i0, Indices... indices) const {
    return contains_impl(dims, i0, indices...);
  }

  std::ptrdiff_t flat_offset_bytes(std::span<index_t> indices) const {
    assert(indices.size() == rank);
    index_t offset = 0;
    for (std::size_t i = 0; i < indices.size(); ++i) {
      offset += dims[i].flat_offset_bytes(indices[i]);
    }
    return offset;
  }
  void* address_at(std::span<index_t> indices) const { return offset_bytes(base, flat_offset_bytes(indices)); }
  bool contains(std::span<index_t> indices) const {
    assert(indices.size() == rank);
    bool result = true;
    for (std::size_t i = 0; i < indices.size(); ++i) {
      result = result && dims[i].contains(indices[i]);
    }
    return result;
  }

  std::size_t size_bytes() const {
    index_t flat_min = 0;
    index_t flat_max = 0;
    for (std::size_t i = 0; i < rank; ++i) {
      index_t extent = dims[i].extent();
      if (dims[i].fold_factor() > 0) {
        extent = std::min(extent, dims[i].fold_factor());
      }
      flat_min += (extent - 1) * std::min<index_t>(0, dims[i].stride());
      flat_max += (extent - 1) * std::max<index_t>(0, dims[i].stride());
    }
    return flat_max - flat_min + elem_size;
  }

  // Does not call constructor or destructor of T!
  void allocate() {
    assert(allocation == nullptr);

    allocation = new char[size_bytes()];
    base = allocation;
  }

  void free() {
    delete[] allocation;
    allocation = nullptr;
    base = nullptr;
  }

  template <typename NewT>
  const buffer<NewT>& cast() const;

  // Make a buffer and space for dims in the same object.
  static raw_buffer_ptr make(std::size_t rank, std::size_t elem_size) {
    char* buf_and_dims = new char[sizeof(raw_buffer) + sizeof(slinky::dim) * rank];
    raw_buffer* buf = new (buf_and_dims) raw_buffer();
    buf->base = nullptr;
    buf->allocation = nullptr;
    buf->rank = rank;
    buf->elem_size = elem_size;
    buf->dims = reinterpret_cast<slinky::dim*>(buf_and_dims + sizeof(raw_buffer));
    new (buf->dims) slinky::dim[rank];
    return {buf, destroy};
  }
};

template <typename T, std::size_t DimsSize>
class buffer : public raw_buffer {
private:
  // TODO: When DimsSize is 0, this still makes sizeof(buffer) bigger than sizeof(raw_buffer).
  // This might be a problem because we can cast raw_buffer to buffer<T>. When DimsSize is 0,
  // we shouldn't actually access this, so it might be harmless, but it still seems ugly.
  slinky::dim dims_storage[DimsSize];

public:
  using raw_buffer::allocate;
  using raw_buffer::cast;
  using raw_buffer::dim;
  using raw_buffer::elem_size;
  using raw_buffer::flat_offset_bytes;
  using raw_buffer::free;
  using raw_buffer::rank;

  buffer() {
    raw_buffer::base = nullptr;
    allocation = nullptr;
    rank = DimsSize;
    elem_size = sizeof(T);
    if (DimsSize > 0) {
      dims = &dims_storage[0];
      new (dims) slinky::dim[rank];
    } else {
      dims = nullptr;
    }
  }

  // Construct a buffer with extents, and strides computed such that the stride of dimension
  // n is the product of all the extents of dimensions [0, n) and elem_size, i.e. the first
  // dimension is "innermost".
  buffer(std::initializer_list<index_t> extents) : buffer() {
    assert(extents.size() <= rank);
    index_t stride = elem_size;
    slinky::dim* d = dims;
    for (index_t extent : extents) {
      d->set_min_extent(0, extent);
      d->set_stride(stride);
      stride *= extent;
      ++d;
    }
  }

  T* base() const { return reinterpret_cast<T*>(raw_buffer::base); }

  // Make a buffer and space for dims in the same allocation.
  static std::unique_ptr<buffer<T>, void (*)(buffer<T>*)> make(std::size_t rank) {
    auto buf = raw_buffer::make(rank, sizeof(T));
    return {static_cast<buffer<T>*>(buf.release()), (void (*)(buffer<T>*))raw_buffer::destroy};
  }

  // These accessors are not designed to be fast. They exist to facilitate testing,
  // and maybe they are useful to compute addresses.
  template <typename... Indices>
  auto& at(index_t i0, Indices... indices) const {
    return *offset_bytes(base(), flat_offset_bytes(i0, indices...));
  }
  template <typename... Indices>
  auto& operator()(index_t i0, Indices... indices) const {
    return at(i0, indices...);
  }

  auto& at(std::span<index_t> indices) const { return *offset_bytes(base(), flat_offset_bytes(indices)); }
  auto& operator()(std::span<index_t> indices) const { return at(indices); }
};

template <typename NewT>
const buffer<NewT>& raw_buffer::cast() const {
  assert(elem_size == sizeof(NewT));
  return *reinterpret_cast<const buffer<NewT>*>(this);
}

// Copy the contents of `src` to `dst`. When the `src` is out of bounds of `dst`, fill with `padding`.
// `padding` should point to `dst.elem_size` bytes, or if `padding` is null, out of bounds regions
// are unmodified.
void copy(const raw_buffer& src, const raw_buffer& dst, const void* padding = nullptr);

// Fill `dst` with `value`. `value` should point to `dst.elem_size` bytes.
void fill(const raw_buffer& dst, const void* value);

namespace internal {

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

}  // namespace internal

// Call `f(std::span<index_t>)` for each index in the range of `dims`, or the dims of `buf`.
// This function is not fast, use it for non-performance critical code. It is useful for
// making rank-agnostic algorithms without a recursive wrapper, which is otherwise difficult.
template <typename F>
void for_each_index(std::span<const dim> dims, F&& f) {
  // Not using alloca for performance, but to avoid including <vector>
  index_t* i = reinterpret_cast<index_t*>(alloca(dims.size() * sizeof(index_t)));
  internal::for_each_index(dims, dims.size() - 1, {i, dims.size()}, f);
}
template <typename F>
void for_each_index(const raw_buffer& buf, F&& f) {
  for_each_index({buf.dims, buf.rank}, f);
}

}  // namespace slinky

#endif  // SLINKY_BUFFER_H
