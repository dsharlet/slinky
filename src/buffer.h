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
  index_t stride_bytes_;
  index_t fold_factor_;

public:
  dim() : min_(0), extent_(0), stride_bytes_(0), fold_factor_(0) {}

  index_t min() const { return min_; }
  index_t max() const { return min_ + extent_ - 1; }
  index_t begin() const { return min_; }
  index_t end() const { return min_ + extent_; }
  index_t extent() const { return extent_; }
  index_t stride_bytes() const { return stride_bytes_; }
  index_t fold_factor() const { return fold_factor_; }

  void set_extent(index_t extent) { extent_ = extent; }
  void set_bounds(index_t min, index_t max) { min_ = min; extent_ = max - min + 1; }
  void set_range(index_t begin, index_t end) {  min_ = begin; extent_ = end - begin; }
  void set_min_extent(index_t min, index_t extent) { min_ = min; extent_ = extent; }
  void set_stride_bytes(index_t stride_bytes) { stride_bytes_ = stride_bytes; }
  void set_fold_factor(index_t fold_factor) { fold_factor_ = fold_factor; }

  void translate(index_t offset) { min_ += offset; }

  constexpr std::ptrdiff_t flat_offset_bytes(index_t i) const {
    assert(i >= min_);
    assert(i <= max());
    // If we use a mask instead of a fold factor, we can just make the mask -1 by default, and
    // always bitwise and to implement the fold factor.
    if (fold_factor_ <= 0) {
      return (i - min_) * stride_bytes_;
    } else {
      return euclidean_mod(i - min_, fold_factor_) * stride_bytes_;
    }
  }
};

template <typename T, std::size_t DimsSize = 0>
class buffer;

class buffer_base;

using buffer_base_ptr = std::unique_ptr<buffer_base, void (*)(buffer_base*)>;

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
class buffer_base : public std::enable_shared_from_this<buffer_base> {
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
    delete[] (char*)buf;
  }

public:
  char* allocation;
  void* base;
  std::size_t elem_size;
  std::size_t rank;
  slinky::dim* dims;

  buffer_base() = default;
  buffer_base(const buffer_base&) = delete;
  buffer_base(buffer_base&&) = delete;
  void operator=(const buffer_base&) = delete;
  void operator=(buffer_base&&) = delete;
  ~buffer_base() { free(); }

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

  std::ptrdiff_t flat_offset_bytes(std::span<index_t> indices) const {
    assert(indices.size() == rank);
    index_t offset = 0;
    for (std::size_t i = 0; i < indices.size(); ++i) {
      offset += dims[i].flat_offset_bytes(indices[i]);
    }
    return offset;
  }
  void* address_at(std::span<index_t> indices) const { return offset_bytes(base, flat_offset_bytes(indices)); }

  std::size_t size_bytes() const {
    index_t flat_min = 0;
    index_t flat_max = 0;
    for (std::size_t i = 0; i < rank; ++i) {
      index_t extent = dims[i].extent();
      if (dims[i].fold_factor() > 0) {
        extent = std::min(extent, dims[i].fold_factor());
      }
      flat_min += (extent - 1) * std::min<index_t>(0, dims[i].stride_bytes());
      flat_max += (extent - 1) * std::max<index_t>(0, dims[i].stride_bytes());
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

  // Make a buffer and space for dims in the same object. Returns a unique_ptr, with the
  // understanding that unique_ptr can be converted to shared_ptr if needed.
  static buffer_base_ptr make(std::size_t rank, std::size_t elem_size) {
    char* buf_and_dims = new char[sizeof(buffer_base) + sizeof(slinky::dim) * rank];
    buffer_base* buf = new (buf_and_dims) buffer_base();
    buf->base = nullptr;
    buf->allocation = nullptr;
    buf->rank = rank;
    buf->elem_size = elem_size;
    buf->dims = reinterpret_cast<slinky::dim*>(buf_and_dims + sizeof(buffer_base));
    memset(&buf->dims[0], 0, sizeof(slinky::dim) * rank);
    return {buf, destroy};
  }
};

template <typename T, std::size_t DimsSize>
class buffer : public buffer_base {
private:
  // TODO: When DimsSize is 0, this still makes sizeof(buffer) bigger than sizeof(buffer_base).
  // This might be a problem because we can cast buffer_base to buffer<T>. When DimsSize is 0,
  // we shouldn't actually access this, so it might be harmless, but it still seems ugly.
  slinky::dim dims_storage[DimsSize];

public:
  using buffer_base::allocate;
  using buffer_base::cast;
  using buffer_base::dim;
  using buffer_base::elem_size;
  using buffer_base::flat_offset_bytes;
  using buffer_base::free;
  using buffer_base::rank;

  buffer() {
    buffer_base::base = nullptr;
    allocation = nullptr;
    rank = DimsSize;
    elem_size = sizeof(T);
    if (DimsSize > 0) {
      dims = &dims_storage[0];
      memset(dims_storage, 0, sizeof(dims_storage));
    } else {
      dims = nullptr;
    }
  }

  // Construct a buffer with extents, and strides computed such that the stride of dimension
  // n is the product of all the extents of dimensions [0, n) and elem_size, i.e. the first
  // dimension is "innermost".
  buffer(std::initializer_list<index_t> extents) : buffer() {
    assert(extents.size() <= rank);
    index_t stride_bytes = elem_size;
    slinky::dim* d = dims;
    for (index_t extent : extents) {
      d->set_min_extent(0, extent);
      d->set_stride_bytes(stride_bytes);
      stride_bytes *= extent;
      ++d;
    }
  }

  T* base() const { return reinterpret_cast<T*>(buffer_base::base); }

  // Make a buffer and space for dims in the same allocation.
  static std::unique_ptr<buffer<T>, void (*)(buffer<T>*)> make(std::size_t rank) {
    auto buf = buffer_base::make(rank, sizeof(T));
    return {static_cast<buffer<T>*>(buf.release()), (void (*)(buffer<T>*))buffer_base::destroy};
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
const buffer<NewT>& buffer_base::cast() const {
  assert(elem_size == sizeof(NewT));
  return *reinterpret_cast<const buffer<NewT>*>(this);
}

}  // namespace slinky

#endif  // SLINKY_BUFFER_H
