#ifndef SLINKY_RUNTIME_BUFFER_H
#define SLINKY_RUNTIME_BUFFER_H

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <type_traits>

#include "slinky/base/arithmetic.h"
#include "slinky/base/function_ref.h"
#include "slinky/base/span.h"
#include "slinky/base/util.h"

namespace slinky {

// index_t needs to at least be as big as a pointer and must be signed.
// Using ptrdiff_t or intptr_t here seems tempting, but those can
// alias to `long` under some compilers which can cause some not so fun
// overloading issues with expr(), so let's use std::conditional
// instead to make it an exact alias of either int32_t or int64_t.
using index_t = std::conditional<sizeof(void*) == 4, std::int32_t, std::int64_t>::type;

// Helper to offset a pointer by a number of bytes.
template <typename T>
T* offset_bytes_non_null(T* x, std::ptrdiff_t bytes) {
  assert(x != nullptr || bytes == 0);
  return reinterpret_cast<T*>(reinterpret_cast<char*>(x) + bytes);
}
template <typename T>
const T* offset_bytes_non_null(const T* x, std::ptrdiff_t bytes) {
  assert(x != nullptr || bytes == 0);
  return reinterpret_cast<const T*>(reinterpret_cast<const char*>(x) + bytes);
}

template <typename T>
T* offset_bytes(T* x, std::ptrdiff_t bytes) {
  return x ? reinterpret_cast<T*>(reinterpret_cast<char*>(x) + bytes) : x;
}
template <typename T>
const T* offset_bytes(const T* x, std::ptrdiff_t bytes) {
  return x ? reinterpret_cast<const T*>(reinterpret_cast<const char*>(x) + bytes) : x;
}

template <typename T>
T* align_up(T* x, std::size_t align) {
  return reinterpret_cast<T*>((reinterpret_cast<uintptr_t>(x) + align - 1) & ~(align - 1));
}
template <typename T>
const T* align_up(const T* x, std::size_t align) {
  return reinterpret_cast<const T*>((reinterpret_cast<uintptr_t>(x) + align - 1) & ~(align - 1));
}

// TODO(https://github.com/dsharlet/slinky/issues/1): This and buffer_expr in pipeline.h should have the same API
// (except for expr instead of index_t).
class dim {
  alignas(16) index_t min_;
  index_t max_;
  index_t stride_;
  index_t fold_factor_;

public:
  static constexpr index_t auto_stride = std::numeric_limits<index_t>::max();
  static constexpr index_t unfolded = -1;

  dim() : min_(0), max_(-1), stride_(auto_stride), fold_factor_(unfolded) {}
  dim(index_t min, index_t max, index_t stride = auto_stride, index_t fold_factor = unfolded)
      : min_(min), max_(max), stride_(stride), fold_factor_(fold_factor) {}

  friend bool operator==(const dim& lhs, const dim& rhs) {
    return std::tie(lhs.min_, lhs.max_, lhs.stride_, lhs.fold_factor_) ==
           std::tie(rhs.min_, rhs.max_, rhs.stride_, rhs.fold_factor_);
  }

  friend bool operator!=(const dim& lhs, const dim& rhs) { return !(lhs == rhs); }

  static const dim& broadcast();

  index_t min() const { return min_; }
  index_t max() const { return max_; }
  index_t begin() const { return min_; }
  index_t end() const { return max_ + 1; }
  index_t extent() const { return end() > begin() ? end() - begin() : 0; }
  index_t stride() const { return stride_; }
  index_t fold_factor() const { return fold_factor_; }
  bool empty() const { return max_ < min_; }
  bool unbounded() const {
    return min_ == std::numeric_limits<index_t>::min() && max_ == std::numeric_limits<index_t>::max();
  }

  void set_extent(index_t extent) { max_ = min_ + extent - 1; }
  void set_point(index_t x) {
    min_ = x;
    max_ = x;
  }
  void set_bounds(index_t min, index_t max) {
    min_ = min;
    max_ = max;
  }
  void set_range(index_t begin, index_t end) {
    min_ = begin;
    max_ = end - 1;
  }
  void set_unbounded() {
    min_ = std::numeric_limits<index_t>::min();
    max_ = std::numeric_limits<index_t>::max();
  }
  void set_min_extent(index_t min, index_t extent) {
    min_ = min;
    max_ = min + extent - 1;
  }
  void set_stride(index_t stride) { stride_ = stride; }
  void set_fold_factor(index_t fold_factor) { fold_factor_ = fold_factor; }

  void translate(index_t offset) {
    min_ += offset;
    max_ += offset;
  }

  // Returns true if the interval [a, b] is in bounds of this dimension.
  bool contains(index_t a, index_t b) const { return fold_factor_ == 0 || (min() <= a && b <= max()); }
  bool contains(index_t x) const { return contains(x, x); }
  bool contains(const dim& other) const { return contains(other.min(), other.max()); }

  std::ptrdiff_t flat_offset_bytes(index_t i) const {
    assert(contains(i));
#ifdef UNDEFINED_BEHAVIOR_SANITIZER
    // Some integer overflow below is harmless when multiplied by zero, but flagged by ubsan.
    if (stride() == 0) return 0;
#endif
    if (stride() == 0 || fold_factor() == unfolded) {
      return (i - min()) * stride();
    } else {
      return euclidean_mod_positive_modulus(i, fold_factor()) * stride();
    }
  }

  // Check if the dimension crosses a fold between min and max.
  bool is_folded(index_t min, index_t max) const {
    if (stride() == 0 || fold_factor() <= 0) return false;
    return euclidean_div_positive_divisor(min, fold_factor()) != euclidean_div_positive_divisor(max, fold_factor());
  }
  bool is_folded(const dim& other) const { return is_folded(other.min(), other.max()); }
  bool is_folded() const { return is_folded(min(), max()); }
};

template <typename T, std::size_t DimsSize = 0>
class buffer;

class raw_buffer;

using raw_buffer_ptr = std::shared_ptr<raw_buffer>;
using const_raw_buffer_ptr = std::shared_ptr<const raw_buffer>;

// This value allows expressing `at` and `address_at` arguments that slice that dimension.
static constexpr struct {
} slice;

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
  static std::ptrdiff_t flat_offset_bytes_impl(const slinky::dim* dims, index_t i0) { return dims->flat_offset_bytes(i0); }
  static std::ptrdiff_t flat_offset_bytes_impl(const slinky::dim*, decltype(slinky::slice)) { return 0; }

  template <typename... Indices>
  static std::ptrdiff_t flat_offset_bytes_impl(const slinky::dim* dims, index_t i0, Indices... indices) {
    return dims->flat_offset_bytes(i0) + flat_offset_bytes_impl(dims + 1, indices...);
  }
  template <typename... Indices>
  static std::ptrdiff_t flat_offset_bytes_impl(const slinky::dim* dims, decltype(slinky::slice), Indices... indices) {
    return flat_offset_bytes_impl(dims + 1, indices...);
  }

  static bool contains_impl(const slinky::dim* dims, index_t i0) { return dims->contains(i0); }

  template <typename... Indices>
  static bool contains_impl(const slinky::dim* dims, index_t i0, Indices... indices) {
    return dims->contains(i0) && contains_impl(dims + 1, indices...);
  }

  static void translate_impl(slinky::dim* dims, index_t o0) { dims->translate(o0); }

  template <typename... Offsets>
  static void translate_impl(slinky::dim* dims, index_t o0, Offsets... offsets) {
    dims->translate(o0);
    translate_impl(dims + 1, offsets...);
  }

public:
  using element = void;
  using pointer = void*;

  alignas(16) void* base;
  std::size_t elem_size;
  std::size_t rank;
  slinky::dim* dims;

  slinky::dim& dim(std::size_t i) {
    assert(i < rank);
    return dims[i];
  }
  const slinky::dim& dim(std::size_t i) const {
    assert(i < rank);
    return dims[i];
  }

  // `indices` may either be integral, or `slice`, indicating that the dimension should be sliced.
  template <typename... Indices>
  std::ptrdiff_t flat_offset_bytes(index_t i0, Indices... indices) const {
    assert(sizeof...(indices) + 1 <= rank);
    return flat_offset_bytes_impl(dims, i0, indices...);
  }
  template <typename... Indices>
  std::ptrdiff_t flat_offset_bytes(decltype(slinky::slice) i0, Indices... indices) const {
    assert(sizeof...(indices) + 1 <= rank);
    return flat_offset_bytes_impl(dims, i0, indices...);
  }
  template <typename... Indices>
  void* address_at(index_t i0, Indices... indices) const {
    assert(sizeof...(indices) + 1 <= rank);
    return offset_bytes(base, flat_offset_bytes(i0, indices...));
  }
  template <typename... Indices>
  void* address_at(decltype(slinky::slice) i0, Indices... indices) const {
    assert(sizeof...(indices) + 1 <= rank);
    return offset_bytes(base, flat_offset_bytes(i0, indices...));
  }
  std::ptrdiff_t flat_offset_bytes() const { return 0; }
  void* address_at() const { return base; }

  template <typename... Indices>
  bool contains(index_t i0, Indices... indices) const {
    assert(sizeof...(indices) + 1 <= rank);
    return contains_impl(dims, i0, indices...);
  }
  template <typename... Indices>
  bool contains(decltype(slinky::slice) i0, Indices... indices) const {
    assert(sizeof...(indices) + 1 <= rank);
    return contains_impl(dims, i0, indices...);
  }
  bool contains() const { return true; }

  std::ptrdiff_t flat_offset_bytes(span<const index_t> indices) const {
    assert(indices.size() <= rank);
    index_t offset = 0;
    for (std::size_t i = 0; i < indices.size(); ++i) {
      offset += dims[i].flat_offset_bytes(indices[i]);
    }
    return offset;
  }
  void* address_at(span<const index_t> indices) const { return offset_bytes(base, flat_offset_bytes(indices)); }
  bool contains(span<const index_t> indices) const {
    assert(indices.size() <= rank);
    bool result = true;
    for (std::size_t i = 0; i < indices.size(); ++i) {
      result = result && dims[i].contains(indices[i]);
    }
    return result;
  }

  template <typename... Offsets>
  raw_buffer& translate(index_t o0, Offsets... offsets) {
    assert(sizeof...(offsets) + 1 <= rank);
    translate_impl(dims, o0, offsets...);
    return *this;
  }
  raw_buffer& translate(span<const index_t> offsets) {
    assert(offsets.size() <= rank);
    for (std::size_t i = 0; i < offsets.size(); ++i) {
      dims[i].translate(offsets[i]);
    }
    return *this;
  }

  // Remove dimensions `ds`. The dimensions must be sorted in ascending order.
  raw_buffer& slice(span<const std::size_t> ds) {
    if (ds.size() == 1) return slice(ds[0]);

    // Handle any slices of leading dimensions by just incrementing the dims pointer.
    std::size_t slice_leading = 0;
    for (std::size_t d : ds) {
      if (d == slice_leading) {
        ++slice_leading;
      } else {
        break;
      }
    }

    for (std::size_t i = slice_leading; i < ds.size(); ++i) {
      std::size_t d = ds[i];
      std::size_t next_d = i + 1 < ds.size() ? ds[i + 1] : rank;
      assert(d < rank);
      assert(next_d <= rank);

      // Move the dimensions between this slice and the next slice down by the number of slices we've done so far.
      // d and next_d are indices in the original dimensions.
      for (std::size_t j = d; j + 1 < next_d; ++j) {
        dims[j - (i - slice_leading)] = dims[j + 1];
      }
    }

    dims += slice_leading;
    rank -= ds.size();

    return *this;
  }
  raw_buffer& slice(std::initializer_list<std::size_t> ds) { return slice({&*ds.begin(), ds.size()}); }

  // Remove dimension `d` and move the base pointer to point to `at` in this dimension.
  // `at` is dim(d).min() by default.
  // If `d` is 0 or rank - 1, the slice does not mutate the dims array.
  raw_buffer& slice(std::size_t d) {
    assert(d < rank);
    rank -= 1;
    if (d == 0) {
      // Slicing the first leading dimension, we can just increment the dims pointer.
      dims += 1;
    } else {
      // We need to move all the dims above `d` down by one.
      for (std::size_t i = d; i < rank; ++i) {
        dims[i] = dims[i + 1];
      }
    }
    return *this;
  }
  raw_buffer& slice(std::size_t d, index_t at) {
    if (base != nullptr) {
      if (dim(d).contains(at)) {
        base = offset_bytes_non_null(base, dim(d).flat_offset_bytes(at));
      } else {
        base = nullptr;
      }
    }
    return slice(d);
  }

  // Crop the buffer in dimension `d` to the bounds `[min, max]`. The bounds will be clamped to the existing bounds.
  // Updates the base pointer to point to the new min.
  raw_buffer& crop(std::size_t d, index_t min, index_t max) {
    min = std::max(min, dim(d).min());
    max = std::min(max, dim(d).max());

    if (base != nullptr) {
      if (max >= min) {
        if (dim(d).fold_factor() == dim::unfolded) {
          index_t offset = dim(d).flat_offset_bytes(min);
          base = offset_bytes_non_null(base, offset);
        }
      } else {
        base = nullptr;
      }
    }

    dim(d).set_bounds(min, max);
    return *this;
  }

  std::size_t size_bytes() const;

  std::size_t elem_count() const;

  // If any strides are `auto_stride`, replace them with automatically determined strides.
  // `alignment` must be a power of 2.
  std::size_t init_strides(index_t alignment = 1);

  // Allocate and set the base pointer using `malloc`. Returns a pointer to the allocated memory, which should
  // be deallocated with `aligned_free`. `base_alignment` and `stride_alignment` must be a power of 2.
  void* allocate(index_t base_alignment = 1, index_t stride_alignment = 1);

  template <typename NewT>
  const buffer<NewT>& cast() const;

  // Make a pointer to a buffer with an allocation for the dims and (optionally) elements in the same allocation.
  static raw_buffer_ptr make(
      std::size_t rank, std::size_t elem_size = 0, const class slinky::dim* dims = nullptr, index_t alignment = 1);

  // Make a deep copy of another buffer, including allocating and copying the data.
  static raw_buffer_ptr make_copy(const raw_buffer& src, index_t alignment = 1);

  // Make a buffer around a scalar value. The resulting buffer will have rank 0. The result is a heap allocated
  // buffer that contains a copy of the scalar value.
  static raw_buffer_ptr make_scalar(std::size_t elem_size, const void* value, index_t alignment = 1);
  template <typename T, typename = typename std::enable_if_t<std::is_trivial_v<T>>>
  static raw_buffer_ptr make_scalar(const T& value, index_t alignment = 1) {
    return make_scalar(sizeof(T), &value, alignment);
  }

  // Make a buffer around a scalar value. The resulting buffer will have rank 0. The result is a buffer that contains a
  // pointer to the value.
  static raw_buffer make_scalar_ref(std::size_t elem_size, void* value) {
    return raw_buffer{value, elem_size, 0, nullptr};
  }
  template <typename T>
  static raw_buffer make_scalar_ref(const T& value) {
    return make_scalar_ref(sizeof(T), &value);
  }
};

// This is a wrapper for a raw_buffer that represents a scalar value, with storage for the scalar value.
template <typename T>
class scalar : public raw_buffer {
public:
  T value;

  scalar(const T& value = T()) : value(value) {
    base = &this->value;
    elem_size = sizeof(T);
    rank = 0;
    dims = nullptr;
  }
};

namespace internal {

template <typename T>
struct type_info {
  static constexpr std::size_t size = sizeof(T);
};
template <>
struct type_info<void> {
  static constexpr std::size_t size = 0;
};
template <>
struct type_info<const void> {
  static constexpr std::size_t size = 0;
};

template <typename T>
void copy_small_n(const T* src, std::size_t n, T* dst) {
  switch (n) {
  case 4: *dst++ = *src++;
  case 3: *dst++ = *src++;
  case 2: *dst++ = *src++;
  case 1: *dst++ = *src++;
  case 0: return;
  default: std::copy_n(src, n, dst); return;
  }
}

template <typename T>
void copy_small_n_backward(const T* src, std::size_t n, T* dst) {
  switch (n) {
  case 4: *(--dst) = *(--src);
  case 3: *(--dst) = *(--src);
  case 2: *(--dst) = *(--src);
  case 1: *(--dst) = *(--src);
  case 0: return;
  default: std::copy_backward(src, src + n, dst + n); return;
  }
}

}  // namespace internal

template <typename T, std::size_t DimsSize>
class buffer : public raw_buffer {
private:
  void* to_free;

  // Avoid default constructor of slinky::dim, we might not use this.
  alignas(slinky::dim) char dims_storage[sizeof(slinky::dim) * DimsSize];

  void assign_dims(std::size_t rank, const slinky::dim* src = nullptr) {
    assert(rank <= DimsSize);
    this->rank = rank;
    if (DimsSize > 0) {
      dims = reinterpret_cast<slinky::dim*>(dims_storage);
      if (src) {
        internal::copy_small_n(src, rank, dims);
      } else {
        new (dims) slinky::dim[rank];
      }
    } else {
      dims = nullptr;
    }
  }

  void copy_construct(const raw_buffer& c) {
    raw_buffer::base = c.base;
    elem_size = c.elem_size;
    assign_dims(c.rank, c.dims);
    to_free = nullptr;
  }

  buffer& assign(const raw_buffer& c) {
    if (static_cast<void*>(this) == static_cast<const void*>(&c)) return *this;
    free();
    copy_construct(c);
    return *this;
  }

  template <std::size_t OtherDimsSize>
  void move_construct(buffer<T, OtherDimsSize>&& m) {
    copy_construct(m);
    // Take ownership of the data.
    to_free = m.to_free;
    m.to_free = nullptr;
  }

  template <std::size_t OtherDimsSize>
  buffer& move(buffer<T, OtherDimsSize>&& m) {
    assign(m);
    // Take ownership of the data.
    std::swap(to_free, m.to_free);
    return *this;
  }

public:
  using element = T;
  using pointer = T*;

  using raw_buffer::cast;
  using raw_buffer::dim;
  using raw_buffer::elem_size;
  using raw_buffer::flat_offset_bytes;
  using raw_buffer::rank;

  buffer() {
    raw_buffer::base = nullptr;
    to_free = nullptr;
    assign_dims(DimsSize);
    elem_size = internal::type_info<T>::size;
  }

  explicit buffer(std::size_t rank, std::size_t elem_size = internal::type_info<T>::size) {
    raw_buffer::base = nullptr;
    to_free = nullptr;
    assign_dims(rank);
    this->elem_size = elem_size;
  }

  // Construct a buffer with extents, and strides computed such that the stride of dimension
  // n is the product of all the extents of dimensions [0, n) and elem_size, i.e. the first
  // dimension is "innermost".
  buffer(span<const index_t> extents, std::size_t elem_size = internal::type_info<T>::size)
      : buffer(extents.size(), elem_size) {
    slinky::dim* d = dims;
    for (index_t extent : extents) {
      d->set_min_extent(0, extent);
      ++d;
    }
    init_strides();
  }
  buffer(std::initializer_list<index_t> extents, std::size_t elem_size = internal::type_info<T>::size)
      : buffer({extents.begin(), extents.end()}, elem_size) {}
  // TODO: A more general version of this constructor would probably be useful.
  buffer(T* base, index_t size, std::size_t elem_size = internal::type_info<T>::size) : buffer({size}) {
    raw_buffer::base = base;
  }
  ~buffer() { free(); }

  // All buffer copy/assignment operators are shallow copies.
  buffer(const raw_buffer& c) { copy_construct(c); }
  buffer(const buffer& c) { copy_construct(c); }
  buffer(buffer&& m) { move_construct(std::move(m)); }
  template <std::size_t OtherDimsSize>
  buffer(const buffer<T, OtherDimsSize>& c) {
    copy_construct(c);
  }
  template <std::size_t OtherDimsSize>
  buffer(buffer<T, OtherDimsSize>&& m) {
    move_construct(std::move(m));
  }

  buffer& operator=(const raw_buffer& c) { return assign(c); }
  buffer& operator=(const buffer& c) { return assign(c); }
  buffer& operator=(buffer&& m) { return move(std::move(m)); }
  template <std::size_t OtherDimsSize>
  buffer& operator=(const buffer<T, OtherDimsSize>& c) {
    return assign(c);
  }
  template <std::size_t OtherDimsSize>
  buffer& operator=(buffer<T, OtherDimsSize>&& m) {
    return move(std::move(m));
  }

  T* base() const { return reinterpret_cast<T*>(raw_buffer::base); }

  // These accessors are not designed to be fast. They exist to facilitate testing,
  // and maybe they are useful to compute addresses.
  // `indices` may either be integral, or `slice`, indicating that the dimension should be sliced.
  template <typename... Indices>
  auto& at(index_t i0, Indices... indices) const {
    return *offset_bytes_non_null(base(), flat_offset_bytes(i0, indices...));
  }
  template <typename... Indices>
  auto& at(decltype(slinky::slice) i0, Indices... indices) const {
    return *offset_bytes_non_null(base(), flat_offset_bytes(i0, indices...));
  }
  template <typename... Indices>
  auto& operator()(index_t i0, Indices... indices) const {
    return at(i0, indices...);
  }
  template <typename... Indices>
  auto& operator()(decltype(slinky::slice) i0, Indices... indices) const {
    return at(i0, indices...);
  }

  auto& at() const { return *base(); }
  auto& operator()() const { return *base(); }

  auto& at(span<const index_t> indices) const { return *offset_bytes_non_null(base(), flat_offset_bytes(indices)); }
  auto& operator()(span<const index_t> indices) const { return at(indices); }

  // Insert a new dimension `dim` at index d, increasing the rank by 1.
  buffer<T, DimsSize>& unslice(std::size_t d, const slinky::dim& dim) {
    assert(d <= rank);
    slinky::dim* dims_storage = reinterpret_cast<slinky::dim*>(this->dims_storage);
    if (d == 0 && &dims_storage[0] < dims) {
      assert(dims < &dims_storage[DimsSize]);
      dims -= 1;
    } else {
      assert(&dims_storage[0] <= dims && dims + 1 < &dims_storage[DimsSize]);
      std::copy_backward(dims + d, dims + rank, dims + rank + 1);
    }
    rank += 1;
    dims[d] = dim;
    return *this;
  }

  void allocate(index_t alignment = 1) {
    assert(!to_free);
    to_free = raw_buffer::allocate(alignment);
  }

  void free() {
    if (to_free) {
      ::free(to_free);
      to_free = nullptr;
    }
  }
};

template <typename NewT>
const buffer<NewT>& raw_buffer::cast() const {
  assert(elem_size == internal::type_info<NewT>::size || internal::type_info<NewT>::size == 0);
  return *reinterpret_cast<const buffer<NewT>*>(this);
}

// Copy the contents of `src` to `dst`.
// If `padding` is `no_padding, every index of `dst` must be in bounds of `src`.
// If `padding` is not `no_padding`, `dst` will be copied from `src` if it is in bounds, otherwise it will be copied
// from `padding`, which must be in bounds.
static constexpr raw_buffer no_padding = {};
void copy(const raw_buffer& src, const raw_buffer& dst, const raw_buffer& pad = no_padding);

// Performs only the padding operation of a copy. The region that would have been copied is unmodified.
void pad(const dim* src_bounds, const raw_buffer& dst, const raw_buffer& pad);

// Returns true if the two dimensions can be fused.
inline bool can_fuse(const dim& inner, const dim& outer) {
  if (outer.min() == outer.max() && outer.fold_factor() != 0) return true;

#ifdef UNDEFINED_BEHAVIOR_SANITIZER
  // Some integer overflow below is harmless when multiplied by zero, but flagged by ubsan.
  index_t next_stride = inner.stride() == 0 ? 0 : inner.stride() * inner.extent();
#else
  index_t next_stride = inner.stride() * (inner.max() - inner.min() + 1);
#endif
  if (next_stride != outer.stride()) return false;

  return next_stride == 0 || inner.fold_factor() == dim::unfolded;
}

// Fuse two dimensions of a buffer.
inline slinky::dim fuse(slinky::dim inner, const slinky::dim& outer) {
  assert(can_fuse(inner, outer));
  if (inner.unbounded()) {
    // Already fused
  } else if (outer == dim::broadcast()) {
    inner = outer;
  } else {
    const index_t inner_extent = inner.extent();
    if (outer.min() != outer.max() && outer.fold_factor() != dim::unfolded) {
      assert(!inner.is_folded());
      inner.set_fold_factor(outer.fold_factor() * inner_extent);
    }
    if (outer.unbounded()) {
      inner.set_unbounded();
    } else {
      inner.set_range(outer.begin() * inner_extent, outer.end() * inner_extent);
    }
  }
  return inner;
}

enum class fuse_type {
  // Leave the outer dimension in its place in an undefined state.
  undef,
  // Replace the outer dimension with a dimension of extent 1 and min 0, preserving the rank.
  keep,
  // Remove the outer dimension, reducing the rank by 1.
  remove,
};

// Fuse two dimensions of a buffer.
template <fuse_type type>
inline void fuse(index_t inner, index_t outer, raw_buffer& buf) {
  if (outer >= static_cast<index_t>(buf.rank)) {
    // Fusing an implicit broadcast, nothing to do.
    assert(inner >= static_cast<index_t>(buf.rank) || can_fuse(buf.dim(inner), dim::broadcast()));
  } else if (inner >= static_cast<index_t>(buf.rank)) {
    // The inner dimension is an implicit broadcast.
    dim& od = buf.dim(outer);
    assert(can_fuse(dim::broadcast(), od));
    if (type == fuse_type::keep) {
      od.set_point(0);
    } else if (type == fuse_type::remove) {
      buf.slice(outer);
    } else {
      assert(type == fuse_type::undef);
    }
  } else {
    dim& id = buf.dim(inner);
    dim& od = buf.dim(outer);
    id = fuse(id, od);
    if (type == fuse_type::keep) {
      od.set_point(0);
    } else if (type == fuse_type::remove) {
      buf.slice(outer);
    } else {
      assert(type == fuse_type::undef);
    }
  }
}

namespace internal {

// Returns true if all buffers have the same rank.
inline bool same_rank(const raw_buffer&) { return true; }
inline bool same_rank(const raw_buffer& buf0, const raw_buffer& buf1) { return buf0.rank == buf1.rank; }
template <typename... Bufs>
bool same_rank(const raw_buffer& buf0, const raw_buffer& buf1, const Bufs&... bufs) {
  return buf0.rank == buf1.rank && same_rank(buf1, bufs...);
}

inline bool same_bounds(const dim& a, const dim& b) { return a.min() == b.min() && a.max() == b.max(); }

inline const dim& dim_or_broadcast(const raw_buffer& buf, std::ptrdiff_t d) {
  return d < static_cast<std::ptrdiff_t>(buf.rank) ? buf.dim(d) : dim::broadcast();
}

// Returns true if all buffers have the same bounds in dimension d.
inline bool same_bounds(std::ptrdiff_t, const raw_buffer&) { return true; }
template <typename... Bufs>
bool same_bounds(std::size_t d, const raw_buffer& buf0, const raw_buffer& buf1, const Bufs&... bufs) {
  return (buf0.rank <= d || buf1.rank <= d || same_bounds(buf0.dim(d), buf1.dim(d))) && same_bounds(d, buf0, bufs...);
}

// Returns true if two dimensions of all buffers can be fused.
inline bool can_fuse(std::ptrdiff_t, std::ptrdiff_t) { return true; }
template <typename... Bufs>
bool can_fuse(std::ptrdiff_t inner, std::ptrdiff_t outer, const raw_buffer& buf, const Bufs&... bufs) {
  return can_fuse(dim_or_broadcast(buf, inner), dim_or_broadcast(buf, outer)) && can_fuse(inner, outer, bufs...);
}

// Fuse two dimensions of all buffers.
template <fuse_type type>
void fuse(std::ptrdiff_t inner, std::ptrdiff_t outer) {}
template <fuse_type type, typename... Bufs>
void fuse(std::ptrdiff_t inner, std::ptrdiff_t outer, raw_buffer& buf, Bufs&... bufs) {
  slinky::fuse<type>(inner, outer, buf);
  fuse<type>(inner, outer, bufs...);
}

template <typename... Bufs>
SLINKY_INLINE bool attempt_fuse(
    std::ptrdiff_t inner, std::ptrdiff_t outer, raw_buffer& buf, Bufs&... bufs) {
  if (same_bounds(inner, buf, bufs...) && can_fuse(inner, outer, buf, bufs...)) {
    fuse<fuse_type::remove>(inner, outer, buf, bufs...);
    return true;
  } else {
    return false;
  }
}

template <typename... Bufs>
SLINKY_INLINE bool attempt_fuse(
    std::ptrdiff_t inner, std::ptrdiff_t outer, span<const int> dim_sets, raw_buffer& buf, Bufs&... bufs) {
  if (static_cast<int>(dim_sets.size()) > outer && dim_sets[outer] != dim_sets[inner]) {
    // These two dims are not part of the same set. Don't fuse them.
    return false;
  }
  return attempt_fuse(inner, outer, buf, bufs...);
}

inline void swap_dims(std::size_t i, std::size_t j, raw_buffer& buf) { std::swap(buf.dim(i), buf.dim(j)); }
template <typename... Bufs>
void swap_dims(std::size_t i, std::size_t j, raw_buffer& buf, Bufs&... bufs) {
  swap_dims(i, j, buf);
  swap_dims(i, j, bufs...);
}

}  // namespace internal

// Jointly sort the dimensions of all buffers such that the strides of the first buffer are in ascending order.
// `dim_sets` is an optional set of integers that indicates the sets that dimensions belong to. Dimensions are only
// sorted within the same set, where dimension `d` is identified by the value of `dim_sets[d]`. By default, all
// dimensions are considered to be in the same set.
// Returns true if the sort changed the ordering of the dimensions.
template <typename... Bufs>
bool sort_dims(span<const int> dim_sets, raw_buffer& buf, Bufs&... bufs) {
  // We only attempt to sort the dimensions that exist in all buffers. We could do better here, sometimes we can
  // swap implicit broadcast dimensions with another broadcast dimension.
  const size_t rank = std::min({buf.rank, bufs.rank...});
  bool modified = false;
  // A bubble sort is appropriate here, because:
  // - Typically, buffer ranks are very small.
  // - To use std::sort or similar, we need to copy the buffer dimensions into a temporary, sort, and copy them back.
  // - This template will be instantiated very frequently, it's worth attempting to minimize code size.
  for (std::size_t i = 0; i < rank; ++i) {
    for (std::size_t j = i + 1; j < rank; ++j) {
      if (j < dim_sets.size() && dim_sets[i] != dim_sets[j]) continue;
      if (buf.dim(i).stride() > buf.dim(j).stride()) {
        internal::swap_dims(i, j, buf, bufs...);
        modified = true;
      }
    }
  }
  return modified;
}
template <typename... Bufs>
bool sort_dims(raw_buffer& buf, Bufs&... bufs) {
  return sort_dims({}, buf, bufs...);
}

// Fuse the dimensions of a set of buffers where the dimensions are contiguous in memory. It only fuses a dimension if
// the same dimension can be fused in all buffers. After this function runs, the buffers may have lower rank, but still
// address the same memory as the original buffers. For "shift invariant" operations, operating on such fused buffers is
// likely to be more efficient. `dim_sets` is an optional span of integers that indicates sets of dimensions that are
// eligible for fusion. By default, all dimensions are considered to be part of the same set.
template <typename... Bufs>
int fuse_contiguous_dims(span<const int> dim_sets, raw_buffer& buf, Bufs&... bufs) {
  int fused = 0;
  for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(buf.rank) - 1; d > 0; --d) {
    fused += internal::attempt_fuse(d - 1, d, dim_sets, buf, bufs...);
  }
  return fused;
}
template <typename... Bufs>
int fuse_contiguous_dims(raw_buffer& buf, Bufs&... bufs) {
  int fused = 0;
  for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(buf.rank) - 1; d > 0; --d) {
    fused += internal::attempt_fuse(d - 1, d, buf, bufs...);
  }
  return fused;
}

// Call both `sort_dims` and `fuse_contiguous_dims` on the buffers.
template <typename... Bufs>
int optimize_dims(span<const int> dim_sets, raw_buffer& buf, Bufs&... bufs) {
  // The order of operations here is for performance: It's a lot faster to fuse dimensions than sort them. So we fuse
  // what we can before sorting, then if the sorting changed the order of the dimensions, attempt to fuse again.
  int fused = fuse_contiguous_dims(dim_sets, buf, bufs...);
  if (sort_dims(dim_sets, buf, bufs...)) {
    fused += fuse_contiguous_dims(dim_sets, buf, bufs...);
  }
  return fused;
}
template <typename... Bufs>
int optimize_dims(raw_buffer& buf, Bufs&... bufs) {
  int fused = fuse_contiguous_dims(buf, bufs...);
  if (sort_dims(buf, bufs...)) {
    fused += fuse_contiguous_dims(buf, bufs...);
  }
  return fused;
}

namespace internal {

template <std::size_t BufsSize>
SLINKY_INLINE void increment_bases(std::size_t n, void** bases, const index_t* strides) {
  n = BufsSize != dynamic_extent ? BufsSize : n;
  bases[0] = offset_bytes(bases[0], strides[0]);
  if (1 < n) bases[1] = offset_bytes(bases[1], strides[1]);
  if (2 < n) bases[2] = offset_bytes(bases[2], strides[2]);
  for (std::size_t i = 3; i < n; ++i) {
    bases[i] = offset_bytes(bases[i], strides[i]);
  }
}

template <typename... Ts, std::size_t... Is>
auto array_to_tuple(void** x, std::index_sequence<Is...>) {
  return std::make_tuple(static_cast<Ts>(x[Is])...);
}

// The implementation of for_each_element involves quite a bit of code. To avoid code size problems, we implement it
// with type erased callbacks. To mitigate the overhead impact of this, the last linear loop is implemented in the
// callbacks without type erasure below.
using for_each_contiguous_slice_callback = function_ref<void(index_t, void**, index_t, const index_t*)>;
using for_each_element_callback = function_ref<void(void**, index_t, const index_t*)>;
template <std::size_t BufsSize>
void for_each_contiguous_slice_impl(span<const raw_buffer*, BufsSize> bufs, for_each_contiguous_slice_callback fn);
template <std::size_t BufsSize>
void for_each_element_impl(span<const raw_buffer*, BufsSize> bufs, for_each_element_callback fn);

// The above templates are only instantiated for a small number of sizes, up to this number. Larger values should use
// 0, which is handled by a runtime parameter instead of a compile-time constant.
static constexpr std::size_t max_bufs_size = 4;

}  // namespace internal

// Call `f(index_t extent, T* base[, Ts* bases, ...])` for each contiguous slice in the domain of `buf[,
// bufs...]`. This function attempts to be efficient to support production quality implementations of callbacks.
//
// When additional buffers are passed, they will be sliced in tandem with the 'main' buffer. Additional buffers can be
// lower rank than the main buffer, these "missing" dimensions are not sliced (i.e. broadcasting in this dimension).
// If the other buffers are out of bounds for a slice, the corresponding argument to the callback will be `nullptr`.
template <typename Buf, typename F, typename... Bufs>
SLINKY_NO_STACK_PROTECTOR void for_each_contiguous_slice(const Buf& buf, const F& f, const Bufs&... bufs) {
  static constexpr std::size_t BufsSize = sizeof...(Bufs) + 1;
  std::array<const raw_buffer*, BufsSize> buf_ptrs = {&buf, &bufs...};

  static constexpr std::size_t ConstBufsSize = BufsSize <= internal::max_bufs_size ? BufsSize : dynamic_extent;

  internal::for_each_contiguous_slice_impl<ConstBufsSize>(
      buf_ptrs, [&f](index_t slice_extent, void** bases, index_t extent, const index_t* strides) {
        for (;;) {
          std::apply(f, std::tuple_cat(std::make_tuple(slice_extent),
                            internal::array_to_tuple<typename Buf::pointer, typename Bufs::pointer...>(
                                bases, std::make_index_sequence<BufsSize>())));
          if (SLINKY_UNLIKELY(--extent <= 0)) break;
          internal::increment_bases<BufsSize>(0, bases, strides);
        }
      });
}

// Call `f` with a pointer to each element of `buf`, and pointers to the same corresponding elements of `bufs`, or
// `nullptr` if `buf` is out of bounds of `bufs`.
template <typename F, typename Buf, typename... Bufs>
SLINKY_NO_STACK_PROTECTOR void for_each_element(const F& f, const Buf& buf, const Bufs&... bufs) {
  static constexpr std::size_t BufsSize = sizeof...(Bufs) + 1;
  std::array<const raw_buffer*, BufsSize> buf_ptrs = {&buf, &bufs...};

  static constexpr std::size_t ConstBufsSize = BufsSize <= internal::max_bufs_size ? BufsSize : dynamic_extent;

  internal::for_each_element_impl<ConstBufsSize>(buf_ptrs, [&f](void** bases, index_t extent, const index_t* strides) {
    for (;;) {
      std::apply(f, internal::array_to_tuple<typename Buf::pointer, typename Bufs::pointer...>(
                        bases, std::make_index_sequence<BufsSize>()));
      if (SLINKY_UNLIKELY(--extent <= 0)) break;
      internal::increment_bases<BufsSize>(0, bases, strides);
    }
  });
}

}  // namespace slinky

#endif  // SLINKY_RUNTIME_BUFFER_H
