#ifndef SLINKY_RUNTIME_BUFFER_H
#define SLINKY_RUNTIME_BUFFER_H

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "base/arithmetic.h"
#include "base/span.h"
#include "base/util.h"

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

// TODO(https://github.com/dsharlet/slinky/issues/1): This and buffer_expr in pipeline.h should have the same API
// (except for expr instead of index_t).
class dim {
  index_t min_;
  index_t max_;
  index_t stride_;
  index_t fold_factor_;

public:
  static constexpr index_t auto_stride = std::numeric_limits<index_t>::min();
  static constexpr index_t unfolded = std::numeric_limits<index_t>::max();

  dim() : min_(0), max_(-1), stride_(auto_stride), fold_factor_(unfolded) {}
  dim(index_t min, index_t max, index_t stride = auto_stride, index_t fold_factor = unfolded)
      : min_(min), max_(max), stride_(stride), fold_factor_(fold_factor) {}

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
  bool contains(index_t a, index_t b) const { return min() <= a && b <= max(); }
  bool contains(index_t x) const { return contains(x, x); }
  bool contains(const dim& other) const { return contains(other.min(), other.max()); }

  std::ptrdiff_t flat_offset_bytes(index_t i) const {
    assert(contains(i));
#ifdef UNDEFINED_BEHAVIOR_SANITIZER
    // Some integer overflow below is harmless when multiplied by zero, but flagged by ubsan.
    if (stride() == 0) return 0;
#endif
    if (fold_factor() == unfolded) {
      return (i - min()) * stride();
    } else {
      return euclidean_mod_positive_modulus(i, fold_factor()) * stride();
    }
  }

  // Check if the dimension crosses a fold between min and max.
  bool is_folded(index_t min, index_t max) const {
    if (fold_factor() == unfolded) return false;
    return euclidean_div(min, fold_factor()) != euclidean_div(max, fold_factor());
  }
  bool is_folded(const dim& other) const { return is_folded(other.min(), other.max()); }
  bool is_folded() const { return is_folded(min(), max()); }
};

template <typename T, std::size_t DimsSize = 0>
class buffer;

class raw_buffer;

using raw_buffer_ptr = std::shared_ptr<raw_buffer>;
using const_raw_buffer_ptr = std::shared_ptr<const raw_buffer>;

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
  static std::ptrdiff_t flat_offset_bytes_impl(const dim* dims, index_t i0) { return dims->flat_offset_bytes(i0); }

  template <typename... Indices>
  static std::ptrdiff_t flat_offset_bytes_impl(const dim* dims, index_t i0, Indices... indices) {
    return dims->flat_offset_bytes(i0) + flat_offset_bytes_impl(dims + 1, indices...);
  }

  static bool contains_impl(const dim* dims, index_t i0) { return dims->contains(i0); }

  template <typename... Indices>
  static bool contains_impl(const dim* dims, index_t i0, Indices... indices) {
    return dims->contains(i0) && contains_impl(dims + 1, indices...);
  }

  static void translate_impl(dim* dims, index_t o0) { dims->translate(o0); }

  template <typename... Offsets>
  static void translate_impl(dim* dims, index_t o0, Offsets... offsets) {
    dims->translate(o0);
    translate_impl(dims + 1, offsets...);
  }

public:
  using pointer = void*;

  void* base;
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

  template <typename... Indices>
  std::ptrdiff_t flat_offset_bytes(index_t i0, Indices... indices) const {
    assert(sizeof...(indices) + 1 <= rank);
    return flat_offset_bytes_impl(dims, i0, indices...);
  }
  template <typename... Indices>
  void* address_at(index_t i0, Indices... indices) const {
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
  void init_strides(index_t alignment = 1);

  // Allocate and set the base pointer using `malloc`. Returns a pointer to the allocated memory, which should
  // be deallocated with `free`.
  void* allocate();

  template <typename NewT>
  const buffer<NewT>& cast() const;

  // Make a pointer to a buffer with an allocation for the dims and (optionally) elements in the same allocation.
  static raw_buffer_ptr make(std::size_t rank, std::size_t elem_size, const class dim* dims = nullptr);

  // Make a deep copy of another buffer, including allocating and copying the data.
  static raw_buffer_ptr make_copy(const raw_buffer& src);
};

namespace internal {

template <typename T>
struct default_elem_size {
  static constexpr std::size_t value = sizeof(T);
};
template <>
struct default_elem_size<void> {
  static constexpr std::size_t value = 0;
};
template <>
struct default_elem_size<const void> {
  static constexpr std::size_t value = 0;
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

}  // namespace internal

template <typename T, std::size_t DimsSize>
class buffer : public raw_buffer {
private:
  void* to_free;
  slinky::dim dims_storage[DimsSize];

  void assign_dims(int rank, const slinky::dim* src = nullptr) {
    assert(rank <= static_cast<int>(DimsSize));
    this->rank = rank;
    if (DimsSize > 0) {
      dims = dims_storage;
      if (src) {
        internal::copy_small_n(src, rank, dims);
      } else {
        new (dims) slinky::dim[rank];
      }
    } else {
      dims = nullptr;
    }
  }

  buffer& shallow_copy(const raw_buffer& c) {
    if (static_cast<void*>(this) == static_cast<const void*>(&c)) return *this;
    free();
    raw_buffer::base = c.base;
    elem_size = c.elem_size;
    assign_dims(c.rank, c.dims);
    return *this;
  }

  template <std::size_t OtherDimsSize>
  buffer& move(buffer<T, OtherDimsSize>&& m) {
    shallow_copy(m);
    // Take ownership of the data.
    std::swap(to_free, m.to_free);
    return *this;
  }

public:
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
    elem_size = internal::default_elem_size<T>::value;
  }

  explicit buffer(std::size_t rank, std::size_t elem_size = internal::default_elem_size<T>::value) {
    raw_buffer::base = nullptr;
    to_free = nullptr;
    assign_dims(rank);
    this->elem_size = elem_size;
  }

  // Construct a buffer with extents, and strides computed such that the stride of dimension
  // n is the product of all the extents of dimensions [0, n) and elem_size, i.e. the first
  // dimension is "innermost".
  buffer(span<const index_t> extents, std::size_t elem_size = internal::default_elem_size<T>::value)
      : buffer(extents.size(), elem_size) {
    slinky::dim* d = dims;
    for (index_t extent : extents) {
      d->set_min_extent(0, extent);
      ++d;
    }
    init_strides();
  }
  buffer(std::initializer_list<index_t> extents, std::size_t elem_size = internal::default_elem_size<T>::value)
      : buffer({extents.begin(), extents.end()}, elem_size) {}
  // TODO: A more general version of this constructor would probably be useful.
  buffer(T* base, index_t size, std::size_t elem_size = internal::default_elem_size<T>::value) : buffer({size}) {
    raw_buffer::base = base;
  }
  ~buffer() { free(); }

  // All buffer copy/assignment operators are shallow copies.
  buffer(const raw_buffer& c) : buffer() { shallow_copy(c); }
  buffer(const buffer& c) : buffer() { shallow_copy(c); }
  buffer(buffer&& m) : buffer() { move(std::move(m)); }
  template <std::size_t OtherDimsSize>
  buffer(const buffer<T, OtherDimsSize>& c) : buffer() {
    shallow_copy(c);
  }
  template <std::size_t OtherDimsSize>
  buffer(buffer<T, OtherDimsSize>&& m) : buffer() {
    move(std::move(m));
  }

  buffer& operator=(const raw_buffer& c) { return shallow_copy(c); }
  buffer& operator=(const buffer& c) { return shallow_copy(c); }
  buffer& operator=(buffer&& m) { return move(std::move(m)); }
  template <std::size_t OtherDimsSize>
  buffer& operator=(const buffer<T, OtherDimsSize>& c) {
    return shallow_copy(c);
  }
  template <std::size_t OtherDimsSize>
  buffer& operator=(buffer<T, OtherDimsSize>&& m) {
    return move(std::move(m));
  }

  T* base() const { return reinterpret_cast<T*>(raw_buffer::base); }

  // These accessors are not designed to be fast. They exist to facilitate testing,
  // and maybe they are useful to compute addresses.
  template <typename... Indices>
  auto& at(index_t i0, Indices... indices) const {
    return *offset_bytes_non_null(base(), flat_offset_bytes(i0, indices...));
  }
  template <typename... Indices>
  auto& operator()(index_t i0, Indices... indices) const {
    return at(i0, indices...);
  }

  auto& at() const { return *base(); }
  auto& operator()() const { return *base(); }

  auto& at(span<const index_t> indices) const { return *offset_bytes_non_null(base(), flat_offset_bytes(indices)); }
  auto& operator()(span<const index_t> indices) const { return at(indices); }

  // Insert a new dimension `dim` at index d, increasing the rank by 1.
  buffer<T, DimsSize>& unslice(std::size_t d, const slinky::dim& dim) {
    assert(d <= rank);
    if (d == 0 && &dims_storage[0] <= dims - 1) {
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

  void allocate() {
    assert(!to_free);
    to_free = raw_buffer::allocate();
  }

  void free() {
    ::free(to_free);
    to_free = nullptr;
  }
};

template <typename NewT>
const buffer<NewT>& raw_buffer::cast() const {
  return *reinterpret_cast<const buffer<NewT>*>(this);
}

// Copy the contents of `src` to `dst`.
// If `padding` is null, `src` must contain every index that `dst` contains.
// If `padding` is non-null, `dst` is filled with the padding when it is out of bounds of `src`.
void copy(const raw_buffer& src, const raw_buffer& dst, const void* padding = nullptr);
template <typename T, typename = typename std::enable_if<!std::is_pointer<T>::value>::type>
void copy(const raw_buffer& src, const raw_buffer& dst, const T& padding) {
  copy(src, dst, &padding);
}

// Performs only the padding operation of a copy. The region that would have been copied is unmodified.
void pad(const dim* in_bounds, const raw_buffer& dst, const void* padding);
template <typename T, typename = typename std::enable_if<!std::is_pointer<T>::value>::type>
void pad(const dim* in_bounds, const raw_buffer& dst, const T& padding) {
  pad(in_bounds, dst, &padding);
}

// Fill `dst` with `value`. `value` should point to `dst.elem_size` bytes.
void fill(const raw_buffer& dst, const void* value);
template <typename T, typename = typename std::enable_if<!std::is_pointer<T>::value>::type>
void fill(const raw_buffer& dst, const T& padding) {
  fill(dst, &padding);
}
// Returns true if the two dimensions can be fused.
inline bool can_fuse(const dim& inner, const dim& outer) {
  if (outer.max() == outer.min() && outer.stride() != 0) return true;
  if (inner.fold_factor() != dim::unfolded) return false;

#ifdef UNDEFINED_BEHAVIOR_SANITIZER
  // Some integer overflow below is harmless when multiplied by zero, but flagged by ubsan.
  index_t next_stride = inner.stride() == 0 ? 0 : inner.stride() * inner.extent();
#else
  index_t next_stride = inner.stride() * (inner.max() - inner.min() + 1);
#endif
  if (next_stride != outer.stride()) return false;
  return true;
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
inline void fuse(int inner, int outer, raw_buffer& buf) {
  dim& id = buf.dim(inner);
  dim& od = buf.dim(outer);
  assert(can_fuse(id, od));
  if (id.stride() == 0) {
    if (id.unbounded()) {
      // Already fused
    } else if (od.unbounded()) {
      id.set_unbounded();
    } else {
      id.set_range(od.begin() * id.extent(), od.end() * id.extent());
    }
  } else {
    const index_t id_extent = id.extent();
    if (od.min() != od.max() && od.fold_factor() != dim::unfolded) {
      assert(id.fold_factor() == dim::unfolded);
      id.set_fold_factor(od.fold_factor() * id_extent);
    }
    id.set_range(od.begin() * id_extent, od.end() * id_extent);
  }
  if (type == fuse_type::keep) {
    od.set_point(0);
  } else if (type == fuse_type::remove) {
    buf.slice(outer);
  } else {
    assert(type == fuse_type::undef);
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

inline bool same_bounds(const dim& a, const dim& b) {
  return a.min() == b.min() && a.max() == b.max() && a.fold_factor() == b.fold_factor();
}

// Returns true if all buffers have the same bounds in dimension d.
inline bool same_bounds(int, const raw_buffer&) { return true; }
inline bool same_bounds(int d, const raw_buffer& buf0, const raw_buffer& buf1) {
  return same_bounds(buf0.dim(d), buf1.dim(d));
}
template <typename... Bufs>
bool same_bounds(int d, const raw_buffer& buf0, const raw_buffer& buf1, const Bufs&... bufs) {
  return same_bounds(buf0.dim(d), buf1.dim(d)) && same_bounds(d, buf1, bufs...);
}

// Returns true if two dimensions of all buffers can be fused.
inline bool can_fuse(int, int) { return true; }
template <typename... Bufs>
bool can_fuse(int inner, int outer, const raw_buffer& buf, const Bufs&... bufs) {
  return can_fuse(buf.dim(inner), buf.dim(outer)) && can_fuse(inner, outer, bufs...);
}

// Fuse two dimensions of all buffers.
template <fuse_type type, typename... Bufs>
void fuse(int inner, int outer, raw_buffer& buf, Bufs&... bufs) {
  fuse<type>(inner, outer, buf);
  fuse<type>(inner, outer, bufs...);
}

template <typename... Bufs>
SLINKY_ALWAYS_INLINE inline void attempt_fuse(int inner, int outer, raw_buffer& buf, Bufs&... bufs) {
  if (same_bounds(inner, buf, bufs...) && can_fuse(inner, outer, buf, bufs...)) {
    fuse<fuse_type::remove>(inner, outer, buf, bufs...);
  }
}

template <typename... Bufs>
SLINKY_ALWAYS_INLINE inline void attempt_fuse(
    int inner, int outer, span<const int> dim_sets, raw_buffer& buf, Bufs&... bufs) {
  if (static_cast<int>(dim_sets.size()) > outer && dim_sets[outer] != dim_sets[inner]) {
    // These two dims are not part of the same set. Don't fuse them.
    return;
  }
  attempt_fuse(inner, outer, buf, bufs...);
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
  assert(internal::same_rank(buf, bufs...));
  bool modified = false;
  // A bubble sort is appropriate here, because:
  // - Typically, buffer ranks are very small.
  // - To use std::sort or similar, we need to copy the buffer dimensions into a temporary, sort, and copy them back.
  // - This template will be instantiated very frequently, it's worth attempting to minimize code size.
  for (std::size_t i = 0; i < buf.rank; ++i) {
    for (std::size_t j = i + 1; j < buf.rank; ++j) {
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
void fuse_contiguous_dims(span<const int> dim_sets, raw_buffer& buf, Bufs&... bufs) {
  assert(internal::same_rank(buf, bufs...));
  for (index_t d = static_cast<index_t>(buf.rank) - 1; d > 0; --d) {
    internal::attempt_fuse(d - 1, d, dim_sets, buf, bufs...);
  }
}
template <typename... Bufs>
void fuse_contiguous_dims(raw_buffer& buf, Bufs&... bufs) {
  assert(internal::same_rank(buf, bufs...));
  for (index_t d = static_cast<index_t>(buf.rank) - 1; d > 0; --d) {
    internal::attempt_fuse(d - 1, d, buf, bufs...);
  }
}

// Call both `sort_dims` and `fuse_contiguous_dims` on the buffers.
template <typename... Bufs>
void optimize_dims(span<const int> dim_sets, raw_buffer& buf, Bufs&... bufs) {
  // The order of operations here is for performance: It's a lot faster to fuse dimensions than sort them. So we fuse
  // what we can before sorting, then if the sorting changed the order of the dimensions, attempt to fuse again.
  fuse_contiguous_dims(dim_sets, buf, bufs...);
  if (sort_dims(dim_sets, buf, bufs...)) {
    fuse_contiguous_dims(dim_sets, buf, bufs...);
  }
}
template <typename... Bufs>
void optimize_dims(raw_buffer& buf, Bufs&... bufs) {
  fuse_contiguous_dims(buf, bufs...);
  if (sort_dims(buf, bufs...)) {
    fuse_contiguous_dims(buf, bufs...);
  }
}

namespace internal {

// The following few helpers implement traversing a mult-dimensional loop nest and then calling a function.
// We often will have two dimensions that traverse memory as if it was one loop, and it is valuable to do so
// to reduce overhead/improve performance.
// To implement this, we first examine the buffers and generate a "plan". The plan is a sequence of these objects
// laid out in memory contiguous, like so:
//
// struct loop_desc {
//   for_each_loop loop;
//   union {
//     dim* dim;
//     index_t stride;
//   } dim_or_stride[buffer_count];
// };
// loop_desc loops[rank];
//
// We don't actually have this struct, because buffer_count needs to be a runtime variable, but we can emulate
// this memory layout with pointer arithmetic.
struct for_each_loop {
  enum {
    // Loop types are combinations of
    call_f = 0x1,
    folded = 0x2,  // Uses dim, extent
    linear = 0,    // Uses stride, extent
  };
  int impl;
  index_t extent;
};

inline std::size_t size_of_plan(std::size_t rank, std::size_t bufs) {
  return std::max<std::size_t>(rank, 1) * (sizeof(for_each_loop) + bufs * sizeof(dim*));
}

index_t make_for_each_contiguous_slice_loops(span<const raw_buffer*> bufs, void** bases, void* plan);
void make_for_each_loops(span<const raw_buffer*> bufs, void** bases, void* plan);

template <typename T>
SLINKY_ALWAYS_INLINE inline const T* read_plan(const void*& x, std::size_t n = 1) {
  const T* result = reinterpret_cast<const T*>(x);
  x = offset_bytes_non_null(x, sizeof(T) * n);
  return result;
}

template <typename F, std::size_t NumBufs>
void for_each_impl(const std::array<void*, NumBufs>& bases, const void* plan, const F& f);

template <bool CallF, typename F, std::size_t NumBufs>
void for_each_impl_linear(const std::array<void*, NumBufs>& bases, index_t extent, const void* plan, const F& f) {
  const index_t* strides = read_plan<index_t>(plan, NumBufs);
  std::array<void*, NumBufs> bases_i = bases;
  // If the next step is to call f, do that eagerly here to avoid an extra call.
  assert(extent >= 1);
  for (index_t i = extent;;) {
    if (CallF) {
      f(bases_i);
    } else {
      for_each_impl(bases_i, plan, f);
    }
    if (--i <= 0) break;
    bases_i[0] = offset_bytes_non_null(bases_i[0], strides[0]);
    // This is a critical loop, and it seems we can't trust the compiler to unroll it. These ifs are constexpr.
    if (1 < NumBufs) bases_i[1] = offset_bytes(bases_i[1], strides[1]);
    if (2 < NumBufs) bases_i[2] = offset_bytes(bases_i[2], strides[2]);
    for (std::size_t n = 3; n < NumBufs; n++) {
      bases_i[n] = offset_bytes(bases_i[n], strides[n]);
    }
  }
}

template <bool CallF, typename F, std::size_t NumBufs>
void for_each_impl_folded(const std::array<void*, NumBufs>& bases, index_t extent, const void* plan, const F& f) {
  if (extent <= 0) return;
  dim* const* dims = read_plan<dim*>(plan, NumBufs);
  index_t begin = dims[0]->begin();
  index_t end = begin + extent;
  std::array<void*, NumBufs> bases_i;
  // If the next step is to call f, do that eagerly here to avoid an extra call.
  for (index_t i = begin; i < end; ++i) {
    bases_i[0] = offset_bytes_non_null(bases[0], dims[0]->flat_offset_bytes(i));
    for (std::size_t n = 1; n < NumBufs; n++) {
      bases_i[n] = dims[n]->contains(i) ? offset_bytes(bases[n], dims[n]->flat_offset_bytes(i)) : nullptr;
    }
    if (CallF) {
      f(bases_i);
    } else {
      for_each_impl(bases_i, plan, f);
    }
  }
}

template <typename F, std::size_t NumBufs>
SLINKY_ALWAYS_INLINE inline void for_each_impl(const std::array<void*, NumBufs>& bases, const void* plan, const F& f) {
  const for_each_loop* loop = read_plan<for_each_loop>(plan);
  if (loop->impl == (for_each_loop::linear | for_each_loop::call_f)) {
    for_each_impl_linear<true>(bases, loop->extent, plan, f);
  } else if (loop->impl == for_each_loop::linear) {
    for_each_impl_linear<false>(bases, loop->extent, plan, f);
  } else if (loop->impl == (for_each_loop::folded | for_each_loop::call_f)) {
    for_each_impl_folded<true>(bases, loop->extent, plan, f);
  } else {
    for_each_impl_folded<false>(bases, loop->extent, plan, f);
  }
}

template <typename... Ts, typename T, std::size_t... Is>
auto tuple_cast(const T& x, std::index_sequence<Is...>) {
  return std::make_tuple(static_cast<Ts>(std::get<Is>(x))...);
}

}  // namespace internal

// Call `f(index_t extent, T* base[, Ts* bases, ...])` for each contiguous slice in the domain of `buf[,
// bufs...]`. This function attempts to be efficient to support production quality implementations of callbacks.
//
// When additional buffers are passed, they will be sliced in tandem with the 'main' buffer. Additional buffers can be
// lower rank than the main buffer, these "missing" dimensions are not sliced (i.e. broadcasting in this dimension).
// If the other buffers are out of bounds for a slice, the corresponding argument to the callback will be `nullptr`.
template <typename Buf, typename F, typename... Bufs>
SLINKY_NO_STACK_PROTECTOR void for_each_contiguous_slice(const Buf& buf, const F& f, const Bufs&... bufs) {
  constexpr std::size_t BufsSize = sizeof...(Bufs) + 1;
  std::array<const raw_buffer*, BufsSize> buf_ptrs = {&buf, &bufs...};

  // We might need a slice dim for each dimension in the buffer, plus one for the call to f.
  auto* plan = SLINKY_ALLOCA(char, internal::size_of_plan(buf.rank, BufsSize));
  std::array<void*, BufsSize> bases;
  index_t slice_extent = internal::make_for_each_contiguous_slice_loops(buf_ptrs, bases.data(), plan);

  internal::for_each_impl(bases, plan, [&f, slice_extent](const std::array<void*, BufsSize>& bases) {
    std::apply(f, std::tuple_cat(std::make_tuple(slice_extent),
                      internal::tuple_cast<typename Buf::pointer, typename Bufs::pointer...>(
                          bases, std::make_index_sequence<BufsSize>())));
  });
}

// Call `f` with a pointer to each element of `buf`, and pointers to the same corresponding elements of `bufs`, or
// `nullptr` if `buf` is out of bounds of `bufs`.
template <typename F, typename Buf, typename... Bufs>
SLINKY_NO_STACK_PROTECTOR void for_each_element(const F& f, const Buf& buf, const Bufs&... bufs) {
  constexpr std::size_t BufsSize = sizeof...(Bufs) + 1;
  std::array<const raw_buffer*, BufsSize> buf_ptrs = {&buf, &bufs...};

  // We might need a slice dim for each dimension in the buffer, plus one for the call to f.
  auto* plan = SLINKY_ALLOCA(char, internal::size_of_plan(buf.rank, BufsSize));
  std::array<void*, BufsSize> bases;
  internal::make_for_each_loops(buf_ptrs, bases.data(), plan);

  internal::for_each_impl(bases, plan, [&](const std::array<void*, BufsSize>& bases) {
    std::apply(f, internal::tuple_cast<typename Buf::pointer, typename Bufs::pointer...>(
                      bases, std::make_index_sequence<BufsSize>()));
  });
}

}  // namespace slinky

#endif  // SLINKY_RUNTIME_BUFFER_H
