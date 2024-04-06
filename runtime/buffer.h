#ifndef SLINKY_RUNTIME_BUFFER_H
#define SLINKY_RUNTIME_BUFFER_H

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "runtime/util.h"

namespace slinky {

using index_t = std::ptrdiff_t;

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
  static constexpr index_t unfolded = std::numeric_limits<index_t>::max();

  dim() : min_(0), extent_(0), stride_(0), fold_factor_(unfolded) {}

  index_t min() const { return min_; }
  index_t max() const { return min_ + extent_ - 1; }
  index_t begin() const { return min_; }
  index_t end() const { return min_ + extent_; }
  index_t extent() const { return extent_; }
  index_t stride() const { return stride_; }
  index_t fold_factor() const { return fold_factor_; }

  void set_extent(index_t extent) { extent_ = extent; }
  void set_point(index_t x) {
    min_ = x;
    extent_ = 1;
  }
  void set_bounds(index_t min, index_t max) {
    min_ = min;
    extent_ = max - min + 1;
  }
  void set_range(index_t begin, index_t end) {
    min_ = begin;
    extent_ = end - begin;
  }
  void set_min_extent(index_t min, index_t extent) {
    min_ = min;
    extent_ = extent;
  }
  void set_stride(index_t stride) { stride_ = stride; }
  void set_fold_factor(index_t fold_factor) { fold_factor_ = fold_factor; }

  void translate(index_t offset) { min_ += offset; }

  bool contains(index_t x) const { return min() <= x && x <= max(); }

  std::ptrdiff_t flat_offset_bytes(index_t i) const {
    // Conceptually, accesses may be out of bounds, but in practice, if the stride is 0, the accesses will not read
    // invalid memory. It's a bit messy to allow this, but this assert feels really overzealous when attempting to
    // implement broadcasting in callbacks.
    assert(i >= min_ || stride_ == 0);
    assert(i <= max() || stride_ == 0);
    if (fold_factor_ == unfolded) {
      return (i - min_) * stride_;
    } else {
      return euclidean_mod(i - min_, fold_factor_) * stride_;
    }
  }
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
    assert(sizeof...(indices) + 1 == rank);
    return flat_offset_bytes_impl(dims, i0, indices...);
  }
  template <typename... Indices>
  void* address_at(index_t i0, Indices... indices) const {
    assert(sizeof...(indices) + 1 == rank);
    return offset_bytes(base, flat_offset_bytes(i0, indices...));
  }
  std::ptrdiff_t flat_offset_bytes() const { return 0; }
  void* address_at() const { return base; }

  template <typename... Indices>
  bool contains(index_t i0, Indices... indices) const {
    assert(sizeof...(indices) + 1 == rank);
    return contains_impl(dims, i0, indices...);
  }
  bool contains() const { return true; }

  std::ptrdiff_t flat_offset_bytes(span<const index_t> indices) const {
    assert(indices.size() == rank);
    index_t offset = 0;
    for (std::size_t i = 0; i < indices.size(); ++i) {
      offset += dims[i].flat_offset_bytes(indices[i]);
    }
    return offset;
  }
  void* address_at(span<const index_t> indices) const { return offset_bytes(base, flat_offset_bytes(indices)); }
  bool contains(span<const index_t> indices) const {
    assert(indices.size() == rank);
    bool result = true;
    for (std::size_t i = 0; i < indices.size(); ++i) {
      result = result && dims[i].contains(indices[i]);
    }
    return result;
  }

  template <typename... Offsets>
  void translate(index_t o0, Offsets... offsets) {
    assert(sizeof...(offsets) + 1 <= rank);
    translate_impl(dims, o0, offsets...);
  }
  void translate(span<const index_t> offsets) {
    assert(offsets.size() <= rank);
    for (std::size_t i = 0; i < offsets.size(); ++i) {
      dims[i].translate(offsets[i]);
    }
  }

  std::size_t size_bytes() const;

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
    } else {
      dims = nullptr;
    }
    if (src) {
      memcpy(dims, src, rank * sizeof(slinky::dim));
    }
  }

public:
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
  buffer(span<const index_t> extents) : buffer() {
    assert(extents.size() <= rank);
    rank = extents.size();
    index_t stride = elem_size;
    slinky::dim* d = dims;
    for (index_t extent : extents) {
      d->set_min_extent(0, extent);
      d->set_stride(stride);
      stride *= extent;
      ++d;
    }
  }
  buffer(std::initializer_list<index_t> extents) : buffer({extents.begin(), extents.end()}) {}
  ~buffer() { free(); }

  // This is a shallow copy.
  buffer(const buffer& c) : buffer() {
    raw_buffer::base = c.base();
    elem_size = c.elem_size;
    assign_dims(c.rank, c.dims);
  }
  void operator=(const buffer& c) {
    if (this == &c) return;
    free();
    raw_buffer::base = c.base();
    elem_size = c.elem_size;
    assign_dims(c.rank, c.dims);
  }

  buffer(buffer&& m) { *this = std::move(m); }
  buffer& operator=(buffer&& m) {
    if (this == &m) return *this;
    free();
    memcpy(static_cast<raw_buffer*>(this), static_cast<const raw_buffer*>(&m), sizeof(raw_buffer));
    if (DimsSize > 0) {
      memcpy(dims_storage, m.dims_storage, DimsSize * sizeof(slinky::dim));
      dims = dims_storage;
    }
    // Take ownership of the data.
    to_free = m.to_free;
    m.to_free = nullptr;
    return *this;
  }

  T* base() const { return reinterpret_cast<T*>(raw_buffer::base); }

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

  auto& at() const { return *base(); }
  auto& operator()() const { return *base(); }

  auto& at(span<const index_t> indices) const { return *offset_bytes(base(), flat_offset_bytes(indices)); }
  auto& operator()(span<const index_t> indices) const { return at(indices); }

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

// Copy the contents of `src` to `dst`. When the `src` is out of bounds of `dst`, fill with `padding`.
// `padding` should point to `dst.elem_size` bytes, or if `padding` is null, out of bounds regions
// are unmodified.
void copy(const raw_buffer& src, const raw_buffer& dst, const void* padding = nullptr);

// Performs only the padding operation of a copy. The region that would have been copied is unmodified.
void pad(const dim* in_bounds, const raw_buffer& dst, const void* padding);

// Fill `dst` with `value`. `value` should point to `dst.elem_size` bytes.
void fill(const raw_buffer& dst, const void* value);

namespace internal {

// Returns true if all buffers have the same rank.
inline bool same_rank(const raw_buffer&) { return true; }
inline bool same_rank(const raw_buffer& buf0, const raw_buffer& buf1) { return buf0.rank == buf1.rank; }
template <typename... Bufs>
bool same_rank(const raw_buffer& buf0, const raw_buffer& buf1, const Bufs&... bufs) {
  return buf0.rank == buf1.rank && same_rank(buf1, bufs...);
}

inline bool same_bounds(const dim& a, const dim& b) { return a.min() == b.min() && a.extent() == b.extent(); }

// Returns true if all buffers have the same bounds in dimension d.
inline bool same_bounds(int, const raw_buffer&) { return true; }
inline bool same_bounds(int d, const raw_buffer& buf0, const raw_buffer& buf1) {
  return same_bounds(buf0.dim(d), buf1.dim(d));
}
template <typename... Bufs>
bool same_bounds(int d, const raw_buffer& buf0, const raw_buffer& buf1, const Bufs&... bufs) {
  return same_bounds(buf0.dim(d), buf1.dim(d)) && same_bounds(d, buf1, bufs...);
}

// Returns true if the two dimensions can be fused.
inline bool can_fuse(const dim& inner, const dim& outer) {
  if (outer.fold_factor() != dim::unfolded || inner.fold_factor() != dim::unfolded) return false;
  if (inner.stride() * inner.extent() != outer.stride()) return false;
  return true;
}

// Returns true if two dimensions of all buffers can be fused.
inline bool can_fuse(int inner, int outer, const raw_buffer& buf) { return can_fuse(buf.dim(inner), buf.dim(outer)); }
template <typename... Bufs>
bool can_fuse(int inner, int outer, const raw_buffer& buf, const Bufs&... bufs) {
  return can_fuse(inner, outer, buf) && can_fuse(inner, outer, bufs...);
}

// Fuse two dimensions of all buffers.
inline void fuse(int inner, int outer, raw_buffer& buf) {
  dim& i = buf.dim(inner);
  const dim& o = buf.dim(outer);
  i.set_min_extent(o.min() * i.extent(), o.extent() * i.extent());
  // Delete the outer dimension.
  for (std::size_t i = outer; i + 1 < buf.rank; ++i) {
    buf.dim(i) = buf.dim(i + 1);
  }
  buf.rank -= 1;
}
template <typename... Bufs>
void fuse(int inner, int outer, raw_buffer& buf, Bufs&... bufs) {
  fuse(inner, outer, buf);
  fuse(inner, outer, bufs...);
}

template <typename... Bufs>
SLINKY_ALWAYS_INLINE inline void attempt_fuse(int inner, int outer, raw_buffer& buf, Bufs&... bufs) {
  if (same_bounds(inner, buf, bufs...) && can_fuse(inner, outer, buf, bufs...)) {
    fuse(inner, outer, buf, bufs...);
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

}  // namespace internal

// This helper fuses the dimensions of a set of buffers where the dimensions are contiguous in memory. It only fuses a
// dimension if the same dimension can be fused in all buffers. After this function runs, the buffers may have lower
// rank, but still address the same memory as the original buffers. For "shift invariant" operations, operating on such
// fused buffers is likely to be more efficient.
// `dim_sets` is an optional span of integers that indicates sets of dimensions that are eligible for fusion. By
// default, all dimensions are considered to be part of the same set.
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

namespace internal {

template <typename F>
void for_each_index(span<const dim> dims, int d, index_t* is, const F& f) {
  if (d == 0) {
    for (index_t i = dims[0].begin(); i < dims[0].end(); ++i) {
      is[0] = i;
      f(span<const index_t>(is, is + dims.size()));
    }
  } else {
    for (index_t i = dims[d].begin(); i < dims[d].end(); ++i) {
      is[d] = i;
      for_each_index(dims, d - 1, is, f);
    }
  }
}

union dim_or_stride {
  // For loop_folded to call flat_offset_bytes
  const slinky::dim* dim;
  // For loop_linear to offset the base.
  index_t stride;
};

struct for_each_slice_dim {
  enum {
    call_f,       // Uses extent
    loop_linear,  // Uses stride, extent
    loop_folded,  // Uses dim, extent
  } impl;
  index_t extent;
};

index_t make_for_each_contiguous_slice_dims(
    span<const raw_buffer*> bufs, void** bases, for_each_slice_dim* slice_dims, dim_or_stride* dims);

bool make_for_each_slice_dims(
    span<const raw_buffer*> bufs, void** bases, for_each_slice_dim* slice_dims, dim_or_stride* dims);

template <typename F, std::size_t NumBufs>
void for_each_slice_impl(
    std::array<void*, NumBufs> bases, const for_each_slice_dim* slice_dim, const dim_or_stride* dims, const F& f) {
  if (slice_dim->impl == for_each_slice_dim::call_f) {
    f(bases);
  } else if (slice_dim->impl == for_each_slice_dim::loop_linear) {
    const auto* next = slice_dim + 1;
    if (next->impl == for_each_slice_dim::call_f) {
      // If the next step is to call f, do that eagerly here to avoid an extra call.
      for (index_t i = 0; i < slice_dim->extent; ++i) {
        f(bases);
        for (std::size_t n = 0; n < NumBufs; n++) {
          bases[n] = offset_bytes(bases[n], dims[n].stride);
        }
      }
    } else {
      for (index_t i = 0; i < slice_dim->extent; ++i) {
        for_each_slice_impl(bases, slice_dim + 1, dims + NumBufs, f);
        for (std::size_t n = 0; n < NumBufs; n++) {
          bases[n] = offset_bytes(bases[n], dims[n].stride);
        }
      }
    }
  } else {
    assert(slice_dim->impl == for_each_slice_dim::loop_folded);

    std::array<void*, NumBufs> offset_bases;

    // TODO: If any buffer if folded in a given dimension, we just take the slow path
    // that handles either folded or unfolded for *all* the buffers in that dimension.
    // It's possible we could special-case and improve the situation somewhat if we
    // see common cases (eg main buffer never folded and one 'other' buffer that is folded).
    index_t begin = dims[0].dim->begin();
    index_t end = begin + slice_dim->extent;
    for (index_t i = begin; i < end; ++i) {
      for (std::size_t n = 0; n < NumBufs; n++) {
        offset_bases[n] = offset_bytes(bases[n], dims[n].dim->flat_offset_bytes(i));
      }
      for_each_slice_impl(offset_bases, slice_dim + 1, dims + NumBufs, f);
    }
  }
}

// Implements the cropping part of a loop over tiles.
template <typename F>
void for_each_tile(const index_t* tile, raw_buffer& buf, int d, const F& f) {
  if (d == -1) {
    f(buf);
    return;
  }

  slinky::dim& dim = buf.dim(d);
  index_t step = tile[d];
  if (dim.extent() <= step) {
    // Don't need to tile this dimension.
    for_each_tile(tile, buf, d - 1, f);
  } else {
    // TODO: Supporting folding here should be possible.
    assert(dim.fold_factor() == dim::unfolded);
    index_t stride = dim.stride() * step;

    // Save the old base and bounds.
    void* old_base = buf.base;
    index_t old_min = dim.min();
    index_t old_max = dim.max();

    // Handle the first tile.
    dim.set_bounds(old_min, old_min + step - 1);
    for_each_tile(tile, buf, d - 1, f);
    for (index_t i = old_min + step; i <= old_max; i += step) {
      buf.base = offset_bytes(buf.base, stride);
      dim.set_bounds(i, std::min(old_max, i + step - 1));
      for_each_tile(tile, buf, d - 1, f);
    }

    // Restore the old base and bounds.
    buf.base = old_base;
    dim.set_bounds(old_min, old_max);
  }
}

}  // namespace internal

// Call `f(span<index_t>)` for each index in the domain of `dims`, or the dims of `buf`.
// This function is not fast, use it for non-performance critical code. It is useful for
// making rank-agnostic algorithms without a recursive wrapper, which is otherwise difficult.
template <typename F>
SLINKY_NO_STACK_PROTECTOR void for_each_index(span<const dim> dims, const F& f) {
  // Not using alloca for performance, but to avoid including <vector>
  index_t* i = SLINKY_ALLOCA(index_t, dims.size());
  internal::for_each_index(dims, dims.size() - 1, i, f);
}
template <typename F>
void for_each_index(const raw_buffer& buf, const F& f) {
  for_each_index(span<const dim>{buf.dims, buf.rank}, f);
}

// Call `f(index_t extent, void* base[, void* bases, ...])` for each contiguous slice in the domain of `buf[,
// bufs...]`. This function attempts to be efficient to support production quality implementations of callbacks.
//
// When additional buffers are passed, they will be sliced in tandem with the 'main' buffer. Additional buffers can be
// lower rank than the main buffer, these "missing" dimensions are not sliced (i.e. broadcasting in this dimension).
template <typename F, typename... Bufs>
SLINKY_NO_STACK_PROTECTOR void for_each_contiguous_slice(const raw_buffer& buf, const F& f, const Bufs&... bufs) {
  constexpr std::size_t BufsSize = sizeof...(Bufs) + 1;
  std::array<const raw_buffer*, BufsSize> buf_ptrs = {&buf, &bufs...};

  // We might need a slice dim for each dimension in the buffer, plus one for the call to f.
  auto* slice_dims = SLINKY_ALLOCA(internal::for_each_slice_dim, buf.rank + 1);
  auto* dims = SLINKY_ALLOCA(internal::dim_or_stride, buf.rank * BufsSize);
  std::array<void*, BufsSize> bases;
  index_t slice_extent = internal::make_for_each_contiguous_slice_dims(buf_ptrs, bases.data(), slice_dims, dims);
  if (slice_extent < 0) {
    return;
  }

  internal::for_each_slice_impl(bases, slice_dims, dims, [&f, slice_extent](const std::array<void*, BufsSize>& bases) {
    std::apply(f, std::tuple_cat(std::make_tuple(slice_extent), bases));
  });
}

// Call `f` for each slice of the first `slice_rank` dimensions of `buf`. The trailing dimensions of `bufs` will also be
// sliced at the same indices as `buf`. Assumes that all of the sliced dimensions of `buf` are in bounds in `bufs...`.
template <typename F, typename... Bufs>
void for_each_slice(std::size_t slice_rank, const raw_buffer& buf, const F& f, const Bufs&... bufs) {
  constexpr std::size_t BufsSize = sizeof...(Bufs) + 1;
  std::array<const raw_buffer*, BufsSize> buf_ptrs;
  // Remove the sliced dimensions from the bufs.
  std::array<raw_buffer, BufsSize> sliced_bufs = {buf, bufs...};
  for (std::size_t i = 0; i < BufsSize; ++i) {
    std::size_t slice_rank_i = slice_rank + std::max(sliced_bufs[i].rank, buf.rank) - buf.rank;
    if (sliced_bufs[i].rank > slice_rank_i) {
      sliced_bufs[i].rank -= slice_rank_i;
      sliced_bufs[i].dims += slice_rank_i;
    } else {
      sliced_bufs[i].rank = 0;
    }
    buf_ptrs[i] = &sliced_bufs[i];
  }

  // We might need a slice dim for each dimension in the buffer, plus one for the call to f.
  auto* slice_dims = SLINKY_ALLOCA(internal::for_each_slice_dim, (buf.rank - slice_rank) + 1);
  auto* dims = SLINKY_ALLOCA(internal::dim_or_stride, (buf.rank - slice_rank) * BufsSize);
  std::array<void*, BufsSize> bases;
  index_t slice_extent = internal::make_for_each_slice_dims(buf_ptrs, bases.data(), slice_dims, dims);
  if (slice_extent < 0) {
    return;
  }

  // TODO: We only need to copy dims and rank here. `elem_size` should already be set, and `base` is set below.
  // I'm not sure if fixing this would be much of an improvement.
  sliced_bufs = {buf, bufs...};
  for (std::size_t i = 0; i < BufsSize; ++i) {
    sliced_bufs[i].rank =
        std::min(sliced_bufs[i].rank, slice_rank + std::max(sliced_bufs[i].rank, buf.rank) - buf.rank);
  }

  internal::for_each_slice_impl(bases, slice_dims, dims, [&](const std::array<void*, BufsSize>& bases) {
    for (std::size_t i = 0; i < BufsSize; ++i) {
      sliced_bufs[i].base = bases[i];
    }
    std::apply(f, sliced_bufs);
  });
}

// Call `f(buf)` for each tile of size `tile` in the domain of `buf`. `tile` is a span of sizes of the tile in each
// dimension.
template <typename F>
SLINKY_NO_STACK_PROTECTOR void for_each_tile(span<const index_t> tile, const raw_buffer& buf, const F& f) {
  assert(buf.rank == tile.size());

  // Copy the buffer so we can mutate it.
  // TODO: We restore the buffer to its original state, so if we can guarantee that this thread has its own copy, it
  // should be OK to just const_cast it.
  raw_buffer buf_;
  buf_.base = buf.base;
  buf_.elem_size = buf.elem_size;
  buf_.rank = buf.rank;
  buf_.dims = SLINKY_ALLOCA(dim, buf.rank);
  memcpy(buf_.dims, buf.dims, sizeof(dim) * buf.rank);

  internal::for_each_tile(tile.data(), buf_, buf_.rank - 1, f);
}

// Value for use in tile tuples indicating the dimension should be passed unmodified.
static constexpr index_t all = std::numeric_limits<index_t>::max();

}  // namespace slinky

#endif  // SLINKY_RUNTIME_BUFFER_H
