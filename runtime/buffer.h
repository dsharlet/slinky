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
    assert(i >= min_);
    assert(i <= max());
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

struct free_deleter {
  void operator()(void* p) { free(p); }
};

using raw_buffer_ptr = std::unique_ptr<raw_buffer, free_deleter>;

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

  // Make a pointer to a buffer with an allocation for the buffer in the same allocation.
  static raw_buffer_ptr make_allocated(std::size_t elem_size, std::size_t rank, const class dim* dims);

  // Make a deep copy of another buffer, including allocating and copying the data.
  static raw_buffer_ptr make_copy(const raw_buffer& src);
};

template <typename T, std::size_t DimsSize>
class buffer : public raw_buffer {
private:
  void* to_free;
  slinky::dim dims_storage[DimsSize];

public:
  using raw_buffer::cast;
  using raw_buffer::dim;
  using raw_buffer::elem_size;
  using raw_buffer::flat_offset_bytes;
  using raw_buffer::rank;

  buffer() {
    raw_buffer::base = nullptr;
    to_free = nullptr;
    rank = DimsSize;
    elem_size = sizeof(T);
    if (DimsSize > 0) {
      dims = &dims_storage[0];
    } else {
      dims = nullptr;
    }
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

  buffer(const buffer&) = delete;
  buffer(buffer&& m) = delete;
  void operator=(const buffer&) = delete;
  void operator=(buffer&& m) = delete;

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

// Value for use in tile tuples indicating the dimension should be passed unmodified.
static constexpr index_t all = std::numeric_limits<index_t>::max();

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

inline void shallow_copy_bufs(raw_buffer* dst, const raw_buffer& src) { *dst = src; }

template <typename... Bufs>
void shallow_copy_bufs(raw_buffer* dst, const raw_buffer& src, const Bufs&... bufs) {
  *dst = src;
  shallow_copy_bufs(dst + 1, bufs...);
}

template <size_t N, typename F>
void for_each_slice(std::size_t slice_rank, std::array<raw_buffer, N>& bufs, const F& f) {
  if (bufs[0].rank <= slice_rank) {
    // We're done slicing.
    std::apply(f, bufs);
    return;
  }

  const slinky::dim& dim = bufs[0].dim(bufs[0].rank - 1);

  index_t min = dim.min();
  index_t max = dim.max();
  if (min > max) {
    // Dimension (and the buffer) is empty.
    return;
  }

  std::array<void*, N> old_bases;
  for (std::size_t n = 0; n < N; ++n) {
    old_bases[n] = bufs[n].base;
    bufs[n].rank -= 1;
  }

  index_t stride = dim.stride();
  assert(dim.min() / dim.fold_factor() == dim.max() / dim.fold_factor());
  for (index_t i = min; i <= max; ++i, bufs[0].base = offset_bytes(bufs[0].base, stride)) {
    for (std::size_t n = 1; n < N; ++n) {
      const slinky::dim& dim_n = bufs[n].dims[bufs[0].rank];
      bufs[n].base = offset_bytes(old_bases[n], dim_n.flat_offset_bytes(i));
    }
    for_each_slice(slice_rank, bufs, f);
  }

  // Restore the buffers' base and rank.
  for (std::size_t n = 0; n < N; ++n) {
    bufs[n].base = old_bases[n];
    bufs[n].rank += 1;
  }
}

struct for_each_contiguous_slice_dim {
  index_t stride;
  index_t extent;
  enum {
    call_f,  // Uses extent
    linear,  // Uses stride, extent
  } impl;
};

template <typename F>
void for_each_contiguous_slice(void* base, const for_each_contiguous_slice_dim* slice_dim, const F& f) {
  if (slice_dim->impl == for_each_contiguous_slice_dim::call_f) {
    f(base, slice_dim->extent);
  } else {
    const for_each_contiguous_slice_dim* next = slice_dim + 1;
    if (next->impl == for_each_contiguous_slice_dim::call_f) {
      // If the next step is to call f, do that eagerly here to avoid an extra call.
      // TODO: This could be implemented by changing `impl` to be flags: folded or not, and call_f or not.
      // Then we wouldn't need to peek at the next slice_dim to see if we should do this, and it would eliminate the
      // need for an extra for_each_contiguous_slice_dim instance on the stack.
      for (index_t i = 0; i < slice_dim->extent; ++i, base = offset_bytes(base, slice_dim->stride)) {
        f(base, next->extent);
      }
    } else {
      for (index_t i = 0; i < slice_dim->extent; ++i, base = offset_bytes(base, slice_dim->stride)) {
        for_each_contiguous_slice(base, slice_dim + 1, f);
      }
    }
  }
}

void make_for_each_contiguous_slice_dims(const raw_buffer& buf, for_each_contiguous_slice_dim* dims);

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
void for_each_index(span<const dim> dims, const F& f) {
  // Not using alloca for performance, but to avoid including <vector>
  index_t* i = SLINKY_ALLOCA(index_t, dims.size());
  internal::for_each_index(dims, dims.size() - 1, i, f);
}
template <typename F>
void for_each_index(const raw_buffer& buf, const F& f) {
  for_each_index(span<const dim>{buf.dims, buf.rank}, f);
}

// Call `f(void* base, index_t extent)` for each contiguous slice in the domain of `buf`.
// This function attempts to be efficient to support production quality implementations of callbacks.
template <typename F>
void for_each_contiguous_slice(const raw_buffer& buf, const F& f) {
  // We might need a slice dim for each dimension in the buffer, plus one for the call to f.
  internal::for_each_contiguous_slice_dim* dims = SLINKY_ALLOCA(internal::for_each_contiguous_slice_dim, buf.rank + 1);
  internal::make_for_each_contiguous_slice_dims(buf, dims);

  internal::for_each_contiguous_slice(buf.base, dims, f);
}

// Call `f` for each slice of the first `slice_rank` dimensions of buf. `bufs` will also be sliced at the same indices
// as `buf`. Assumes that all of the sliced dimensions of `buf` are in bounds in `bufs...`.
template <typename F, typename... Bufs>
void for_each_slice(std::size_t slice_rank, const raw_buffer& buf, const F& f, const Bufs&... bufs) {
  std::array<raw_buffer, sizeof...(Bufs) + 1> bufs_;
  // Shallow copy is OK because we don't modify the dims.
  internal::shallow_copy_bufs(bufs_.data(), buf, bufs...);

  internal::for_each_slice(slice_rank, bufs_, f);
}

// Call `f(buf)` for each tile of size `tile` in the domain of `buf`. `tile` is a span of sizes of the tile in each
// dimension.
template <typename F>
void for_each_tile(span<const index_t> tile, const raw_buffer& buf, const F& f) {
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

}  // namespace slinky

#endif  // SLINKY_RUNTIME_BUFFER_H
