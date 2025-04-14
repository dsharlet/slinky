#include "runtime/buffer.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>

#include "base/util.h"

namespace slinky {

namespace {

dim broadcast_dim(std::numeric_limits<index_t>::min(), std::numeric_limits<index_t>::max(), 0);

}  // namespace

const dim& dim::broadcast() { return broadcast_dim; }

namespace {

index_t alloc_extent(const dim& dim) {
  if (dim.fold_factor() != dim::unfolded) {
    // TODO: We can do better than this if the dim doesn't cross a fold boundary.
    return dim.fold_factor();
  } else {
    return dim.extent();
  }
}

std::size_t alloc_size(std::size_t rank, std::size_t elem_size, const dim* dims) {
  index_t flat_min = 0;
  index_t flat_max = 0;
  for (std::size_t i = 0; i < rank; ++i) {
    if (dims[i].stride() == 0) continue;
    index_t extent = alloc_extent(dims[i]);
    assert(extent >= 0);
    if (extent == 0) return 0;
    flat_min += (extent - 1) * std::min<index_t>(0, dims[i].stride());
    flat_max += (extent - 1) * std::max<index_t>(0, dims[i].stride());
  }
  return flat_max - flat_min + elem_size;
}

}  // namespace

std::size_t raw_buffer::size_bytes() const { return alloc_size(rank, elem_size, dims); }

std::size_t raw_buffer::elem_count() const {
  std::size_t result = 1;
  for (std::size_t i = 0; i < rank; ++i) {
    result *= std::max<index_t>(0, dims[i].extent());
  }
  return result;
}

raw_buffer_ptr raw_buffer::make(std::size_t rank, std::size_t elem_size, const class dim* dims) {
  const std::size_t data_size = rank == 0 || dims ? alloc_size(rank, elem_size, dims) : 0;
  const std::size_t size = sizeof(raw_buffer) + sizeof(slinky::dim) * rank + data_size;
  char* mem = reinterpret_cast<char*>(malloc(size));
  raw_buffer* buf = new (mem) raw_buffer();
  mem += sizeof(raw_buffer);
  buf->elem_size = elem_size;
  buf->rank = rank;
  buf->dims = reinterpret_cast<slinky::dim*>(mem);
  if (rank > 0 && dims) {
    internal::copy_small_n(dims, rank, buf->dims);
    mem += sizeof(slinky::dim) * rank;
  } else {
    new (buf->dims) slinky::dim[buf->rank];
  }
  buf->base = data_size > 0 ? mem : nullptr;
  return raw_buffer_ptr(buf, free);
}

raw_buffer_ptr raw_buffer::make_copy(const raw_buffer& src) {
  auto buf = make(src.rank, src.elem_size, src.dims);
  copy(src, *buf);
  return buf;
}

raw_buffer_ptr raw_buffer::make_scalar(std::size_t elem_size, const void* value) {
  auto buf = make(0, elem_size);
  memcpy(buf->base, value, elem_size);
  return buf;
}

namespace {

// This algorithm is the same idea as [1], but here it does not need to attempt to support compile-time constant
// strides, which makes it a lot easier to implement.
//
// 1. https://github.com/dsharlet/array/blob/38f8ce332fc4e26af08325ad0654c8516a445e8c/include/array/array.h#L835-L907

// A proposed stride is "OK" w.r.t. `dim` if the proposed stride does not cause this dimension's memory to overlap with
// any other dimension's memory.
struct init_stride_dim {
  index_t stride;
  index_t dim_stride;  // stride * extent

  init_stride_dim(index_t stride, index_t extent) : stride(stride), dim_stride(stride * extent) {}

  bool operator<(const init_stride_dim& r) const { return dim_stride < r.dim_stride; }
};

SLINKY_ALWAYS_INLINE inline bool is_stride_ok(index_t stride, index_t extent, span<const init_stride_dim> dims) {
  const index_t dim_stride = stride * extent;
  for (const init_stride_dim& d : dims) {
    if (d.stride >= dim_stride) {
      // The dim is completely outside the proposed stride.
    } else if (d.dim_stride <= stride) {
      // The dim is completely inside the proposed stride.
    } else {
      return false;
    }
  }
  return true;
}

}  // namespace

std::size_t raw_buffer::init_strides(index_t alignment) {
  // We remember the strides of the dims we know about, in sorted order.
  init_stride_dim* dims = SLINKY_ALLOCA(init_stride_dim, rank);
  init_stride_dim* dims_end = dims;
  // Insert d into dims, sorted by dim_stride. Also track the flat max index of the buffer, to compute the size.
  std::size_t flat_max = 0;
  auto learn_dim = [&](const init_stride_dim& d) {
    init_stride_dim* at = std::lower_bound(dims, dims_end, d);
    internal::copy_small_n_backward(dims_end, dims_end - at, dims_end + 1);
    *at = d;
    flat_max += d.dim_stride - d.stride;
    ++dims_end;
  };

  std::size_t unknown_begin = rank;
  std::size_t unknown_end = 0;
  for (std::size_t i = 0; i < rank; ++i) {
    if (dim(i).stride() == 0) continue;

    index_t alloc_extent_i = alloc_extent(dim(i));
    if (alloc_extent_i <= 1) {
      // The buffer is empty or has extent 1, we don't care about the stride.
      if (dim(i).stride() == dim::auto_stride) dim(i).set_stride(elem_size);
    } else if (dim(i).stride() != dim::auto_stride) {
      learn_dim(init_stride_dim(std::abs(dim(i).stride()), alloc_extent_i));
    } else {
      // Track the range of dimensions we need to find the stride of.
      unknown_begin = std::min(unknown_begin, i);
      unknown_end = i + 1;
    }
  }

  for (std::size_t i = unknown_begin; i < unknown_end; ++i) {
    if (dim(i).stride() != dim::auto_stride) continue;

    const index_t alloc_extent_i = alloc_extent(dim(i));
    assert(alloc_extent_i > 1);

    span<const init_stride_dim> known_dims{dims, dims_end};
    if (is_stride_ok(elem_size, alloc_extent_i, known_dims)) {
      // This dimension can have stride elem_size, no other stride could be better.
      dim(i).set_stride(elem_size);
      learn_dim(init_stride_dim(elem_size, alloc_extent_i));
      continue;
    }

    // Loop through all the dimensions and see if a stride that is just outside any dimension is OK.
    for (const init_stride_dim& dim_j : known_dims) {
      const index_t candidate = (dim_j.dim_stride + (alignment - 1)) & ~(alignment - 1);
      if (&dim_j == &known_dims.back() || is_stride_ok(candidate, alloc_extent_i, known_dims)) {
        dim(i).set_stride(candidate);
        learn_dim(init_stride_dim(candidate, alloc_extent_i));
        // The dims are sorted, so no subsequent candidate will be better.
        break;
      }
    }
    assert(dim(i).stride() != dim::auto_stride);
  }

  return (flat_max + elem_size + (alignment - 1)) & ~(alignment - 1);
}

void* raw_buffer::allocate(index_t alignment) {
  std::size_t size = init_strides(alignment);
  void* allocation = malloc(size + alignment - 1);
  base = align_up(allocation, alignment);
  return allocation;
}

namespace {

// Returns true if `value` is `size` repeats of the same byte. This probably will only ever be used for fills of 0.
bool is_repeated_byte(const void* value, std::size_t size) {
  const char* bytes = reinterpret_cast<const char*>(value);
  for (std::size_t i = 1; i < size; ++i) {
    if (bytes[0] != bytes[i]) {
      // The value is not the same repeated byte.
      return false;
    }
  }
  return true;
}

void fill(void* dst, const void* value, index_t elem_size, index_t size) {
  if (elem_size == 1) {
    memset(dst, *reinterpret_cast<const uint8_t*>(value), size);
  } else {
    assert(elem_size > 0);
    while (size >= elem_size) {
      memcpy(dst, value, elem_size);
      dst = offset_bytes_non_null(dst, elem_size);
      size -= elem_size;
    }
    // The elem_size might not divide the size if optimize_fill_value replicated it.
    memcpy(dst, value, size);
  }
}

// When copying broadcasted buffers, it would be slow if we copied each element one at a time. We can avoid this by
// duplicating the value into a buffer and copying from that larger buffer instead.
using fill_value_buffer = std::array<uint8_t, 256>;
void optimize_fill_value(const void*& value, index_t& elem_size, index_t fill_size, fill_value_buffer& buffer) {
  if (is_repeated_byte(value, elem_size)) {
    // This value can be filled with memset.
    elem_size = 1;
    return;
  }
  assert(elem_size != 1);
  const index_t target_size = std::min<index_t>(buffer.size(), fill_size);
  if (elem_size * 2 <= target_size) {
    // Repeatedly duplicate the constant as long as it fits in the buffer.
    memcpy(buffer.data(), value, elem_size);
    while (elem_size * 2 <= target_size) {
      memcpy(&buffer[elem_size], buffer.data(), elem_size);
      elem_size *= 2;
    }
    value = buffer.data();
  }
}

const slinky::dim& slice_dim0(raw_buffer& buf) {
  if (buf.rank > 0) {
    buf.rank--;
    return *buf.dims++;
  } else {
    return dim::broadcast();
  }
}

void unslice_dim0(raw_buffer& buf, const slinky::dim& d) {
  if (&d + 1 == buf.dims) {
    buf.rank++;
    buf.dims--;
  } else {
    // The dimension didn't come from this buffer (rank 0 broadcast).
  }
}

// Perform an unpadded copy.
void copy_impl(raw_buffer& src, raw_buffer& dst) {
  assert(src.elem_size == dst.elem_size);
  assert(dst.base || dst.elem_count() == 0);
  index_t elem_size = dst.elem_size;

  if (dst.rank == 0) {
    memcpy(dst.base, src.base, elem_size);
  } else {
    const slinky::dim& dst_dim0 = dst.dim(0);
    const slinky::dim& src_dim0 = src.rank > 0 ? src.dim(0) : dim::broadcast();

    if (dst_dim0.empty()) {
      // Empty destination, nothing to do.
    } else if (dst_dim0.fold_factor() != dim::unfolded || src_dim0.fold_factor() != dim::unfolded ||
               dst_dim0.stride() != elem_size || (src_dim0.stride() != 0 && src_dim0.stride() != elem_size)) {
      // There is some complication to the innermost dimension's copy.
      for_each_contiguous_slice(
          dst, [elem_size](index_t extent, void* dst, const void* src) { memcpy(dst, src, extent * elem_size); }, src);
    } else {
      slice_dim0(dst);
      slice_dim0(src);

      const index_t dst_size = dst_dim0.extent() * elem_size;

      if (src_dim0.stride() == 0) {
        // The inner dimension is a fill call.
        fill_value_buffer buffer;
        const void* buffer_value;
        index_t buffer_elem_size;
        const void* cached_src = nullptr;
        for_each_element(
            [&](void* dst, const void* src) {
              if (cached_src != src) {
                cached_src = src;
                buffer_elem_size = elem_size;
                buffer_value = src;
                optimize_fill_value(buffer_value, buffer_elem_size, dst_size, buffer);
              }
              fill(dst, buffer_value, buffer_elem_size, dst_size);
            },
            dst, src);
      } else {
        // The inner dimension is a memcpy.
        assert(src_dim0.stride() == elem_size);
        assert(src_dim0.begin() <= dst_dim0.begin());
        assert(src_dim0.end() >= dst_dim0.end());

        slinky::index_t src_offset = src_dim0.flat_offset_bytes(dst_dim0.min());
        for_each_element(
            [=](void* dst, const void* src) { memcpy(dst, offset_bytes_non_null(src, src_offset), dst_size); }, dst,
            src);
      }
      unslice_dim0(dst, dst_dim0);
      unslice_dim0(src, src_dim0);
    }
  }
}

// The strategy used here for padding is to start with the last dimension (which should have the largest stride), and
// fill the padded area. Once this padded area is filled, it can be cropped from the buffer, which reduces the area
// the next dimension needs to pad. Proceeding in this order minimizes the area we need to fill with small stride, which
// is where this is slow (because we can only copy one or a few elements at a time.

// This function copies `pad` to `dst` where `dst` is out of bounds of `src` in dimension `d`, and then crops `dst` such
// that only the unpadded area remains in dimension `d`.
void pad_impl(raw_buffer& src, raw_buffer& dst, raw_buffer& pad) {
  for (int d = static_cast<int>(std::min(src.rank, dst.rank)) - 1; d >= 0; --d) {
    const slinky::dim& src_d = src.dim(d);
    // TODO: Try to implement this without saving and restoring the dst buffer between each crop.
    void* dst_base = dst.base;
    slinky::dim dst_d = dst.dim(d);
    if (dst_d.min() < src_d.min()) {
      // There's padding before the min in this dimension.
      dst.crop(d, dst_d.min(), src_d.min() - 1);
      copy_impl(pad, dst);
      dst.base = dst_base;
      dst.dim(d) = dst_d;
    }
    if (dst_d.max() > src_d.max()) {
      // There's padding after the max in this dimension.
      dst.crop(d, src_d.max() + 1, dst_d.max());
      copy_impl(pad, dst);
      dst.base = dst_base;
      dst.dim(d) = dst_d;
    }
    // Crop off the padded areas we've filled in this dimension.
    dst.crop(d, src_d.min(), src_d.max());
    // If the src was bigger, crop it, so the bounds should match.
    src.crop(d, dst_d.min(), dst_d.max());
  }
}

}  // namespace

SLINKY_NO_STACK_PROTECTOR void copy(const raw_buffer& src, const raw_buffer& dst, const raw_buffer* pad) {
  assert(dst.elem_size == src.elem_size);
  assert(dst.base || dst.elem_count() == 0);
  if (dst.rank == 0) {
    assert(src.base || (pad && pad->base));
    memcpy(dst.base, !src.base && pad ? pad->base : src.base, dst.elem_size);
    return;
  }

  // Make (shallow) copies of the buffers, so we can optimize the dimensions.
  raw_buffer dst_opt = dst;
  dst_opt.dims = SLINKY_ALLOCA(dim, dst.rank);
  internal::copy_small_n(dst.dims, dst.rank, dst_opt.dims);

  raw_buffer src_opt = src;
  src_opt.dims = SLINKY_ALLOCA(dim, src.rank);
  internal::copy_small_n(src.dims, src.rank, src_opt.dims);

  // If the src has rank 0, then the padding is irrelevant, nothing is out of bounds.
  if (src_opt.rank > 0 && pad && pad->base) {
    assert(dst_opt.elem_size == pad->elem_size);

    raw_buffer pad_opt = *pad;
    pad_opt.dims = SLINKY_ALLOCA(dim, pad->rank);
    internal::copy_small_n(pad->dims, pad->rank, pad_opt.dims);

    optimize_dims(dst_opt, src_opt, pad_opt);

    // Implement the padding in all but the first dimension.
    pad_impl(src_opt, dst_opt, pad_opt);
    if (src_opt.base == dst_opt.base) {
      // This is an in-place padded copy, we're done.
      return;
    }
  } else {
    optimize_dims(dst_opt, src_opt);
  }
  copy_impl(src_opt, dst_opt);
}

void pad(const dim* in_bounds, const raw_buffer& dst, const raw_buffer& pad) {
  assert(dst.elem_size == pad.elem_size);
  if (dst.rank == 0) {
    return;
  }

  // To implement pad, we'll make a buffer that looks like dst, but cropped to the bounds, and copy it with pad.
  raw_buffer dst_opt = dst;
  dst_opt.dims = SLINKY_ALLOCA(dim, dst.rank);
  internal::copy_small_n(dst.dims, dst.rank, dst_opt.dims);

  raw_buffer src = dst;
  src.dims = SLINKY_ALLOCA(dim, dst.rank);
  internal::copy_small_n(dst.dims, dst.rank, src.dims);
  for (std::size_t d = 0; d < dst.rank; ++d) {
    src.crop(d, in_bounds[d].min(), in_bounds[d].max());
  }

  raw_buffer pad_opt = pad;
  pad_opt.dims = SLINKY_ALLOCA(dim, pad.rank);
  internal::copy_small_n(pad.dims, pad.rank, pad_opt.dims);

  optimize_dims(dst_opt, src, pad_opt);

  // Implement the padding in all but the first dimension.
  pad_impl(src, dst_opt, pad_opt);
}

namespace internal {

namespace {

template <std::size_t BufsSize>
SLINKY_ALWAYS_INLINE inline bool is_contiguous_slice(span<const raw_buffer*, BufsSize> bufs, std::size_t d) {
  const raw_buffer& buf = *bufs[0];
  if (buf.dim(d).stride() != static_cast<index_t>(buf.elem_size)) {
    // This dimension is not contiguous.
    return false;
  }
  for (std::size_t n = 1; n < bufs.size(); n++) {
    const raw_buffer& buf_n = *bufs[n];
    if (&buf_n == &buf || !buf_n.base) {
      // This is the same buffer as the base, or the base pointer is nullptr.
      continue;
    } else if (d >= buf_n.rank) {
      // This dimension is broadcasted, it's not contiguous.
      return false;
    } else if (buf_n.dim(d).stride() != static_cast<index_t>(buf_n.elem_size)) {
      // This dimension is not contiguous.
      return false;
    }
  }
  return true;
}

template <std::size_t BufsSize>
SLINKY_ALWAYS_INLINE inline bool can_fuse(span<const raw_buffer*, BufsSize> bufs, std::size_t d) {
  assert(d > 0);
  const raw_buffer& buf = *bufs[0];
  const dim& base_inner = buf.dim(d - 1);
  const dim& base_outer = buf.dim(d);
  if (base_inner.fold_factor() != dim::unfolded) {
    // One of the dimensions is folded.
    return false;
  }
  const index_t inner_extent = base_inner.extent();
  if (base_inner.stride() * inner_extent != base_outer.stride()) {
    // The dimensions are not contiguous in memory.
    return false;
  }

  for (std::size_t n = 1; n < bufs.size(); n++) {
    const raw_buffer& buf_n = *bufs[n];
    if (&buf_n == &buf || !buf_n.base) {
      // This is the same buffer as the base, or the base pointer is nullptr.
      continue;
    }
    const std::size_t rank = buf_n.rank;
    if (d > rank) {
      // Both dimensions are broadcasts, they can be fused.
      continue;
    }

    const dim& inner = buf_n.dim(d - 1);
    if (inner.min() != base_inner.min() || inner.max() != base_inner.max()) {
      // The bounds of the inner dimension are not equal.
      return false;
    } else if (inner.fold_factor() != dim::unfolded) {
      // One of the dimensions is folded.
      return false;
    }

    const index_t outer_stride = d < rank ? buf_n.dim(d).stride() : 0;
    if (inner.stride() * inner_extent != outer_stride) {
      // The dimensions are not contiguous in memory.
      return false;
    }
  }
  return true;
}

// We need a non-linear loop if we can't compute a pointer for any dimension via multiplying the stride and adding it to
// the beginning of the dimension:
// - A dimension is folded
// - A dimension is partially out of bounds
template <std::size_t BufsSize>
SLINKY_ALWAYS_INLINE inline bool use_nonlinear_loop(span<const raw_buffer*, BufsSize> bufs, std::size_t d) {
  const raw_buffer& buf = *bufs[0];
  const dim& buf_dim = buf.dim(d);
  if (buf_dim.is_folded()) {
    // The main buffer is folded.
    return true;
  }
  for (std::size_t n = 1; n < bufs.size(); ++n) {
    const raw_buffer& buf_n = *bufs[n];
    if (&buf_n == &buf || !buf_n.base) {
      // This is the same buffer as the base, or the base pointer is nullptr.
      continue;
    } else if (d >= buf_n.rank) {
      // Broadcast dimension.
      continue;
    }
    const dim& buf_n_dim = buf_n.dim(d);
    if (buf_n_dim.is_folded(buf_dim)) {
      // There's a folded buffer, we need a folded loop.
      return true;
    } else if (!buf_n_dim.contains(buf_dim)) {
      // One of the extra buffers is out of bounds, use a folded loop.
      return true;
    }
  }
  return false;
}

template <std::size_t BufsSize = dynamic_extent>
struct for_each_loop;

// These functions may modify the first parameter in-place.
// TODO: The last parameter (slice_extent) is unused by `for_each_element_callback`. Find a way to eliminate the
// parameter in that case. Adding a template parameter to everything affected to control this is painful.
template <std::size_t BufsSize>
using for_each_loop_impl = void (*)(mutable_span<void*, BufsSize>, const for_each_loop<BufsSize>*, index_t);

template <std::size_t BufsSize>
struct for_each_loop {
  union {
    // For linear loops, the extent of the loop.
    index_t extent;
    // For nonlinear loops, the intersection of all the folds in this loop.
    index_t fold_factor;
  };
  for_each_loop_impl<BufsSize> impl;
  union {
    index_t strides[1];  // [bufs_size]
    const dim* dims[1];  // [bufs_size]
  };
};

std::ptrdiff_t sizeof_for_each_loop(std::size_t bufs_size) {
  return sizeof(for_each_loop<>) - sizeof(for_each_loop<>::dims) + sizeof(for_each_loop<>::dims) * bufs_size;
}

// We store a plan for a parallel for loop in a structure of the following layout, for N buffers and rank R loops:
// for_each_loop<N> loops[R];
// F f;
//
// We can't make a simple struct for this, because N and R are not necessarily compile-time constants.
template <typename F>
SLINKY_ALWAYS_INLINE inline std::size_t size_of_plan(std::size_t bufs_size, std::size_t rank) {
  return sizeof_for_each_loop(bufs_size) * std::max<std::size_t>(1, rank) + sizeof(F);
}

// Compile-time dispatch to either for_each_contiguous_slice_callback or for_each_element_callback.
SLINKY_ALWAYS_INLINE inline void call_f(
    for_each_element_callback f, void** bases, index_t extent, const index_t* strides, index_t slice_extent) {
  assert(slice_extent == 1);
  f(bases, extent, strides);
}
SLINKY_ALWAYS_INLINE inline void call_f(
    for_each_contiguous_slice_callback f, void** bases, index_t extent, const index_t* strides, index_t slice_extent) {
  f(slice_extent, bases, extent, strides);
}

template <typename F, std::size_t BufsSize>
void for_each_impl_call_f(
    mutable_span<void*, BufsSize> bases, const for_each_loop<BufsSize>* loop, index_t slice_extent) {
  const F& f = *reinterpret_cast<const F*>(offset_bytes_non_null(loop, sizeof_for_each_loop(bases.size())));

  call_f(f, bases.data(), loop->extent, loop->strides, slice_extent);
}

template <std::size_t BufsSize>
SLINKY_NO_STACK_PROTECTOR void call_impl_linear(index_t extent, mutable_span<void*, BufsSize> bases,
    const for_each_loop<BufsSize>* loop, const index_t* strides, index_t slice_extent) {
  assert(extent >= 1);

  for_each_loop_impl<BufsSize> impl = loop->impl;

  void** bases_i = SLINKY_ALLOCA(void*, bases.size());
  for (;;) {
    copy_small_n(bases.data(), bases.size(), bases_i);
    impl({bases_i, bases.size()}, loop, slice_extent);
    if (SLINKY_UNLIKELY(--extent <= 0)) break;
    increment_bases<BufsSize>(bases.size(), bases.data(), strides);
  }
}

template <typename F, std::size_t BufsSize>
SLINKY_NO_STACK_PROTECTOR void for_each_impl_linear(
    mutable_span<void*, BufsSize> bases, const for_each_loop<BufsSize>* loop, index_t slice_extent) {
  const index_t* strides = loop->strides;
  index_t extent = loop->extent;
  assert(extent >= 1);

  loop = offset_bytes_non_null(loop, sizeof_for_each_loop(bases.size()));

  call_impl_linear(extent, bases, loop, strides, slice_extent);
}

template <typename F, std::size_t BufsSize, bool CallF, bool Contiguous>
SLINKY_NO_STACK_PROTECTOR void for_each_impl_nonlinear(
    mutable_span<void*, BufsSize> bases, const for_each_loop<BufsSize>* loop, index_t slice_extent) {
  const dim* const* dims = loop->dims;
  const index_t fold_factor = loop->fold_factor;

  loop = offset_bytes_non_null(loop, sizeof_for_each_loop(bases.size()));

  const F& f = *reinterpret_cast<const F*>(loop);

  index_t* strides = SLINKY_ALLOCA(index_t, bases.size());
  for (std::size_t n = 0; n < bases.size(); ++n) {
    strides[n] = dims[n]->stride();
  }

  const dim& dim_0 = *dims[0];
  void** bases_i = SLINKY_ALLOCA(void*, bases.size());

  // To handle non-linear loops, we process an interval [min, max] in blocks of the `fold_factor`, within which we can
  // compute the base pointers linearly from the strides, if the buffers are fully in-bounds or out-of-bounds.
  // We need to handle buffers going in and out of bounds too, so we break the blocks into smaller chunks at those
  // boundaries.
  auto run_one_fold = [&](index_t min, index_t max) {
    for (index_t i = min; i <= max;) {
      index_t max_i = max;
      assert(!dim_0.is_folded(i, max_i));
      bases_i[0] = offset_bytes_non_null(bases[0], dim_0.flat_offset_bytes(i));
      for (std::size_t n = 1; n < bases.size(); n++) {
        if (!bases[n]) {
          bases_i[n] = nullptr;
          continue;
        }
        const dim& dim_n = *dims[n];
        assert(!dim_n.is_folded(i, max_i));
        if (dim_n.contains(i)) {
          // This interval starts out non-null, but could become null before the end of the interval.
          bases_i[n] = offset_bytes_non_null(bases[n], dim_n.flat_offset_bytes(i));
          max_i = std::min(max_i, dim_n.max());
        } else {
          bases_i[n] = nullptr;
          if (dim_n.min() > i) {
            // This interval starts out null, but could become non-null before the end of the interval.
            max_i = std::min(max_i, dim_n.min() - 1);
          }
        }
      }
      index_t extent_i = Contiguous ? 1 : max_i - i + 1;
      index_t slice_extent_i = (Contiguous ? (max_i - i + 1) : 1) * slice_extent;
      if (CallF) {
        // If the next step is to call f, do that eagerly here to avoid an extra call.
        call_f(f, bases_i, extent_i, strides, slice_extent_i);
      } else {
        call_impl_linear<BufsSize>(extent_i, {bases_i, bases.size()}, loop, strides, slice_extent_i);
      }
      i = max_i + 1;
    }
  };

  index_t min = dim_0.min();
  index_t max = dim_0.max();
  if (fold_factor == dim::unfolded) {
    // Not folded, we can treat the whole range as one chunk.
    run_one_fold(min, max);
  } else {
    index_t first_fold = align_up(min, fold_factor);
    if (min != first_fold) {
      // Handle the partial fold before the first fold boundary.
      run_one_fold(min, std::min(first_fold - 1, max));
    }

    for (index_t i = first_fold; i <= max; i += fold_factor) {
      // Process up to the end without crossing a fold boundary.
      run_one_fold(i, std::min(i + fold_factor - 1, max));
    }
  }
}

index_t gcd_fold_factor(index_t a, index_t b) {
  if (a == dim::unfolded) return b;
  if (b == dim::unfolded) return a;
  return gcd(a, b);
}

template <bool SkipContiguous, std::size_t BufsSize, typename F>
SLINKY_NO_STACK_PROTECTOR SLINKY_ALWAYS_INLINE inline void for_each_impl(span<const raw_buffer*, BufsSize> bufs, F f) {
  const raw_buffer& buf = *bufs[0];

  auto* loop = reinterpret_cast<for_each_loop<BufsSize>*>(SLINKY_ALLOCA(char, size_of_plan<F>(bufs.size(), buf.rank)));

  void** bases = SLINKY_ALLOCA(void*, bufs.size());
  bases[0] = buf.base;
  for (std::size_t n = 1; n < bufs.size(); ++n) {
    bases[n] = bufs[n]->base;
  }

  for_each_loop_impl<BufsSize> inner_impl;
  for_each_loop<BufsSize>* outer_loop = loop;

  index_t slice_extent = 1;
  index_t extent = 1;
  for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(buf.rank) - 1; d >= 0; --d) {
    const dim& buf_dim = buf.dim(d);

    if (buf_dim.min() == buf_dim.max()) {
      // extent 1, we don't need any of the logic here, skip to below.
    } else if (buf_dim.max() < buf_dim.min() || use_nonlinear_loop(bufs, d)) {
      // extent > 1 and there is a folded dimension in one of the buffers, or we need to crop one of the buffers, or the
      // loops are empty.
      if (SkipContiguous && is_contiguous_slice(bufs, d)) {
        loop->impl = for_each_impl_nonlinear<F, BufsSize, false, true>;
        inner_impl = for_each_impl_nonlinear<F, BufsSize, true, true>;
      } else {
        loop->impl = for_each_impl_nonlinear<F, BufsSize, false, false>;
        inner_impl = for_each_impl_nonlinear<F, BufsSize, true, false>;
      }
      extent = 1;

      const dim** dims = loop->dims;
      dims[0] = &buf.dim(d);
      loop->fold_factor = buf_dim.fold_factor();
      for (std::size_t n = 1; n < bufs.size(); n++) {
        const raw_buffer& buf_n = *bufs[n];
        if (d < static_cast<std::ptrdiff_t>(buf_n.rank)) {
          const dim& buf_n_dim = buf_n.dim(d);
          dims[n] = &buf_n_dim;
          loop->fold_factor = gcd_fold_factor(loop->fold_factor, buf_n_dim.fold_factor());
        } else {
          dims[n] = &broadcast_dim;
        }
      }
      loop = offset_bytes_non_null(loop, sizeof_for_each_loop(bufs.size()));
      continue;
    } else {
      // Use a linear, possibly fused loop below.
      extent *= buf_dim.max() - buf_dim.min() + 1;
    }

    // Align the bases for dimensions we will access via linear pointer arithmetic.
    if (buf_dim.fold_factor() != dim::unfolded) {
      // This function is expected to adjust all bases to point to the min of `buf_dim`. For non-folded dimensions, that
      // is true by construction, but not for folded dimensions.
      index_t offset = buf_dim.flat_offset_bytes(buf_dim.min());
      bases[0] = offset_bytes_non_null(bases[0], offset);
    }
    for (std::size_t n = 1; n < bufs.size(); n++) {
      const raw_buffer& buf_n = *bufs[n];
      if (SLINKY_LIKELY(bases[n] && d < static_cast<std::ptrdiff_t>(buf_n.rank))) {
        const dim& buf_n_dim = buf_n.dim(d);
        if (SLINKY_LIKELY(buf_n_dim.contains(buf_dim))) {
          index_t offset = buf_n_dim.flat_offset_bytes(buf_dim.min());
          bases[n] = offset_bytes_non_null(bases[n], offset);
        } else {
          // If we got here, we need to say the buffer is always out of bounds. If it is partially out of bounds,
          // use_nonlinear_loop should have returned true above.
          assert(buf_n_dim.empty() || buf_n_dim.min() > buf_dim.max() || buf_n_dim.max() < buf_dim.min());
          bases[n] = nullptr;
        }
      }
    }

    if (extent == 1 || (d > 0 && can_fuse(bufs, d))) {
      // Let this fuse with the next dimension.
    } else if (SkipContiguous && is_contiguous_slice(bufs, d)) {
      // This is the slice dimension.
      slice_extent *= extent;
      extent = 1;
    } else {
      // For the "output" buf, we can't cross a fold boundary, which means we can treat it as linear.
      assert(!buf_dim.is_folded());

      loop->extent = extent;
      extent = 1;
      loop->impl = for_each_impl_linear<F, BufsSize>;
      inner_impl = for_each_impl_call_f<F, BufsSize>;

      index_t* strides = loop->strides;
      strides[0] = buf_dim.stride();
      for (std::size_t n = 1; n < bufs.size(); n++) {
        const raw_buffer& buf_n = *bufs[n];
        strides[n] = d < static_cast<std::ptrdiff_t>(buf_n.rank) ? buf_n.dim(d).stride() : 0;
      }
      loop = offset_bytes_non_null(loop, sizeof_for_each_loop(bufs.size()));
    }
  }
  if (loop == outer_loop) {
    // There are no loops, just call f. This is an edge case below branch which assumes there is at least one loop.
    call_f(f, bases, 1, nullptr, slice_extent);
  } else {
    // Put the callback at the end of the plan, where the inner loop expects to find it.
    *reinterpret_cast<F*>(loop) = f;

    // We need to replace the implementation of the last loop.
    for_each_loop<BufsSize>* inner_loop = offset_bytes_non_null(loop, -sizeof_for_each_loop(bufs.size()));
    (void)inner_impl;
    inner_loop->impl = inner_impl;

    // Run the outer loop.
    outer_loop->impl({bases, bufs.size()}, outer_loop, slice_extent);
  }
}

}  // namespace

template <std::size_t BufsSize>
SLINKY_NO_STACK_PROTECTOR void for_each_contiguous_slice_impl(
    span<const raw_buffer*, BufsSize> bufs, for_each_contiguous_slice_callback f) {
  for_each_impl<true, BufsSize>(bufs, f);
}

template <size_t BufsSize>
SLINKY_NO_STACK_PROTECTOR void for_each_element_impl(
    span<const raw_buffer*, BufsSize> bufs, for_each_element_callback f) {
  for_each_impl<false, BufsSize>(bufs, f);
}

// These are templates defined in an implementation file, explicitly instantiate the templates we want to exist.
template void for_each_contiguous_slice_impl<dynamic_extent>(
    span<const raw_buffer*> bufs, for_each_contiguous_slice_callback f);
template void for_each_contiguous_slice_impl<1>(span<const raw_buffer*, 1> bufs, for_each_contiguous_slice_callback f);
template void for_each_contiguous_slice_impl<2>(span<const raw_buffer*, 2> bufs, for_each_contiguous_slice_callback f);
template void for_each_contiguous_slice_impl<3>(span<const raw_buffer*, 3> bufs, for_each_contiguous_slice_callback f);
template void for_each_contiguous_slice_impl<4>(span<const raw_buffer*, 4> bufs, for_each_contiguous_slice_callback f);

template void for_each_element_impl<dynamic_extent>(span<const raw_buffer*> bufs, for_each_element_callback f);
template void for_each_element_impl<1>(span<const raw_buffer*, 1> bufs, for_each_element_callback f);
template void for_each_element_impl<2>(span<const raw_buffer*, 2> bufs, for_each_element_callback f);
template void for_each_element_impl<3>(span<const raw_buffer*, 3> bufs, for_each_element_callback f);
template void for_each_element_impl<4>(span<const raw_buffer*, 4> bufs, for_each_element_callback f);

}  // namespace internal
}  // namespace slinky
