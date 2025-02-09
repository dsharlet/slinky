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
  std::size_t size = sizeof(raw_buffer) + sizeof(slinky::dim) * rank;
  if (dims) {
    size += alloc_size(rank, elem_size, dims);
  }
  char* mem = reinterpret_cast<char*>(malloc(size));
  raw_buffer* buf = new (mem) raw_buffer();
  mem += sizeof(raw_buffer);
  buf->rank = rank;
  buf->elem_size = elem_size;
  buf->dims = reinterpret_cast<slinky::dim*>(mem);
  if (dims) {
    internal::copy_small_n(dims, rank, buf->dims);
    mem += sizeof(slinky::dim) * rank;
    buf->base = mem;
  } else {
    new (buf->dims) slinky::dim[buf->rank];
  }
  return raw_buffer_ptr(buf, free);
}

raw_buffer_ptr raw_buffer::make_copy(const raw_buffer& src) {
  auto buf = make(src.rank, src.elem_size, src.dims);
  copy(src, *buf);
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

// We represent scalar values that get used in `fill` as a pointer to bytes. The simplest implementation of such a fill
// is to call `memcpy(dst++, value, elem_size)` in a loop. However, this is very inefficient for small values of
// `elem_size`. This function attempts to rewrite such fills as a different scalar value that is more efficient.
using constant_buffer = std::array<uint8_t, 64>;
void optimize_fill_value(const void*& value, index_t& elem_size, constant_buffer& buffer) {
  if (is_repeated_byte(value, elem_size)) {
    // We can use memset to implement this fill value.
    elem_size = 1;
  } else if (elem_size * 2 <= static_cast<index_t>(buffer.size())) {
    // Repeatedly duplicate the constant as long as it fits in the buffer.
    memcpy(buffer.data(), value, elem_size);
    while (elem_size * 2 <= static_cast<index_t>(buffer.size())) {
      memcpy(&buffer[elem_size], buffer.data(), elem_size);
      elem_size *= 2;
    }
    value = buffer.data();
  }
}

void fill(void* dst, const void* value, index_t elem_size, index_t size) {
  if (elem_size == 1) {
    memset(dst, *reinterpret_cast<const uint8_t*>(value), size);
  } else {
    while (size >= elem_size) {
      memcpy(dst, value, elem_size);
      dst = offset_bytes_non_null(dst, elem_size);
      size -= elem_size;
    }
    // The elem_size might not divide the size if replicate_constant replicated it.
    memcpy(dst, value, size);
  }
}

// This function modifies the dims of src and dst.
void copy_impl(raw_buffer& src, raw_buffer& dst, const void* padding) {
  assert(src.rank == dst.rank);
  assert(src.elem_size == dst.elem_size);
  assert(dst.base || dst.elem_count() == 0);
  const std::size_t rank = dst.rank;
  index_t elem_size = dst.elem_size;

  if (rank == 0) {
    memcpy(dst.base, padding && !src.base ? padding : src.base, elem_size);
    return;
  }

  // Make a (shallow) copy of the buffers and optimize the dimensions.
  optimize_dims(dst, src);
  dim& dst_dim0 = dst.dim(0);
  dim& src_dim0 = src.dim(0);

  if (dst_dim0.empty()) {
    // Empty destination, nothing to do.
  } else if (dst_dim0.fold_factor() != dim::unfolded || src_dim0.fold_factor() != dim::unfolded ||
             dst_dim0.stride() != elem_size || src_dim0.stride() != elem_size) {
    // The inner copy dimension is not a linear copy, let for_each_element handle it.
    for_each_element(
        [elem_size, padding](void* dst, const void* src) { memcpy(dst, padding && !src ? padding : src, elem_size); },
        dst, src);
  } else {
    // The inner dimension is a linear copy. Slice off that dimension and handle it ourselves.
    src.crop(0, dst_dim0.min(), dst_dim0.max());
    dst.slice(0);
    src.slice(0);

    const index_t dst_size = dst_dim0.extent() * elem_size;

    if (padding) {
      const index_t src_dim0_begin = std::min(src_dim0.begin(), dst_dim0.end());
      const index_t src_dim0_end = std::max(src_dim0.end(), dst_dim0.begin());

      const index_t pad_before =
          src_dim0_begin > dst_dim0.begin() ? (src_dim0_begin - dst_dim0.begin()) * elem_size : 0;
      const index_t pad_after = dst_dim0.end() > src_dim0_end ? (dst_dim0.end() - src_dim0_end) * elem_size : 0;
      const index_t src_size = dst_size - pad_before - pad_after;

      constant_buffer buffer;
      optimize_fill_value(padding, elem_size, buffer);

      for_each_element(
          [=](void* dst, const void* src) {
            // TDOO: There are a lot of branches in here. They could possibly be lifted out of the for_each_element
            // loops, but we need to find ways to do it that avoids increasing the number of cases we need to handle too
            // much.
            if (src) {
              if (pad_before > 0) {
                fill(dst, padding, elem_size, pad_before);
                dst = offset_bytes_non_null(dst, pad_before);
              }
              memcpy(dst, src, src_size);
              if (pad_after > 0) {
                dst = offset_bytes_non_null(dst, src_size);
                fill(dst, padding, elem_size, pad_after);
              }
            } else {
              fill(dst, padding, elem_size, dst_size);
            }
          },
          dst, src);
    } else {
      assert(src_dim0.begin() == dst_dim0.begin());
      assert(src_dim0.end() == dst_dim0.end());

      for_each_element([=](void* dst, const void* src) { memcpy(dst, src, dst_size); }, dst, src);
    }
  }
}

}  // namespace

SLINKY_NO_STACK_PROTECTOR void copy(const raw_buffer& src, const raw_buffer& dst, const void* padding) {
  // Make (shallow) copies of the buffers and optimize the dimensions.
  raw_buffer src_opt = src;
  src_opt.dims = SLINKY_ALLOCA(dim, src.rank);
  internal::copy_small_n(src.dims, src.rank, src_opt.dims);
  raw_buffer dst_opt = dst;
  dst_opt.dims = SLINKY_ALLOCA(dim, dst.rank);
  internal::copy_small_n(dst.dims, dst.rank, dst_opt.dims);

  copy_impl(src_opt, dst_opt, padding);
}

void pad(const dim* in_bounds, const raw_buffer& dst, const void* padding) {
  // To implement pad, we'll make a buffer that looks like dst, but cropped to the bounds, and copy it with padding.
  raw_buffer src = dst;
  src.dims = SLINKY_ALLOCA(dim, dst.rank);
  internal::copy_small_n(dst.dims, dst.rank, src.dims);
  for (std::size_t d = 0; d < dst.rank; ++d) {
    src.crop(d, in_bounds[d].min(), in_bounds[d].max());
  }

  raw_buffer dst_opt = dst;
  dst_opt.dims = SLINKY_ALLOCA(dim, dst.rank);
  internal::copy_small_n(dst.dims, dst.rank, dst_opt.dims);

  copy_impl(src, dst_opt, padding);
}

SLINKY_NO_STACK_PROTECTOR void fill(const raw_buffer& dst, const void* value) {
  assert(value);
  assert(dst.base || dst.elem_count() == 0);
  const std::size_t rank = dst.rank;
  index_t elem_size = dst.elem_size;

  if (rank == 0) {
    memcpy(dst.base, value, elem_size);
    return;
  }

  // Make a (shallow) copy of the buffer and optimize the dimensions.
  raw_buffer dst_opt = dst;
  dst_opt.dims = SLINKY_ALLOCA(dim, dst.rank);
  internal::copy_small_n(dst.dims, dst.rank, dst_opt.dims);

  optimize_dims(dst_opt);
  dim& dst_dim0 = dst_opt.dim(0);

  if (dst_dim0.empty()) {
    // Empty destination, nothing to do.
  } else if (dst_dim0.fold_factor() != dim::unfolded || dst_dim0.stride() != elem_size) {
    // The inner dimension is not a linear fill, let for_each_element handle it.
    for_each_element([elem_size, value](void* dst) { memcpy(dst, value, elem_size); }, dst_opt);
  } else {
    // The inner dimension is a linear fill. Slice off that dimension and handle it ourselves.
    const index_t size = dst_dim0.extent() * elem_size;
    dst_opt.slice(0);

    constant_buffer buffer;
    optimize_fill_value(value, elem_size, buffer);

    for_each_element([=](void* dst) { fill(dst, value, elem_size, size); }, dst_opt);
  }
}

namespace internal {

namespace {

SLINKY_ALWAYS_INLINE inline const dim& get_dim(const raw_buffer& buf, std::size_t d) {
  // Dimensions beyond the rank are broadcasts.
  return d < buf.rank ? buf.dim(d) : broadcast_dim;
}

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

template <std::size_t BufsSize>
SLINKY_ALWAYS_INLINE inline bool use_folded_loop(span<const raw_buffer*, BufsSize> bufs, std::size_t d) {
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

// These functions may modify the second parameter in-place.
template <std::size_t BufsSize>
using for_each_loop_impl = void (*)(mutable_span<void*, BufsSize>, const for_each_loop<BufsSize>*);

template <std::size_t BufsSize>
struct for_each_loop {
  index_t extent;
  for_each_loop_impl<BufsSize> impl;
  union {
    index_t strides[1];  // [bufs_size]
    const dim* dims[1];  // [bufs_size]
  };
};

template <typename F>
struct callback {
  F f;
  index_t slice_extent;
};

std::ptrdiff_t sizeof_for_each_loop(std::size_t bufs_size) {
  return sizeof(for_each_loop<>) - sizeof(for_each_loop<>::dims) + sizeof(for_each_loop<>::dims) * bufs_size;
}

// We store a plan for a parallel for loop in a structure of the following layout, for N buffers and rank R loops:
// struct {
//   for_each_loop loop;
//   union {
//     index_t strides[N];  // Used by linear loops
//     const dim* dims[N];  // Used by folded loops
//   };
// } loops[R];
// callback<F> f;
//
// We can't make a simple struct for this, because N and R are not necessarily compile-time constants.
template <typename F>
SLINKY_ALWAYS_INLINE inline std::size_t size_of_plan(std::size_t bufs_size, std::size_t rank) {
  return sizeof_for_each_loop(bufs_size) * std::max<std::size_t>(1, rank) + sizeof(callback<F>);
}

// Compile-time dispatch to either for_each_contiguous_slice_callback or for_each_element_callback
SLINKY_ALWAYS_INLINE inline void call_f(
    const callback<for_each_element_callback>& f, void** bases, index_t extent, const index_t* strides) {
  f.f(bases, extent, strides);
}
SLINKY_ALWAYS_INLINE inline void call_f(
    const callback<for_each_contiguous_slice_callback>& f, void** bases, index_t extent, const index_t* strides) {
  f.f(f.slice_extent, bases, extent, strides);
}

template <typename F, std::size_t BufsSize>
void for_each_impl_call_f(mutable_span<void*, BufsSize> bases, const for_each_loop<BufsSize>* loop) {
  const callback<F>& f =
      *reinterpret_cast<const callback<F>*>(offset_bytes_non_null(loop, sizeof_for_each_loop(bases.size())));

  call_f(f, bases.data(), loop->extent, loop->strides);
}

template <typename F, std::size_t BufsSize>
SLINKY_NO_STACK_PROTECTOR void for_each_impl_linear(
    mutable_span<void*, BufsSize> bases, const for_each_loop<BufsSize>* loop) {
  const index_t* strides = loop->strides;
  index_t extent = loop->extent;
  assert(extent >= 1);

  loop = offset_bytes_non_null(loop, sizeof_for_each_loop(bases.size()));

  for_each_loop_impl<BufsSize> impl = loop->impl;

  void** bases_i = SLINKY_ALLOCA(void*, bases.size());
  for (;;) {
    copy_small_n(bases.data(), bases.size(), bases_i);
    impl({bases_i, bases.size()}, loop);
    if (SLINKY_UNLIKELY(--extent <= 0)) break;
    increment_bases<BufsSize>(bases.size(), bases.data(), strides);
  }
}

template <typename F, std::size_t BufsSize, bool CallF>
SLINKY_NO_STACK_PROTECTOR void for_each_impl_folded(
    mutable_span<void*, BufsSize> bases, const for_each_loop<BufsSize>* loop) {
  index_t extent = loop->extent;
  const dim* const* dims = loop->dims;
  loop = offset_bytes_non_null(loop, sizeof_for_each_loop(bases.size()));

  for_each_loop_impl<BufsSize> impl = loop->impl;
  const callback<F>& f = *reinterpret_cast<const callback<F>*>(loop);

  index_t begin = dims[0]->begin();
  index_t end = begin + extent;
  void** bases_i = SLINKY_ALLOCA(void*, bases.size());
  for (index_t i = begin; i < end; ++i) {
    bases_i[0] = offset_bytes_non_null(bases[0], dims[0]->flat_offset_bytes(i));
    for (std::size_t n = 1; n < bases.size(); n++) {
      bases_i[n] = dims[n]->contains(i) ? offset_bytes(bases[n], dims[n]->flat_offset_bytes(i)) : nullptr;
    }
    if (CallF) {
      // If the next step is to call f, do that eagerly here to avoid an extra call.
      call_f(f, bases_i, 1, nullptr);
    } else {
      impl({bases_i, bases.size()}, loop);
    }
  }
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
    } else if (buf_dim.max() < buf_dim.min() || use_folded_loop(bufs, d)) {
      // extent > 1 and there is a folded dimension in one of the buffers, or we need to crop one of the buffers, or the
      // loops are empty.
      loop->extent = buf_dim.extent();
      loop->impl = for_each_impl_folded<F, BufsSize, false>;
      inner_impl = for_each_impl_folded<F, BufsSize, true>;
      extent = 1;

      const dim** dims = loop->dims;
      dims[0] = &buf.dim(d);
      for (std::size_t n = 1; n < bufs.size(); n++) {
        dims[n] = &get_dim(*bufs[n], d);
      }
      loop = offset_bytes_non_null(loop, sizeof_for_each_loop(bufs.size()));
      continue;
    } else {
      // Not folded, use a linear, possibly fused loop below.
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
          // use_folded_loop should have returned true above.
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
    call_f(callback<F>{f, slice_extent}, bases, 1, nullptr);
  } else {
    // Put the callback at the end of the plan, where the inner loop expects to find it.
    reinterpret_cast<callback<F>*>(loop)->f = f;
    if (SkipContiguous) {
      reinterpret_cast<callback<F>*>(loop)->slice_extent = slice_extent;
    }

    // We need to replace the implementation of the last loop.
    for_each_loop<BufsSize>* inner_loop = offset_bytes_non_null(loop, -sizeof_for_each_loop(bufs.size()));
    inner_loop->impl = inner_impl;

    // Run the outer loop.
    outer_loop->impl({bases, bufs.size()}, outer_loop);
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
