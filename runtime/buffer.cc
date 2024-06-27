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
bool is_stride_ok(index_t stride, index_t extent, const dim& dim) {
  if (dim.stride() == dim::auto_stride) {
    // If the dimension has an unknown stride, it's OK, we're
    // resolving the current dim first.
    return true;
  } else if (extent == 1 && std::abs(stride) == std::abs(dim.stride()) && alloc_extent(dim) > 1) {
    // If a dimension is extent 1, avoid giving this dimension the same stride
    // as another dimension with extent greater than 1. This doesn't affect the
    // results of most programs (because the stride only ever multiplied with
    // zero), but it makes the strides less objectionable to asserts in some
    // other libraries that make extra assumptions about images, and may be
    // easier to understand.
    return false;
  } else if (alloc_extent(dim) * std::abs(dim.stride()) <= stride) {
    // The dim is completely inside the proposed stride.
    return true;
  } else if (std::abs(dim.stride()) >= extent * stride) {
    // The dim is completely outside the proposed stride.
    return true;
  } else {
    return false;
  }
}

bool is_stride_ok(index_t stride, index_t extent, span<const dim> dims) {
  for (const dim& i : dims) {
    if (!is_stride_ok(stride, extent, i)) {
      return false;
    }
  }
  return true;
}

}  // namespace

void raw_buffer::init_strides(index_t alignment) {
  for (std::size_t i = 0; i < rank; ++i) {
    if (dim(i).stride() != dim::auto_stride) continue;

    index_t alloc_extent_i = alloc_extent(dim(i));

    if (is_stride_ok(elem_size, alloc_extent_i, {dims, rank})) {
      // This dimension can have stride elem_size, no other stride could be better.
      dim(i).set_stride(elem_size);
      continue;
    }

    // Loop through all the dimensions and see if a stride that is just outside any dimension is OK.
    index_t min = std::numeric_limits<index_t>::max();
    for (std::size_t j = 0; j < rank; ++j) {
      if (dim(j).stride() == dim::auto_stride) {
        // We don't know the stride of this dimension, it can't help us decide a stride for this dimension.
        continue;
      } else if (dim(j).max() < dim(j).min()) {
        // This dimension (and this buffer) is empty.
        min = 0;
        break;
      }

      index_t candidate = align_up(std::abs(dim(j).stride()) * alloc_extent(dim(j)), alignment);
      if (candidate >= min) {
        // This candidate stride is not better than the current stride.
        continue;
      } else if (!is_stride_ok(candidate, alloc_extent_i, {dims, rank})) {
        continue;
      }
      min = candidate;
    }
    assert(min < std::numeric_limits<index_t>::max());
    dim(i).set_stride(min);
  }
}

void* raw_buffer::allocate() {
  init_strides();
  void* allocation = malloc(size_bytes());
  base = allocation;
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
  const std::size_t rank = dst.rank;
  index_t elem_size = dst.elem_size;

  if (rank == 0) {
    if (src.base) {
      memcpy(dst.base, src.base, elem_size);
    } else if (padding) {
      memcpy(dst.base, padding, elem_size);
    }
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
        [elem_size, padding](void* dst, const void* src) {
          if (src) {
            memcpy(dst, src, elem_size);
          } else {
            memcpy(dst, padding, elem_size);
          }
        },
        dst, src);
  } else {
    // The inner dimension is a linear copy. Slice off that dimension and handle it ourselves.

    // Eliminate the case we need to consider where src is bigger than dst.
    src.crop(0, dst_dim0.min(), dst_dim0.max());

    const index_t padded_size = dst_dim0.extent() * elem_size;
    const index_t pad_before = (src_dim0.begin() - dst_dim0.begin()) * elem_size;
    const index_t pad_after = (dst_dim0.end() - src_dim0.end()) * elem_size;
    const index_t size = padded_size - pad_before - pad_after;
    dst.slice(0);
    src.slice(0);

    constant_buffer buffer;
    if (padding) {
      optimize_fill_value(padding, elem_size, buffer);
    } else {
      assert(size == padded_size);
      assert(pad_before == 0);
      assert(pad_after == 0);
    }

    for_each_element(
        [=](void* dst, const void* src) {
          // TDOO: There are a lot of branches in here. They could possibly be lifted out of the for_each_element loops,
          // but we need to find ways to do it that avoids increasing the number of cases we need to handle too much.
          if (src) {
            if (pad_before > 0) {
              fill(dst, padding, elem_size, pad_before);
              dst = offset_bytes_non_null(dst, pad_before);
            }
            memcpy(dst, src, size);
            if (pad_after > 0) {
              dst = offset_bytes_non_null(dst, size);
              fill(dst, padding, elem_size, pad_after);
            }
          } else {
            fill(dst, padding, elem_size, padded_size);
          }
        },
        dst, src);
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

SLINKY_ALWAYS_INLINE inline bool is_contiguous_slice(const raw_buffer* const* bufs, std::size_t size, int d) {
  if (bufs[0]->dim(d).stride() != static_cast<index_t>(bufs[0]->elem_size)) {
    // This dimension is not contiguous.
    return false;
  }
  for (std::size_t n = 1; n < size; n++) {
    if (d >= static_cast<int>(bufs[n]->rank)) {
      // This dimension is broadcasted, it's not contiguous.
      return false;
    } else if (bufs[n]->dim(d).stride() != static_cast<index_t>(bufs[n]->elem_size)) {
      // This dimension is not contiguous.
      return false;
    }
  }
  return true;
}

SLINKY_ALWAYS_INLINE inline bool can_fuse(const raw_buffer* const* bufs, std::size_t size, int d) {
  assert(d > 0);
  const dim& base_inner = bufs[0]->dim(d - 1);
  const dim& base_outer = bufs[0]->dim(d);
  if (base_inner.fold_factor() != dim::unfolded) {
    // One of the dimensions is folded.
    return false;
  }
  const index_t inner_extent = base_inner.extent();
  if (base_inner.stride() * inner_extent != base_outer.stride()) {
    // The dimensions are not contiguous in memory.
    return false;
  }

  for (std::size_t n = 1; n < size; n++) {
    if (d - 1 >= static_cast<int>(bufs[n]->rank)) {
      // Both dimensions are broadcasts, they can be fused.
      continue;
    }

    const dim& inner = bufs[n]->dim(d - 1);
    if (inner.min() != base_inner.min() || inner.max() != base_inner.max()) {
      // The bounds of the inner dimension are not equal.
      return false;
    } else if (inner.fold_factor() != dim::unfolded) {
      // One of the dimensions is folded.
      return false;
    }

    const index_t outer_stride = d < static_cast<int>(bufs[n]->rank) ? bufs[n]->dim(d).stride() : 0;
    if (inner.stride() * inner_extent != outer_stride) {
      // The dimensions are not contiguous in memory.
      return false;
    }
  }
  return true;
}

SLINKY_ALWAYS_INLINE inline bool use_folded_loop(const raw_buffer* const* bufs, std::size_t size, int d) {
  if (bufs[0]->dim(d).fold_factor() != dim::unfolded) {
    // The main buffer is folded.
    return true;
  }
  for (std::size_t i = 1; i < size; ++i) {
    if (d >= static_cast<int>(bufs[i]->rank)) {
      // Broadcast dimension.
      continue;
    } else if (bufs[i]->dim(d).fold_factor() != dim::unfolded) {
      // There's a folded buffer, we need a folded loop.
      return true;
    } else if (!bufs[i]->dim(d).contains(bufs[0]->dim(d))) {
      // One of the extra buffers is out of bounds, use a folded loop.
      return true;
    }
  }
  return false;
}

template <typename T>
SLINKY_ALWAYS_INLINE inline T* increment_plan(void*& x, std::size_t n = 1) {
  T* result = reinterpret_cast<T*>(x);
  x = offset_bytes_non_null(x, sizeof(T) * n);
  return result;
}

// Helper function to write a plan that does nothing when interpreted by for_each_impl.
void write_empty_plan(void* plan, std::size_t bufs_size) {
  for_each_loop* next = increment_plan<for_each_loop>(plan);
  next->impl = for_each_loop::linear | for_each_loop::call_f;
  next->extent = 0;
}

template <bool SkipContiguous, std::size_t BufsSize>
SLINKY_NO_INLINE index_t make_for_each_loops_impl(
    const raw_buffer* const* bufs, void** bases, std::size_t bufs_size_dynamic, void* plan_base) {
  std::size_t bufs_size = BufsSize == 0 ? bufs_size_dynamic : BufsSize;
  const auto* buf = bufs[0];
  bases[0] = buf->base;
  for (std::size_t n = 1; n < bufs_size; ++n) {
    bases[n] = bufs[n]->base;
  }

  // Start out with a loop of extent 1, in case the buffer is rank 0.
  for_each_loop* prev_loop = reinterpret_cast<for_each_loop*>(plan_base);
  prev_loop->impl = for_each_loop::linear;
  prev_loop->extent = 1;

  void* plan = plan_base;
  index_t slice_extent = 1;
  index_t extent = 1;
  for (index_t d = static_cast<index_t>(buf->rank) - 1; d >= 0; --d) {
    const dim& buf_dim = buf->dim(d);

    if (buf_dim.min() == buf_dim.max()) {
      // extent 1, we don't need any of the logic here, skip to below.
    } else if (buf_dim.max() > buf_dim.min()) {
      if (use_folded_loop(bufs, bufs_size, d)) {
        // extent > 1 and there is a folded dimension in one of the buffers, or we need to crop one of the buffers.
        assert(extent == 1);
        for_each_loop* loop = increment_plan<for_each_loop>(plan);
        loop->impl = for_each_loop::folded;
        loop->extent = buf_dim.extent();
        prev_loop = loop;

        const dim** dims = increment_plan<const dim*>(plan, bufs_size);
        dims[0] = &buf->dim(d);
        for (std::size_t n = 1; n < bufs_size; n++) {
          dims[n] = d < static_cast<index_t>(bufs[n]->rank) ? &bufs[n]->dim(d) : &broadcast_dim;
        }
        continue;
      } else {
        // Not folded, use a linear, possibly fused loop below.
        extent *= buf_dim.extent();
      }
    } else {
      // extent <= 0.
      assert(buf_dim.empty());
      write_empty_plan(plan_base, bufs_size);
      return 0;
    }

    // Align the bases for dimensions we will access via linear pointer arithmetic.
    if (bases[0]) {
      // This function is expected to adjust all bases to point to the min of `buf_dim`. For non-folded dimensions, that
      // is true by construction, but not for folded dimensions.
      index_t offset = buf_dim.flat_offset_bytes(buf_dim.min());
      bases[0] = offset_bytes_non_null(bases[0], offset);
    }
    for (std::size_t n = 1; n < bufs_size; n++) {
      if (bases[n] && d < static_cast<index_t>(bufs[n]->rank)) {
        const dim& buf_n_dim = bufs[n]->dim(d);
        if (buf_n_dim.contains(buf_dim)) {
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

    if (d > 0 && (extent == 1 || can_fuse(bufs, bufs_size, d))) {
      // Let this fuse with the next dimension.
    } else if (SkipContiguous && is_contiguous_slice(bufs, bufs_size, d)) {
      // This is the slice dimension.
      slice_extent *= extent;
      extent = 1;
    } else {
      // For the "output" buf, we can't cross a fold boundary, which means we can treat it as linear.
      assert(!buf_dim.is_folded());

      for_each_loop* loop = increment_plan<for_each_loop>(plan);
      loop->impl = for_each_loop::linear;
      loop->extent = extent;
      prev_loop = loop;
      extent = 1;

      index_t* strides = increment_plan<index_t>(plan, bufs_size);
      strides[0] = buf->dim(d).stride();
      for (std::size_t n = 1; n < bufs_size; n++) {
        strides[n] = d < static_cast<index_t>(bufs[n]->rank) ? bufs[n]->dim(d).stride() : 0;
      }
    }
  }
  prev_loop->impl |= for_each_loop::call_f;
  assert(extent == 1);
  return SkipContiguous ? slice_extent : 1;
}

}  // namespace

index_t make_for_each_contiguous_slice_loops(span<const raw_buffer*> bufs, void** bases, void* plan) {
  // The implementation of this function benefits from knowing the size of the bufs span is constant.
  // By far the common case of this function is implementing elementwise unary or binary operations.
  // So, we provide special cases for those use cases, and use a slightly slower implementation otherwise.
  switch (bufs.size()) {
  case 1: return make_for_each_loops_impl<true, 1>(bufs.data(), bases, 0, plan);
  case 2: return make_for_each_loops_impl<true, 2>(bufs.data(), bases, 0, plan);
  case 3: return make_for_each_loops_impl<true, 3>(bufs.data(), bases, 0, plan);
  default: return make_for_each_loops_impl<true, 0>(bufs.data(), bases, bufs.size(), plan);
  }
}

void make_for_each_loops(span<const raw_buffer*> bufs, void** bases, void* plan) {
  // The implementation of this function benefits from knowing the size of the bufs span is constant.
  // By far the common case of this function is implementing elementwise unary or binary operations.
  // So, we provide special cases for those use cases, and use a slightly slower implementation otherwise.
  switch (bufs.size()) {
  case 1: make_for_each_loops_impl<false, 1>(bufs.data(), bases, 0, plan); return;
  case 2: make_for_each_loops_impl<false, 2>(bufs.data(), bases, 0, plan); return;
  case 3: make_for_each_loops_impl<false, 3>(bufs.data(), bases, 0, plan); return;
  default: make_for_each_loops_impl<false, 0>(bufs.data(), bases, bufs.size(), plan); return;
  }
}

}  // namespace internal

internal::iterator_range<internal::index_iterator> index_range(const raw_buffer& buf, std::size_t min_dim) {
  std::vector<index_t> min(buf.rank - min_dim);
  std::vector<index_t> max(buf.rank - min_dim);
  for (std::size_t d = min_dim; d < buf.rank; ++d) {
    min[d - min_dim] = buf.dim(d).min();
    max[d - min_dim] = buf.dim(d).max();
  }
  internal::index_iterator begin(min, min, max);
  std::vector<index_t> end_i = min;
  end_i.back() = max.back() + 1;
  internal::index_iterator end(std::move(end_i), std::move(min), std::move(max));
  return {std::move(begin), std::move(end)};
}

}  // namespace slinky