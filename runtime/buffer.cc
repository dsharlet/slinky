#include "runtime/buffer.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "runtime/util.h"

namespace slinky {

namespace {

std::size_t alloc_size(std::size_t rank, std::size_t elem_size, const dim* dims) {
  index_t flat_min = 0;
  index_t flat_max = 0;
  for (std::size_t i = 0; i < rank; ++i) {
    index_t extent = std::min(dims[i].extent(), dims[i].fold_factor());
    flat_min += (extent - 1) * std::min<index_t>(0, dims[i].stride());
    flat_max += (extent - 1) * std::max<index_t>(0, dims[i].stride());
  }
  return flat_max - flat_min + elem_size;
}

}  // namespace

std::size_t raw_buffer::size_bytes() const { return alloc_size(rank, elem_size, dims); }

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
    memcpy(buf->dims, dims, sizeof(slinky::dim) * rank);
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

void* raw_buffer::allocate() {
  void* allocation = malloc(size_bytes());
  base = allocation;
  return allocation;
}

namespace {

struct copy_dim {
  index_t pad_before;
  index_t size;
  index_t pad_after;
  index_t total_size;
  index_t src_stride;
  index_t dst_stride;

  bool operator<(const copy_dim& r) const { return dst_stride < r.dst_stride; }
};

void fill(char* dst, index_t stride, index_t elem_size, const void* value, index_t size) {
  if (!value) return;

  if (stride == elem_size) {
    switch (elem_size) {
    case 1: std::fill_n(reinterpret_cast<uint8_t*>(dst), size, *reinterpret_cast<const uint8_t*>(value)); return;
    case 2: std::fill_n(reinterpret_cast<uint16_t*>(dst), size, *reinterpret_cast<const uint16_t*>(value)); return;
    case 4: std::fill_n(reinterpret_cast<uint32_t*>(dst), size, *reinterpret_cast<const uint32_t*>(value)); return;
    case 8: std::fill_n(reinterpret_cast<uint64_t*>(dst), size, *reinterpret_cast<const uint64_t*>(value)); return;
    }
  }
  for (index_t i = 0; i < size; ++i) {
    memcpy(dst, value, elem_size);
    dst += stride;
  }
}

void copy(const char* src, index_t src_stride, char* dst, index_t dst_stride, index_t elem_size, index_t size) {
  for (index_t i = 0; i < size; ++i) {
    memcpy(dst, src, elem_size);
    dst += dst_stride;
    src += src_stride;
  }
}

void fill(char* dst, const copy_dim* dims, index_t elem_size, const void* value, int dim) {
  if (!value) return;

  const copy_dim& d = dims[dim];
  if (dim == 0) {
    fill(dst, d.dst_stride, elem_size, value, d.total_size);
  } else {
    for (index_t i = 0; i < d.total_size; ++i) {
      fill(dst, dims, elem_size, value, dim - 1);
      dst += d.dst_stride;
    }
  }
}

void copy(const char* src, char* dst, const copy_dim* dims, index_t elem_size, const void* padding, int dim) {
  // src can be nullptr, in which case we should only fill the padding.
  const copy_dim& d = dims[dim];
  if (dim == 0) {
    if (d.pad_before > 0) {
      fill(dst, d.dst_stride, elem_size, padding, d.pad_before);
      dst += d.dst_stride * d.pad_before;
    }
    if (src) {
      if (d.dst_stride == elem_size && d.src_stride == elem_size) {
        // src and dst are both dense, this can be implemented by memcpy.
        memcpy(dst, src, d.size * elem_size);
      } else if (d.src_stride == 0) {
        // Special case for broadcasting.
        fill(dst, d.dst_stride, elem_size, src, d.size);
      } else {
        // Need to copy one element at a time to skip padding.
        copy(src, d.src_stride, dst, d.dst_stride, elem_size, d.size);
      }
    }
    dst += d.size * d.dst_stride;
    if (d.pad_after > 0) {
      fill(dst, d.dst_stride, elem_size, padding, d.pad_after);
    }
  } else {
    for (index_t i = 0; i < d.pad_before; ++i) {
      fill(dst, dims, elem_size, padding, dim - 1);
      dst += d.dst_stride;
    }
    for (index_t i = 0; i < d.size; ++i) {
      copy(src, dst, dims, elem_size, padding, dim - 1);
      if (src) src += d.src_stride;
      dst += d.dst_stride;
    }
    for (index_t i = 0; i < d.pad_after; ++i) {
      fill(dst, dims, elem_size, padding, dim - 1);
      dst += d.dst_stride;
    }
  }
}

// For sorting tiny arrays of dimension metadata, this is faster than std::sort.
template <class It>
void bubble_sort(It begin, It end) {
  for (It i = begin; i != end; ++i) {
    for (It j = i; j != end; ++j) {
      if (*j < *i) {
        std::swap(*i, *j);
      }
    }
  }
}

void compute_padding(
    index_t src_begin, index_t src_end, const dim& dst, copy_dim& dim, index_t src_fold_factor = dim::unfolded) {
  if (dst.end() <= src_begin || dst.begin() >= src_end) {
    // This dimension is all padding.
    dim.pad_before = dim.total_size;
    dim.size = 0;
    dim.pad_after = 0;
  } else {
    index_t copy_begin = std::max(src_begin, dst.begin());
    index_t copy_end = std::min(src_end, dst.end());
    // TODO(https://github.com/dsharlet/slinky/issues/41): Enable storage folding in copies.
    assert(dst.min() / dst.fold_factor() == dst.max() / dst.fold_factor());
    assert(copy_begin / src_fold_factor == (copy_end - 1) / src_fold_factor);
    dim.size = std::max<index_t>(0, copy_end - copy_begin);
    dim.pad_before = std::max<index_t>(0, copy_begin - dst.begin());
    dim.pad_after = std::max<index_t>(0, dst.end() - copy_end);
  }
  assert(dim.pad_before + dim.pad_after + dim.size == dim.total_size);
}

int optimize_copy_dims(copy_dim* dims, int rank) {
  if (rank <= 1) return rank;

  // Sort the dims by (dst) stride.
  bubble_sort(dims, dims + rank);

  // Find dimensions we can fuse.
  for (int d = 0; d + 1 < rank;) {
    copy_dim& a = dims[d];
    const copy_dim& b = dims[d + 1];
    if (b.dst_stride != a.dst_stride * a.total_size || b.src_stride != a.src_stride * a.total_size) {
      // There are gaps between these dimensions, we can't fuse them.
      ++d;
      continue;
    }

    if (a.pad_before == 0 && a.pad_after == 0) {
      // a is entirely copied in this dimension.
      assert(a.size == a.total_size);
      a.pad_before = b.pad_before * a.size;
      a.pad_after = b.pad_after * a.size;
      a.total_size = b.total_size * a.size;
      a.size = b.size * a.size;
    } else if (a.size == 0 && a.pad_after == 0) {
      // a is entirely padded in this dimension.
      assert(a.pad_before == a.total_size);
      a.pad_before *= b.total_size;
      a.total_size = a.pad_before;
    } else {
      // Make sure we didn't use pad_after for all the padding.
      assert(a.pad_after < a.total_size);
      ++d;
      continue;
    }

    // Remove the now-fused dimension.
    for (int i = d + 1; i + 1 < rank; ++i) {
      dims[i] = dims[i + 1];
    }
    --rank;
  }
  return rank;
}

}  // namespace

SLINKY_NO_STACK_PROTECTOR void copy(const raw_buffer& src, const raw_buffer& dst, const void* padding) {
  assert(src.rank == dst.rank);
  assert(src.elem_size == dst.elem_size);

  const char* src_base = reinterpret_cast<const char*>(src.base);
  char* dst_base = reinterpret_cast<char*>(dst.base);

  // Make a list of pointers to dims that we are going to copy.
  copy_dim* dims = SLINKY_ALLOCA(copy_dim, dst.rank);

  int rank = 0;
  for (std::size_t i = 0; i < dst.rank; ++i) {
    const dim& dst_dim = dst.dims[i];
    if (dst_dim.max() < dst_dim.min()) {
      // Output is empty.
      return;
    }
    const dim& src_dim = src.dims[i];
    if (dst_dim.stride() == 0) {
      // Copying a broadcast to a broadcast is OK.
      assert(src_dim.stride() == 0);
      continue;
    } else {
      dims[rank].src_stride = src_dim.stride();
      dims[rank].dst_stride = dst_dim.stride();
      dims[rank].total_size = dst_dim.extent();
      compute_padding(src_dim.begin(), src_dim.end(), dst_dim, dims[rank]);
      if (src_dim.min() < dst_dim.min() && src_dim.contains(dst_dim.min())) {
        src_base += src_dim.flat_offset_bytes(dst_dim.min());
      }
      assert(dst.dims[rank].extent() <= dst.dims[rank].fold_factor());
      ++rank;
    }
  }

  rank = optimize_copy_dims(dims, rank);
  if (rank <= 0) {
    // The buffers are scalar.
    memcpy(dst.base, src.base, dst.elem_size);
    return;
  }

  // Now we have an optimized set of dimensions to copy. Run the copy.
  copy(src_base, dst_base, dims, dst.elem_size, padding, rank - 1);
}

SLINKY_NO_STACK_PROTECTOR void pad(const dim* in_bounds, const raw_buffer& dst, const void* padding) {
  char* dst_base = reinterpret_cast<char*>(dst.base);

  // Make a list of pointers to dims that we are going to pad.
  copy_dim* dims = SLINKY_ALLOCA(copy_dim, dst.rank);
  int rank = 0;
  for (std::size_t i = 0; i < dst.rank; ++i) {
    const dim& dst_dim = dst.dims[i];
    if (dst_dim.max() < dst_dim.min()) return;
    if (dst_dim.stride() == 0) continue;
    dims[rank].src_stride = 0;
    dims[rank].dst_stride = dst_dim.stride();
    dims[rank].total_size = dst_dim.extent();
    compute_padding(in_bounds[i].begin(), in_bounds[i].end(), dst_dim, dims[rank]);
    ++rank;
  }

  rank = optimize_copy_dims(dims, rank);
  if (rank <= 0) {
    // The buffer is scalar.
    return;
  }

  // Now we have an optimized set of dimensions to pad. Run the pad.
  copy(nullptr, dst_base, dims, dst.elem_size, padding, rank - 1);
}

SLINKY_NO_STACK_PROTECTOR void fill(const raw_buffer& dst, const void* value) {
  char* dst_base = reinterpret_cast<char*>(dst.base);

  // Make a list of pointers to dims that we are going to copy.
  copy_dim* dims = SLINKY_ALLOCA(copy_dim, dst.rank);
  int rank = 0;
  for (std::size_t i = 0; i < dst.rank; ++i) {
    const dim& dst_dim = dst.dims[i];
    if (dst_dim.max() < dst_dim.min()) return;
    if (dst_dim.stride() == 0) continue;
    dims[rank].dst_stride = dst_dim.stride();
    dims[rank].src_stride = 0;  // For optimize_copy_dims
    dims[rank].total_size = dst_dim.extent();

    dims[rank].pad_before = dims[rank].total_size;
    dims[rank].size = 0;
    dims[rank].pad_after = 0;
    rank++;
  }

  rank = optimize_copy_dims(dims, rank);
  if (rank <= 0) {
    // The buffer is scalar.
    memcpy(dst.base, value, dst.elem_size);
    return;
  }

  fill(dst_base, dims, dst.elem_size, value, rank - 1);
}

namespace internal {

namespace {

#ifndef NDEBUG
bool can_slice_with(const raw_buffer& buf, const raw_buffer& other_buf) {
  if (other_buf.rank > buf.rank) return false;
  for (std::size_t d = 0; d < other_buf.rank; d++) {
    const dim& other_dim = other_buf.dim(d);

    // Allow stride 0 dimensions to be accessed "out of bounds".
    if (other_dim.stride() == 0) continue;

    const dim& buf_dim = buf.dim(d);
    if (other_dim.min() > buf_dim.min()) return false;
    if (other_dim.max() < buf_dim.max()) return false;
  }
  return true;
}
#endif

bool is_contiguous_slice(const raw_buffer* const* bufs, std::size_t size, int d) {
  for (std::size_t n = 0; n < size; n++) {
    if (d >= static_cast<int>(bufs[n]->rank) || bufs[n]->dim(d).stride() != static_cast<index_t>(bufs[n]->elem_size))
      return false;
  }
  return true;
}

bool can_fuse(const raw_buffer* const* bufs, std::size_t size, int d) {
  assert(d > 0);
  const dim& base_inner = bufs[0]->dim(d - 1);
  for (std::size_t n = 0; n < size; n++) {
    if (d >= static_cast<int>(bufs[n]->rank)) return false;

    const dim& outer = bufs[n]->dim(d);
    const dim& inner = bufs[n]->dim(d - 1);
    // Our caller should have ensured this
    assert(outer.fold_factor() == dim::unfolded);
    if (inner.fold_factor() != dim::unfolded) return false;
    if (inner.min() != base_inner.min() || inner.extent() != base_inner.extent()) return false;
    if (inner.stride() * inner.extent() != outer.stride()) return false;
  }
  return true;
}

bool any_folded(const raw_buffer* const* bufs, std::size_t size, int d) {
  for (std::size_t i = 0; i < size; ++i) {
    if (d < static_cast<int>(bufs[i]->rank) && bufs[i]->dim(d).fold_factor() != dim::unfolded) return true;
  }
  return false;
}

static dim stride_0_dim;

template <typename T>
SLINKY_ALWAYS_INLINE inline T* get_plan(void*& x, std::size_t n = 1) {
  T* result = reinterpret_cast<T*>(x);
  x = offset_bytes(x, sizeof(T) * n);
  return result;
}

template <bool SkipContiguous, std::size_t BufsSize>
index_t make_for_each_slice_dims_impl(
    const raw_buffer* const* bufs, void** bases, std::size_t bufs_size_dynamic, void* plan) {
  std::size_t bufs_size = BufsSize == 0 ? bufs_size_dynamic : BufsSize;
  const auto* buf = bufs[0];
  for (std::size_t n = 0; n < bufs_size; ++n) {
    bases[n] = bufs[n]->base;
  }
  for_each_slice_dim* next = get_plan<for_each_slice_dim>(plan);
  dim_or_stride* next_dims = get_plan<dim_or_stride>(plan, bufs_size);
  index_t slice_extent = 1;
  index_t extent = 1;
  for (index_t d = static_cast<index_t>(buf->rank) - 1; d >= 0; --d) {
    const dim& buf_dim = buf->dim(d);
    if (buf_dim.extent() > 1 && any_folded(bufs, bufs_size, d)) {
      // There is a folded dimension in one of the buffers.
      assert(extent == 1);
      next->impl = for_each_slice_dim::loop_folded;
      next->extent = buf_dim.extent();
      for (std::size_t n = 0; n < bufs_size; n++) {
        next_dims[n].dim = d < static_cast<index_t>(bufs[n]->rank) ? &bufs[n]->dim(d) : &stride_0_dim;
      }
      next = get_plan<for_each_slice_dim>(plan);
      next_dims = get_plan<dim_or_stride>(plan, bufs_size);
      extent = 1;
      continue;
    }

    extent *= buf_dim.extent();
    // Align the bases for dimensions we will access via linear pointer arithmetic.
    for (std::size_t n = 1; n < bufs_size; n++) {
      if (d < static_cast<index_t>(bufs[n]->rank)) {
        index_t offset = bufs[n]->dim(d).flat_offset_bytes(buf_dim.min());
        bases[n] = offset_bytes(bases[n], offset);
      }
    }

    if (d > 0 && buf_dim.extent() == 1) {
      // This dimension has only one element, nothing to do.
    } else if (SkipContiguous && is_contiguous_slice(bufs, bufs_size, d)) {
      // This is the slice dimension.
      slice_extent = extent;
      extent = 1;
    } else if (d > 0 && can_fuse(bufs, bufs_size, d)) {
      // Let this dimension fuse with the next dimension.
    } else {
      // For the "output" buf, we can't cross a fold boundary, which means we can treat it as linear.
      assert(buf_dim.min() / buf_dim.fold_factor() == buf_dim.max() / buf_dim.fold_factor());
      next->impl = for_each_slice_dim::loop_linear;
      next->extent = extent;
      for (std::size_t n = 0; n < bufs_size; n++) {
        next_dims[n].stride = d < static_cast<index_t>(bufs[n]->rank) ? bufs[n]->dim(d).stride() : 0;
      }
      next = get_plan<for_each_slice_dim>(plan);
      next_dims = get_plan<dim_or_stride>(plan, bufs_size);
      extent = 1;
    }
  }
  next->impl = for_each_slice_dim::call_f;
  assert(extent == 1);
  return slice_extent;
}

}  // namespace

index_t make_for_each_contiguous_slice_dims(span<const raw_buffer*> bufs, void** bases, void* plan) {
  for (std::size_t n = 1; n < bufs.size(); n++) {
    assert(can_slice_with(*bufs[0], *bufs[n]));
  }

  // The implementation of this function benefits from knowing the size of the bufs span is constant.
  // By far the common case of this function is implementing elementwise unary or binary operations.
  // So, we provide special cases for those use cases, and use a slightly slower implementation otherwise.
  switch (bufs.size()) {
  case 1: return make_for_each_slice_dims_impl<true, 1>(bufs.data(), bases, 0, plan);
  case 2: return make_for_each_slice_dims_impl<true, 2>(bufs.data(), bases, 0, plan);
  case 3: return make_for_each_slice_dims_impl<true, 3>(bufs.data(), bases, 0, plan);
  default: return make_for_each_slice_dims_impl<true, 0>(bufs.data(), bases, bufs.size(), plan);
  }
}

void make_for_each_slice_dims(span<const raw_buffer*> bufs, void** bases, void* plan) {
  for (std::size_t n = 1; n < bufs.size(); n++) {
    assert(can_slice_with(*bufs[0], *bufs[n]));
  }

  // The implementation of this function benefits from knowing the size of the bufs span is constant.
  // By far the common case of this function is implementing elementwise unary or binary operations.
  // So, we provide special cases for those use cases, and use a slightly slower implementation otherwise.
  switch (bufs.size()) {
  case 1: make_for_each_slice_dims_impl<false, 1>(bufs.data(), bases, 0, plan); return;
  case 2: make_for_each_slice_dims_impl<false, 2>(bufs.data(), bases, 0, plan); return;
  case 3: make_for_each_slice_dims_impl<false, 3>(bufs.data(), bases, 0, plan); return;
  default: make_for_each_slice_dims_impl<false, 0>(bufs.data(), bases, bufs.size(), plan); return;
  }
}

}  // namespace internal
}  // namespace slinky