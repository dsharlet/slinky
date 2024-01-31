#include "runtime/buffer.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "runtime/util.h"

namespace slinky {

std::size_t raw_buffer::size_bytes() const {
  index_t flat_min = 0;
  index_t flat_max = 0;
  for (std::size_t i = 0; i < rank; ++i) {
    index_t extent = std::min(dims[i].extent(), dims[i].fold_factor());
    flat_min += (extent - 1) * std::min<index_t>(0, dims[i].stride());
    flat_max += (extent - 1) * std::max<index_t>(0, dims[i].stride());
  }
  return flat_max - flat_min + elem_size;
}

std::ptrdiff_t raw_buffer::allocation_offset_bytes() const {
  std::ptrdiff_t offset = 0;
  for (std::size_t d = 0; d < rank; ++d) {
    if (dims[d].fold_factor() == dim::unfolded) {
      offset -= dims[d].flat_offset_bytes(dims[d].min());
    }
  }
  return offset;
}

// Does not call constructor or destructor of T!
void raw_buffer::allocate() {
  assert(allocation == nullptr);

  allocation = new char[size_bytes()];
  base = allocation + allocation_offset_bytes();
}

void raw_buffer::free() {
  delete[] allocation;
  allocation = nullptr;
  base = nullptr;
}

raw_buffer_ptr raw_buffer::make(std::size_t rank, std::size_t elem_size) {
  char* buf_and_dims = new char[sizeof(raw_buffer) + sizeof(slinky::dim) * rank];
  raw_buffer* buf = new (buf_and_dims) raw_buffer();
  buf->base = nullptr;
  buf->allocation = nullptr;
  buf->rank = rank;
  buf->elem_size = elem_size;
  buf->dims = reinterpret_cast<slinky::dim*>(buf_and_dims + sizeof(raw_buffer));
  new (buf->dims) slinky::dim[rank];
  return buf;
}

void raw_buffer::destroy(raw_buffer* buf) {
  buf->~raw_buffer();
  delete[] (char*)buf;
}

raw_buffer_ptr raw_buffer::make(std::size_t elem_size, span<const index_t> extents) {
  raw_buffer_ptr result = make(extents.size(), elem_size);
  index_t stride = elem_size;
  for (std::size_t d = 0; d < extents.size(); ++d) {
    result->dims[d].set_min_extent(0, extents[d]);
    result->dims[d].set_stride(stride);
    stride *= extents[d];
  }
  return result;
}

raw_buffer_ptr raw_buffer::make(const raw_buffer& src) {
  raw_buffer_ptr result = make(src.rank, src.elem_size);
  for (std::size_t d = 0; d < src.rank; ++d) {
    result->dims[d] = src.dims[d];
  }
  if (src.base) {
    result->allocate();
    copy(src, *result);
  }
  return result;
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

template <typename T>
void fill(T* dst, T value, index_t size) {
  std::fill(dst, dst + size, value);
}

void fill(char* dst, index_t stride, index_t elem_size, const void* value, index_t size) {
  if (!value) return;

  if (stride == elem_size) {
    switch (elem_size) {
    case 1: fill(reinterpret_cast<uint8_t*>(dst), *reinterpret_cast<const uint8_t*>(value), size); return;
    case 2: fill(reinterpret_cast<uint16_t*>(dst), *reinterpret_cast<const uint16_t*>(value), size); return;
    case 4: fill(reinterpret_cast<uint32_t*>(dst), *reinterpret_cast<const uint32_t*>(value), size); return;
    case 8: fill(reinterpret_cast<uint64_t*>(dst), *reinterpret_cast<const uint64_t*>(value), size); return;
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

void compute_padding(index_t src_begin, index_t src_end, const dim& dst, copy_dim& dim) {
  if (dst.end() <= src_begin || dst.begin() >= src_end) {
    // This dimension is all padding.
    dim.pad_before = dim.total_size;
    dim.size = 0;
    dim.pad_after = 0;
  } else {
    index_t copy_begin = std::max(src_begin, dst.begin());
    index_t copy_end = std::min(src_end, dst.end());
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
    if (a.pad_before == 0 && a.pad_after == 0 && b.dst_stride == a.dst_stride * a.total_size &&
        b.src_stride == a.src_stride * a.total_size) {
      assert(a.size == a.total_size);
      a.pad_before = b.pad_before * a.size;
      a.pad_after = b.pad_after * a.size;
      a.total_size = b.total_size * a.size;
      a.size = b.size * a.size;

      // Remove the now-fused dimension.
      for (int i = d + 1; i + 1 < rank; ++i) {
        dims[i] = dims[i + 1];
      }
      --rank;
    } else {
      ++d;
    }
  }
  return rank;
}

}  // namespace

void copy(const raw_buffer& src, const raw_buffer& dst, const void* padding) {
  assert(src.rank == dst.rank);
  assert(src.elem_size == dst.elem_size);

  if (dst.rank == 0) {
    // The buffers are scalar.
    memcpy(dst.base, src.base, dst.elem_size);
    return;
  }

  const char* src_base = reinterpret_cast<const char*>(src.base);
  char* dst_base = reinterpret_cast<char*>(dst.base);

  // Make a list of pointers to dims that we are going to copy.
  copy_dim* dims = reinterpret_cast<copy_dim*>(alloca(sizeof(copy_dim) * dst.rank));

  int rank = 0;
  for (std::size_t i = 0; i < dst.rank; ++i) {
    if (dst.dims[i].max() < dst.dims[i].min()) return;
    if (dst.dims[i].stride() == 0) {
      // Copying a broadcast to a broadcast is OK.
      assert(src.dims[i].stride() == 0);
      continue;
    } else {
      dims[rank].src_stride = src.dims[i].stride();
      dims[rank].dst_stride = dst.dims[i].stride();
      dims[rank].total_size = dst.dims[i].extent();
      compute_padding(src.dims[i].begin(), src.dims[i].end(), dst.dims[i], dims[rank]);
      src_base += src.dims[i].flat_offset_bytes(std::max(dst.dims[i].min(), src.dims[i].min()));
      dst_base += dst.dims[i].flat_offset_bytes(dst.dims[i].min());
      assert(dst.dims[rank].extent() <= dst.dims[rank].fold_factor());
      ++rank;
    }
  }

  rank = optimize_copy_dims(dims, rank);

  // Now we have an optimized set of dimensions to copy. Run the copy.
  copy(src_base, dst_base, dims, dst.elem_size, padding, rank - 1);
}

void pad(const dim* in_bounds, const raw_buffer& dst, const void* padding) {
  if (dst.rank == 0) {
    // The buffer is scalar.
    return;
  }

  char* dst_base = reinterpret_cast<char*>(dst.base);

  // Make a list of pointers to dims that we are going to pad.
  copy_dim* dims = reinterpret_cast<copy_dim*>(alloca(sizeof(copy_dim) * dst.rank));
  int rank = 0;
  for (std::size_t i = 0; i < dst.rank; ++i) {
    if (dst.dims[i].max() < dst.dims[i].min()) return;
    if (dst.dims[i].stride() == 0) continue;
    dst_base += dst.dims[i].flat_offset_bytes(dst.dims[i].min());
    assert(dst.dims[rank].extent() <= dst.dims[rank].fold_factor());
    dims[rank].src_stride = 0;
    dims[rank].dst_stride = dst.dims[i].stride();
    dims[rank].total_size = dst.dims[i].extent();
    compute_padding(in_bounds[i].begin(), in_bounds[i].end(), dst.dims[i], dims[rank]);
    ++rank;
  }

  rank = optimize_copy_dims(dims, rank);

  // Now we have an optimized set of dimensions to pad. Run the pad.
  copy(nullptr, dst_base, dims, dst.elem_size, padding, rank - 1);
}

void fill(const raw_buffer& dst, const void* value) {
  if (dst.rank == 0) {
    // The buffer is scalar.
    memcpy(dst.base, value, dst.elem_size);
    return;
  }

  char* dst_base = reinterpret_cast<char*>(dst.base);

  // Make a list of pointers to dims that we are going to copy.
  copy_dim* dims = reinterpret_cast<copy_dim*>(alloca(sizeof(copy_dim) * dst.rank));
  int rank = 0;
  for (std::size_t i = 0; i < dst.rank; ++i) {
    if (dst.dims[i].max() < dst.dims[i].min()) return;
    if (dst.dims[i].stride() == 0) continue;
    dst_base += dst.dims[i].flat_offset_bytes(dst.dims[i].min());
    assert(dst.dims[rank].extent() <= dst.dims[rank].fold_factor());
    dims[rank].dst_stride = dst.dims[i].stride();
    dims[rank].src_stride = 0;  // For optimize_copy_dims
    dims[rank].total_size = dst.dims[i].extent();

    dims[rank].pad_before = dims[rank].total_size;
    dims[rank].size = 0;
    dims[rank].pad_after = 0;
    rank++;
  }

  rank = optimize_copy_dims(dims, rank);

  fill(dst_base, dims, dst.elem_size, value, rank - 1);
}

}  // namespace slinky