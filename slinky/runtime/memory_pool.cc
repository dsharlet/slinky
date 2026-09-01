#include "slinky/runtime/memory_pool.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace slinky {

namespace {

// Stored immediately before the pointer returned to the caller, so `free` can find the underlying block.
struct block_header {
  void* block;
  std::size_t capacity;
};

block_header& header_of(void* ptr) { return reinterpret_cast<block_header*>(ptr)[-1]; }

// The capacity a block needs to serve `size` bytes at `alignment`, with room for the header.
std::size_t capacity_for(std::size_t size, std::size_t alignment) {
  return size + alignment - 1 + sizeof(block_header);
}

void* place(void* block, std::size_t capacity, std::size_t alignment) {
  uintptr_t ptr = reinterpret_cast<uintptr_t>(block) + sizeof(block_header);
  ptr = (ptr + alignment - 1) & ~static_cast<uintptr_t>(alignment - 1);
  void* result = reinterpret_cast<void*>(ptr);
  header_of(result) = {block, capacity};
  return result;
}

}  // namespace

memory_pool::~memory_pool() { assert(free_.empty()); }

void* memory_pool::allocate(std::size_t size, std::size_t alignment, function_ref<void*(std::size_t)> alloc_block,
    function_ref<void(void*)> free_block) {
  alignment = std::max(alignment, alignof(block_header));
  const std::size_t capacity = capacity_for(size, alignment);
  // Take the smallest retained block that can serve this request, but don't let a small request squat on a much
  // bigger block.
  auto best = free_.end();
  for (auto i = free_.begin(); i != free_.end(); ++i) {
    if (i->capacity >= capacity && i->capacity <= capacity * 2 &&
        (best == free_.end() || i->capacity < best->capacity)) {
      best = i;
    }
  }
  if (best != free_.end()) {
    void* block = best->block;
    const std::size_t block_capacity = best->capacity;
    retained_size_ -= block_capacity;
    *best = free_.back();
    free_.pop_back();
    return place(block, block_capacity, alignment);
  }
  // No retained block fits this request, so the sizes being asked for have changed. Retained blocks that also went
  // unused through the whole previous such episode are not part of the working set any more: release them, so their
  // pages can serve fresh allocations, instead of inflating the memory footprint. Blocks that are reused keep their
  // stamp fresh and are never evicted.
  ++miss_epoch_;
  for (auto i = free_.begin(); i != free_.end();) {
    if (i->freed_at + 2 <= miss_epoch_) {
      free_block(i->block);
      retained_size_ -= i->capacity;
      *i = free_.back();
      free_.pop_back();
    } else {
      ++i;
    }
  }
  void* block = alloc_block(capacity);
  if (!block) return nullptr;
  return place(block, capacity, alignment);
}

void memory_pool::free(void* ptr) {
  if (!ptr) return;
  const block_header header = header_of(ptr);
  free_.push_back({header.capacity, header.block, miss_epoch_});
  retained_size_ += header.capacity;
}

void memory_pool::trim(function_ref<void(void*)> free_block) {
  for (const retained_block& i : free_) {
    free_block(i.block);
  }
  free_.clear();
  retained_size_ = 0;
}

}  // namespace slinky
