#include "slinky/runtime/memory_pool.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace slinky {

memory_pool::~memory_pool() { assert(free_.empty()); }

memory_pool::block memory_pool::remove(std::size_t i) {
  block result = {free_[i].block, free_[i].size};
  retained_size_ -= free_[i].size;
  free_[i] = free_.back();
  free_.pop_back();
  return result;
}

memory_pool::block memory_pool::allocate(std::size_t size, std::size_t alignment) {
  // Take the smallest retained block that can serve this request, but don't let a small request squat on a much
  // bigger block.
  std::size_t best = free_.size();
  for (std::size_t i = 0; i < free_.size(); ++i) {
    const retained_block& b = free_[i];
    if (b.size >= size && b.size <= size * 2 && (reinterpret_cast<uintptr_t>(b.block) & (alignment - 1)) == 0 &&
        (best == free_.size() || b.size < free_[best].size)) {
      best = i;
    }
  }
  if (best < free_.size()) {
    return remove(best);
  }
  ++miss_epoch_;
  // Partition the blocks that just became stale to the back of the list, so `evict_stale` can pop them without
  // scanning. We're already paying for a scan on this miss; this keeps each eviction constant time.
  std::size_t end = free_.size();
  for (std::size_t i = 0; i < end;) {
    if (free_[i].freed_at + 2 <= miss_epoch_) {
      std::swap(free_[i], free_[--end]);
    } else {
      ++i;
    }
  }
  return {};
}

void memory_pool::free(void* ptr, std::size_t size) {
  free_.push_back({size, ptr, miss_epoch_});
  retained_size_ += size;
}

memory_pool::block memory_pool::evict_stale() {
  if (!free_.empty() && free_.back().freed_at + 2 <= miss_epoch_) {
    return remove(free_.size() - 1);
  }
  return {};
}

memory_pool::block memory_pool::evict_any() {
  if (free_.empty()) return {};
  return remove(free_.size() - 1);
}

}  // namespace slinky
