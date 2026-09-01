#ifndef SLINKY_BASE_ARENA_H
#define SLINKY_BASE_ARENA_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <new>

#include "slinky/base/ref_count.h"

namespace slinky {

// Allocates memory incrementally from a block of memory owned by this object.
class arena : public ref_counted<arena> {
  std::size_t next_;

  arena() : next_(sizeof(arena)) {}

public:
  static constexpr std::size_t block_size = 4096;

  // A block is aligned to `block_size`, so it can align anything it holds up to this much.
  static constexpr std::size_t max_alignment = block_size;

  static slinky::ref_count<arena> make() {
    return slinky::ref_count<arena>(new (::operator new(block_size, std::align_val_t(block_size))) arena());
  }

  // Because blocks are aligned to `block_size`, the arena an object lives in is the beginning of the block containing
  // it. This lets objects find the arena that owns them without storing a pointer to it.
  static arena* from(const void* p) {
    return reinterpret_cast<arena*>(reinterpret_cast<std::uintptr_t>(p) & ~(block_size - 1));
  }

  // Returns memory for an object, padding to align it, or null if there is not enough space in this block.
  void* allocate(std::size_t size, std::size_t alignment) {
    assert(alignment <= max_alignment);
    std::size_t offset = (next_ + alignment - 1) & ~(alignment - 1);
    if (offset + size > block_size) {
      // Out of space
      return nullptr;
    }
    next_ = offset + size;
    return reinterpret_cast<char*>(this) + offset;
  }

  static void destroy(arena* a) {
    a->~arena();
    ::operator delete(a, std::align_val_t(block_size));
  }
};

}  // namespace slinky

#endif  // SLINKY_BASE_ARENA_H
