#ifndef SLINKY_RUNTIME_MEMORY_POOL_H
#define SLINKY_RUNTIME_MEMORY_POOL_H

#include <cstddef>
#include <vector>

#include "slinky/base/function_ref.h"

namespace slinky {

// A pool of heap blocks that are kept for reuse when freed instead of being returned to the system.
//
// A pipeline allocates and frees every internal buffer on every evaluation. The system allocator typically serves
// large allocations with fresh pages and returns them to the OS on free, so each evaluation pays for the system calls
// and, worse, for the page faults of first touching those pages, which land inside the kernels that write the buffers.
// A pool keeps freed blocks alive for the next allocation.
//
// The pool does not choose an allocator: blocks it cannot serve from its free list are allocated with the callback
// passed to `allocate`, and blocks it releases (evicted or trimmed) are freed with the callback passed to the
// corresponding call, so the pool composes with any allocator.
//
// A request is served from a retained block if the block is at least as large as the request and at most twice as
// large. When no retained block can serve a request, blocks that also sat unused through the whole previous such
// episode are released: they are no longer part of the working set, and holding them would only inflate the memory
// footprint.
//
// The pool is not thread safe and cannot be copied. Each `eval_context` owns one, including the per-worker context
// copies made for parallel loops, keeping every block on the thread that freed it without any synchronization.
class memory_pool {
public:
  memory_pool() = default;
  // The pool must be empty when destroyed: it cannot free blocks itself. `eval_context` trims its pool on
  // destruction.
  ~memory_pool();

  memory_pool(const memory_pool&) = delete;
  memory_pool& operator=(const memory_pool&) = delete;

  // Returns a pointer to at least `size` bytes aligned to `alignment` (a power of 2), or nullptr on failure. The
  // block comes from the free list when a retained block fits, and from `alloc_block` otherwise; `alloc_block` needs
  // to provide at least `alignof(std::max_align_t)` alignment. Retained blocks evicted as stale are released through
  // `free_block`.
  void* allocate(std::size_t size, std::size_t alignment, function_ref<void*(std::size_t)> alloc_block,
      function_ref<void(void*)> free_block);

  // Releases a pointer returned by `allocate`, retaining the block for reuse.
  void free(void* ptr);

  // Releases all retained blocks through `free_block`.
  void trim(function_ref<void(void*)> free_block);

  // The total size in bytes of the blocks currently retained for reuse.
  std::size_t retained_size() const { return retained_size_; }

private:
  struct retained_block {
    std::size_t capacity;
    void* block;
    // The value of `miss_epoch_` when this block was freed.
    std::size_t freed_at;
  };

  // Free blocks. This is a handful of entries, so linear searches beat a map and its per-node allocations.
  std::vector<retained_block> free_;
  std::size_t retained_size_ = 0;
  // Counts the allocations that no retained block could serve. Retained blocks that go unused across two of these
  // are released.
  std::size_t miss_epoch_ = 0;
};

}  // namespace slinky

#endif  // SLINKY_RUNTIME_MEMORY_POOL_H
