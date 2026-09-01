#ifndef SLINKY_RUNTIME_MEMORY_POOL_H
#define SLINKY_RUNTIME_MEMORY_POOL_H

#include <cstddef>
#include <vector>

namespace slinky {

// A pool of heap blocks that are kept for reuse when freed instead of being returned to the system.
//
// A pipeline allocates and frees every internal buffer on every evaluation. The system allocator typically serves
// large allocations with fresh pages and returns them to the OS on free, so each evaluation pays for the system calls
// and, worse, for the page faults of first touching those pages, which land inside the kernels that write the buffers.
// A pool keeps freed blocks alive for the next allocation.
//
// A request is served from a retained block if the block is at least as large as the request and at most twice as
// large. When no retained block can serve a request, blocks that also sat unused through the whole previous such
// episode are returned to the system: they are no longer part of the working set, and holding them would only inflate
// the memory footprint. `trim` returns all retained blocks to the system.
//
// The pool is not thread safe, and copying a pool does not copy its retained blocks: each copy starts empty. This is
// so each `eval_context` can own a pool, including the per-worker context copies made for parallel loops, keeping
// every block on the thread that freed it without any synchronization.
class memory_pool {
public:
  memory_pool() = default;
  ~memory_pool();

  memory_pool(const memory_pool&) {}
  memory_pool& operator=(const memory_pool&) {
    trim();
    return *this;
  }

  // Returns a pointer to at least `size` bytes aligned to `alignment` (a power of 2), or nullptr on failure.
  void* allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t));
  // Releases a pointer returned by `allocate` of any `memory_pool` instance, retaining the block for reuse.
  void free(void* ptr);

  // Returns all retained blocks to the system.
  void trim();

  // The total size in bytes of the blocks currently retained for reuse.
  std::size_t retained_size() const;

private:
  struct retained_block {
    std::size_t capacity;
    void* block;
    // The value of `miss_epoch_` when this block was freed.
    std::size_t freed_at;
  };

  // Free blocks.
  std::vector<retained_block> free_;
  // Counts the allocations that no retained block could serve. Retained blocks that go unused across two of these
  // are returned to the system.
  std::size_t miss_epoch_ = 0;
};

}  // namespace slinky

#endif  // SLINKY_RUNTIME_MEMORY_POOL_H
