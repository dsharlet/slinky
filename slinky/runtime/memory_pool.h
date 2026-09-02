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
// The pool does not allocate or free memory itself: `free` takes ownership of a block the caller allocated, and
// blocks leave the pool either through `allocate` (reuse) or through `evict_stale`/`evict_any`, which hand them back
// to the caller to release. When no retained block can
// serve a request, blocks that also sat unused through the whole previous such episode become stale: they are no
// longer part of the working set, and holding them would only inflate the memory footprint.
//
// The pool is not thread safe and cannot be copied. Each `eval_context` owns one, including the per-worker context
// copies made for parallel loops, keeping every block on the thread that freed it without any synchronization.
class memory_pool {
public:
  memory_pool() = default;
  // The pool must be empty when destroyed: it cannot free blocks itself. `eval_context` empties its pool on
  // destruction.
  ~memory_pool();

  memory_pool(const memory_pool&) = delete;
  memory_pool& operator=(const memory_pool&) = delete;

  // Returns a retained block of at least `size` bytes whose address is aligned to `alignment` (a power of 2), and
  // removes it from the pool, or nullptr if no retained block fits. The caller should allocate normally in the
  // latter case, and release the blocks `evict_stale` produces as a result.
  void* allocate(std::size_t size, std::size_t alignment);

  // Adds `block` of `size` bytes to the pool, taking ownership of it.
  void free(void* block, std::size_t size);

  // Removes and returns a retained block that is no longer part of the working set, or nullptr if there is none.
  // The caller owns releasing it. Constant time when called right after a failed `allocate`.
  void* evict_stale();

  // Removes and returns any retained block, or nullptr if the pool is empty. The caller owns releasing it.
  void* evict_any();

  // The total size in bytes of the blocks currently retained for reuse.
  std::size_t retained_size() const { return retained_size_; }

private:
  struct retained_block {
    std::size_t size;
    void* block;
    // The value of `miss_epoch_` when this block was freed.
    std::size_t freed_at;
  };

  void* remove(std::size_t i);

  // Free blocks. This is a handful of entries, so linear searches beat a map and its per-node allocations.
  std::vector<retained_block> free_;
  std::size_t retained_size_ = 0;
  // Counts the `allocate` calls that no retained block could serve. Retained blocks that go unused across two of
  // these are stale.
  std::size_t miss_epoch_ = 0;
};

}  // namespace slinky

#endif  // SLINKY_RUNTIME_MEMORY_POOL_H
