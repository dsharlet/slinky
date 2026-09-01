#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "slinky/runtime/memory_pool.h"

namespace slinky {

namespace {

bool is_aligned(void* ptr, std::size_t alignment) { return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0; }

// Counts the blocks allocated and freed on behalf of the pool. `function_ref` invokes its callee as const.
struct counting_allocator {
  mutable int allocs = 0;
  mutable int frees = 0;

  void* alloc(std::size_t size) const {
    ++allocs;
    return std::malloc(size);
  }
  void free(void* block) const {
    ++frees;
    std::free(block);
  }
};

}  // namespace

TEST(memory_pool, blocks_are_reused) {
  counting_allocator heap;
  auto alloc_cb = [&](std::size_t size) { return heap.alloc(size); };
  auto free_cb = [&](void* block) { heap.free(block); };
  memory_pool pool;

  void* a = pool.allocate(4096, 16, alloc_cb, free_cb);
  ASSERT_NE(a, nullptr);
  std::memset(a, 0, 4096);
  pool.free(a);
  ASSERT_GE(pool.retained_size(), 4096);
  ASSERT_EQ(heap.allocs, 1);

  // An identical request gets the block back without allocating.
  void* b = pool.allocate(4096, 16, alloc_cb, free_cb);
  ASSERT_EQ(a, b);
  ASSERT_EQ(pool.retained_size(), 0);
  ASSERT_EQ(heap.allocs, 1);
  pool.free(b);

  // A slightly smaller request gets the same block.
  void* c = pool.allocate(3000, 16, alloc_cb, free_cb);
  ASSERT_EQ(a, c);
  ASSERT_EQ(heap.allocs, 1);
  pool.free(c);

  // A much smaller request does not squat on the big block.
  void* d = pool.allocate(1100, 16, alloc_cb, free_cb);
  ASSERT_NE(a, d);
  ASSERT_GE(pool.retained_size(), 4096);
  ASSERT_EQ(heap.allocs, 2);
  pool.free(d);

  pool.trim(free_cb);
  ASSERT_EQ(pool.retained_size(), 0);
  ASSERT_EQ(heap.frees, heap.allocs);
}

TEST(memory_pool, alignment) {
  counting_allocator heap;
  auto alloc_cb = [&](std::size_t size) { return heap.alloc(size); };
  auto free_cb = [&](void* block) { heap.free(block); };
  memory_pool pool;

  for (std::size_t alignment : {16, 64, 256, 4096}) {
    for (std::size_t size : {1, 100, 1000, 5000, 100000}) {
      void* a = pool.allocate(size, alignment, alloc_cb, free_cb);
      ASSERT_NE(a, nullptr);
      ASSERT_TRUE(is_aligned(a, alignment));
      std::memset(a, 0, size);
      pool.free(a);
    }
  }
  // Reused blocks are re-aligned to the new request.
  void* a = pool.allocate(4096, 16, alloc_cb, free_cb);
  pool.free(a);
  void* b = pool.allocate(4096, 4096, alloc_cb, free_cb);
  ASSERT_TRUE(is_aligned(b, 4096));
  std::memset(b, 0, 4096);
  pool.free(b);
  pool.trim(free_cb);
}

TEST(memory_pool, free_null) {
  memory_pool pool;
  pool.free(nullptr);
}

TEST(memory_pool, stale_blocks_are_evicted) {
  counting_allocator heap;
  auto alloc_cb = [&](std::size_t size) { return heap.alloc(size); };
  auto free_cb = [&](void* block) { heap.free(block); };
  memory_pool pool;

  void* a = pool.allocate(4096, 16, alloc_cb, free_cb);
  pool.free(a);

  // The first mismatched request leaves the unused block retained.
  pool.free(pool.allocate(100, 16, alloc_cb, free_cb));
  ASSERT_GE(pool.retained_size(), 4096);
  ASSERT_EQ(heap.frees, 0);

  // A block that goes unused through a second mismatch is released; the recently freed one is kept.
  void* b = pool.allocate(100000, 16, alloc_cb, free_cb);
  ASSERT_EQ(heap.frees, 1);
  ASSERT_LT(pool.retained_size(), 4096);
  ASSERT_GT(pool.retained_size(), 0);
  pool.free(b);

  pool.trim(free_cb);
  ASSERT_EQ(heap.frees, heap.allocs);
}

TEST(memory_pool, reused_blocks_are_not_evicted) {
  counting_allocator heap;
  auto alloc_cb = [&](std::size_t size) { return heap.alloc(size); };
  auto free_cb = [&](void* block) { heap.free(block); };
  memory_pool pool;

  void* a = pool.allocate(4096, 16, alloc_cb, free_cb);
  pool.free(a);

  // A block that keeps being reused stays retained through any number of mismatched requests.
  for (int i = 0; i < 10; ++i) {
    // A request that no retained block can serve.
    pool.free(pool.allocate(10000 + i * 25000, 16, alloc_cb, free_cb));
    void* b = pool.allocate(4096, 16, alloc_cb, free_cb);
    ASSERT_EQ(a, b);
    pool.free(b);
  }

  pool.trim(free_cb);
  ASSERT_EQ(heap.frees, heap.allocs);
}

}  // namespace slinky
