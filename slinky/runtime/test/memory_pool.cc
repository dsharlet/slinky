#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>

#include "slinky/base/util.h"
#include "slinky/runtime/memory_pool.h"

namespace slinky {

namespace {

// Drain the pool, freeing the blocks it retained.
void empty(memory_pool& pool) {
  for (memory_pool::block b = pool.evict_any(); b.ptr; b = pool.evict_any()) {
    aligned_free(b.ptr);
  }
}

}  // namespace

TEST(memory_pool, blocks_are_reused) {
  memory_pool pool;

  // An empty pool serves nothing.
  ASSERT_EQ(pool.allocate(4096, 16).ptr, nullptr);

  void* a = aligned_alloc(16, 4096);
  pool.free(a, 4096);
  ASSERT_EQ(pool.retained_size(), 4096);

  // An exact fit gets the block back.
  ASSERT_EQ(pool.allocate(4096, 16).ptr, a);
  ASSERT_EQ(pool.retained_size(), 0);
  pool.free(a, 4096);

  // A slightly smaller request gets the same block, reported with the block's own size so it can be freed with it.
  memory_pool::block b = pool.allocate(3000, 16);
  ASSERT_EQ(b.ptr, a);
  ASSERT_EQ(b.size, 4096);
  pool.free(b.ptr, b.size);

  // A much smaller request does not squat on the big block.
  ASSERT_EQ(pool.allocate(1100, 16).ptr, nullptr);
  ASSERT_EQ(pool.retained_size(), 4096);

  // A bigger request cannot be served by the block either.
  ASSERT_EQ(pool.allocate(5000, 16).ptr, nullptr);

  empty(pool);
  ASSERT_EQ(pool.retained_size(), 0);
}

TEST(memory_pool, best_fit) {
  memory_pool pool;
  void* small = aligned_alloc(16, 3000);
  void* big = aligned_alloc(16, 4096);
  pool.free(big, 4096);
  pool.free(small, 3000);

  // The smallest block that fits is taken, not the first retained.
  ASSERT_EQ(pool.allocate(2500, 16).ptr, small);
  aligned_free(small);
  empty(pool);
}

TEST(memory_pool, alignment_is_respected) {
  memory_pool pool;
  void* a = aligned_alloc(64, 4096);
  ASSERT_TRUE(a);
  pool.free(a, 4096);

  // The block can only serve requests its address is aligned for.
  const std::size_t alignment = (reinterpret_cast<uintptr_t>(a) & 4095) == 0 ? 8192 : 4096;
  ASSERT_EQ(pool.allocate(4096, alignment).ptr, nullptr);
  ASSERT_EQ(pool.allocate(4096, 64).ptr, a);
  aligned_free(a);
  empty(pool);
}

TEST(memory_pool, stale_blocks_are_evicted) {
  memory_pool pool;
  void* a = aligned_alloc(16, 4096);
  pool.free(a, 4096);

  // The first mismatched request does not make the block stale.
  ASSERT_EQ(pool.allocate(100, 16).ptr, nullptr);
  ASSERT_EQ(pool.evict_stale().ptr, nullptr);
  void* b = aligned_alloc(16, 100);
  pool.free(b, 100);

  // A block that goes unused through a second mismatch is stale; the recently freed one is not.
  ASSERT_EQ(pool.allocate(100000, 16).ptr, nullptr);
  ASSERT_EQ(pool.evict_stale().ptr, a);
  ASSERT_EQ(pool.evict_stale().ptr, nullptr);
  ASSERT_EQ(pool.retained_size(), 100);
  aligned_free(a);
  empty(pool);
}

TEST(memory_pool, reused_blocks_are_not_evicted) {
  memory_pool pool;
  void* a = aligned_alloc(16, 4096);
  pool.free(a, 4096);

  // A block that keeps being reused stays retained through any number of mismatched requests.
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(pool.allocate(100000 + i, 16).ptr, nullptr);
    ASSERT_EQ(pool.evict_stale().ptr, nullptr);
    ASSERT_EQ(pool.allocate(4096, 16).ptr, a);
    pool.free(a, 4096);
  }
  empty(pool);
}

}  // namespace slinky
