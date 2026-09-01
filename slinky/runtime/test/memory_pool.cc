#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include "slinky/runtime/memory_pool.h"

namespace slinky {

namespace {

bool is_aligned(void* ptr, std::size_t alignment) { return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0; }

}  // namespace

TEST(memory_pool, allocations_are_reused) {
  memory_pool pool;
  void* a = pool.allocate(4096);
  ASSERT_NE(a, nullptr);
  std::memset(a, 0, 4096);
  pool.free(a);
  ASSERT_GE(pool.retained_size(), 4096);

  void* b = pool.allocate(4096);
  ASSERT_EQ(a, b);
  ASSERT_EQ(pool.retained_size(), 0);
  pool.free(b);

  // A slightly smaller request gets the same block.
  void* c = pool.allocate(3000);
  ASSERT_EQ(a, c);
  pool.free(c);

  // A much smaller request does not squat on the big block.
  void* d = pool.allocate(1100);
  ASSERT_NE(a, d);
  ASSERT_GE(pool.retained_size(), 4096);
  pool.free(d);

  pool.trim();
  ASSERT_EQ(pool.retained_size(), 0);
}

TEST(memory_pool, stale_blocks_are_evicted) {
  memory_pool pool;
  void* a = pool.allocate(4096);
  std::memset(a, 0, 4096);
  pool.free(a);

  // The first mismatched request leaves the unused block retained.
  pool.free(pool.allocate(100));
  ASSERT_GE(pool.retained_size(), 4096);

  // A block that goes unused through a second mismatch is returned to the system; the recently freed one is kept.
  void* b = pool.allocate(100000);
  ASSERT_LT(pool.retained_size(), 4096);
  ASSERT_GT(pool.retained_size(), 0);
  pool.free(b);
}

TEST(memory_pool, alignment) {
  memory_pool pool;
  for (std::size_t alignment : {16, 64, 256, 4096}) {
    for (std::size_t size : {1, 100, 1000, 5000, 100000}) {
      void* a = pool.allocate(size, alignment);
      ASSERT_NE(a, nullptr);
      ASSERT_TRUE(is_aligned(a, alignment));
      std::memset(a, 0, size);
      pool.free(a);
    }
  }
  // Reused blocks are re-aligned to the new request.
  void* a = pool.allocate(4096, 16);
  pool.free(a);
  void* b = pool.allocate(4096, 4096);
  ASSERT_TRUE(is_aligned(b, 4096));
  std::memset(b, 0, 4096);
  pool.free(b);
}

TEST(memory_pool, free_null) {
  memory_pool pool;
  pool.free(nullptr);
}

TEST(memory_pool, copies_start_empty) {
  memory_pool pool;
  void* a = pool.allocate(4096);
  pool.free(a);
  ASSERT_GE(pool.retained_size(), 4096);

  memory_pool copy = pool;
  ASSERT_EQ(copy.retained_size(), 0);
  // The original still has its block.
  ASSERT_GE(pool.retained_size(), 4096);

  // Assigning discards what the destination held.
  copy.free(copy.allocate(1024));
  ASSERT_GT(copy.retained_size(), 0);
  copy = pool;
  ASSERT_EQ(copy.retained_size(), 0);
  ASSERT_GE(pool.retained_size(), 4096);
}

TEST(memory_pool, blocks_can_move_between_pools) {
  // A block allocated from one pool may be freed into another (e.g. an early free inside a worker context).
  memory_pool a, b;
  void* p = a.allocate(4096);
  b.free(p);
  ASSERT_EQ(a.retained_size(), 0);
  ASSERT_GE(b.retained_size(), 4096);
}

}  // namespace slinky
