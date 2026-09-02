#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#ifdef _MSC_VER
#include <malloc.h>
#endif

#include "slinky/runtime/memory_pool.h"

namespace slinky {

namespace {

// Drain the pool, freeing the blocks it retained.
void empty(memory_pool& pool) {
  while (void* block = pool.evict_any()) {
    std::free(block);
  }
}

}  // namespace

TEST(memory_pool, blocks_are_reused) {
  memory_pool pool;

  // An empty pool serves nothing.
  ASSERT_EQ(pool.allocate(4096, 16), nullptr);

  void* a = std::malloc(4096);
  pool.free(a, 4096);
  ASSERT_EQ(pool.retained_size(), 4096);

  // An exact fit gets the block back.
  ASSERT_EQ(pool.allocate(4096, 16), a);
  ASSERT_EQ(pool.retained_size(), 0);
  pool.free(a, 4096);

  // A slightly smaller request gets the same block.
  ASSERT_EQ(pool.allocate(3000, 16), a);
  pool.free(a, 4096);

  // A much smaller request does not squat on the big block.
  ASSERT_EQ(pool.allocate(1100, 16), nullptr);
  ASSERT_EQ(pool.retained_size(), 4096);

  // A bigger request cannot be served by the block either.
  ASSERT_EQ(pool.allocate(5000, 16), nullptr);

  empty(pool);
  ASSERT_EQ(pool.retained_size(), 0);
}

TEST(memory_pool, best_fit) {
  memory_pool pool;
  void* small = std::malloc(3000);
  void* big = std::malloc(4096);
  pool.free(big, 4096);
  pool.free(small, 3000);

  // The smallest block that fits is taken, not the first retained.
  ASSERT_EQ(pool.allocate(2500, 16), small);
  std::free(small);
  empty(pool);
}

TEST(memory_pool, alignment_is_respected) {
  memory_pool pool;
#ifdef _MSC_VER
  void* a = _aligned_malloc(4096, 64);
#else
  void* a = std::aligned_alloc(64, 4096);
#endif
  ASSERT_TRUE(a);
  pool.free(a, 4096);

  // The block can only serve requests its address is aligned for.
  const std::size_t alignment = (reinterpret_cast<uintptr_t>(a) & 4095) == 0 ? 8192 : 4096;
  ASSERT_EQ(pool.allocate(4096, alignment), nullptr);
  ASSERT_EQ(pool.allocate(4096, 64), a);
#ifdef _MSC_VER
  _aligned_free(a);
#else
  std::free(a);
#endif
  empty(pool);
}

TEST(memory_pool, stale_blocks_are_evicted) {
  memory_pool pool;
  void* a = std::malloc(4096);
  pool.free(a, 4096);

  // The first mismatched request does not make the block stale.
  ASSERT_EQ(pool.allocate(100, 16), nullptr);
  ASSERT_EQ(pool.evict_stale(), nullptr);
  void* b = std::malloc(100);
  pool.free(b, 100);

  // A block that goes unused through a second mismatch is stale; the recently freed one is not.
  ASSERT_EQ(pool.allocate(100000, 16), nullptr);
  ASSERT_EQ(pool.evict_stale(), a);
  ASSERT_EQ(pool.evict_stale(), nullptr);
  ASSERT_EQ(pool.retained_size(), 100);
  std::free(a);
  empty(pool);
}

TEST(memory_pool, reused_blocks_are_not_evicted) {
  memory_pool pool;
  void* a = std::malloc(4096);
  pool.free(a, 4096);

  // A block that keeps being reused stays retained through any number of mismatched requests.
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(pool.allocate(100000 + i, 16), nullptr);
    ASSERT_EQ(pool.evict_stale(), nullptr);
    ASSERT_EQ(pool.allocate(4096, 16), a);
    pool.free(a, 4096);
  }
  empty(pool);
}

}  // namespace slinky
