#ifndef SLINKY_BUILDER_TEST_CONTEXT_H
#define SLINKY_BUILDER_TEST_CONTEXT_H

#include <cassert>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "slinky/runtime/evaluate.h"

namespace slinky {

void setup_tracing(eval_config& config, const std::string& filename);

struct memory_info {
  std::atomic<index_t> live_count = 0;
  std::atomic<index_t> live_size = 0;
  std::mutex m;
  std::vector<index_t> allocs;
  // The sizes of the live allocations, so `track_free` only needs the pointer.
  std::map<void*, index_t> live_allocs;

  void track_allocate(void* ptr, index_t size) {
    live_count += 1;
    live_size += size;
    std::unique_lock l(m);
    allocs.push_back(size);
    live_allocs[ptr] = size;
  }

  void track_free(void* ptr) {
    index_t size;
    {
      std::unique_lock l(m);
      auto i = live_allocs.find(ptr);
      assert(i != live_allocs.end());
      size = i->second;
      live_allocs.erase(i);
    }
    live_count -= 1;
    live_size -= size;
  }
};

class test_context : public eval_context {
public:
  memory_info heap;
  int copy_calls = 0;
  int copy_elements = 0;

  eval_config config;

  copy_stmt::callable copy;

  test_context();
};

}  // namespace slinky

#endif  // SLINKY_BUILDER_TEST_CONTEXT_H