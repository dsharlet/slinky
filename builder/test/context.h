#ifndef SLINKY_BUILDER_TEST_CONTEXT_H
#define SLINKY_BUILDER_TEST_CONTEXT_H

#include <mutex>
#include <string>
#include <vector>

#include "runtime/evaluate.h"

namespace slinky {

void setup_tracing(eval_context& ctx, const std::string& filename);

struct memory_info {
  std::atomic<index_t> live_count = 0;
  std::atomic<index_t> live_size = 0;
  std::mutex m;
  std::vector<index_t> allocs;

  void track_allocate(index_t size) {
    live_count += 1;
    live_size += size;
    std::unique_lock l(m);
    allocs.push_back(size);
  }

  void track_free(index_t size) {
    live_count -= 1;
    live_size -= size;
  }
};

class test_context : public eval_context {
public:
  memory_info heap;
  int copy_calls = 0;
  int copy_elements = 0;
  int pad_calls = 0;

  test_context();
};

}  // namespace slinky

#endif  // SLINKY_BUILDER_TEST_CONTEXT_H