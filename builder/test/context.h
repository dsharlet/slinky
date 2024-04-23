#ifndef SLINKY_BUILDER_TEST_CONTEXT_H
#define SLINKY_BUILDER_TEST_CONTEXT_H

#include <string>

#include "runtime/evaluate.h"

namespace slinky {

void setup_tracing(eval_context& ctx, const std::string& filename);

struct memory_info {
  std::atomic<index_t> live_count = 0;
  std::atomic<index_t> live_size = 0;
  std::atomic<index_t> total_count = 0;
  std::atomic<index_t> total_size = 0;

  void track_allocate(index_t size) {
    live_count += 1;
    live_size += size;
    total_count += 1;
    total_size += size;
  }

  void track_free(index_t size) {
    live_count -= 1;
    live_size -= size;
  }
};

class test_context : public eval_context {
public:
  memory_info heap;

  test_context();
};

}  // namespace slinky

#endif  // SLINKY_BUILDER_TEST_CONTEXT_H