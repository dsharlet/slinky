#include "base/thread_pool.h"

namespace slinky {

void thread_pool::parallel_for(std::size_t n, task_body body, int max_workers) {
  if (n == 0) {
    return;
  } else if (n == 1) {
    body(0);
    return;
  }
  if (max_workers == 1 || thread_count() == 0) {
    // We aren't going to get any worker threads, just run the loop.
    for (std::size_t i = 0; i < n; ++i) {
      body(i);
    }
  } else {
    auto loop = enqueue(n, std::move(body), max_workers);
    // Working on the loop here guarantees forward progress on the loop even if no threads in the thread pool are
    // available.
    wait_for(loop.get());
  }
}

}  // namespace slinky
