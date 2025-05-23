#ifndef SLINKY_BASE_THREAD_POOL_H
#define SLINKY_BASE_THREAD_POOL_H

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <deque>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>

#include "base/function_ref.h"

namespace slinky {

class task {
public:
  // Work on the loop. This returns when work on all items in the loop has started, but may return before all items are
  // complete. Returns true if this call resulted in the loop being done (any subsequent calls to `run` will do no
  // work), but the loop may not be done.
  virtual bool run(function_ref<void(std::size_t)> body) = 0;

  // Returns true if all work is complete.
  virtual bool done() const = 0;
};

// This implements a simple thread pool that maps easily to the eval_context thread pool interface.
// It is not directly used by anything except for testing.
class thread_pool {
public:
  using task_body = std::function<void(std::size_t)>;
  using predicate_ref = function_ref<bool()>;

  virtual ~thread_pool() = default;

  virtual int thread_count() const = 0;

  // Enqueues a loop task over `n` iterations. The tasks queued in this thread pool are instances of `task`
  // plus a task body `t` that executes each iteration. Tasks are complete when all `n` of the iterations are done.
  virtual std::shared_ptr<task> enqueue(
      std::size_t n, task_body t, int max_workers = std::numeric_limits<int>::max()) = 0;
  // Run the task on the current thread, and prevents tasks enqueued by `enqueue` from running recursively.
  // Does not return until the loop is complete.
  virtual void run(task* l, task_body body) = 0;
  // Waits for `condition` to become true. While waiting, executes tasks on the queue.
  // The condition is executed atomically.
  virtual void wait_for(predicate_ref condition) = 0;
  // Run `t` on the calling thread, but atomically w.r.t. other `atomic_call` and `wait_for` conditions.
  virtual void atomic_call(function_ref<void()> t) = 0;

  // Enqueues a singleton task.
  template <typename Fn>
  void enqueue(Fn t) {
    // Make a dummy loop task with one iteration.
    enqueue(1, [t = std::move(t)](std::size_t) { t(); });
  }

  template <typename Fn>
  void parallel_for(std::size_t n, Fn&& body, int max_workers = std::numeric_limits<int>::max()) {
    if (n == 0) {
      return;
    } else if (n == 1) {
      body(0);
      return;
    }
    max_workers = std::min(max_workers - 1, thread_count());
    if (max_workers == 0) {
      // We aren't going to get any worker threads, just run the loop.
      for (std::size_t i = 0; i < n; ++i) {
        body(i);
      }
    } else {
      auto loop = enqueue(n, body, max_workers);
      // Working on the loop here guarantees forward progress on the loop even if no threads in the thread pool are
      // available.
      run(loop.get(), std::move(body));
    }
  }
};

}  // namespace slinky

#endif  // SLINKY_BASE_THREAD_POOL_H
