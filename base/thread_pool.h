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

// This is an abstract base class for a thread pool used by slinky.
class thread_pool {
public:
  // A task is a set of work items indexed by an integer i, where a `body` is called for each i in the set.
  class task {
  public:
    virtual ~task() = default;

    // Work on the task. This returns when work on all items in the task have started, but may return before all items
    // are complete. Returns true if this call resulted in the loop being done (any subsequent calls to `run` will do no
    // work), but the loop may not be done.
    virtual bool run(function_ref<void(std::size_t)> body) = 0;

    // Returns true if all work is complete.
    virtual bool done() const = 0;
  };

  using task_body = std::function<void(std::size_t)>;
  using predicate_ref = function_ref<bool()>;

  virtual ~thread_pool() = default;

  virtual int thread_count() const = 0;

  // Enqueues a loop task over `n` work items `[0, n)`. The tasks queued in this thread pool are instances of `task`
  // plus a task body `t` that executes each iteration. Tasks are complete when all work items are done. Each thread
  // calling `t` uses its own instance of `t`.
  virtual std::shared_ptr<task> enqueue(
      std::size_t n, task_body t, int max_workers = std::numeric_limits<int>::max()) = 0;
  // Run the task on the current thread, and prevents tasks enqueued by `enqueue` from running recursively.
  // Does not return until the loop is complete.
  virtual void wait_for(task* l, task_body body) = 0;
  // Waits for `condition` to become true. While waiting, executes tasks on the queue.
  // The condition is executed atomically.
  virtual void wait_for(predicate_ref condition) = 0;
  // Run `t` on the calling thread, but atomically w.r.t. other `atomic_call` and `wait_for` conditions.
  virtual void atomic_call(function_ref<void()> t) = 0;

  // Enqueues a singleton task.
  template <typename Fn>
  std::shared_ptr<task> enqueue(Fn t) {
    // Make a dummy loop task with one iteration.
    return enqueue(1, [t = std::move(t)](std::size_t) { t(); });
  }

  void parallel_for(std::size_t n, task_body body, int max_workers = std::numeric_limits<int>::max());
};

}  // namespace slinky

#endif  // SLINKY_BASE_THREAD_POOL_H
