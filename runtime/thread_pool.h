#ifndef SLINKY_RUNTIME_TEST_THREAD_POOL_H
#define SLINKY_RUNTIME_TEST_THREAD_POOL_H

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

namespace slinky {

// This implements a simple thread pool that maps easily to the eval_context thread pool interface.
// It is not directly used by anything except for testing.
class thread_pool {
public:
  using task = std::function<void()>;
  using task_id = std::size_t;

private:
  std::vector<std::thread> workers_;
  std::atomic<bool> stop_;

  task_id next_task_id_ = 1;
  using queued_task = std::tuple<int, task, task_id>;
  std::deque<queued_task> task_queue_;
  std::mutex mutex_;
  std::condition_variable cv_;

  bool dequeue(task& t, std::vector<task_id>& task_stack);

public:
  thread_pool(int workers = 3);
  ~thread_pool();

  int thread_count() const { return workers_.size(); }

  // Enqueues `n` copies of task `t` on the thread pool queue. This guarantees that `t` will not
  // be run recursively on the same thread while in `wait_for`.
  void enqueue(int n, task t); 
  void enqueue(task t);
  // Waits for `condition` to become true. While waiting, executes tasks on the queue.
  // The condition is executed atomically.
  void wait_for(const std::function<bool()>& condition);
  // Run `t` on the calling thread, but atomically w.r.t. other `atomic_call` and `wait_for` conditions.
  void atomic_call(const task& t);
};

}  // namespace slinky

#endif  // SLINKY_RUNTIME_TEST_THREAD_POOL_H
