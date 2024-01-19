#ifndef SLINKY_TEST_THREAD_POOL_H
#define SLINKY_TEST_THREAD_POOL_H

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

// This file provides a simple thread pool that can be used for testing parallel work.
// Note that this is in the test folder, not the src folder. Slinky does not provide a
// thread pool, to support parallelism in slinky pipelines, you must provide implementations of
// `eval_context::enqueue_many`, `eval_context::enqueue_one`, and `eval_context::wait_for`.

namespace slinky {

class thread_pool {
public:
  using task = std::function<void()>;

private:
  std::vector<std::thread> workers_;
  std::atomic<bool> stop_;

  std::deque<task> task_queue_;
  std::mutex mutex_;
  std::condition_variable cv_;

public:
  thread_pool(int workers = 3);
  ~thread_pool();

  int thread_count() const { return workers_.size(); }

  // Enqueues `n` copies of task `t` on the thread pool queue.
  void enqueue(int n, const task& t);
  void enqueue(task t);
  // Waits for `condition` to become true. While waiting, executes tasks on the queue.
  void wait_for(std::function<bool()> condition);
};

}  // namespace slinky

#endif  // SLINKY_TEST_THREAD_POOL_H
