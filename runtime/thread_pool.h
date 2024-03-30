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

  // Enqueues `n` copies of task `t` on the thread pool queue.
  void enqueue(int n, const task& t);
  void enqueue(task t);
  // Waits for `condition` to become true. While waiting, executes tasks on the queue.
  void wait_for(std::function<bool()> condition);
};

}  // namespace slinky

#endif  // SLINKY_RUNTIME_TEST_THREAD_POOL_H
