#ifndef SLINKY_TEST_THREAD_POOL_H
#define SLINKY_TEST_THREAD_POOL_H

#include <deque>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>

// This file provides a simple thread pool that can be used for testing parallel work.

namespace slinky {

class thread_pool {
public:
  using task = std::function<void()>;

private:
  std::vector<std::thread> workers_;
  std::atomic<bool> run_;

  std::deque<task> task_queue_;
  std::mutex mutex_;
  std::condition_variable cv_;

public:
  thread_pool(int workers = 3);
  ~thread_pool();

  int thread_count() const { return workers_.size(); }

  void enqueue(int n, const task& t);
  void enqueue(task t);
  void work_on_tasks(std::function<bool()> while_true);
};

}  // namespace slinky

#endif  // SLINKY_TEST_THREAD_POOL_H
