#include "runtime/thread_pool.h"

#include <thread>
#include <functional>
#include <mutex>
#include <vector>

namespace slinky {

thread_pool::thread_pool(int workers) : stop_(false) {
  auto worker = [this]() { wait_for([this]() -> bool { return stop_; }); };
  for (int i = 0; i < workers; ++i) {
    workers_.push_back(std::thread(worker));
  }
}

thread_pool::~thread_pool() {
  stop_ = true;
  cv_.notify_all();
  for (std::thread& i : workers_) {
    i.join();
  }
}

void thread_pool::wait_for(std::function<bool()> condition) {
  std::unique_lock l(mutex_);
  while (!condition()) {
    if (!task_queue_.empty()) {
      auto task = std::move(task_queue_.front());
      task_queue_.pop_front();
      l.unlock();
      task();
      l.lock();
      // This is pretty inefficient. It is here to wake up threads that are waiting for a condition to become true, that
      // may have become true due to the task completing. There may be a better way to do this.
      cv_.notify_all();
    } else {
      cv_.wait(l);
    }
  }
}

void thread_pool::enqueue(int n, const task& t) {
  std::unique_lock l(mutex_);
  task_queue_.insert(task_queue_.end(), n, t);
  cv_.notify_all();
}

void thread_pool::enqueue(task t) {
  std::unique_lock l(mutex_);
  task_queue_.push_back(std::move(t));
  cv_.notify_one();
}

}  // namespace slinky
