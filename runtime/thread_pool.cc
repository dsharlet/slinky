#include "runtime/thread_pool.h"

#include <cassert>
#include <functional>
#include <mutex>
#include <thread>
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

thread_pool::task thread_pool::dequeue() {
  auto& next = task_queue_.front();
  if (next.first == 1) {
    task t = std::move(next.second);
    task_queue_.pop_front();
    return t;
  } else {
    assert(next.first > 1);
    next.first -= 1;
    return next.second;
  }
}

void thread_pool::wait_for(std::function<bool()> condition) {
  std::unique_lock l(mutex_);
  while (!condition()) {
    if (!task_queue_.empty()) {
      task t = dequeue();
      l.unlock();
      task();
      l.lock();
      // This is pretty inefficient. It is here to wake up threads that are waiting for a condition to become true, that
      // may have become true due to the task completing. There may be a better way to do this.
      cv_.notify_all();
    } else if (!stop_) {
      cv_.wait(l);
    }
  }
}

void thread_pool::enqueue(int n, const task& t) {
  std::unique_lock l(mutex_);
  task_queue_.push_back({n, t});
  cv_.notify_all();
}

void thread_pool::enqueue(task t) {
  std::unique_lock l(mutex_);
  task_queue_.push_back({1, std::move(t)});
  cv_.notify_one();
}

}  // namespace slinky
