#include "thread_pool.h"

namespace slinky {

thread_pool::thread_pool(int workers) : run_(true) {
  auto worker = [this]() { work_on_tasks([this]() -> bool { return run_; }); };
  for (int i = 0; i < workers; ++i) {
    workers_.emplace_back(worker);
  }
}

thread_pool::~thread_pool() {
  run_ = false;
  cv_.notify_all();
  for (std::thread& i : workers_) {
    i.join();
  }
}

void thread_pool::work_on_tasks(std::function<bool()> while_true) {
  std::unique_lock l(mutex_);
  while (while_true()) {
    if (!task_queue_.empty()) {
      auto task = std::move(task_queue_.front());
      task_queue_.pop_front();
      l.unlock();
      task();
      l.lock();
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
