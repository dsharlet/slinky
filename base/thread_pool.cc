#include "base/thread_pool.h"

#include <cassert>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace slinky {

thread_pool_impl::thread_pool_impl(int workers) : stop_(false) {
  auto worker = [this]() { wait_for([this]() -> bool { return stop_; }); };
  for (int i = 0; i < workers; ++i) {
    workers_.push_back(std::thread(worker));
  }
}

thread_pool_impl::~thread_pool_impl() {
  atomic_call([this]() { stop_ = true; });
  for (std::thread& i : workers_) {
    i.join();
  }
}

bool thread_pool_impl::dequeue(task& t, std::vector<thread_pool_impl::task_id>& task_stack) {
  for (auto i = task_queue_.begin(); i != task_queue_.end(); ++i) {
    if (std::find(task_stack.begin(), task_stack.end(), std::get<2>(*i)) != task_stack.end()) {
      // Don't enqueue the same task multiple times on the same thread.
      continue;
    }
    if (std::get<0>(*i) == 1) {
      t = std::move(std::get<1>(*i));
      task_queue_.erase(i);
      task_stack.push_back(0);
      return true;
    } else {
      assert(std::get<0>(*i) > 1);
      std::get<0>(*i) -= 1;
      task_stack.push_back(std::get<2>(*i));
      t = std::get<1>(*i);
      return true;
    }
  }
  return false;
}

void thread_pool_impl::wait_for(const thread_pool::predicate& condition) {
  thread_local std::vector<std::size_t> task_stack;
  std::unique_lock l(mutex_);
  while (!condition()) {
    task t;
    if (dequeue(t, task_stack)) {
      l.unlock();
      t();
      task_stack.pop_back();
      l.lock();
      // This is pretty inefficient. It is here to wake up threads that are waiting for a condition to become true,
      // that may have become true due to the task completing. There may be a better way to do this.
      cv_.notify_all();
    } else if (!stop_) {
      cv_.wait(l);
    }
  }
}

void thread_pool_impl::atomic_call(const task& t) {
  std::unique_lock l(mutex_);
  t();
  cv_.notify_all();
}

void thread_pool_impl::enqueue(int n, task t) {
  if (n <= 0) return;
  std::unique_lock l(mutex_);
  task_id id = next_task_id_++;
  task_queue_.push_back({n, std::move(t), id});
  cv_.notify_all();
}

void thread_pool_impl::enqueue(task t) {
  std::unique_lock l(mutex_);
  task_id id = next_task_id_++;
  task_queue_.push_back({1, std::move(t), id});
  cv_.notify_one();
}

}  // namespace slinky
