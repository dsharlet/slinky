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
  atomic_call([this]() { stop_ = true; });
  for (std::thread& i : workers_) {
    i.join();
  }
}

bool thread_pool::dequeue(task& t, std::vector<const queued_task*>& task_stack) {
  for (auto i = task_queue_.begin(); i != task_queue_.end(); ++i) {
    if (std::find(task_stack.begin(), task_stack.end(), &*i) != task_stack.end()) {
      // Don't enqueue the same task multiple times on the same thread.
      continue;
    }
    if (i->first == 1) {
      t = std::move(i->second);
      task_queue_.erase(i);
      task_stack.push_back(nullptr);
      return true;
    } else {
      assert(i->first > 1);
      i->first -= 1;
      task_stack.push_back(&*i);
      t = i->second;
      return true;
    }
  }
  return false;
}

void thread_pool::wait_for(std::function<bool()> condition) {
  thread_local std::vector<const queued_task*> task_stack;
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

void thread_pool::atomic_call(task t) {
  std::unique_lock l(mutex_);
  t();
  cv_.notify_all();
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
