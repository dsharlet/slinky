#include "base/thread_pool.h"

#include <cassert>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace slinky {

thread_pool_impl::thread_pool_impl(int workers, const task& init) : stop_(false) {
  auto worker = [this, init]() {
    if (init) init();
    wait_for([this]() -> bool { return stop_; }, cv_worker_);
  };
  for (int i = 0; i < workers; ++i) {
    workers_.push_back(std::thread(worker));
  }
}

thread_pool_impl::~thread_pool_impl() {
  atomic_call([this]() { stop_ = true; });
  cv_worker_.notify_all();
  for (std::thread& i : workers_) {
    i.join();
  }
}

namespace {

thread_local std::vector<const void*> task_stack;

}

const void* thread_pool_impl::dequeue(task& t) {
  for (auto i = task_queue_.begin(); i != task_queue_.end(); ++i) {
    const task_id id = std::get<2>(*i);
    if (id && std::find(task_stack.begin(), task_stack.end(), id) != task_stack.end()) {
      // Don't enqueue the same task multiple times on the same thread.
      continue;
    }
    if (std::get<0>(*i) == 1) {
      t = std::move(std::get<1>(*i));
      task_queue_.erase(i);
      return id;
    } else {
      assert(std::get<0>(*i) > 1);
      std::get<0>(*i) -= 1;
      t = std::get<1>(*i);
      return id;
    }
  }
  return nullptr;
}

void thread_pool_impl::wait_for(const thread_pool::predicate& condition, std::condition_variable& cv) {
  // We want to spin a few times before letting the OS take over.
  const int spin_count = 1000;
  int spins = 0;

  std::unique_lock l(mutex_);
  while (!condition()) {
    task t;
    if (task_id id = dequeue(t)) {
      l.unlock();
      task_stack.push_back(id);
      t();
      task_stack.pop_back();
      l.lock();
      // Notify the helper CV, helpers might be waiting for a condition that the task changed the value of.
      cv_helper_.notify_all();
      // We did a task, reset the spin counter.
      spins = spin_count;
    } else if (spins-- > 0) {
      l.unlock();
      std::this_thread::yield();
      l.lock();
    } else {
      cv.wait(l);
    }
  }
}

void thread_pool_impl::atomic_call(const task& t) {
  std::unique_lock l(mutex_);
  t();
  cv_helper_.notify_all();
}

void thread_pool_impl::enqueue(int n, const task& t) {
  if (n <= 0) return;
  std::unique_lock l(mutex_);
  task_queue_.push_back({n, t, &t});
  cv_worker_.notify_all();
  cv_helper_.notify_all();
}

void thread_pool_impl::enqueue(task t) {
  std::unique_lock l(mutex_);
  task_queue_.push_back({1, std::move(t), nullptr});
  cv_worker_.notify_one();
  cv_helper_.notify_one();
}

void thread_pool_impl::run(const task& t) {
  assert(std::find(task_stack.begin(), task_stack.end(), &t) == task_stack.end());
  task_stack.push_back(&t);
  t();
  task_stack.pop_back();
}

}  // namespace slinky
