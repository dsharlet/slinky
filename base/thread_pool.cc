#include "base/thread_pool.h"
#include "base/atomic_wait.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace slinky {

thread_pool::task_id thread_pool::unique_task_id = &thread_pool::unique_task_id;

thread_pool_impl::thread_pool_impl(int workers, task_ref init) : stop_(false) {
  auto worker = [this, init]() {
    if (init) init();
    run_worker([this]() -> bool { return stop_; });
  };
  for (int i = 0; i < workers; ++i) {
    threads_.push_back(std::thread(worker));
  }
}

thread_pool_impl::~thread_pool_impl() {
  atomic_call([this]() { stop_ = true; });
  cv_worker_.notify_all();
  for (std::thread& i : threads_) {
    i.join();
  }
}

void thread_pool_impl::run_worker(predicate_ref condition) {
  ++worker_count_;
  wait_for(condition, cv_worker_);
  --worker_count_;
}

namespace {

thread_local std::vector<thread_pool::task_id> task_stack;

}  // namespace

thread_pool::task_id thread_pool_impl::dequeue(task& t) {
  for (auto i = task_queue_.begin(); i != task_queue_.end(); ++i) {
    task_id id = std::get<2>(*i);
    if (id != unique_task_id && std::find(task_stack.begin(), task_stack.end(), id) != task_stack.end()) {
      // Don't enqueue the same task multiple times on the same thread.
      continue;
    }
    int& task_count = std::get<0>(*i);
    if (task_count == 1) {
      t = std::move(std::get<1>(*i));
      task_queue_.erase(i);
      return id;
    } else {
      assert(task_count > 1);
      task_count -= 1;
      t = std::get<1>(*i);
      return id;
    }
  }
  return nullptr;
}

void thread_pool_impl::wait_for(predicate_ref condition, std::condition_variable& cv) {
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

void thread_pool_impl::wait_for(predicate_ref condition, std::atomic<bool>& flag) {
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
      l.unlock();
      slinky::atomic_wait(&flag, false);
      l.lock();
    }
  }
}

void thread_pool_impl::atomic_call(task_ref t) {
  std::unique_lock l(mutex_);
  t();
  cv_worker_.notify_all();
  cv_helper_.notify_all();
}

void thread_pool_impl::enqueue(int n, task t, task_id id) {
  if (n <= 0) return;
  std::unique_lock l(mutex_);
  task_queue_.push_back({n, std::move(t), id});
  cv_worker_.notify_all();
  cv_helper_.notify_all();
}

void thread_pool_impl::enqueue(task t, task_id id) {
  std::unique_lock l(mutex_);
  task_queue_.push_back({1, std::move(t), id});
  cv_worker_.notify_one();
  cv_helper_.notify_one();
}

void thread_pool_impl::run(task_ref t, task_id id) {
  assert(id == unique_task_id || std::find(task_stack.begin(), task_stack.end(), id) == task_stack.end());
  task_stack.push_back(id);
  t();
  task_stack.pop_back();
}

void thread_pool_impl::cancel(task_id id) {
  std::unique_lock l(mutex_);
  for (auto i = task_queue_.begin(); i != task_queue_.end();) {
    if (std::get<2>(*i) == id) {
      i = task_queue_.erase(i);
    } else {
      ++i;
    }
  }
}

}  // namespace slinky
