#include "base/thread_pool.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace slinky {

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

thread_local std::vector<const void*> task_stack;

}  // namespace

std::shared_ptr<thread_pool::loop> thread_pool_impl::dequeue(loop_task& t) {
  for (auto i = task_queue_.begin(); i != task_queue_.end();) {
    std::shared_ptr<thread_pool::loop> loop = std::get<1>(*i);
    if (std::find(task_stack.begin(), task_stack.end(), &*loop) != task_stack.end()) {
      // Don't run the same loop multiple times on the same thread.
      ++i;
      continue;
    }
    size_t iterations_remaining = loop->count_remaining_iterations();
    if (iterations_remaining == 0) {
      // No more threads can start working on this loop.
      i = task_queue_.erase(i);
      continue;
    }
    t = std::get<2>(*i);
    int& max_workers = std::get<0>(*i);
    assert(max_workers > 0);
    if (--max_workers == 0 || iterations_remaining == 1) {
      // No more workers for this loop.
      task_queue_.erase(i);
    }
    return loop;
  }
  return nullptr;
}

void thread_pool_impl::wait_for(predicate_ref condition, std::condition_variable& cv) {
  // We want to spin a few times before letting the OS take over.
  const int spin_count = 1000;
  int spins = 0;

  std::unique_lock l(mutex_);
  while (!condition()) {
    loop_task t;
    if (auto loop = dequeue(t)) {
      l.unlock();
      task_stack.push_back(&*loop);
      loop->run(t);
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

void thread_pool_impl::atomic_call(task_ref t) {
  std::unique_lock l(mutex_);
  t();
  cv_worker_.notify_all();
  cv_helper_.notify_all();
}


std::shared_ptr<thread_pool::loop> thread_pool_impl::enqueue(std::size_t n, loop_task t, int max_workers) {
  auto loop = std::make_shared<thread_pool::loop>(n);
  std::unique_lock l(mutex_);
  task_queue_.push_back({max_workers, loop, std::move(t)});
  if (n == 1 || max_workers == 1) {
    cv_worker_.notify_one();
    cv_helper_.notify_one();
  } else {
    cv_worker_.notify_all();
    cv_helper_.notify_all();
  }
  return loop;
}

void thread_pool_impl::work_on_loop(loop& l, loop_task body) {
  assert(std::find(task_stack.begin(), task_stack.end(), &l) == task_stack.end());
  task_stack.push_back(&l);
  l.run(body);
  task_stack.pop_back();
}

}  // namespace slinky
