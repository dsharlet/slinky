#include "base/thread_pool_impl.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace slinky {

thread_pool_impl::thread_pool_impl(int workers, function_ref<void()> init) : stop_(false) {
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

std::shared_ptr<task> thread_pool_impl::dequeue(task_body& t) {
  for (auto i = task_queue_.begin(); i != task_queue_.end();) {
    std::shared_ptr<task_impl>& loop = std::get<1>(*i);
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

    // We are going to work on this loop.
    int& max_workers = std::get<0>(*i);
    assert(max_workers > 0);
    if (--max_workers == 0 || iterations_remaining == 1) {
      // We're the last worker for this loop, remove it from the queue.
      t = std::move(std::get<2>(*i));
      std::shared_ptr<task> result = std::move(loop);
      task_queue_.erase(i);
      return result;
    } else {
      // Leave the loop in the queue for other workers to work on.
      t = std::get<2>(*i);
      return loop;
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
    task_body t;
    if (auto loop = dequeue(t)) {
      l.unlock();

      // Run the task.
      task_stack.push_back(&*loop);
      bool completed = loop->run(t);
      task_stack.pop_back();

      // We did a task, reset the spin counter.
      spins = spin_count;

      l.lock();

      if (completed) {
        // We completed the loop, notify the helper CV in case it is waiting for this loop to complete.
        cv_helper_.notify_all();
      }
    } else if (spins-- > 0) {
      l.unlock();
      std::this_thread::yield();
      l.lock();
    } else {
      cv.wait(l);
    }
  }
}

void thread_pool_impl::atomic_call(function_ref<void()> t) {
  std::unique_lock l(mutex_);
  t();
  cv_worker_.notify_all();
  cv_helper_.notify_all();
}

std::shared_ptr<task> thread_pool_impl::enqueue(std::size_t n, task_body t, int max_workers) {
  auto loop = std::make_shared<task_impl>(n);
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

void thread_pool_impl::wait_for(task* l, task_body body) {
  assert(std::find(task_stack.begin(), task_stack.end(), &l) == task_stack.end());
  task_stack.push_back(&l);
  bool completed = l->run(body);
  task_stack.pop_back();
  if (!completed || !l->done()) {
    // The loop isn't done, work on other tasks while waiting for it to complete.
    wait_for([&]() { return l->done(); });
  }
}

}  // namespace slinky
