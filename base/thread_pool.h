#ifndef SLINKY_BASE_THREAD_POOL_H
#define SLINKY_BASE_THREAD_POOL_H

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <deque>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>

#include "base/function_ref.h"

namespace slinky {

constexpr std::size_t cache_line_size = 64;

// This is a helper class for implementing a work stealing scheduler for a parallel for loop. It divides the work among
// `K` task objects, which can be executed independently by separate threads. When the task is complete, the thread will
// try to steal work from other tasks.
template <size_t K = 1>
class parallel_for {
  struct task {
    // i is the next iteration to run.
    alignas(cache_line_size) std::atomic<std::size_t> i;

    // The last iteration to run in this task.
    std::size_t end;
  };
  task tasks_[K];
  alignas(cache_line_size) std::atomic<std::size_t> worker_ = 0;
  std::atomic<std::size_t> todo_ = 0;

public:
  // Set up a parallel for loop over `n` items.
  parallel_for(std::size_t n) : todo_(n) {
    // Divide the work evenly among the tasks we have.
    if (K > 1 && n < K) {
      for (std::size_t i = 0; i < n; ++i) {
        task& k = tasks_[i];
        k.i = i;
        k.end = i + 1;
      }
      for (std::size_t i = n; i < K; ++i) {
        task& k = tasks_[i];
        k.i = 0;
        k.end = 0;
      }
    } else {
      std::size_t begin = 0;
      for (std::size_t i = 0; i < K; ++i) {
        task& k = tasks_[i];
        k.i = begin;
        k.end = ((i + 1) * n) / K;
        begin = k.end;
      }
    }
  }

  // Work on the loop. This returns when work on all items in the loop has started, but may return before all items are
  // complete.
  template <typename Fn>
  void run(const Fn& body) {
    std::size_t w = K == 1 ? 0 : worker_++;
    std::size_t done = 0;
    // The first iteration of this loop runs the work allocated to this worker. Subsequent iterations of this loop are
    // stealing work from other workers.
    for (std::size_t i = 0; i < K; ++i) {
      task& k = tasks_[(i + w) % K];
      while (true) {
        std::size_t i = k.i++;
        if (i >= k.end) {
          // There are no more iterations to run.
          break;
        }
        body(i);
        ++done;
      }
    }
    todo_ -= done;
  }

  // Return the number of loop iterations that have not started yet. This returns an upper bound, by the time the
  // function returns some of the counted iterations may have already been started by another thread.
  std::size_t count_remaining_iterations() const {
    std::size_t result = 0;
    for (const task& k : tasks_) {
      const std::size_t i = k.i;
      result += std::max(i, k.end) - i;
    }
    return result;
  }

  bool done() const { return todo_ == 0; }
};

// This implements a simple thread pool that maps easily to the eval_context thread pool interface.
// It is not directly used by anything except for testing.
class thread_pool {
public:
  using loop = slinky::parallel_for<>;
  using loop_task = std::function<void(std::size_t)>;
  using task_ref = function_ref<void()>;
  using predicate = std::function<bool()>;
  using predicate_ref = function_ref<bool()>;

  virtual ~thread_pool() = default;

  virtual int thread_count() const = 0;

  // Enqueues a loop task over `n` iterations. The tasks queued in this thread pool are instances of `parallel_for<>`
  // plus a loop task `t` that executes each iteration. Tasks are complete when all `n` of the iterations are done.
  virtual std::shared_ptr<loop> enqueue(
      std::size_t n, loop_task t, int max_workers = std::numeric_limits<int>::max()) = 0;
  // Run the task on the current thread, and prevents tasks enqueued by `enqueue` from running recursively.
  virtual void work_on_loop(loop& l, loop_task body) = 0;
  // Waits for `condition` to become true. While waiting, executes tasks on the queue.
  // The condition is executed atomically.
  virtual void wait_for(predicate_ref condition) = 0;
  // Run `t` on the calling thread, but atomically w.r.t. other `atomic_call` and `wait_for` conditions.
  virtual void atomic_call(task_ref t) = 0;

  // Enqueues a singleton task.
  template <typename Fn>
  void enqueue(Fn t) {
    // Make a dummy loop task with one iteration.
    enqueue(1, [t = std::move(t)](std::size_t) { t(); });
  }

  template <typename Fn>
  void parallel_for(std::size_t n, Fn&& body, int max_workers = std::numeric_limits<int>::max()) {
    if (n == 0) {
      return;
    } else if (n == 1) {
      body(0);
      return;
    }
    max_workers = std::min(max_workers - 1, thread_count());
    if (max_workers == 0) {
      // We aren't going to get any worker threads, just run the loop.
      for (std::size_t i = 0; i < n; ++i) {
        body(i);
      }
    } else {
      auto loop = enqueue(n, body, max_workers);
      // Working on the loop here guarantees forward progress on the loop even if no threads in the thread pool are
      // available.
      work_on_loop(*loop, std::move(body));
      // While the loop still isn't done, work on other tasks. Checking before calling `wait_for` helps because we
      // don't need to call `loop->done` atomically.
      if (!loop->done()) {
        wait_for([&]() { return loop->done(); });
      }
    }
  }
};

// This implements a simple thread pool that maps easily to the eval_context thread pool interface.
// It is not directly used by anything except for testing.
class thread_pool_impl : public thread_pool {
private:
  int expected_thread_count_ = 0;
  std::atomic<int> worker_count_{0};
  std::vector<std::thread> threads_;
  std::atomic<bool> stop_;

  using queued_task = std::tuple<int, std::shared_ptr<loop>, loop_task>;
  std::deque<queued_task> task_queue_;
  std::mutex mutex_;
  // We have two condition variables in an attempt to minimize unnecessary thread wakeups:
  // - cv_helper_ is waited on by threads that are helping the worker threads while waiting for a condition.
  // - cv_worker_ is waited on by worker threads.
  // Enqueuing a task wakes up a thread from each condition variable.
  // cv_helper_ is only notified when the state of a condition may change (a task completes, or an `atomic_call` runs).
  std::condition_variable cv_helper_;
  std::condition_variable cv_worker_;

  void wait_for(predicate_ref condition, std::condition_variable& cv);

  std::shared_ptr<loop> dequeue(loop_task& t);

public:
  // `workers` indicates how many worker threads the thread pool will have.
  // `init` is a task that is run on each newly created thread.
  // Pass workers = 0 to have a thread pool with no worker threads and
  // use `run_worker` to enter a thread into the thread pool.
  thread_pool_impl(int workers = 3, task_ref init = nullptr);
  ~thread_pool_impl() override;

  // Enters the calling thread into the thread pool as a worker. Does not return until `condition` returns true.
  void run_worker(predicate_ref condition);
  // Because the above API allows adding workers to the thread pool, we might not know how many threads there will be
  // when starting up a task. This allows communicating that information.
  void expect_workers(int n) { expected_thread_count_ = n; }

  int thread_count() const override { return std::max<int>(expected_thread_count_, worker_count_); }

  std::shared_ptr<loop> enqueue(std::size_t n, loop_task t, int max_workers) override;
  void work_on_loop(loop& l, loop_task body) override;
  void wait_for(predicate_ref condition) override { wait_for(condition, cv_helper_); }
  void atomic_call(task_ref t) override;
};

}  // namespace slinky

#endif  // SLINKY_BASE_THREAD_POOL_H
