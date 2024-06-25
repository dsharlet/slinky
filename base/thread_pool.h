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

namespace slinky {

// This implements a simple thread pool that maps easily to the eval_context thread pool interface.
// It is not directly used by anything except for testing.
class thread_pool {
public:
  using task_id = const void*;
  static task_id unique_task_id;
  using task = std::function<void()>;
  using predicate = std::function<bool()>;

  virtual int thread_count() const = 0;

  // Enqueues `n` copies of task `t` on the thread pool queue. This guarantees that `t` will not
  // be run recursively on the same thread while in `wait_for`.
  virtual void enqueue(int n, task t, task_id id = unique_task_id) = 0;
  virtual void enqueue(task t, task_id id = unique_task_id) = 0;
  // Run the task on the current thread, and prevents tasks enqueued by `enqueue` from running recursively.
  virtual void run(const task& t, task_id id = unique_task_id) = 0;
  // Cancel tasks previously enqueued with the given `id`.
  virtual void cancel(task_id id) {}
  // Waits for `condition` to become true. While waiting, executes tasks on the queue.
  // The condition is executed atomically.
  virtual void wait_for(const predicate& condition) = 0;
  // Run `t` on the calling thread, but atomically w.r.t. other `atomic_call` and `wait_for` conditions.
  virtual void atomic_call(const task& t) = 0;

  template <typename Fn>
  void parallel_for(std::size_t n, Fn&& body, int max_workers = std::numeric_limits<int>::max()) {
    if (n == 0) {
      return;
    } else if (n == 1) {
      body(0);
      return;
    }

    struct shared_state {
      // We track the loop progress with two variables: `i` is the next iteration to run, and `done` is the number of
      // iterations completed. This allows us to check if the loop is done without relying on the workers actually
      // running. If the thread pool is busy, then we might enqueue workers that never run until after the loop is
      // done. Waiting for these to return (after doing nothing) would risk deadlock.
      std::atomic<std::size_t> i = 0;
      std::atomic<std::size_t> done = 0;
    };
    auto state = std::make_shared<shared_state>();
    // Capture n by value becuase this may run after the parallel_for call returns.
    auto worker = [this, state, n, body = std::move(body)]() mutable {
      while (true) {
        std::size_t i = state->i++;
        if (i >= n) {
          // There are no more iterations to run.
          if (i == n) {
            // We hit the end of the loop, cancel any queued workers to save ourselves some work.
            cancel(state.get());
          }
          break;
        }
        body(i);
        ++state->done;
      }
    };
    int workers = std::min<int>(max_workers, std::min<std::size_t>(thread_count() + 1, n));
    if (workers > 1) {
      enqueue(workers - 1, worker, state.get());
    }
    // Running the worker here guarantees forward progress on the loop even if no threads in the thread pool are
    // available.
    run(worker, state.get());
    // While the loop still isn't done, work on other tasks.
    wait_for([&]() { return state->done >= n; });
  }

  template <typename Iterator, typename Fn>
  void parallel_for(Iterator begin, Iterator end, Fn&& body, int max_workers = std::numeric_limits<int>::max()) {
    using size_type = typename std::iterator_traits<Iterator>::difference_type;
    size_type n = std::distance(begin, end);
    if (n == 0) {
      return;
    } else if (n == 1) {
      body(*begin);
      return;
    }

    struct shared_state {
      // Similar to the above, but we can't track the iterator atomically.
      std::mutex m;
      Iterator i;
      Iterator end;
      std::atomic<size_type> done = 0;

      shared_state(Iterator begin, Iterator end) : i(begin), end(end) {}
    };
    auto state = std::make_shared<shared_state>(begin, end);
    // Capture n by value becuase this may run after the parallel_for call returns.
    auto worker = [this, state, n, body = std::move(body)]() mutable {
      while (true) {
        std::unique_lock l(state->m);
        if (state->i == state->end) break;
        Iterator i = state->i++;
        bool done = state->i == state->end;
        l.unlock();
        if (done) {
          // We hit the end of the loop, cancel any queued workers to save ourselves some work.
          cancel(state.get());
        }
        body(*i);
        ++state->done;
      }
    };
    int workers = std::min<int>(max_workers, std::min<size_type>(thread_count() + 1, n));
    if (workers > 1) {
      enqueue(workers - 1, worker, state.get());
    }
    // Running the worker here guarantees forward progress on the loop even if no threads in the thread pool are
    // available.
    run(worker, state.get());
    // While the loop still isn't done, work on other tasks.
    wait_for([&]() { return state->done >= n; });
  }
};

// This implements a simple thread pool that maps easily to the eval_context thread pool interface.
// It is not directly used by anything except for testing.
class thread_pool_impl : public thread_pool {
private:
  std::vector<std::thread> workers_;
  std::atomic<bool> stop_;

  using queued_task = std::tuple<int, task, task_id>;
  std::deque<queued_task> task_queue_;
  std::mutex mutex_;
  // We have two condition variables in an attempt to minimize unnecessary thread wakeups:
  // - cv_helper_ is waited on by threads that are helping the worker threads while waiting for a condition.
  // - cv_worker_ is waited on by worker threads.
  // Enqueuing a task wakes up a thread from each condition variable.
  // cv_helper_ is only notified when the state of a condition may change (a task completes, or an `atomic_call` runs).
  std::condition_variable cv_helper_;
  std::condition_variable cv_worker_;

  void wait_for(const predicate& condition, std::condition_variable& cv);

  task_id dequeue(task& t);

public:
  // `workers` indicates how many worker threads the thread pool will have.
  // `init` is a task that is run on each newly created thread.
  thread_pool_impl(int workers = 3, const task& init = nullptr);
  ~thread_pool_impl();

  int thread_count() const override { return workers_.size(); }

  void enqueue(int n, task t, task_id id) override;
  void enqueue(task t, task_id id) override;
  void run(const task& t, task_id id) override;
  void cancel(task_id id) override;
  void wait_for(const predicate& condition) override { wait_for(condition, cv_helper_); }
  void atomic_call(const task& t) override;
};

}  // namespace slinky

#endif  // SLINKY_BASE_THREAD_POOL_H
