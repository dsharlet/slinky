#ifndef SLINKY_BASE_THREAD_POOL_H
#define SLINKY_BASE_THREAD_POOL_H

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <deque>
#include <functional>
#include <limits>
#include <mutex>
#include <set>
#include <thread>

namespace slinky {

// This implements a simple thread pool that maps easily to the eval_context thread pool interface.
// It is not directly used by anything except for testing.
class thread_pool {
public:
  using task = std::function<void()>;
  using predicate = std::function<bool()>;

  virtual int thread_count() const = 0;

  // Enqueues `n` copies of task `t` on the thread pool queue. This guarantees that `t` will not
  // be run recursively on the same thread while in `wait_for`.
  virtual void enqueue(int n, task t) = 0;
  virtual void enqueue(task t) = 0;
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

      // Which threads are working on this loop.
      std::set<std::thread::id> working_threads;
      std::mutex m;

      // This should be called when entering a worker. If it returns false, we are already in the call stack of a
      // worker on this loop, and should return to work on other tasks instead.
      bool begin_work() {
        std::unique_lock l(m);
        std::thread::id tid = std::this_thread::get_id();
        return working_threads.emplace(tid).second;
      }

      void end_work() {
        std::unique_lock l(m);
        auto i = working_threads.find(std::this_thread::get_id());
        assert(i != working_threads.end());
        working_threads.erase(i);
      }
    };
    auto state = std::make_shared<shared_state>();
    // Capture n by value becuase this may run after the parallel_for call returns.
    auto worker = [state, n, body = std::move(body)]() mutable {
      if (!state->begin_work()) return;

      while (true) {
        std::size_t i = state->i++;
        if (i >= n) break;
        body(i);
        ++state->done;
      }

      state->end_work();
    };
    int workers = std::min<int>(max_workers, std::min<std::size_t>(thread_count() + 1, n));
    if (workers > 1) {
      enqueue(workers - 1, worker);
    }
    worker();
    // While the loop still isn't done, work on other tasks.
    wait_for([&]() { return state->done >= n; });
  }
};

// This implements a simple thread pool that maps easily to the eval_context thread pool interface.
// It is not directly used by anything except for testing.
class thread_pool_impl : public thread_pool {
private:
  using task_id = std::size_t;

  std::vector<std::thread> workers_;
  std::atomic<bool> stop_;

  task_id next_task_id_ = 1;
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

  bool dequeue(task& t, std::vector<task_id>& task_stack);

public:
  thread_pool_impl(int workers = 3);
  ~thread_pool_impl();

  int thread_count() const override { return workers_.size(); }

  // Enqueues `n` copies of task `t` on the thread pool queue. This guarantees that `t` will not
  // be run recursively on the same thread while in `wait_for`.
  void enqueue(int n, task t) override;
  void enqueue(task t) override;
  // Waits for `condition` to become true. While waiting, executes tasks on the queue.
  // The condition is executed atomically.
  void wait_for(const predicate& condition) override { wait_for(condition, cv_helper_); }
  // Run `t` on the calling thread, but atomically w.r.t. other `atomic_call` and `wait_for` conditions.
  void atomic_call(const task& t) override;
};

}  // namespace slinky

#endif  // SLINKY_BASE_THREAD_POOL_H
