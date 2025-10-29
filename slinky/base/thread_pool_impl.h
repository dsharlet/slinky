#ifndef SLINKY_BASE_THREAD_POOL_IMPL_H
#define SLINKY_BASE_THREAD_POOL_IMPL_H

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <deque>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#include "slinky/base/function_ref.h"
#include "slinky/base/ref_count.h"
#include "slinky/base/thread_pool.h"

namespace slinky {

constexpr std::size_t cache_line_size = 64;

// This implements a simple thread pool that maps easily to the eval_context thread pool interface.
class thread_pool_impl final : public thread_pool {
public:
  // This is a helper class for implementing a work stealing scheduler for a parallel for loop. It divides the work
  // among `shards`, which can be executed independently by separate threads. When the task is complete, the
  // thread will try to steal work from other shards.
  class task_impl final : public task {
  private:
    task_body body_;
    std::size_t shard_count_;
    // How many workers can start working on this loop. Decremented as workers begin working.
    std::atomic<int> max_workers_;

    alignas(cache_line_size) std::atomic<std::size_t> todo_;

    struct shard {
      // i is the next iteration to run.
      alignas(cache_line_size) std::atomic<std::size_t> i;

      // One past the last iteration to run in this shard.
      std::size_t end;

      // Execute the body on each work item in this shard.
      std::size_t work(task_body& body);
    };
    // This memory follows the task_impl object.
    shard shards_[1];

    // Set up a parallel for loop over `n` items.
    task_impl(std::size_t shard_count, std::size_t n, task_body body, int max_workers);

  public:
    static slinky::ref_count<task_impl> make(
        std::size_t shard_count, std::size_t n, task_body body, int max_workers = std::numeric_limits<int>::max());

    void destroy() override;

    // Return a unique worker ID for this loop. Negative worker IDs are invalid, indicating no more workers should work
    // on this loop.
    int allocate_worker() { return --max_workers_; }

    // Work on the loop. This returns when work on all items in the loop has started, but may return before all items
    // are complete. Returns true if this call resulted in the loop being done, but the loop may not be done.
    bool work(std::size_t worker);
    bool work();

    // Returns true if there is no work left to start.
    bool all_work_started() const;

    bool done() const override { return todo_ == 0; }
  };

private:
  int expected_thread_count_ = 0;
  std::atomic<int> worker_count_{0};
  std::vector<std::thread> threads_;
  std::atomic<bool> stop_;

  std::deque<ref_count<task_impl>> task_queue_;
  std::mutex mutex_;
  // We have two condition variables in an attempt to minimize unnecessary thread wakeups:
  // - cv_helper_ is waited on by threads that are helping the worker threads while waiting for a condition.
  // - cv_worker_ is waited on by worker threads.
  // Enqueuing a task wakes up a thread from each condition variable.
  // cv_helper_ is only notified when the state of a condition may change (a task completes, or an `atomic_call` runs).
  std::condition_variable cv_helper_;
  std::condition_variable cv_worker_;

  void wait_for(predicate_ref condition, std::condition_variable& cv);

  ref_count<task_impl> dequeue(int& worker);

public:
  // `workers` indicates how many worker threads the thread pool will have.
  // `init` is a task that is run on each newly created thread.
  // Pass workers = 0 to have a thread pool with no worker threads and
  // use `run_worker` to enter a thread into the thread pool.
  thread_pool_impl(int workers = 3, function_ref<void()> init = nullptr);
  ~thread_pool_impl() override;

  // Enters the calling thread into the thread pool as a worker. Does not return until `condition` returns true.
  void run_worker(predicate_ref condition);

  // Enters the calling thread into the thread pool as a worker. Returns when there is no work to do.
  void work_until_idle();

  // Because the above API allows adding workers to the thread pool, we might not know how many threads there will be
  // when starting up a task. This allows communicating that information.
  void expect_workers(int n) { expected_thread_count_ = n; }

  int thread_count() const override { return std::max<int>(expected_thread_count_, worker_count_); }

  ref_count<task> enqueue(std::size_t n, task_body t, int max_workers = std::numeric_limits<int>::max()) override;
  using thread_pool::enqueue;
  void wait_for(task* t) override;
  void wait_for(predicate_ref condition) override { wait_for(condition, cv_helper_); }
  void atomic_call(function_ref<void()> t) override;
};

}  // namespace slinky

#endif  // SLINKY_BASE_THREAD_POOL_IMPL_H
