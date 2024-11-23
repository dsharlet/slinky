#ifndef SLINKY_BASE_THREAD_POOL_H
#define SLINKY_BASE_THREAD_POOL_H

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <deque>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>

namespace slinky {

// This implements a simple thread pool that maps easily to the eval_context thread pool interface.
// It is not directly used by anything except for testing.
class thread_pool {
public:
  using task = std::function<void()>;
  using predicate = std::function<bool()>;

private:
  struct task_queue {
    std::deque<task> queue;
    std::mutex mutex;
    std::condition_variable cv;

    // task queues form a linked list.
    std::atomic<task_queue*> prev = nullptr;
    std::atomic<task_queue*> next = nullptr;

    bool dequeue(task& t, bool allow_steal = true);
  };

  std::atomic<int> worker_count_{0};
  std::vector<std::thread> threads_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::atomic<bool> stop_;

  // This is a pointer to a random element in a circular linked list.
  std::atomic<task_queue*> queue_ = nullptr;

  task_queue* get_task_queue(std::optional<task_queue*> init = std::nullopt);

  void wait_for(const predicate& condition, task_queue* queue, std::condition_variable& cv);

public:
  // `workers` indicates how many worker threads the thread pool will have.
  // `init` is a task that is run on each newly created thread.
  // Pass workers = 0 to have a thread pool with no worker threads and 
  // use `run_worker` to enter a thread into the thread pool.
  thread_pool(int workers = 3, const task& init = nullptr);
  ~thread_pool();

  // Enters the calling thread into the thread pool as a worker. Does not return until `condition` returns true.
  void run_worker(const predicate& condition);
  // Wait until this thread pool has at least one worker.
  void wait_for_worker();

  int thread_count() const { return worker_count_; }

  void enqueue(int n, const task& t);
  void enqueue(task t);
  void wait_for(const predicate& condition) { wait_for(condition, get_task_queue(), cv_); }

  void atomic_call(const task& t);

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
    auto worker = [state, n, body = std::move(body)]() mutable {
      while (true) {
        std::size_t i = state->i++;
        if (i >= n) {
          // There are no more iterations to run.
          break;
        }
        body(i);
        ++state->done;
      }
    };
    int workers = std::min<int>(max_workers, std::min<std::size_t>(thread_count() + 1, n));
    if (workers > 1) {
      enqueue(workers - 1, worker);
    }
    // Running the worker here guarantees forward progress on the loop even if no threads in the thread pool are
    // available.
    worker();
    // While the loop still isn't done, work on other tasks.
    wait_for([&]() { return state->done >= n; });
  }
};

}  // namespace slinky

#endif  // SLINKY_BASE_THREAD_POOL_H
