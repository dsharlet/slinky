#include "base/thread_pool.h"

#include <iostream>

#include <cassert>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace slinky {

thread_pool::thread_pool(int workers, const task& init) : stop_(false) {
  if (workers <= 0) return;

  auto worker = [this, init]() {
    if (init) init();
    run_worker([this]() -> bool { return stop_; });
  };
  for (int i = 0; i < workers; ++i) {
    threads_.push_back(std::thread(worker));
  }

  wait_for_worker();
}

thread_pool::~thread_pool() {
  atomic_call([this]() { stop_ = true; });
  for (std::thread& i : threads_) {
    i.join();
  }
}

void thread_pool::wait_for_worker() {
  std::unique_lock l(mutex_);
  while (!queue_.load()) {
    cv_.wait(l);
  }
}

thread_pool::task_queue* thread_pool::get_task_queue(std::optional<task_queue*> init) {
  thread_local task_queue* queue;
  if (init) {
    assert(!queue || !*init);
    queue = *init;
  }
  if (queue) {
    return queue;
  } else {
    std::unique_lock l(mutex_);
    task_queue* result = queue_.load();
    queue_ = result->next.load();
    return result;
  }
}

void thread_pool::run_worker(const predicate& condition) {
  task_queue queue;
  std::unique_lock l(mutex_);
  ++worker_count_;
  if (queue_) {
    task_queue* prev = queue_.load();
    task_queue* next = prev->next.load();
    // queue is new, set up its pointers first.
    queue.next = next;
    queue.prev = prev;
    // Now splice it in.
    next->prev = &queue;
    prev->next = &queue;
  } else {
    // This is the first queue.
    queue.next = &queue;
    queue.prev = &queue;
    queue_ = &queue;
  }
  l.unlock();
  cv_.notify_all();

  // This actually initializes the thread_local task queue to be this queue.
  get_task_queue(&queue);

  wait_for(condition, &queue, queue.cv);

  get_task_queue(nullptr);

  l.lock();
  task_queue* this_queue = &queue;
  task_queue* prev = queue.prev.load();
  task_queue* next = queue.next.load();
  if (next != &queue) {
    queue_.compare_exchange_strong(this_queue, next);
    prev->next = next;
    next->prev = prev;
    --worker_count_;
  } else {
    // This is the last queue.
    queue_ = nullptr;
  }
}

bool thread_pool::task_queue::dequeue(task& t, bool allow_steal) {
  if (!queue.empty()) {
    t = std::move(queue.front());
    queue.pop_front();
    return true;
  }
  if (allow_steal) {
    // Redistribute tasks between prev or next queues.

    for (task_queue* i : {next.load(), prev.load()}) {
      if (i->mutex.try_lock()) {
        if (i->queue.size() > 0) {
          t = std::move(i->queue.front());
          i->queue.pop_front();

          if (i->queue.size() > 0) {
            queue.insert(queue.end(), i->queue.begin() + i->queue.size() / 2, i->queue.end());
            i->queue.resize(i->queue.size() / 2);
          }
          i->mutex.unlock();
          return true;
        }
        i->mutex.unlock();
      }
    }
  }
  return false;
}

void thread_pool::wait_for(const thread_pool::predicate& condition, task_queue* queue, std::condition_variable& cv) {
  // We want to spin a few times before letting the OS take over.
  const int spin_count = 100;
  int spins = 0;

  std::unique_lock l(queue->mutex);
  while (!condition()) {
    task t;
    if (queue->dequeue(t)) {
      l.unlock();
      t();
      l.lock();
      // Notify the helper CV, helpers might be waiting for a condition that the task changed the value of.
      cv_.notify_all();
      // We did a task, reset the spin counter.
      spins = spin_count;
    } else if (spins-- > 0) {
      l.unlock();
      std::this_thread::yield();
      l.lock();
    } else {
      cv.wait_for(l, std::chrono::milliseconds(1));
    }
  }
}

void thread_pool::atomic_call(const task& t) {
  std::unique_lock l(mutex_);
  t();
  cv_.notify_all();
}

void thread_pool::enqueue(int n, const task& t) {
  while (n-- > 0) {
    task_queue* queue = get_task_queue();
    std::unique_lock l(queue->mutex);
    queue->queue.push_back(t);
    queue->cv.notify_one();
  }
}

void thread_pool::enqueue(task t) {
  task_queue* queue = get_task_queue();
  std::unique_lock l(queue->mutex);
  queue->queue.push_back(std::move(t));
  queue->cv.notify_one();
}

}  // namespace slinky
