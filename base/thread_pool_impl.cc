#include "base/thread_pool_impl.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <thread>
#include <vector>

#include "base/thread_synchronization.h"

namespace slinky {

thread_pool_impl::task_impl::task_impl(std::size_t shard_count, std::size_t n, task_body body, int max_workers)
    : body_(std::move(body)), shard_count_(shard_count), max_workers_(max_workers), todo_(n) {
  std::size_t begin = 0;
  // Divide the work evenly among the shards we have.
  for (std::size_t i = 0; i < shard_count_; ++i) {
    shard& s = shards_[i];
    s.i = begin;
    s.end = ((i + 1) * n) / shard_count_;
    begin = s.end;
  }
}

// We want to allocate memory and use placement new, that is aligned to a cache line. To do that, we'll allocate an
// array of this struct, and then placement new into that array.
struct cache_line {
  alignas(cache_line_size) char data[cache_line_size];
};

slinky::ref_count<thread_pool_impl::task_impl> thread_pool_impl::task_impl::make(
    std::size_t shard_count, std::size_t n, task_body body, int max_workers) {
  static_assert(sizeof(cache_line) == cache_line_size);
  static_assert(sizeof(shard) == cache_line_size, "");
  cache_line* memory = new cache_line[sizeof(task_impl) / cache_line_size + (shard_count - 1)];
  return new (memory) task_impl(shard_count, n, std::move(body), max_workers);
}

void thread_pool_impl::task_impl::destroy() {
  this->~task_impl();
  delete[] reinterpret_cast<cache_line*>(this);
}

std::size_t thread_pool_impl::task_impl::shard::work(task_body& body) {
  std::size_t done = 0;
  while (true) {
    std::size_t i = this->i++;
    if (i >= end) {
      // There are no more iterations to run.
      break;
    }
    body(i);
    ++done;
  }
  return done;
}

bool thread_pool_impl::task_impl::work(std::size_t worker) {
  task_body body = body_;
  std::size_t done = 0;
  // The first iteration of this loop runs the work allocated to this worker. Subsequent iterations of this loop are
  // stealing work from other workers.
  const std::size_t i0 = worker % shard_count_;
  for (std::size_t i = i0; i < shard_count_; ++i) {
    done += shards_[i].work(body);
  }
  for (std::size_t i = 0; i < i0; ++i) {
    done += shards_[i].work(body);
  }
  return done > 0 && (todo_ -= done) == 0;
}

bool thread_pool_impl::task_impl::work() {
  int worker = allocate_worker();
  if (worker >= 0) {
    return work(worker);
  } else {
    return false;
  }
}

bool thread_pool_impl::task_impl::all_work_started() const {
  if (done()) return true;
  for (std::size_t i = 0; i < shard_count_; ++i) {
    const shard& s = shards_[i];
    if (s.i < s.end) return false;
  }
  return true;
}

thread_pool_impl::thread_pool_impl(int workers, function_ref<void()> init) : stop_(false) {
  expect_workers(workers);
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

thread_local std::vector<const thread_pool::task*> task_stack;

template <typename... Args>
bool work_on_task(thread_pool_impl::task_impl* t, Args... args) {
  assert(std::find(task_stack.begin(), task_stack.end(), t) == task_stack.end());
  task_stack.push_back(&*t);
  bool completed = t->work(args...);
  task_stack.pop_back();
  return completed;
}

}  // namespace

ref_count<thread_pool_impl::task_impl> thread_pool_impl::dequeue(int& worker) {
  for (auto i = task_queue_.begin(); i != task_queue_.end();) {
    ref_count<task_impl>& loop = *i;
    if (loop->all_work_started()) {
      // No more threads can start working on this loop.
      i = task_queue_.erase(i);
    } else if (std::find(task_stack.begin(), task_stack.end(), &*loop) != task_stack.end()) {
      // Don't run the same loop multiple times on the same thread.
      ++i;
    } else {
      // We want to work on this loop, find out which worker we will be.
      worker = loop->allocate_worker();
      if (worker < 0) {
        // No more threads can start working on this loop.
        i = task_queue_.erase(i);
      } else if (worker == 0) {
        // This is the last worker for this loop.
        auto result = std::move(loop);
        i = task_queue_.erase(i);
        return result;
      } else {
        return loop;
      }
    }
  }
  return nullptr;
}

void thread_pool_impl::wait_for(predicate_ref condition, condition_variable& cv) {
  // We want to spin a few times before letting the OS take over.
  const int spin_count = 1000;
  int spins = 0;

  unique_lock l(mutex_);
  while (!condition()) {
    int worker;
    if (auto task = dequeue(worker)) {
      l.unlock();

      // Run the task.
      bool completed = work_on_task(task, worker);

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

void thread_pool_impl::work_until_idle() {
  unique_lock l(mutex_);
  int worker;
  while (auto task = dequeue(worker)) {
    l.unlock();

    // Run the task.
    bool completed = work_on_task(task, worker);

    l.lock();

    if (completed) {
      // We completed the loop, notify the helper CV in case it is waiting for this loop to complete.
      cv_helper_.notify_all();
    }
  }
}

void thread_pool_impl::atomic_call(function_ref<void()> t) {
  unique_lock l(mutex_);
  t();
  cv_worker_.notify_all();
  cv_helper_.notify_all();
}

ref_count<thread_pool::task> thread_pool_impl::enqueue(std::size_t n, task_body t, int max_workers) {
  assert(n > 0);
  constexpr int max_shards = 8;

  // If the number of workers is less than "infinite", we assume the caller wants a single loop counter.
  // TODO: This is a hack that should be removed when we remove pipelined loops.
  const bool ordered = max_workers < std::numeric_limits<int>::max();

  // Don't try to run more workers than there are work items.
  max_workers = std::min<std::size_t>(n, max_workers);
  const std::size_t shard_count = ordered ? 1 : std::min(max_shards, max_workers);
  auto loop = task_impl::make(shard_count, n, std::move(t), max_workers);
  unique_lock l(mutex_);
  task_queue_.push_back(loop);
  if (max_workers == 1) {
    cv_worker_.notify_one();
    cv_helper_.notify_one();
  } else {
    cv_worker_.notify_all();
    cv_helper_.notify_all();
  }
  return loop;
}

void thread_pool_impl::wait_for(task* t) {
  task_impl* task = reinterpret_cast<task_impl*>(t);
  bool completed = work_on_task(task);
  if (!completed || !task->done()) {
    // We want to spin a few times before letting the OS take over. This spinning is especially useful because we don't
    // need to lock the mutex.
    const int spin_count = 1000;
    for (int i = 0; i < spin_count; ++i) {
      std::this_thread::yield();
      if (task->done()) return;
    }

    // The loop isn't done, work on other tasks while waiting for it to complete.
    wait_for([&]() { return task->done(); });
  }
}

}  // namespace slinky
