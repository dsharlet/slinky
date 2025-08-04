#ifndef SLINKY_BASE_THREAD_SYNCHRONIZATION_H
#define SLINKY_BASE_THREAD_SYNCHRONIZATION_H

#ifdef SLINKY_USE_ABSL_SYNCHRONIZATION

#include "absl/synchronization/mutex.h"

#else

#include <condition_variable>
#include <mutex>

#endif

namespace slinky {

#ifdef SLINKY_USE_ABSL_SYNCHRONIZATION

using mutex = absl::Mutex;

class unique_lock {
  friend class condition_variable;
  mutex& m_;

public:
  unique_lock(mutex& m) : m_(m) { m_.Lock(); }
  unique_lock(const unique_lock&) = delete;
  unique_lock(unique_lock&&) = default;
  ~unique_lock() { m_.Unlock(); }

  unique_lock& operator=(const unique_lock&) = delete;
  unique_lock& operator=(unique_lock&&) = default;

  void lock() { m_.Lock(); }
  void unlock() { m_.Unlock(); }
};

class condition_variable {
  absl::CondVar impl_;

public:
  void notify_one() { impl_.Signal(); }
  void notify_all() { impl_.SignalAll(); }
  void wait(unique_lock& l) { impl_.Wait(&l.m_); }
};

#else

using mutex = std::mutex;
using condition_variable = std::condition_variable;
using unique_lock = std::unique_lock<std::mutex>;

#endif

}  // namespace slinky

#endif  // SLINKY_BASE_THREAD_SYNCHRONIZATION_H
