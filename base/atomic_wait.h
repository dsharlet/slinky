#ifndef SLINKY_BASE_ATOMIC_WAIT_H
#define SLINKY_BASE_ATOMIC_WAIT_H

#include <atomic>
#include <cassert>
#include <chrono>

namespace slinky {

// C++20 atomic API.
template <typename T>
void wait(std::atomic<T>& a, T old) {
  while (true) {
    int result = futex(&a, FUTEX_WAIT, old, nullptr, nullptr, 0);
    if (result == -1) {
      assert(errno == EAGAIN);
    } else {
      assert(result == 0);
      return;
    }
  }
}

template <typename T>
void timed_wait(std::atomic<T>& a, T old, std::chrono::nanoseconds timeout) {
  struct timespec ts;
  ts.tv_sec = timeout.count() / 1000000000;
  ts.tv_nsec = timeout.count() % 1000000000;
  while (true) {
    int result = futex(&a, FUTEX_WAIT, old, &ts, nullptr, 0);
    if (result == -1) {
      assert(errno == EAGAIN);
    } else {
      assert(result == 0);
      return;
    }
  }
}

template <typename T>
void notify_n(std::atomic<T>& a, int n) {
  while (true) {
    int result = futex(&a, FUTEX_WAKE, n, NULL, NULL, 0);
    if (result > 0) {
      return;
    } else {
      assert(result == 0);
    }
  }
}

template <typename T>
void notify_one(std::atomic<T>& a) {
  notify_n(a, 1);
}
template <typename T>
void notify_all(std::atomic<T>& a) {
  notify_n(a, std::numeric_limits<int>::max());
}

}  // namespace slinky

#endif  // SLINKY_BASE_ATOMIC_WAIT_H
