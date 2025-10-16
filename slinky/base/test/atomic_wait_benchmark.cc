#include <benchmark/benchmark.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "slinky/base/atomic_wait.h"

namespace {

constexpr int PING = 0;
constexpr int PONG = 1;

class atomic_ping_pong {
  alignas(64) std::atomic<int> turn_{PING};
  alignas(64) std::atomic<int> rounds_left_;

public:
  explicit atomic_ping_pong(int rounds) : rounds_left_(rounds) {}

  void ping() {
    while (rounds_left_ > 0) {
      slinky::atomic_wait(&turn_, PONG);
      --rounds_left_;
      turn_.store(PONG);
      slinky::atomic_notify_one(&turn_);
    }
  }

  void pong() {
    while (rounds_left_ > 0) {
      slinky::atomic_wait(&turn_, PING);
      turn_.store(PING);
      slinky::atomic_notify_one(&turn_);
    }
  }
};

class cond_var_ping_pong {
  std::mutex m_;
  std::condition_variable cv_;
  int turn_{PING};

  alignas(64) std::atomic<int> rounds_left_;

public:
  explicit cond_var_ping_pong(int rounds) : rounds_left_(rounds) {}

  void ping() {
    while (rounds_left_ > 0) {

      std::unique_lock<std::mutex> lock(m_);
      cv_.wait(lock, [&] { return turn_ == PING; });

      --rounds_left_;
      turn_ = PONG;

      lock.unlock();
      cv_.notify_one();
    }
  }

  void pong() {
    while (rounds_left_ > 0) {

      std::unique_lock<std::mutex> lock(m_);
      cv_.wait(lock, [&] { return turn_ == PONG; });

      --rounds_left_;
      turn_ = PING;

      lock.unlock();
      cv_.notify_one();
    }
  }
};

template <class PingPong> void BM_ping_pong(benchmark::State &state) {
  const int rounds = 100000;
  while (state.KeepRunningBatch(rounds)) {
    PingPong game(rounds);
    std::thread ping_thread([&]() { game.ping(); });
    std::thread pong_thread([&]() { game.pong(); });
    ping_thread.join();
    pong_thread.join();
  }
}

BENCHMARK(BM_ping_pong<atomic_ping_pong>);
BENCHMARK(BM_ping_pong<cond_var_ping_pong>);

} // anonymous namespace
