#ifndef SLINKY_BASE_TEST_SEEDED_TEST_H
#define SLINKY_BASE_TEST_SEEDED_TEST_H

#include <gtest/gtest.h>

#include <random>

#include "base/util.h"

namespace slinky {

// This class is a wrapper for an STL random number generator that gets its seed from the --gtest_random_seed argument,
// and logs the seed if there is a failure.
template <typename Rng>
class gtest_seeded_rng {
  Rng rng_;
  testing::ScopedTrace trace_;

public:
  using result_type = typename Rng::result_type;

  gtest_seeded_rng() : rng_(seed()), trace_(__FILE__, __LINE__, "seed=" + std::to_string(seed())) {}

  int seed() const { return testing::UnitTest::GetInstance()->random_seed(); }

  result_type operator()() { return rng_(); }
  static constexpr result_type min() { return Rng::min(); }
  static constexpr result_type max() { return Rng::max(); }
};

using gtest_seeded_mt19937 = gtest_seeded_rng<std::mt19937>;

// Helper to run a test for a specified amount of time. Example usage:
//
// for (auto _ : fuzz_test(std::chrono::seconds(1))) {
//   // Will run for at least one second.
// }
class fuzz_test {
public:
  class fuzz_iterator {
    struct SLINKY_UNUSED dummy_value {};
    int iters_ = -1;

  public:
    explicit fuzz_iterator(fuzz_test* parent) : parent_(parent) {}
    ~fuzz_iterator() {
      if (iters_ >= 0) {
        std::cout << "fuzz_test: executed " << iters_ << " iterations" << std::endl;
      }
    }

    void operator++() { ++iters_; }

    bool operator!=(const fuzz_iterator& other) const { return !parent_->done(); }

    dummy_value operator*() const { return dummy_value{}; }

  private:
    fuzz_test* parent_;
  };

  template <typename Duration>
  explicit fuzz_test(Duration duration) : expire_at_(clock::now() + duration) {}

  auto begin() { return fuzz_iterator(this); }
  auto end() { return fuzz_iterator(this); }

  bool done() const { return clock::now() >= expire_at_; }

private:
  using clock = std::chrono::steady_clock;
  clock::time_point expire_at_;
};

}  // namespace slinky

#endif
