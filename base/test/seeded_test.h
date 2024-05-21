#ifndef SLINKY_BASE_TEST_SEEDED_TEST_H
#define SLINKY_BASE_TEST_SEEDED_TEST_H

#include <gtest/gtest.h>

#include <random>

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

}  // namespace slinky

#endif
