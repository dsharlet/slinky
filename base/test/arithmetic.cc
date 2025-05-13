#include <gtest/gtest.h>

#include <cstdlib>
#include <random>

#include "base/arithmetic.h"
#include "base/test/seeded_test.h"

namespace slinky {

TEST(arithmetic, euclidean_div_mod) {
  for (int i = 0; i < 1000; ++i) {
    int a = rand() / 2;
    int b = rand() / 2;

    ASSERT_EQ(euclidean_div(a, b), a / b);
    ASSERT_EQ(euclidean_mod(a, b), a % b);
    ASSERT_EQ(euclidean_mod_positive_modulus(a, b), a % b);

    // These are the properties we want from euclidean_div/mod:
    // 1. 0 <= a % b < |b|
    ASSERT_GE(euclidean_mod(a, b), 0);
    ASSERT_GE(euclidean_mod(-a, b), 0);
    ASSERT_GE(euclidean_mod(a, -b), 0);
    ASSERT_GE(euclidean_mod(-a, -b), 0);
    ASSERT_LT(euclidean_mod(a, b), std::abs(b));
    ASSERT_LT(euclidean_mod(-a, b), std::abs(b));
    ASSERT_LT(euclidean_mod(a, -b), std::abs(b));
    ASSERT_LT(euclidean_mod(-a, -b), std::abs(b));
    ASSERT_GE(euclidean_mod_positive_modulus(a, b), 0);
    ASSERT_GE(euclidean_mod_positive_modulus(-a, b), 0);
    ASSERT_LT(euclidean_mod_positive_modulus(a, b), std::abs(b));
    ASSERT_LT(euclidean_mod_positive_modulus(-a, b), std::abs(b));

    // 2. (a / b) * b + a % b == a
    ASSERT_EQ(euclidean_div(a, b) * b + euclidean_mod(a, b), a);
    ASSERT_EQ(euclidean_div(-a, b) * b + euclidean_mod(-a, b), -a);
    ASSERT_EQ(euclidean_div_positive_divisor(a, b) * b + euclidean_mod(a, b), a);
    ASSERT_EQ(euclidean_div_positive_divisor(-a, b) * b + euclidean_mod(-a, b), -a);
    ASSERT_EQ(euclidean_div(a, -b) * -b + euclidean_mod(a, -b), a);
    ASSERT_EQ(euclidean_div(-a, -b) * -b + euclidean_mod(-a, -b), -a);
    ASSERT_EQ(euclidean_div(a, b) * b + euclidean_mod_positive_modulus(a, b), a);
    ASSERT_EQ(euclidean_div(-a, b) * b + euclidean_mod_positive_modulus(-a, b), -a);
  }
}

template <typename Dst, typename Src>
Dst saturate(Src x) {
  return std::max<Src>(std::min<Src>(x, std::numeric_limits<Dst>::max()), std::numeric_limits<Dst>::min());
}

template <typename Dst, typename Src>
bool saturates(Src x) {
  return x < std::numeric_limits<Dst>::min() || x > std::numeric_limits<Dst>::max();
}

TEST(arithmetic, saturate) {
  constexpr int32_t min = std::numeric_limits<int32_t>::min();
  constexpr int32_t max = std::numeric_limits<int32_t>::max();
  const int32_t values[] = {min, min + 1, min + 2, -2, -1, 0, 1, 2, max - 2, max - 1, max};
  for (int32_t a : values) {
    for (int32_t b : values) {
      ASSERT_EQ(saturate_add(a, b), saturate<int32_t>(static_cast<int64_t>(a) + static_cast<int64_t>(b)));
      ASSERT_EQ(saturate_sub(a, b), saturate<int32_t>(static_cast<int64_t>(a) - static_cast<int64_t>(b)));
      ASSERT_EQ(saturate_mul(a, b), saturate<int32_t>(static_cast<int64_t>(a) * static_cast<int64_t>(b)));
      ASSERT_EQ(saturate_div(a, b), saturate<int32_t>(euclidean_div(static_cast<int64_t>(a), static_cast<int64_t>(b))));
      ASSERT_EQ(saturate_mod(a, b), saturate<int32_t>(euclidean_mod(static_cast<int64_t>(a), static_cast<int64_t>(b))));

      ASSERT_EQ(add_overflows(a, b), saturates<int32_t>(static_cast<int64_t>(a) + static_cast<int64_t>(b)));
      ASSERT_EQ(sub_overflows(a, b), saturates<int32_t>(static_cast<int64_t>(a) - static_cast<int64_t>(b)));
      ASSERT_EQ(mul_overflows(a, b), saturates<int32_t>(static_cast<int64_t>(a) * static_cast<int64_t>(b)));
    }
  }
}

TEST(arithmetic, gcd) {
  ASSERT_EQ(gcd(1, 1), 1);
  ASSERT_EQ(gcd(2, 1), 1);
  ASSERT_EQ(gcd(1, 2), 1);
  ASSERT_EQ(gcd(2, 2), 2);
  ASSERT_EQ(gcd(3, 5), 1);
  ASSERT_EQ(gcd(4, 8), 4);
  ASSERT_EQ(gcd(15, 25), 5);
}

TEST(arithmetic, staircase_sum_bounds) {
  // Not the same slope
  ASSERT_EQ(staircase_sum_bounds(0, 1, 1, 0, 1, 1), std::nullopt);
  ASSERT_EQ(staircase_sum_bounds(0, 2, 1, 0, 1, -1), std::nullopt);

  ASSERT_EQ(staircase_sum_bounds(0, 1, 1, 0, 1, -1), std::make_pair(0, 0));
  ASSERT_EQ(staircase_sum_bounds(0, 1, 3, 0, 1, -3), std::make_pair(0, 0));
  ASSERT_EQ(staircase_sum_bounds(0, 4, 1, 0, -4, 1), std::make_pair(0, 0));

  // Generate random staircases, check against brute force.
  const int max_abs_bc = 16;
  gtest_seeded_mt19937 rng;
  std::uniform_int_distribution<int> a_dist{-1000, 1000};
  std::uniform_int_distribution<int> bc_dist{-max_abs_bc, max_abs_bc};
  for (auto _ : fuzz_test(std::chrono::seconds(1))) {
    const int a1 = a_dist(rng);
    const int a2 = a_dist(rng);
    const int b1 = bc_dist(rng);
    const int c1 = bc_dist(rng);
    const int b2 = bc_dist(rng);
    const int c2 = -euclidean_div(c1 * b2, b1);

    int min = std::numeric_limits<int>::max();
    int max = std::numeric_limits<int>::min();
    for (int x = -max_abs_bc * max_abs_bc; x < max_abs_bc * max_abs_bc; ++x) {
      const int y = euclidean_div(x + a1, b1) * c1 + euclidean_div(x + a2, b2) * c2;
      min = std::min(y, min);
      max = std::max(y, max);
    }

    auto bounds = staircase_sum_bounds(a1, b1, c1, a2, b2, c2);
    if (b1 * c2 == -b2 * c1 && ((b1 != 0) == (b2 != 0))) {
      ASSERT_TRUE(bounds);
      ASSERT_EQ(min, bounds->first);
      ASSERT_EQ(max, bounds->second);
    } else {
      ASSERT_FALSE(bounds);
    }
  }
}

}  // namespace slinky
