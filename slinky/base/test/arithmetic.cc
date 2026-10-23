#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>

#include "slinky/base/arithmetic.h"
#include "slinky/base/test/seeded_test.h"

namespace slinky {

TEST(arithmetic, euclidean_div_mod_negative_one) {
  constexpr int64_t min = std::numeric_limits<int64_t>::min();
  constexpr int64_t max = std::numeric_limits<int64_t>::max();
  for (int64_t a : {min, min + 1, int64_t{-5}, int64_t{-1}, int64_t{0}, int64_t{1}, int64_t{5}, max}) {
    // Negation must wrap rather than overflow, so min maps to min.
    const int64_t neg = static_cast<int64_t>(uint64_t{0} - static_cast<uint64_t>(a));
    ASSERT_EQ(euclidean_div<int64_t>(a, -1), neg);
    ASSERT_EQ(euclidean_mod<int64_t>(a, -1), 0);
  }
  ASSERT_TRUE(div_overflows<int64_t>(min, -1));
  ASSERT_FALSE(div_overflows<int64_t>(min + 1, -1));
  ASSERT_FALSE(div_overflows<int64_t>(min, 1));
  ASSERT_FALSE(div_overflows<int64_t>(min, -2));
  ASSERT_EQ(euclidean_div<int8_t>(-128, -1), -128);
  ASSERT_EQ(euclidean_mod<int8_t>(-128, -1), 0);
  ASSERT_EQ(euclidean_div<uint32_t>(7, 2), 3u);
  ASSERT_EQ(euclidean_mod<uint32_t>(0xFFFFFFFFu, 0xFFFFFFFFu), 0u);
}

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
      ASSERT_EQ(add_sat(a, b), saturate<int32_t>(static_cast<int64_t>(a) + static_cast<int64_t>(b)));
      ASSERT_EQ(sub_sat(a, b), saturate<int32_t>(static_cast<int64_t>(a) - static_cast<int64_t>(b)));
      ASSERT_EQ(mul_sat(a, b), saturate<int32_t>(static_cast<int64_t>(a) * static_cast<int64_t>(b)));
      ASSERT_EQ(div_sat(a, b), saturate<int32_t>(euclidean_div(static_cast<int64_t>(a), static_cast<int64_t>(b))));

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
  ASSERT_EQ(staircase_sum_bounds(0, 1, 1, 0, 1, 1), interval<int64_t>{});
  ASSERT_EQ(staircase_sum_bounds(0, 2, 1, 0, 1, -1), interval<int64_t>{});

  ASSERT_EQ(staircase_sum_bounds(0, 1, 1, 0, 1, -1), interval<int64_t>(0));
  ASSERT_EQ(staircase_sum_bounds(0, 1, 3, 0, 1, -3), interval<int64_t>(0));
  ASSERT_EQ(staircase_sum_bounds(0, 4, 1, 0, -4, 1), interval<int64_t>(0));

  // Brute force over one period (x in [0, lcm(|b1|, |b2|))), evaluating the expression directly in int64_t. Only valid
  // for periodic sums, and for arguments small enough that the direct evaluation can't overflow.
  auto brute_force = [](int64_t a1, int64_t b1, int64_t c1, int64_t a2, int64_t b2, int64_t c2) {
    if (b1 == 0 && b2 == 0) return interval<int64_t>(0);
    const int64_t period = lcm<int64_t>(b1 < 0 ? -b1 : b1, b2 < 0 ? -b2 : b2);
    int64_t min = std::numeric_limits<int64_t>::max();
    int64_t max = std::numeric_limits<int64_t>::min();
    for (int64_t x = 0; x < period; ++x) {
      const int64_t y = euclidean_div(x + a1, b1) * c1 + euclidean_div(x + a2, b2) * c2;
      min = std::min(min, y);
      max = std::max(max, y);
    }
    return interval<int64_t>(min, max);
  };
  // Checks that the result is bounded and exactly matches brute force. The arguments must be safe for brute_force.
  auto check = [&](int64_t a1, int64_t b1, int64_t c1, int64_t a2, int64_t b2, int64_t c2) {
    EXPECT_EQ(staircase_sum_bounds(a1, b1, c1, a2, b2, c2), brute_force(a1, b1, c1, a2, b2, c2))
        << a1 << " " << b1 << " " << c1 << " " << a2 << " " << b2 << " " << c2;
  };

  // Large offsets are handled exactly (|a| <= 2^40 and small c keep the brute force from overflowing).
  const int64_t big = int64_t{1} << 31;
  const int64_t imax = std::numeric_limits<int64_t>::max();
  const int64_t imin = std::numeric_limits<int64_t>::min();
  for (int64_t a : {big, -big, big + 1, int64_t{1} << 40, -(int64_t{1} << 40), (int64_t{1} << 40) - 7}) {
    check(a, 3, 5, 0, 3, -5);
    check(a, 4, 2, 7, 6, -3);
    check(a, 4, 1, a + 1, 4, -1);
    check(a, 1, 1, 0, 1, -1);
    check(a, -4, 1, a + 3, 4, 1);
    check(0, 6, 1, a, 6, -1);
  }
  ASSERT_EQ(staircase_sum_bounds(big, 1, big, big, 1, -big), interval<int64_t>(0));
  // Large coefficients.
  check(1, 2, int64_t{1} << 40, 0, 2, -(int64_t{1} << 40));

  // Extreme values: fixed expectations (the direct evaluation would overflow). These must not trigger UB, and must
  // return exactly these values.
  // Offsets near the extremes whose quotient terms don't overflow. For imax = 4 * (2^61 - 1) + 3, the quotient terms
  // cancel and the residual staircase is floor((x + 3) / 4) - floor((x + 2) / 4), with range [0, 1].
  ASSERT_EQ(staircase_sum_bounds(imax, 4, 1, imax - 1, 4, -1), interval<int64_t>(0, 1));
  // imin = -4 * 2^61, residual floor(x / 4) - floor((x + 1) / 4), with range [-1, 0].
  ASSERT_EQ(staircase_sum_bounds(imin, 4, 1, imin + 1, 4, -1), interval<int64_t>(-1, 0));
  // Quotient 2^61 - 1, residual floor((x + 3) / 4) - floor(x / 4), with range [0, 1].
  ASSERT_EQ(staircase_sum_bounds(imax, 4, 1, 0, 4, -1), interval<int64_t>((int64_t{1} << 61) - 1, int64_t{1} << 61));
  ASSERT_EQ(staircase_sum_bounds(0, 1, imax, 0, 1, -imax), interval<int64_t>(0));
  ASSERT_EQ(staircase_sum_bounds(0, 1, int64_t{1} << 61, 0, 1, -(int64_t{1} << 61)), interval<int64_t>(0));
  // INT64_MIN / -1 does not fit.
  ASSERT_EQ(staircase_sum_bounds(imin, -1, 1, 0, -1, -1), interval<int64_t>{});
  ASSERT_EQ(staircase_sum_bounds(0, -1, 1, imin, -1, -1), interval<int64_t>{});
  // Overflowing offsets, products and sums are unbounded.
  ASSERT_EQ(staircase_sum_bounds(imax, 1, 1, imin, 1, -1), interval<int64_t>{});
  ASSERT_EQ(staircase_sum_bounds(imax, 1, imax, 0, 1, -imax), interval<int64_t>{});
  ASSERT_EQ(staircase_sum_bounds(0, 1, imin, 0, 1, imin), interval<int64_t>{});
  ASSERT_EQ(staircase_sum_bounds(0, 2, imin, 0, 2, imax), interval<int64_t>{});
  // Throw the extreme values at every parameter. This must not be UB (checked by UBSan), and the result must be
  // unbounded or the exact result for the periodic cases that are constant in x.
  auto neg = [](int64_t x) { return static_cast<int64_t>(uint64_t{0} - static_cast<uint64_t>(x)); };
  for (int64_t v : {big, -big, imax, imin, imax - 1, imin + 1}) {
    staircase_sum_bounds(0, 1, v, 0, 1, neg(v));
    staircase_sum_bounds(v, 1, 1, 0, 1, -1);
    staircase_sum_bounds(0, 1, 1, v, 1, -1);
    staircase_sum_bounds(0, 1, 1, 0, 1, v);
    staircase_sum_bounds(0, v, 1, 0, 1, 1);
    staircase_sum_bounds(0, 1, 1, 0, v, 1);
    staircase_sum_bounds(0, 0, v, 0, 0, 1);
    staircase_sum_bounds(v, 3, v, v, 3, neg(v));
    staircase_sum_bounds(v, -1, v, v, 1, v);
    staircase_sum_bounds(v, -1, 1, 0, -1, -1);
    staircase_sum_bounds(0, -1, v, 0, -1, neg(v));
    // With equal offsets and opposite coefficients the sum is identically zero, unless it overflows.
    auto r = staircase_sum_bounds(v, 1, v, v, 1, neg(v));
    if (r.min || r.max) ASSERT_EQ(r, interval<int64_t>(0));
  }

  // Generate random staircases, check against brute force.
  const int max_abs_bc = 16;
  gtest_seeded_mt19937 rng;
  std::uniform_int_distribution<int> a_dist{-1000, 1000};
  std::uniform_int_distribution<int64_t> large_a_dist{-(int64_t{1} << 40), int64_t{1} << 40};
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
      ASSERT_TRUE(bounds.min);
      ASSERT_TRUE(bounds.max);
      ASSERT_EQ(min, *bounds.min);
      ASSERT_EQ(max, *bounds.max);
    } else {
      ASSERT_FALSE(bounds.min);
      ASSERT_FALSE(bounds.max);
    }
  }

  // Large offsets, checked against brute force over one period.
  std::uniform_int_distribution<int> small_dist{-24, 24};
  for (auto _ : fuzz_test(std::chrono::seconds(1))) {
    const int64_t a1 = large_a_dist(rng);
    const int64_t a2 = rng() % 2 ? large_a_dist(rng) : a1 + small_dist(rng);
    const int64_t b1 = bc_dist(rng);
    const int64_t b2 = bc_dist(rng);
    const int64_t c1 = bc_dist(rng) * (rng() % 2 ? 1 : 1003);
    if (b1 == 0 || b2 == 0 || (c1 * b2) % b1 != 0) continue;
    const int64_t c2 = -(c1 * b2) / b1;
    check(a1, b1, c1, a2, b2, c2);
    // Offsets over the whole range may overflow, so we can't use brute force. This must not be UB.
    const int64_t e1 = std::uniform_int_distribution<int64_t>{imin, imax}(rng);
    staircase_sum_bounds(e1, b1, c1, static_cast<int64_t>(static_cast<uint64_t>(e1) + small_dist(rng)), b2, c2);
  }
}

}  // namespace slinky
