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
  constexpr int64_t imin = std::numeric_limits<int64_t>::min();
  constexpr int64_t imax = std::numeric_limits<int64_t>::max();

  // Not periodic.
  ASSERT_EQ(staircase_sum_bounds(0, 1, 1, 0, 1, 1), interval<int64_t>{});
  ASSERT_EQ(staircase_sum_bounds(0, 2, 1, 0, 1, -1), interval<int64_t>{});

  ASSERT_EQ(staircase_sum_bounds(0, 1, 1, 0, 1, -1), interval<int64_t>(0));
  ASSERT_EQ(staircase_sum_bounds(0, 1, 3, 0, 1, -3), interval<int64_t>(0));
  ASSERT_EQ(staircase_sum_bounds(0, 4, 1, 0, -4, 1), interval<int64_t>(0));

  // The divisors must be at most 1024.
  ASSERT_EQ(staircase_sum_bounds(0, 1024, 1, 0, -1024, 1), interval<int64_t>(0));
  ASSERT_EQ(staircase_sum_bounds(0, 1025, 1, 0, -1025, 1), interval<int64_t>{});
  ASSERT_EQ(staircase_sum_bounds(0, -1025, 1, 0, 1025, 1), interval<int64_t>{});

  // Extreme offsets and coefficients.
  ASSERT_EQ(staircase_sum_bounds(imax, 4, 1, imax - 1, 4, -1), interval<int64_t>(0, 1));
  ASSERT_EQ(staircase_sum_bounds(imin, 4, 1, imin + 1, 4, -1), interval<int64_t>(-1, 0));
  ASSERT_EQ(staircase_sum_bounds(imax, 4, 1, 0, 4, -1), interval<int64_t>((int64_t{1} << 61) - 1, int64_t{1} << 61));
  ASSERT_EQ(staircase_sum_bounds(0, 1, imax, 0, 1, -imax), interval<int64_t>(0));
  ASSERT_EQ(staircase_sum_bounds(0, -1, 1, imin, -1, -1), interval<int64_t>(imin));

  // The result doesn't fit in an int64.
  ASSERT_EQ(staircase_sum_bounds(imin, -1, 1, 0, -1, -1), interval<int64_t>{});
  ASSERT_EQ(staircase_sum_bounds(imax, 1, 1, imin, 1, -1), interval<int64_t>{});
}

#ifdef __SIZEOF_INT128__
namespace {

using int128 = __int128;

// The bounds that staircase_sum_bounds should return, computed by brute force in 128 bits, which can't overflow.
interval<int128> exact_staircase_sum_bounds(int128 a1, int128 b1, int128 c1, int128 a2, int128 b2, int128 c2) {
  auto abs = [](int128 x) { return x < 0 ? -x : x; };
  if (b1 == 0 && b2 == 0) return interval<int128>(0);
  if (b1 == 0 || b2 == 0 || abs(b1) > 1024 || abs(b2) > 1024 || c1 * b2 != -c2 * b1) return {};
  interval<int128> result;
  const int128 period = lcm(abs(b1), abs(b2));
  for (int128 x = 0; x < period; ++x) {
    const int128 y = staircase(x, a1, b1, c1) + staircase(x, a2, b2, c2);
    result.min = result.min ? std::min(*result.min, y) : y;
    result.max = result.max ? std::max(*result.max, y) : y;
  }
  return result;
}

bool fits_in_int64(int128 x) {
  return std::numeric_limits<int64_t>::min() <= x && x <= std::numeric_limits<int64_t>::max();
}

}  // namespace

TEST(arithmetic, staircase_sum_bounds_fuzz) {
  constexpr int64_t imin = std::numeric_limits<int64_t>::min();
  constexpr int64_t imax = std::numeric_limits<int64_t>::max();
  // With offsets and coefficients this small, no intermediate value can overflow.
  constexpr int64_t medium = int64_t{1} << 30;
  auto is_medium = [](int64_t x) { return -medium <= x && x <= medium; };

  gtest_seeded_mt19937 rng;
  auto random_in = [&](int64_t min, int64_t max) { return std::uniform_int_distribution<int64_t>{min, max}(rng); };
  auto random_value = [&]() {
    switch (rng() % 5) {
    case 0: return random_in(-16, 16);
    case 1: return random_in(-medium, medium);
    case 2: return random_in(imin, imin + 16);
    case 3: return random_in(imax - 16, imax);
    default: return random_in(imin, imax);
    }
  };

  for (auto _ : fuzz_test(std::chrono::seconds(1))) {
    const int64_t b1 = rng() % 8 ? random_in(-16, 16) : random_value();
    const int64_t b2 = rng() % 8 ? random_in(-16, 16) : random_value();
    const int64_t a1 = random_value();
    // Offsets that are close together can cancel even when they are large.
    const int64_t a2 =
        rng() % 2 ? random_value() : static_cast<int64_t>(static_cast<uint64_t>(a1) + random_in(-16, 16));
    int64_t c1 = random_value();
    int64_t c2 = random_value();
    if (rng() % 4 && b1 != 0 && b2 != 0 && is_medium(b1) && is_medium(b2)) {
      // Make the sum periodic, with c1/b1 = -c2/b2.
      const int64_t k = random_value();
      const int64_t g = gcd(std::abs(b1), std::abs(b2));
      int64_t p1, p2;
      if (!mul_with_overflow(k, b1 / g, p1) && !mul_with_overflow(k, -(b2 / g), p2)) {
        c1 = p1;
        c2 = p2;
      }
    }

    SCOPED_TRACE(testing::Message() << "staircase_sum_bounds(" << a1 << ", " << b1 << ", " << c1 << ", " << a2 << ", "
                                    << b2 << ", " << c2 << ")");
    const interval<int64_t> actual = staircase_sum_bounds(a1, b1, c1, a2, b2, c2);
    const interval<int128> exact = exact_staircase_sum_bounds(a1, b1, c1, a2, b2, c2);
    ASSERT_EQ(actual.min.has_value(), actual.max.has_value());
    if (actual.min) {
      ASSERT_TRUE(exact.min && *exact.min == *actual.min && *exact.max == *actual.max);
    } else if (exact.min && fits_in_int64(*exact.min) && fits_in_int64(*exact.max)) {
      // Giving up on a bound that fits is only allowed if an intermediate value might overflow.
      ASSERT_FALSE(is_medium(a1) && is_medium(a2) && is_medium(c1) && is_medium(c2));
    }
  }
}
#endif  // __SIZEOF_INT128__

}  // namespace slinky
