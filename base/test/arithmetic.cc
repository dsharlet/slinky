#include <gtest/gtest.h>

#include <cstdlib>

#include "base/arithmetic.h"

namespace slinky {

TEST(arithmetic, euclidean_div_mod) {
  for (int i = 0; i < 1000; ++i) {
    int a = rand() / 2;
    int b = rand() / 2;

    ASSERT_EQ(euclidean_div(a, b), a / b);
    ASSERT_EQ(euclidean_mod(a, b), a % b);

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

    // 2. (a / b) * b + a % b == a
    ASSERT_EQ(euclidean_div(a, b) * b + euclidean_mod(a, b), a);
    ASSERT_EQ(euclidean_div(-a, b) * b + euclidean_mod(-a, b), -a);
    ASSERT_EQ(euclidean_div(a, -b) * -b + euclidean_mod(a, -b), a);
    ASSERT_EQ(euclidean_div(-a, -b) * -b + euclidean_mod(-a, -b), -a);
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

}  // namespace slinky
