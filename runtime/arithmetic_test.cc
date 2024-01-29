#include <gtest/gtest.h>

#include <cstdlib>

#include "runtime/util.h"

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

}
