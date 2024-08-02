#include <gtest/gtest.h>

#include <cstdlib>

#include "base/modulus_remainder.h"

namespace slinky {

TEST(modulus_remainder, p) {
    ASSERT_EQ(modulus_remainder<int64_t>(30, 3) + modulus_remainder<int64_t>(40, 2), modulus_remainder<int64_t>(10, 5));
    ASSERT_EQ(modulus_remainder<int64_t>(6, 3) * modulus_remainder<int64_t>(4, 1), modulus_remainder<int64_t>(2, 1));
    ASSERT_EQ(modulus_remainder<int64_t>(30, 6) | modulus_remainder<int64_t>(40, 31), modulus_remainder<int64_t>(5, 1));
    ASSERT_EQ(modulus_remainder<int64_t>(10, 0) - modulus_remainder<int64_t>(33, 0), modulus_remainder<int64_t>(1, 0));
    ASSERT_EQ(modulus_remainder<int64_t>(10, 0) - modulus_remainder<int64_t>(35, 0), modulus_remainder<int64_t>(5, 0));
    // // Check overflow
    ASSERT_EQ(modulus_remainder<int64_t>(5045320, 4) * modulus_remainder<int64_t>(405713, 3) * modulus_remainder<int64_t>(8000123, 4354), modulus_remainder<int64_t>(1, 0));
}

}  // namespace slinky
