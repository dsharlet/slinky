#include <gtest/gtest.h>

#include <cstdlib>

#include "base/modulus_remainder.h"

namespace slinky {

TEST(modulus_remainder, p) {
    ASSERT_EQ(modulus_remainder(30, 3) + modulus_remainder(40, 2), modulus_remainder(10, 5));
    ASSERT_EQ(modulus_remainder(6, 3) * modulus_remainder(4, 1), modulus_remainder(2, 1));
    ASSERT_EQ(modulus_remainder::unify(modulus_remainder(30, 6), modulus_remainder(40, 31)), modulus_remainder(5, 1));
    ASSERT_EQ(modulus_remainder(10, 0) - modulus_remainder(33, 0), modulus_remainder(1, 0));
    ASSERT_EQ(modulus_remainder(10, 0) - modulus_remainder(35, 0), modulus_remainder(5, 0));
    // // Check overflow
    ASSERT_EQ(modulus_remainder(5045320, 4) * modulus_remainder(405713, 3) * modulus_remainder(8000123, 4354), modulus_remainder(1, 0));
}

}  // namespace slinky
