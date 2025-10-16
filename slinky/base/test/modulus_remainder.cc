#include <gtest/gtest.h>

#include <cstdlib>

#include "slinky/base/modulus_remainder.h"

namespace slinky {

using mod_rem = modulus_remainder<int64_t>;

std::ostream& operator<<(std::ostream& os, const mod_rem& mr) {
  return os << "(" << mr.modulus << ", " << mr.remainder << ")";
}

TEST(modulus_remainder, p) {
  ASSERT_EQ(mod_rem(0, 0) + mod_rem(0, 6), mod_rem(0, 6));
  ASSERT_EQ(mod_rem(30, 3) + mod_rem(40, 2), mod_rem(10, 5));
  ASSERT_EQ(mod_rem(6, 3) * mod_rem(4, 1), mod_rem(2, 1));
  ASSERT_EQ(mod_rem(30, 6) | mod_rem(40, 31), mod_rem(5, 1));
  ASSERT_EQ(mod_rem(10, 0) - mod_rem(33, 0), mod_rem(1, 0));
  ASSERT_EQ(mod_rem(10, 0) - mod_rem(35, 0), mod_rem(5, 0));
  // // Check overflow
  ASSERT_EQ(mod_rem(5045320, 4) * mod_rem(405713, 3) * mod_rem(8000123, 4354), mod_rem(1, 0));

  ASSERT_EQ(mod_rem(2, 0) & mod_rem(2, 0), mod_rem(2, 0));
  ASSERT_EQ(mod_rem(2, 0) & mod_rem(3, 0), mod_rem(6, 0));
  ASSERT_EQ(mod_rem(2, 1) & mod_rem(3, 1), mod_rem(3, 1));
}

}  // namespace slinky
