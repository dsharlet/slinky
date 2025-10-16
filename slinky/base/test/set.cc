#include <gtest/gtest.h>

#include "slinky/base/set.h"

namespace slinky {

TEST(empty_intersection, basic) {
  ASSERT_TRUE(empty_intersection<int>({}, {}));
  ASSERT_TRUE(empty_intersection<int>({}, {1}));
  ASSERT_TRUE(empty_intersection<int>({1}, {}));
  ASSERT_TRUE(empty_intersection<int>({}, {1, 2}));
  ASSERT_TRUE(empty_intersection<int>({1, 2}, {}));

  ASSERT_FALSE(empty_intersection<int>({1}, {1}));
  ASSERT_TRUE(empty_intersection<int>({1}, {2}));
  ASSERT_FALSE(empty_intersection<int>({1}, {1, 2}));
  ASSERT_TRUE(empty_intersection<int>({1}, {2, 3}));
  ASSERT_FALSE(empty_intersection<int>({1, 2}, {1}));
  ASSERT_TRUE(empty_intersection<int>({1, 3}, {2}));
  ASSERT_TRUE(empty_intersection<int>({1, 3}, {2, 4}));

  ASSERT_TRUE(empty_intersection<int>({1, 3, 5}, {2, 4, 6}));
  ASSERT_TRUE(empty_intersection<int>({3, 5, 7}, {2, 4, 6}));
}

}  // namespace slinky
