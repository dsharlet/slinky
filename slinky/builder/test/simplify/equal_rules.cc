#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cassert>

#include "slinky/builder/simplify_rules.h"
#include "slinky/builder/test/simplify/expr_generator.h"
#include "slinky/builder/test/simplify/rule_tester.h"

namespace slinky {

TEST(fuzz_rules, equal) { ASSERT_FALSE(apply_equal_rules(rule_tester())); }

}  // namespace slinky
