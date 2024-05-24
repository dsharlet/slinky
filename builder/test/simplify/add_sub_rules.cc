#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cassert>

#include "builder/simplify_rules.h"
#include "builder/test/simplify/expr_generator.h"
#include "builder/test/simplify/rule_tester.h"

namespace slinky {

TEST(fuzz_rules, add) { ASSERT_FALSE(apply_add_rules(rule_tester())); }
TEST(fuzz_rules, sub) { ASSERT_FALSE(apply_sub_rules(rule_tester())); }

}  // namespace slinky
