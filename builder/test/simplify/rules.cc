#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cassert>

#include "builder/simplify_rules.h"
#include "builder/test/simplify/expr_generator.h"
#include "builder/test/simplify/rule_tester.h"

namespace slinky {

TEST(fuzz_rules, logical_and) { ASSERT_FALSE(apply_logical_and_rules(rule_tester())); }
TEST(fuzz_rules, logical_or) { ASSERT_FALSE(apply_logical_or_rules(rule_tester())); }
TEST(fuzz_rules, logical_not) { ASSERT_FALSE(apply_logical_not_rules(rule_tester())); }
TEST(fuzz_rules, select) { ASSERT_FALSE(apply_select_rules(rule_tester())); }
TEST(fuzz_rules, call) { ASSERT_FALSE(apply_call_rules(rule_tester())); }

}  // namespace slinky
