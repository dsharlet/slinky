#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cassert>

#include "slinky/builder/simplify_rules.h"
#include "slinky/builder/test/simplify/expr_generator.h"
#include "slinky/builder/test/simplify/rule_tester.h"

namespace slinky {

TEST(fuzz_rules, mul) { ASSERT_FALSE(apply_mul_rules(rule_tester())); }
TEST(fuzz_rules, div) { ASSERT_FALSE(apply_div_rules(rule_tester())); }
TEST(fuzz_rules, mod) { ASSERT_FALSE(apply_mod_rules(rule_tester())); }

}  // namespace slinky
