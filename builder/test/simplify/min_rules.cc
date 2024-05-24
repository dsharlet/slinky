#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cassert>

#include "builder/simplify_rules.h"
#include "builder/test/simplify/expr_generator.h"
#include "builder/test/simplify/rule_tester.h"

namespace slinky {

TEST(fuzz_rules, min) { ASSERT_FALSE(apply_min_rules(rule_tester())); }

}  // namespace slinky
