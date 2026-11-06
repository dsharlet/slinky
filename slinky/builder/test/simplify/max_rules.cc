#include <gtest/gtest.h>

#include <ostream>
#include <string>
#include <tuple>

#include "slinky/builder/simplify_rules.h"
#include "slinky/builder/test/simplify/rule_tester.h"

namespace slinky {

TEST(fuzz_rules, max) { ASSERT_FALSE(apply_max_rules(rule_tester())); }

}  // namespace slinky
