#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cassert>

#include "base/test/seeded_test.h"
#include "builder/simplify.h"
#include "builder/simplify_rules.h"
#include "builder/substitute.h"
#include "runtime/evaluate.h"
#include "runtime/expr.h"
#include "runtime/print.h"

namespace slinky {

namespace {

bool contains_infinity(expr x) {
  expr no_infinity = x;
  no_infinity = substitute(no_infinity, positive_infinity(), std::numeric_limits<index_t>::max());
  no_infinity = substitute(no_infinity, negative_infinity(), std::numeric_limits<index_t>::min());
  return !no_infinity.same_as(x);
}

}  // namespace

class rule_tester {
public:
  static constexpr std::size_t var_count = 6;
  static constexpr double constant_mean = 4.0;
  static constexpr double constant_stddev = 10.0;

  gtest_seeded_mt19937 rng_;
  std::normal_distribution<> constant_distribution_{constant_mean, constant_stddev};

  eval_context ctx;

  rewrite::match_context m;
  std::vector<expr> exprs;
  std::array<index_t, rewrite::constant_count> constants;

  rule_tester() {
    for (int i = 0; i < 1000; ++i) {
      if (rng_() % 4 != 0) {
        exprs.push_back(variable::make(var(rng_() % var_count)));
      } else {
        exprs.push_back(constant::make(random_constant()));
      }
    }
    for (std::size_t i = 0; i < m.constants.size(); ++i) {
      m.constants[i] = &constants[i];
    }
  }

  index_t random_constant() {
    return std::round(constant_distribution_(rng_));
  }

  bool test_expr(expr e, expr simplified) {
    if (contains_infinity(e)) {
      // TODO: Maybe there's a way to test this...
      return true;
    }

    for (int test = 0; test < 100; ++test) {
      for (std::size_t i = 0; i < var_count; ++i) {
        ctx[var(i)] = random_constant();
      }

      index_t value = evaluate(e, ctx);
      index_t simplified_value = evaluate(simplified, ctx);
      if (value != simplified_value) return false;
    }
    return true;
  }

  template <typename Pattern, typename Replacement>
  bool operator()(const Pattern& p, const Replacement& r) {
    return operator()(p, r, 1);
  }

  template <typename Pattern, typename Replacement, typename Predicate>
  bool operator()(const Pattern& p, const Replacement& r, const Predicate& pr) {
    for (int test = 0; test < 100; ++test) {
      for (std::size_t i = 0; i < m.constants.size(); ++i) {
        constants[i] = random_constant();
      }
      for (std::size_t i = 0; i < m.vars.size(); ++i) {
        m.vars[i] = exprs[rng_() % exprs.size()].get();
      }
      if (substitute(pr, m)) {
        expr e = substitute(p, m);
        expr simplified = simplify(e);

        // Make sure the rule did something.
        EXPECT_FALSE(e.same_as(simplified)) << p << " -> " << r << " if " << pr;

        // Make sure the expressions have the same value.
        EXPECT_TRUE(test_expr(e, simplified)) << p << " -> " << r << " if " << pr;

        return true;
      }
    }
    const bool rule_applied = false;
    // We failed to apply the rule to any expressions.
    EXPECT_TRUE(rule_applied) << p << " -> " << r << " if " << pr;
    // Returning false stops any more tests.
    return false;
  }
};

TEST(fuzz_rules, min) { apply_min_rules(rule_tester()); }
TEST(fuzz_rules, max) { apply_max_rules(rule_tester()); }
TEST(fuzz_rules, add) { apply_add_rules(rule_tester()); }
TEST(fuzz_rules, sub) { apply_sub_rules(rule_tester()); }
TEST(fuzz_rules, mul) { apply_mul_rules(rule_tester()); }
TEST(fuzz_rules, div) { apply_div_rules(rule_tester()); }
TEST(fuzz_rules, mod) { apply_mod_rules(rule_tester()); }
TEST(fuzz_rules, less) { apply_less_rules(rule_tester()); }
TEST(fuzz_rules, equal) { apply_equal_rules(rule_tester()); }
//TEST(fuzz_rules, logical_and) { apply_logical_and_rules(rule_tester()); }
//TEST(fuzz_rules, logical_or) { apply_logical_or_rules(rule_tester()); }
//TEST(fuzz_rules, logical_not) { apply_logical_not_rules(rule_tester()); }
TEST(fuzz_rules, select) { apply_select_rules(rule_tester()); }
TEST(fuzz_rules, call) { apply_call_rules(rule_tester()); }

}  // namespace slinky
