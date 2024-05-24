#ifndef SLINKY_BUILDER_TEST_SIMPLIFY_RULE_TESTER_H
#define SLINKY_BUILDER_TEST_SIMPLIFY_RULE_TESTER_H

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cassert>

#include "base/test/seeded_test.h"
#include "builder/simplify.h"
#include "builder/substitute.h"
#include "builder/test/simplify/expr_generator.h"
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
  static constexpr std::size_t var_count = 6;

  gtest_seeded_mt19937 rng_;
  expr_generator<gtest_seeded_mt19937> expr_gen_;

  std::array<expr, rewrite::symbol_count> exprs;
  std::array<index_t, rewrite::constant_count> constants;
  rewrite::match_context m;

  void init_match_context() {
    for (std::size_t i = 0; i < m.constants.size(); ++i) {
      constants[i] = expr_gen_.random_constant();
      m.constants[i] = &constants[i];
    }
    for (std::size_t i = 0; i < rewrite::symbol_count; ++i) {
      exprs[i] = expr_gen_.random_expr(2);
      m.vars[i] = exprs[i].get();
    }
  }

public:
  rule_tester() : expr_gen_(rng_, var_count) { init_match_context(); }

  SLINKY_NO_INLINE void test_expr(expr e, const std::string& rule_str) {
    if (contains_infinity(e)) {
      // TODO: Maybe there's a way to test this...
      return;
    }

    expr simplified = simplify(e);
    ASSERT_FALSE(e.same_as(simplified)) << "Rule did not apply: " << rule_str << "\nTo: " << e << "\n";

    eval_context ctx;
    for (int test = 0; test < 100; ++test) {
      for (std::size_t i = 0; i < var_count; ++i) {
        ctx[var(i)] = expr_gen_.random_constant();
      }

      index_t value = evaluate(e, ctx);
      index_t simplified_value = evaluate(simplified, ctx);
      ASSERT_EQ(value, simplified_value) << "Incorrect rule: " << rule_str << "\n" << e << " -> " << simplified << "\n";
    }
  }

  template <typename Pattern, typename Replacement>
  bool operator()(const Pattern& p, const Replacement& r) {
    // This function needs to be kept small and simple, because it is instantiated by hundreds of different rules.
    std::stringstream rule_str;
    rule_str << p << " -> " << r;

    expr e = substitute(p, m);

    // Make sure the expressions have the same value when evaluated.
    test_expr(e, rule_str.str());

    // Returning false means the rule applicator will continue to the next rule.
    return false;
  }

  template <typename Pattern, typename Replacement, typename Predicate>
  bool operator()(const Pattern& p, const Replacement& r, const Predicate& pr) {
    // This function needs to be kept small and simple, because it is instantiated by hundreds of different rules.
    std::stringstream rule_str;
    rule_str << p << " -> " << r << " if " << pr;

    // Some rules are very picky about a large number of constants, which makes it very unlikely to generate an
    // expression that the rule applies to.
    for (int test = 0; test < 100000; ++test) {
      init_match_context();
      if (substitute(pr, m)) {
        expr e = substitute(p, m);

        // Make sure the expressions have the same value when evaluated.
        test_expr(e, rule_str.str());

        // Returning false means the rule applicator will continue to the next rule.
        return false;
      }
    }
    const bool rule_applied = false;
    // We failed to apply the rule to an expression.
    EXPECT_TRUE(rule_applied) << rule_str.str();
    // Returning true stops any more tests.
    return true;
  }
};

}  // namespace slinky

#endif  // SLINKY_BUILDER_TEST_SIMPLIFY_RULE_TESTER_H