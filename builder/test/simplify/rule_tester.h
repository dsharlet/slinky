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
public:
  static constexpr std::size_t var_count = 6;

  gtest_seeded_mt19937 rng_;
  expr_generator<gtest_seeded_mt19937> expr_gen_;

  rule_tester() : expr_gen_(rng_, var_count) {
  }

  SLINKY_NO_INLINE bool test_expr(expr e, expr simplified) {
    if (contains_infinity(e)) {
      // TODO: Maybe there's a way to test this...
      return true;
    }

    eval_context ctx;
    for (int test = 0; test < 100; ++test) {
      for (std::size_t i = 0; i < var_count; ++i) {
        ctx[var(i)] = expr_gen_.random_constant();
      }

      index_t value = evaluate(e, ctx);
      index_t simplified_value = evaluate(simplified, ctx);
      if (value != simplified_value) {
        std::cout << value << " != " << simplified_value << std::endl;
        return false;
      }
    }
    return true;
  }

  template <typename Pattern, typename Replacement>
  bool operator()(const Pattern& p, const Replacement& r) {
    return operator()(p, r, 1);
  }

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

  template <typename Pattern, typename Replacement, typename Predicate>
  bool operator()(const Pattern& p, const Replacement& r, const Predicate& pr) {
    for (int test = 0; test < 100000; ++test) {
      init_match_context();
      if (substitute(pr, m)) {
        expr e = substitute(p, m);
        expr simplified = simplify(e);

        // Make sure the expressions have the same value.
        EXPECT_TRUE(test_expr(e, simplified)) << p << " -> " << r << " if " << pr << "\n" << e << " -> " << simplified;

        // Returning false means the rule applicator will continue to the next rule.
        return false;
      }
    }
    const bool rule_applied = false;
    // We failed to apply the rule to any expressions.
    EXPECT_TRUE(rule_applied) << p << " if " << pr;
    // Returning true stops any more tests.
    return true;
  }
};

}  // namespace slinky

#endif  // SLINKY_BUILDER_TEST_SIMPLIFY_RULE_TESTER_H