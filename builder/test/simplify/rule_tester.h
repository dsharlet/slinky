#ifndef SLINKY_BUILDER_TEST_SIMPLIFY_RULE_TESTER_H
#define SLINKY_BUILDER_TEST_SIMPLIFY_RULE_TESTER_H

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cassert>

#include "base/test/seeded_test.h"
#include "builder/rewrite.h"
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

namespace rewrite {

// We need a version of substitute from rewrite.h that does not eagerly apply optimization.
SLINKY_UNIQUE index_t substitute(index_t p, const match_context&) { return p; }

template <int N>
SLINKY_UNIQUE expr_ref substitute(const pattern_wildcard<N>&, const match_context& ctx) {
  return ctx.vars[N];
}

template <int N>
SLINKY_UNIQUE index_t substitute(const pattern_constant<N>&, const match_context& ctx) {
  return ctx.constants[N];
}

template <typename A, index_t Default>
SLINKY_UNIQUE auto substitute(const pattern_optional<A, Default>& p, const match_context& ctx) {
  return substitute(p.a, ctx);
}

template <typename T, typename A, typename B>
SLINKY_UNIQUE auto substitute(const pattern_binary<T, A, B>& p, const match_context& ctx) {
  return make_binary<T>(substitute(p.a, ctx), substitute(p.b, ctx));
}

template <typename T, typename A>
SLINKY_UNIQUE auto substitute(const pattern_unary<T, A>& p, const match_context& ctx) {
  return make_unary<T>(substitute(p.a, ctx));
}

template <typename C, typename T, typename F>
SLINKY_UNIQUE expr substitute(const pattern_select<C, T, F>& p, const match_context& ctx) {
  return select::make(substitute(p.c, ctx), substitute(p.t, ctx), substitute(p.f, ctx));
}

SLINKY_UNIQUE expr substitute(const pattern_call<>& p, const match_context& ctx) { return call::make(p.fn, {}); }
template <typename A>
SLINKY_UNIQUE expr substitute(const pattern_call<A>& p, const match_context& ctx) {
  return call::make(p.fn, {substitute(std::get<0>(p.args), ctx)});
}
template <typename A, typename B>
SLINKY_UNIQUE expr substitute(const pattern_call<A, B>& p, const match_context& ctx) {
  return call::make(p.fn, {substitute(std::get<0>(p.args), ctx), substitute(std::get<1>(p.args), ctx)});
}

template <typename X, typename A, typename B, typename C>
SLINKY_UNIQUE auto substitute(const pattern_staircase<X, A, B, C>& p, const match_context& ctx) {
  expr x = substitute(p.x, ctx);
  index_t a = substitute(p.a, ctx);
  index_t b = substitute(p.b, ctx);
  index_t c = substitute(p.c, ctx);

  if (a != 0) x += a;
  if (b != 1) x /= b;
  if (c != 1) x *= c;
  return x;
}

template <typename T, typename Fn>
SLINKY_UNIQUE bool substitute(const replacement_predicate<T, Fn>& r, const match_context& ctx) {
  return r.fn(substitute(r.a, ctx));
}

template <typename T>
SLINKY_UNIQUE index_t substitute(const replacement_eval<T>& r, const match_context& ctx) {
  return substitute(r.a, ctx);
}

template <typename T>
SLINKY_UNIQUE auto substitute(const replacement_boolean<T>& r, const match_context& ctx) {
  return boolean(substitute(r.a, ctx));
}

template <typename A1, typename B1, typename C1, typename A2, typename B2, typename C2>
SLINKY_UNIQUE index_t substitute(
    const replacement_staircase_sum_bound<A1, B1, C1, A2, B2, C2>& r, const match_context& ctx) {
  index_t a1 = substitute(r.a1, ctx);
  index_t b1 = substitute(r.b1, ctx);
  index_t c1 = substitute(r.c1, ctx);
  index_t a2 = substitute(r.a2, ctx);
  index_t b2 = substitute(r.b2, ctx);
  index_t c2 = substitute(r.c2, ctx);
  auto bounds = staircase_sum_bounds(a1, b1, c1, a2, b2, c2);
  if (r.bound_sign < 0) {
    return bounds.min ? *bounds.min : std::numeric_limits<index_t>::min();
  } else {
    return bounds.max ? *bounds.max : std::numeric_limits<index_t>::max();
  }
}

}  // namespace rewrite

class rule_tester {
  static constexpr std::size_t var_count = 6;

  gtest_seeded_mt19937 rng_;
  expr_generator<gtest_seeded_mt19937> expr_gen_;

  std::array<expr, rewrite::symbol_count> exprs;
  rewrite::match_context m;

  void init_match_context() {
    for (std::size_t i = 0; i < m.constants.size(); ++i) {
      m.constants[i] = expr_gen_.random_constant();
    }
    for (std::size_t i = 0; i < rewrite::symbol_count; ++i) {
      exprs[i] = expr_gen_.random_expr(0);
      m.vars[i] = exprs[i].get();
    }
  }

public:
  rule_tester() : expr_gen_(rng_, var_count) { init_match_context(); }

  SLINKY_NO_INLINE void test_expr(expr pattern, expr replacement, const std::string& rule_str) {
    if (contains_infinity(pattern)) {
      // TODO: Maybe there's a way to test this...
      return;
    }

    expr simplified = simplify(pattern);
    ASSERT_FALSE(pattern.same_as(simplified)) << "Rule did not apply: " << rule_str << "\nTo: " << pattern << "\n";

    eval_context ctx;
    for (int test = 0; test < 100; ++test) {
      for (std::size_t i = 0; i < var_count; ++i) {
        ctx[var(i)] = expr_gen_.random_constant();
      }

      auto dump_ctx = [&]() {
        std::stringstream ss;
        for (std::size_t i = 0; i < var_count; ++i) {
          ss << ", " << var(i) << "=" << ctx[var(i)];
        }
        return ss.str();
      };

      index_t value = evaluate(pattern, ctx);
      index_t replacement_value = evaluate(replacement, ctx);
      index_t simplified_value = evaluate(simplified, ctx);
      ASSERT_EQ(value, replacement_value) << "Incorrect rule: " << rule_str << "\n"
                                          << pattern << " -> " << replacement << dump_ctx() << "\n";
      ASSERT_EQ(value, simplified_value) << "Incorrect simplification: " << rule_str << "\n"
                                         << pattern << " -> " << simplified << dump_ctx() << "\n";
    }
  }

  template <typename Pattern, typename Replacement>
  bool operator()(const Pattern& p, const Replacement& r) {
    // This function needs to be kept small and simple, because it is instantiated by hundreds of different rules.
    std::stringstream rule_str;
    rule_str << p << " -> " << r;

    expr pattern = expr(substitute(p, m));
    bool overflowed = false;
    expr replacement = expr(substitute(r, m, overflowed));
    assert(!overflowed);

    // Make sure the expressions have the same value when evaluated.
    test_expr(pattern, replacement, rule_str.str());

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
      bool overflowed = false;
      if (substitute(pr, m, overflowed) && !overflowed) {
        expr pattern = expr(substitute(p, m));
        expr replacement = expr(substitute(r, m, overflowed));
        assert(!overflowed);

        // Make sure the expressions have the same value when evaluated.
        test_expr(pattern, replacement, rule_str.str());

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

  template <typename Pattern, typename Replacement, typename Predicate, typename... ReplacementPredicate>
  bool operator()(const Pattern& p, const Replacement& r, const Predicate& pr, ReplacementPredicate... r_pr) {
    return operator()(p, r, pr) && operator()(p, r_pr...);
  }
};

}  // namespace slinky

#endif  // SLINKY_BUILDER_TEST_SIMPLIFY_RULE_TESTER_H