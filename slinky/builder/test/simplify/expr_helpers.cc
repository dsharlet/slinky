#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "slinky/base/arithmetic.h"
#include "slinky/base/span.h"
#include "slinky/base/test/seeded_test.h"
#include "slinky/builder/simplify.h"
#include "slinky/builder/substitute.h"
#include "slinky/builder/test/simplify/expr_generator.h"
#include "slinky/runtime/evaluate.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/print.h"

namespace slinky {

const node_context* set_default_print_context(const node_context* ctx);

namespace {

node_context symbols;

var x(symbols, "x");
var y(symbols, "y");
var z(symbols, "z");
var w(symbols, "w");

auto _ = []() {
  set_default_print_context(&symbols);
  return 0;
}();

MATCHER_P(matches, expected, "") { return match(arg, expected); }

constexpr auto fuzz_duration = std::chrono::milliseconds(500);
constexpr int eval_checks = 8;

// Wraps expr_generator to also produce helper-specific constructs (abs, boolean, non-boolean logical ops).
template <typename Rng>
class helper_expr_generator {
  Rng& rng_;
  expr_generator<Rng> gen_;

public:
  explicit helper_expr_generator(Rng& rng, int var_count = 4) : rng_(rng), gen_(rng, var_count) {}

  void init_context(eval_context& ctx) { gen_.init_context(ctx); }

  expr random_expr(int depth) {
    if (depth <= 0) {
      switch (rng_() % 5) {
      case 0: return expr(0);
      case 1: return expr(1);
      case 2: return expr(-1);
      case 3: return constant::make(gen_.random_constant());
      default: return gen_.random_variable();
      }
    }
    switch (rng_() % 7) {
    case 0: return abs(random_expr(depth - 1));
    case 1: return boolean(random_expr(depth - 1));
    case 2: return !random_expr(depth - 1);
    case 3: return and_then(random_expr(depth - 1), random_expr(depth - 1));
    case 4: return or_else(random_expr(depth - 1), random_expr(depth - 1));
    default: return gen_.random_expr(depth);
    }
  }

  // Produces an already-simplified operand, suitable for passing directly to single-op simplify(const Op*, ...).
  expr random_simplified_operand() {
    switch (rng_() % 7) {
    case 0: return expr(0);
    case 1: return expr(1);
    case 2: return expr(-1);
    case 3: return constant::make(gen_.random_constant());
    default: return simplify(random_expr(1));
    }
  }

  interval_expr random_finite_interval() {
    expr a = random_expr(1);
    expr b = random_expr(1);
    if (rng_() % 4 == 0) return point(a);
    return bounds(a, b);
  }

  // Produces finite, half-unbounded, or fully unbounded intervals.
  interval_expr random_interval() {
    switch (rng_() % 6) {
    case 0: return interval_expr::all();
    case 1: return bounds(expr(), random_expr(1));
    case 2: return bounds(random_expr(1), expr());
    case 3: return point(random_expr(1));
    default: return bounds(random_expr(1), random_expr(1));
    }
  }
};

std::vector<expr> edge_case_exprs() {
  return {
      expr(),
      expr(0),
      expr(1),
      expr(-1),
      expr(2),
      expr(-2),
      expr(x),
      expr(x) + 1,
      expr(x) < y,
      expr(x) == y,
      expr(x) != 0,
      !expr(x),
      expr(x) && y,
      expr(x) || y,
      and_then(x, y),
      or_else(x, y),
      select(x, y, z),
      abs(x),
  };
}

std::vector<expr> simplified_edge_operands() {
  return {
      expr(0),
      expr(1),
      expr(-1),
      expr(2),
      expr(-2),
      expr(x),
      expr(y),
      expr(x) < y,
      abs(x),
  };
}

std::string format_ctx(const eval_context& ctx, int var_count = 4) {
  std::ostringstream ss;
  for (int i = 0; i < var_count; ++i) {
    ss << (i ? ", " : "") << symbols.name(var(i)) << "=" << ctx.lookup(var(i));
  }
  return ss.str();
}

}  // namespace

// Properties of boolean(), is_boolean(), is_true(), is_false().
TEST(expr_helpers, boolean_and_truthiness) {
  gtest_seeded_mt19937 rng;
  helper_expr_generator<gtest_seeded_mt19937> gen(rng);
  eval_context ctx;

  auto check_expr = [&](const expr& e) {
    if (!e.defined()) {
      EXPECT_FALSE(boolean(e).defined());
      EXPECT_FALSE(is_boolean(e));
      EXPECT_FALSE(is_true(e));
      EXPECT_FALSE(is_false(e));
      return true;
    }

    const expr b = boolean(e);
    const expr sb = simplify(b);
    const expr sne = simplify(e != 0);
    const expr snotnot = simplify(!!e);
    const expr ssel = simplify(select(e, 1, 0));

    bool ok = true;
    auto check = [&](bool cond, const std::string& msg) {
      if (!cond) {
        ok = false;
        ADD_FAILURE() << msg << " (e=" << e << ", boolean(e)=" << b << ")";
      }
    };

    // Idempotence of boolean().
    check(match(boolean(b), b), "boolean(boolean(e)) != boolean(e)");
    check(is_boolean(b), "is_boolean(boolean(e)) should hold");

    // Algebraic agreement of boolean(e) with simplify.
    auto equiv = [&](const expr& lhs, const expr& rhs) {
      if (match(lhs, rhs)) return true;
      for (int i = 0; i < eval_checks; ++i) {
        gen.init_context(ctx);
        if (evaluate(lhs, ctx) != evaluate(rhs, ctx)) return false;
      }
      return true;
    };
    check(equiv(sb, sne), "simplify(boolean(e)) != simplify(e != 0)");
    check(equiv(sb, snotnot), "simplify(boolean(e)) != simplify(!!e)");
    check(equiv(sb, ssel), "simplify(boolean(e)) != simplify(select(e, 1, 0))");

    // On constants, boolean(e) should fold directly to the simplified value (0 or 1).
    if (as_constant(e)) {
      const expr expected = simplify(static_cast<const not_equal*>(nullptr), e, 0);
      check(match(b, expected), "boolean(constant) did not fold to simplify(constant != 0)");
    }

    // Soundness of is_boolean(e).
    if (is_boolean(e)) {
      check(b.same_as(e), "boolean(e) should be a no-op when is_boolean(e)");
      check(match(simplify(e), sne), "is_boolean(e) implies simplify(e) == simplify(e != 0)");
    }

    // Mutual exclusion and consistency of is_true / is_false.
    check(!(is_true(e) && is_false(e)), "is_true(e) and is_false(e) must be mutually exclusive");
    check(is_false(e) == is_zero(e), "is_false(e) should agree with is_zero(e)");
    if (is_true(e)) {
      check(match(sb, 1), "is_true(e) implies simplify(boolean(e)) == 1");
      check(match(simplify(!e), 0), "is_true(e) implies simplify(!e) == 0");
      check(match(b, 1), "is_true(e) implies boolean(e) == 1");
    }
    if (is_false(e)) {
      check(match(sb, 0), "is_false(e) implies simplify(boolean(e)) == 0");
      check(match(simplify(!e), 1), "is_false(e) implies simplify(!e) == 1");
      check(match(b, 0), "is_false(e) implies boolean(e) == 0");
    }

    // Provably nonzero expressions (positive/negative) are truthy.
    if (is_positive(e) || is_negative(e)) {
      check(is_true(e), "is_positive(e) || is_negative(e) implies is_true(e)");
    }

    // Agreement with evaluate.
    for (int i = 0; i < eval_checks; ++i) {
      gen.init_context(ctx);
      const index_t ve = evaluate(e, ctx);
      const index_t vb = evaluate(b, ctx);
      const index_t vsb = evaluate(sb, ctx);
      check(vb == boolean(ve), "evaluate(boolean(e)) != boolean(evaluate(e)) at " + format_ctx(ctx));
      check(vb == vsb, "evaluate(boolean(e)) != evaluate(simplify(boolean(e))) at " + format_ctx(ctx));
      if (is_boolean(e)) {
        check(ve == 0 || ve == 1, "is_boolean(e) evaluated outside {0, 1} at " + format_ctx(ctx));
      }
      if (is_true(e)) {
        check(ve != 0, "is_true(e) evaluated to 0 at " + format_ctx(ctx));
      }
      if (is_false(e)) {
        check(ve == 0, "is_false(e) evaluated to nonzero at " + format_ctx(ctx));
      }
    }
    return ok;
  };

  for (const expr& e : edge_case_exprs()) {
    check_expr(e);
  }
  for (auto _ : fuzz_test(fuzz_duration)) {
    if (!check_expr(gen.random_expr(2))) break;
  }
}

// Properties of is_positive(), is_non_negative(), is_negative(), is_non_positive(), and constant predicates.
TEST(expr_helpers, sign_predicates) {
  gtest_seeded_mt19937 rng;
  helper_expr_generator<gtest_seeded_mt19937> gen(rng);
  eval_context ctx;

  auto check_expr = [&](const expr& e) {
    const bool pos = is_positive(e);
    const bool non_neg = is_non_negative(e);
    const bool neg = is_negative(e);
    const bool non_pos = is_non_positive(e);
    const bool zero = is_zero(e);
    const bool one = is_one(e);

    bool ok = true;
    auto check = [&](bool cond, const std::string& msg) {
      if (!cond) {
        ok = false;
        ADD_FAILURE() << msg << " (e=" << e << ")";
      }
    };

    if (!e.defined()) {
      check(!pos && !non_neg && !neg && !non_pos && !zero && !one,
          "undefined must not satisfy any sign or constant predicate");
      return ok;
    }

    // Constant predicate consistency.
    if (auto c = as_constant(e)) {
      check(is_constant(e, *c), "is_constant(e, *as_constant(e)) should hold");
      if (*c < std::numeric_limits<index_t>::max()) {
        check(!is_constant(e, *c + 1), "is_constant(e, c + 1) should be false");
      }
      check(zero == (*c == 0), "is_zero mismatch on constant");
      check(one == (*c == 1), "is_one mismatch on constant");
      check(pos == (*c > 0), "is_positive mismatch on constant");
      check(non_neg == (*c >= 0), "is_non_negative mismatch on constant");
      check(neg == (*c < 0), "is_negative mismatch on constant");
      check(non_pos == (*c <= 0), "is_non_positive mismatch on constant");
    } else {
      check(!zero && !one, "non-constant must not satisfy is_zero or is_one");
    }

    // Sign lattice implications.
    if (pos) check(non_neg && !neg && !non_pos && !zero, "is_positive lattice violation");
    if (neg) check(non_pos && !pos && !non_neg && !zero, "is_negative lattice violation");
    if (zero) check(non_neg && non_pos && !pos && !neg, "is_zero lattice violation");
    check((non_neg && non_pos) == zero, "(is_non_negative && is_non_positive) <=> is_zero");

    // Soundness against simplify.
    if (pos) check(match(simplify(e > 0), 1), "is_positive(e) implies simplify(e > 0) == 1");
    if (non_neg) check(match(simplify(e >= 0), 1), "is_non_negative(e) implies simplify(e >= 0) == 1");
    if (neg) check(match(simplify(e < 0), 1), "is_negative(e) implies simplify(e < 0) == 1");
    if (non_pos) check(match(simplify(e <= 0), 1), "is_non_positive(e) implies simplify(e <= 0) == 1");

    // Consistency with abs(e).
    const expr ae = abs(e);
    if (pos || neg) {
      check(is_positive(ae), "is_positive(e) || is_negative(e) implies is_positive(abs(e))");
    }
    check(is_non_negative(ae), "abs of defined expression should satisfy is_non_negative");

    // Soundness against evaluate.
    for (int i = 0; i < eval_checks; ++i) {
      gen.init_context(ctx);
      const index_t v = evaluate(e, ctx);
      if (pos) check(v > 0, "is_positive(e) evaluated <= 0 at " + format_ctx(ctx));
      if (non_neg) check(v >= 0, "is_non_negative(e) evaluated < 0 at " + format_ctx(ctx));
      if (neg) check(v < 0, "is_negative(e) evaluated >= 0 at " + format_ctx(ctx));
      if (non_pos) check(v <= 0, "is_non_positive(e) evaluated > 0 at " + format_ctx(ctx));
    }
    return ok;
  };

  for (const expr& e : edge_case_exprs()) {
    check_expr(e);
  }
  EXPECT_FALSE(is_non_negative(call::make(intrinsic::abs, {expr()})));

  // Helpers stay O(1) without digging through nested abs(); simplification still folds them.
  expr nested_x = x;
  for (int i = 0; i < 100; ++i) {
    nested_x = abs(nested_x);
  }
  EXPECT_FALSE(is_true(nested_x));
  EXPECT_FALSE(is_positive(nested_x));
  EXPECT_TRUE(is_non_negative(nested_x));
  EXPECT_THAT(simplify(nested_x), matches(abs(x)));

  expr dag = x;
  for (int i = 0; i < 40; ++i) {
    dag = dag + dag;
  }
  EXPECT_TRUE(is_non_negative(abs(dag)));

  for (auto _ : fuzz_test(fuzz_duration)) {
    if (!check_expr(gen.random_expr(2))) break;
  }
}

// Constructors (::make / operators) perform eager constant/identity folding; their simplified result must agree
// algebraically with calling simplify(const Op*, ...) directly on the operands.
TEST(expr_helpers, constructors_agree_with_simplify) {
  gtest_seeded_mt19937 rng;
  helper_expr_generator<gtest_seeded_mt19937> gen(rng);
  eval_context ctx;

  auto equivalent = [&](const expr& lhs, const expr& rhs) {
    if (match(lhs, rhs)) return true;
    for (int i = 0; i < eval_checks; ++i) {
      gen.init_context(ctx);
      if (evaluate(lhs, ctx) != evaluate(rhs, ctx)) return false;
    }
    return true;
  };

  auto check_binary_pair = [&](const expr& a, const expr& b) {
    bool ok = true;
    auto check_op = [&](auto tag, const char* name, expr via_make) {
      using Op = std::remove_pointer_t<decltype(tag)>;
      expr lhs = simplify(via_make);
      expr rhs = simplify(simplify(static_cast<const Op*>(nullptr), a, b));
      if (!equivalent(lhs, rhs)) {
        ok = false;
        ADD_FAILURE() << name << "::make(" << a << ", " << b << ") simplified to " << lhs
                      << ", but simplify(" << name << ", ...) gave " << rhs;
      }
    };

    check_op(static_cast<const add*>(nullptr), "add", add::make(a, b));
    check_op(static_cast<const sub*>(nullptr), "sub", sub::make(a, b));
    check_op(static_cast<const mul*>(nullptr), "mul", mul::make(a, b));
    check_op(static_cast<const div*>(nullptr), "div", div::make(a, b));
    check_op(static_cast<const mod*>(nullptr), "mod", mod::make(a, b));
    check_op(static_cast<const class min*>(nullptr), "min", min::make(a, b));
    check_op(static_cast<const class max*>(nullptr), "max", max::make(a, b));
    check_op(static_cast<const equal*>(nullptr), "equal", equal::make(a, b));
    check_op(static_cast<const not_equal*>(nullptr), "not_equal", not_equal::make(a, b));
    check_op(static_cast<const less*>(nullptr), "less", less::make(a, b));
    check_op(static_cast<const less_equal*>(nullptr), "less_equal", less_equal::make(a, b));
    check_op(static_cast<const logical_and*>(nullptr), "logical_and", logical_and::make(a, b));
    check_op(static_cast<const logical_or*>(nullptr), "logical_or", logical_or::make(a, b));

    // Short-circuit intrinsics.
    auto check_call = [&](intrinsic fn, const char* name, expr via_helper) {
      expr lhs = simplify(via_helper);
      expr rhs = simplify(simplify(static_cast<const call*>(nullptr), fn, {a, b}));
      if (!equivalent(lhs, rhs)) {
        ok = false;
        ADD_FAILURE() << name << "(" << a << ", " << b << ") simplified to " << lhs
                      << ", but simplify(" << name << ", ...) gave " << rhs;
      }
    };
    check_call(intrinsic::and_then, "and_then", and_then(a, b));
    check_call(intrinsic::or_else, "or_else", or_else(a, b));

    // Derived comparison & arithmetic operator helpers.
    auto check_match = [&](const expr& lhs, const expr& rhs, const char* label) {
      if (!equivalent(lhs, rhs)) {
        ok = false;
        ADD_FAILURE() << label << " mismatch for (" << a << ", " << b << "): " << lhs << " vs " << rhs;
      }
    };
    check_match(simplify(a > b), simplify(b < a), "a > b vs b < a");
    check_match(simplify(a >= b), simplify(b <= a), "a >= b vs b <= a");
    check_match(simplify(euclidean_div(a, b)), simplify(a / b), "euclidean_div vs /");
    check_match(simplify(euclidean_mod(a, b)), simplify(a % b), "euclidean_mod vs %");

    return ok;
  };

  auto check_unary = [&](const expr& a) {
    bool ok = true;
    auto check_match = [&](const expr& lhs, const expr& rhs, const char* label) {
      if (!equivalent(lhs, rhs)) {
        ok = false;
        ADD_FAILURE() << label << " mismatch for " << a << ": " << lhs << " vs " << rhs;
      }
    };
    check_match(simplify(logical_not::make(a)), simplify(simplify(static_cast<const logical_not*>(nullptr), a)),
        "logical_not");
    check_match(simplify(-a), simplify(simplify(static_cast<const sub*>(nullptr), 0, a)), "unary -");
    check_match(simplify(abs(a)), simplify(simplify(static_cast<const call*>(nullptr), intrinsic::abs, {a})), "abs");
    return ok;
  };

  auto check_select = [&](const expr& c, const expr& t, const expr& f) {
    expr via_make = select::make(c, t, f);
    expr lhs = simplify(via_make);
    expr rhs = simplify(simplify(static_cast<const class select*>(nullptr), c, t, f));
    if (!equivalent(lhs, rhs)) {
      ADD_FAILURE() << "select::make(" << c << ", " << t << ", " << f << ") simplified to " << lhs
                    << ", but simplify(select, ...) gave " << rhs;
      return false;
    }
    return true;
  };

  const std::vector<expr> operands = simplified_edge_operands();
  for (const expr& a : operands) {
    check_unary(a);
    for (const expr& b : operands) {
      check_binary_pair(a, b);
    }
  }

  for (const expr& c : {expr(0), expr(1), expr(-2), expr(x)}) {
    for (const expr& t : {expr(0), expr(x)}) {
      for (const expr& f : {expr(1), expr(y)}) {
        check_select(c, t, f);
      }
    }
  }

  for (auto _ : fuzz_test(fuzz_duration)) {
    expr a = gen.random_simplified_operand();
    expr b = gen.random_simplified_operand();
    expr c = gen.random_simplified_operand();
    if (!check_unary(a)) break;
    if (!check_binary_pair(a, b)) break;
    if (!check_select(c, a, b)) break;

    gen.init_context(ctx);
    ASSERT_EQ(evaluate(a + b, ctx), evaluate(simplify(a + b), ctx));
    ASSERT_EQ(evaluate(a - b, ctx), evaluate(simplify(a - b), ctx));
    ASSERT_EQ(evaluate(a * b, ctx), evaluate(simplify(a * b), ctx));
    ASSERT_EQ(evaluate(a / b, ctx), evaluate(simplify(a / b), ctx));
    ASSERT_EQ(evaluate(a % b, ctx), evaluate(simplify(a % b), ctx));
    ASSERT_EQ(evaluate(a && b, ctx), evaluate(simplify(a && b), ctx));
    ASSERT_EQ(evaluate(a || b, ctx), evaluate(simplify(a || b), ctx));
    ASSERT_EQ(evaluate(and_then(a, b), ctx), evaluate(simplify(and_then(a, b)), ctx));
    ASSERT_EQ(evaluate(or_else(a, b), ctx), evaluate(simplify(or_else(a, b)), ctx));
  }
}

// Non-annihilating operations propagate undefined operands, while annihilating operations (0 * x, 0 / x, x / 0,
// 0 % x, x % 0, x % +/-1, false && x, true || x, and_then(false, x), or_else(true, x)) fold to the annihilator
// in both constructors and simplify().
TEST(expr_helpers, undefined_propagation) {
  const expr undef;

  // Annihilating instances in constructors.
  EXPECT_THAT(mul::make(undef, 0), matches(0));
  EXPECT_THAT(mul::make(0, undef), matches(0));
  EXPECT_THAT(div::make(undef, 0), matches(0));
  EXPECT_THAT(div::make(0, undef), matches(0));
  EXPECT_THAT(mod::make(undef, 0), matches(0));
  EXPECT_THAT(mod::make(0, undef), matches(0));
  EXPECT_THAT(mod::make(undef, 1), matches(0));
  EXPECT_THAT(mod::make(undef, -1), matches(0));
  EXPECT_THAT(logical_and::make(undef, 0), matches(0));
  EXPECT_THAT(logical_and::make(0, undef), matches(0));
  EXPECT_THAT(and_then(0, undef), matches(0));
  for (const expr& truthy : {expr(1), expr(-1), expr(2)}) {
    EXPECT_THAT(logical_or::make(undef, truthy), matches(1)) << "undef || " << truthy;
    EXPECT_THAT(logical_or::make(truthy, undef), matches(1)) << truthy << " || undef";
    EXPECT_THAT(or_else(truthy, undef), matches(1)) << "or_else(" << truthy << ", undef)";
  }

  // Annihilating instances inside simplify() when an operand simplifies to undefined
  // (semaphore_wait(undef) is a defined call node that simplifies to undef, while y - y simplifies to 0).
  expr undef_expr = semaphore_wait(undef);
  ASSERT_TRUE(undef_expr.defined());
  ASSERT_FALSE(simplify(undef_expr).defined());
  expr zero_expr = y - y;
  expr one_expr = y - y + 1;
  expr neg_one_expr = y - y - 1;
  EXPECT_THAT(simplify(undef_expr * zero_expr), matches(0));
  EXPECT_THAT(simplify(zero_expr * undef_expr), matches(0));
  EXPECT_THAT(simplify(undef_expr / zero_expr), matches(0));
  EXPECT_THAT(simplify(zero_expr / undef_expr), matches(0));
  EXPECT_THAT(simplify(undef_expr % zero_expr), matches(0));
  EXPECT_THAT(simplify(zero_expr % undef_expr), matches(0));
  EXPECT_THAT(simplify(undef_expr % one_expr), matches(0));
  EXPECT_THAT(simplify(undef_expr % neg_one_expr), matches(0));
  EXPECT_THAT(simplify(undef_expr && zero_expr), matches(0));
  EXPECT_THAT(simplify(zero_expr && undef_expr), matches(0));
  EXPECT_THAT(simplify(undef_expr || one_expr), matches(1));
  EXPECT_THAT(simplify(one_expr || undef_expr), matches(1));
  EXPECT_THAT(simplify(and_then(zero_expr, undef_expr)), matches(0));
  EXPECT_THAT(simplify(or_else(one_expr, undef_expr)), matches(1));
  EXPECT_THAT(simplify(select(one_expr, x, undef_expr)), matches(x));
  EXPECT_THAT(simplify(select(zero_expr, undef_expr, x)), matches(x));
  EXPECT_FALSE(simplify(select(x, undef_expr, y)).defined());
  EXPECT_FALSE(simplify(select(x, y, undef_expr)).defined());
  EXPECT_FALSE(simplify(abs(undef_expr)).defined());
  EXPECT_FALSE(simplify(and_then(one_expr, undef_expr)).defined());
  EXPECT_FALSE(simplify(and_then(undef_expr, zero_expr)).defined());
  EXPECT_FALSE(simplify(or_else(zero_expr, undef_expr)).defined());
  EXPECT_FALSE(simplify(or_else(undef_expr, one_expr)).defined());

  // Non-annihilating instances must propagate undefined.
  for (const expr& a : {expr(0), expr(1), expr(-1), expr(2), expr(x)}) {
    EXPECT_FALSE(add::make(undef, a).defined()) << "add(undef, " << a << ")";
    EXPECT_FALSE(add::make(a, undef).defined()) << "add(" << a << ", undef)";
    EXPECT_FALSE(sub::make(undef, a).defined()) << "sub(undef, " << a << ")";
    EXPECT_FALSE(sub::make(a, undef).defined()) << "sub(" << a << ", undef)";
    if (!is_zero(a)) {
      EXPECT_FALSE(mul::make(undef, a).defined()) << "mul(undef, " << a << ")";
      EXPECT_FALSE(mul::make(a, undef).defined()) << "mul(" << a << ", undef)";
      EXPECT_FALSE(div::make(undef, a).defined()) << "div(undef, " << a << ")";
      EXPECT_FALSE(div::make(a, undef).defined()) << "div(" << a << ", undef)";
      EXPECT_FALSE(mod::make(a, undef).defined()) << "mod(" << a << ", undef)";
    }
    if (!is_zero(a) && !is_one(a) && !is_constant(a, -1)) {
      EXPECT_FALSE(mod::make(undef, a).defined()) << "mod(undef, " << a << ")";
    }
    EXPECT_FALSE(min::make(undef, a).defined()) << "min(undef, " << a << ")";
    EXPECT_FALSE(min::make(a, undef).defined()) << "min(" << a << ", undef)";
    EXPECT_FALSE(max::make(undef, a).defined()) << "max(undef, " << a << ")";
    EXPECT_FALSE(max::make(a, undef).defined()) << "max(" << a << ", undef)";
    EXPECT_FALSE(equal::make(undef, a).defined()) << "equal(undef, " << a << ")";
    EXPECT_FALSE(equal::make(a, undef).defined()) << "equal(" << a << ", undef)";
    EXPECT_FALSE(not_equal::make(undef, a).defined()) << "not_equal(undef, " << a << ")";
    EXPECT_FALSE(not_equal::make(a, undef).defined()) << "not_equal(" << a << ", undef)";
    EXPECT_FALSE(less::make(undef, a).defined()) << "less(undef, " << a << ")";
    EXPECT_FALSE(less::make(a, undef).defined()) << "less(" << a << ", undef)";
    EXPECT_FALSE(less_equal::make(undef, a).defined()) << "less_equal(undef, " << a << ")";
    EXPECT_FALSE(less_equal::make(a, undef).defined()) << "less_equal(" << a << ", undef)";
    if (!is_false(a)) {
      EXPECT_FALSE(logical_and::make(undef, a).defined()) << "logical_and(undef, " << a << ")";
      EXPECT_FALSE(logical_and::make(a, undef).defined()) << "logical_and(" << a << ", undef)";
      EXPECT_FALSE(and_then(a, undef).defined()) << "and_then(" << a << ", undef)";
    }
    EXPECT_FALSE(and_then(undef, a).defined()) << "and_then(undef, " << a << ")";
    if (!is_true(a)) {
      EXPECT_FALSE(logical_or::make(undef, a).defined()) << "logical_or(undef, " << a << ")";
      EXPECT_FALSE(logical_or::make(a, undef).defined()) << "logical_or(" << a << ", undef)";
      EXPECT_FALSE(or_else(a, undef).defined()) << "or_else(" << a << ", undef)";
    }
    EXPECT_FALSE(or_else(undef, a).defined()) << "or_else(undef, " << a << ")";
    EXPECT_THAT(select::make(undef, a, a), matches(a)) << "select(undef, " << a << ", " << a << ")";
    EXPECT_FALSE(select::make(undef, a, a + 1).defined()) << "select(undef, " << a << ", " << a + 1 << ")";
    EXPECT_THAT(select::make(1, a, undef), matches(a)) << "select(1, " << a << ", undef)";
    EXPECT_THAT(select::make(0, undef, a), matches(a)) << "select(0, undef, " << a << ")";
    EXPECT_FALSE(select::make(x, undef, a).defined()) << "select(x, undef, " << a << ")";
    EXPECT_FALSE(select::make(x, a, undef).defined()) << "select(x, " << a << ", undef)";
  }
  EXPECT_FALSE(abs(undef).defined()) << "abs(undef)";
  EXPECT_FALSE(call::make(intrinsic::abs, {undef}).defined()) << "call::make(abs, {undef})";
  EXPECT_THAT(call::make(intrinsic::and_then, {expr(0), undef}), matches(0));
  EXPECT_FALSE(call::make(intrinsic::and_then, {expr(x), undef}).defined());
  EXPECT_THAT(call::make(intrinsic::or_else, {expr(1), undef}), matches(1));
  EXPECT_FALSE(call::make(intrinsic::or_else, {expr(x), undef}).defined());
  EXPECT_FALSE(let::make(x, undef, expr(x) + 1).defined()) << "let(x = undef, x + 1)";
  EXPECT_FALSE(let::make(x, 1, undef).defined()) << "let(x = 1, undef)";
  EXPECT_FALSE(logical_not::make(undef).defined()) << "logical_not(undef)";
  EXPECT_FALSE(select::make(x, undef, undef).defined()) << "select(x, undef, undef)";
  EXPECT_FALSE(boolean(undef).defined()) << "boolean(undef)";
}

// bounds_of(const select*, c, t, f) must handle arbitrary integer intervals for c (not just [0, 1]),
// consistent with evaluate() and is_true()'s truthiness semantics.
TEST(expr_helpers, bounds_of_select) {
  const expr inf;
  const interval_expr t = bounds(10, 20);
  const interval_expr f = bounds(100, 200);
  const interval_expr both = bounds(10, 200);

  // Condition intervals containing both 0 and nonzero values must include both arms.
  EXPECT_THAT(bounds_of(static_cast<const class select*>(nullptr), bounds(-1, 1), t, f), matches(both));
  EXPECT_THAT(bounds_of(static_cast<const class select*>(nullptr), bounds(-5, 0), t, f), matches(both));
  EXPECT_THAT(bounds_of(static_cast<const class select*>(nullptr), bounds(0, 5), t, f), matches(both));
  EXPECT_THAT(bounds_of(static_cast<const class select*>(nullptr), interval_expr::all(), t, f), matches(both));

  // Condition intervals excluding 0 select t; [0, 0] selects f.
  EXPECT_THAT(bounds_of(static_cast<const class select*>(nullptr), bounds(1, inf), t, f), matches(t));
  EXPECT_THAT(bounds_of(static_cast<const class select*>(nullptr), bounds(inf, -1), t, f), matches(t));
  EXPECT_THAT(bounds_of(static_cast<const class select*>(nullptr), bounds(2, 5), t, f), matches(t));
  EXPECT_THAT(bounds_of(static_cast<const class select*>(nullptr), bounds(-5, -2), t, f), matches(t));
  EXPECT_THAT(bounds_of(static_cast<const class select*>(nullptr), bounds(0, 0), t, f), matches(f));

  // Exhaustive soundness check over small integer condition intervals for select, logical_not, logical_and, logical_or.
  for (int c_min = -3; c_min <= 3; ++c_min) {
    for (int c_max = c_min; c_max <= 3; ++c_max) {
      interval_expr b = bounds_of(static_cast<const class select*>(nullptr), bounds(c_min, c_max), t, f);
      index_t bmin = evaluate(b.min);
      index_t bmax = evaluate(b.max);
      interval_expr bn = bounds_of(static_cast<const logical_not*>(nullptr), bounds(c_min, c_max));
      index_t bn_min = evaluate(bn.min);
      index_t bn_max = evaluate(bn.max);
      EXPECT_LE(bn_min, bn_max) << "logical_not inverted for c=[" << c_min << ", " << c_max << "]: " << bn;
      for (int c_val = c_min; c_val <= c_max; ++c_val) {
        index_t arm_min = c_val != 0 ? 10 : 100;
        index_t arm_max = c_val != 0 ? 20 : 200;
        EXPECT_LE(bmin, arm_min) << "c=[" << c_min << ", " << c_max << "], c_val=" << c_val << ", b=" << b;
        EXPECT_GE(bmax, arm_max) << "c=[" << c_min << ", " << c_max << "], c_val=" << c_val << ", b=" << b;
        index_t not_val = c_val == 0 ? 1 : 0;
        EXPECT_LE(bn_min, not_val) << "logical_not c=[" << c_min << ", " << c_max << "], c_val=" << c_val;
        EXPECT_GE(bn_max, not_val) << "logical_not c=[" << c_min << ", " << c_max << "], c_val=" << c_val;
      }

      for (int d_min = -2; d_min <= 2; ++d_min) {
        for (int d_max = d_min; d_max <= 2; ++d_max) {
          interval_expr ba =
              bounds_of(static_cast<const logical_and*>(nullptr), bounds(c_min, c_max), bounds(d_min, d_max));
          interval_expr bo =
              bounds_of(static_cast<const logical_or*>(nullptr), bounds(c_min, c_max), bounds(d_min, d_max));
          index_t ba_min = evaluate(ba.min), ba_max = evaluate(ba.max);
          index_t bo_min = evaluate(bo.min), bo_max = evaluate(bo.max);
          EXPECT_LE(ba_min, ba_max);
          EXPECT_LE(bo_min, bo_max);
          for (int cv = c_min; cv <= c_max; ++cv) {
            for (int dv = d_min; dv <= d_max; ++dv) {
              index_t and_v = (cv != 0 && dv != 0) ? 1 : 0;
              index_t or_v = (cv != 0 || dv != 0) ? 1 : 0;
              EXPECT_LE(ba_min, and_v);
              EXPECT_GE(ba_max, and_v);
              EXPECT_LE(bo_min, or_v);
              EXPECT_GE(bo_max, or_v);
            }
          }
        }
      }
    }
  }
}

// Properties of clamp(), abs(), align_down(), align_up(), min(span), max(span), and_then(), or_else().
TEST(expr_helpers, arithmetic_helpers) {
  gtest_seeded_mt19937 rng;
  helper_expr_generator<gtest_seeded_mt19937> gen(rng);
  eval_context ctx;

  // Empty and singleton span min/max.
  std::vector<expr> empty;
  EXPECT_FALSE(min(span<expr>(empty)).defined());
  EXPECT_FALSE(max(span<expr>(empty)).defined());
  std::vector<expr> single = {expr(x)};
  EXPECT_TRUE(min(span<expr>(single)).same_as(single[0]));
  EXPECT_TRUE(max(span<expr>(single)).same_as(single[0]));

  for (auto _ : fuzz_test(fuzz_duration)) {
    expr e = gen.random_expr(1);
    expr lo = gen.random_expr(1);
    expr hi = gen.random_expr(1);
    // Positive alignment (sometimes 1 to hit the fast path).
    expr align_expr = (rng() % 4 == 0) ? expr(1) : (abs(gen.random_expr(1)) + 1);

    expr clamped = clamp(e, lo, hi);
    expr clamped_simplified = simplify(clamped);
    expr abs_e = abs(e);
    expr abs_simplified = simplify(abs_e);
    expr down = align_down(e, align_expr);
    expr down_simplified = simplify(down);
    expr up = align_up(e, align_expr);
    expr up_simplified = simplify(up);

    std::vector<expr> list = {e, lo, hi};
    expr min_list = min(span<expr>(list));
    expr max_list = max(span<expr>(list));

    for (int i = 0; i < eval_checks; ++i) {
      gen.init_context(ctx);
      const index_t ve = evaluate(e, ctx);
      const index_t vlo = evaluate(lo, ctx);
      const index_t vhi = evaluate(hi, ctx);
      const index_t va = evaluate(align_expr, ctx);
      ASSERT_GT(va, 0);

      // clamp
      const index_t vc = evaluate(clamped, ctx);
      ASSERT_EQ(vc, std::max(std::min(ve, vhi), vlo)) << clamped;
      ASSERT_EQ(vc, evaluate(clamped_simplified, ctx)) << clamped << " -> " << clamped_simplified;
      if (vlo <= vhi) {
        ASSERT_GE(vc, vlo);
        ASSERT_LE(vc, vhi);
        if (vlo <= ve && ve <= vhi) ASSERT_EQ(vc, ve);
      }

      // abs
      const index_t vabs = evaluate(abs_e, ctx);
      ASSERT_EQ(vabs, std::abs(ve)) << abs_e;
      ASSERT_EQ(vabs, evaluate(abs_simplified, ctx)) << abs_e << " -> " << abs_simplified;
      ASSERT_EQ(evaluate(abs(-e), ctx), vabs);
      ASSERT_EQ(evaluate(abs(abs_e), ctx), vabs);

      // align_down / align_up
      const index_t vdown = evaluate(down, ctx);
      const index_t vup = evaluate(up, ctx);
      ASSERT_EQ(vdown, slinky::align_down(ve, va)) << down;
      ASSERT_EQ(vup, slinky::align_up(ve, va)) << up;
      ASSERT_EQ(vdown, evaluate(down_simplified, ctx)) << down << " -> " << down_simplified;
      ASSERT_EQ(vup, evaluate(up_simplified, ctx)) << up << " -> " << up_simplified;
      ASSERT_LE(vdown, ve);
      ASSERT_LT(ve, vdown + va);
      ASSERT_EQ(euclidean_mod(vdown, va), 0);
      ASSERT_LE(ve, vup);
      ASSERT_LT(vup, ve + va);
      ASSERT_EQ(euclidean_mod(vup, va), 0);
      ASSERT_EQ(vup, -evaluate(align_down(-e, align_expr), ctx));

      // min/max(span)
      ASSERT_EQ(evaluate(min_list, ctx), std::min({ve, vlo, vhi}));
      ASSERT_EQ(evaluate(max_list, ctx), std::max({ve, vlo, vhi}));
      ASSERT_EQ(evaluate(simplify(min_list), ctx), std::min({ve, vlo, vhi}));
      ASSERT_EQ(evaluate(simplify(max_list), ctx), std::max({ve, vlo, vhi}));

      // and_then / or_else
      ASSERT_EQ(evaluate(and_then(e, lo), ctx), (ve != 0 && vlo != 0) ? 1 : 0);
      ASSERT_EQ(evaluate(or_else(e, lo), ctx), (ve != 0 || vlo != 0) ? 1 : 0);
    }
  }
}

// Algebraic properties of interval_expr helpers and agreement with simplify / bounds_of / evaluate.
TEST(expr_helpers, interval_expr_algebra) {
  gtest_seeded_mt19937 rng;
  helper_expr_generator<gtest_seeded_mt19937> gen(rng);
  eval_context ctx;
  const expr inf;

  // Default interval_expr{} == all() == intersection_identity().
  EXPECT_TRUE(interval_expr().same_as(interval_expr::all()));
  EXPECT_TRUE(interval_expr(expr(), expr()).same_as(interval_expr::all()));
  EXPECT_TRUE(point(expr()).same_as(interval_expr::all()));
  EXPECT_FALSE(point(expr()).is_point());
  EXPECT_FALSE(interval_expr::all().min.defined());
  EXPECT_FALSE(interval_expr::all().max.defined());
  EXPECT_THAT(interval_expr::all().empty(), matches(0));
  EXPECT_THAT(interval_expr::all().contains(expr(x)), matches(1));
  EXPECT_THAT(bounds(inf, 5).empty(), matches(0));
  EXPECT_THAT(bounds(5, inf).empty(), matches(0));
  EXPECT_THAT(bounds(inf, 5).contains(expr(x)), matches(x <= 5));
  EXPECT_THAT(bounds(5, inf).contains(expr(x)), matches(5 <= x));

  // Identity and idempotence laws.
  for (const interval_expr& iv :
      {bounds(x, y), point(x), bounds(0, 10), bounds(inf, 5), bounds(5, inf), interval_expr::all()}) {
    EXPECT_THAT(simplify(iv & interval_expr::intersection_identity()), matches(simplify(iv)));
    EXPECT_THAT(simplify(interval_expr::intersection_identity() & iv), matches(simplify(iv)));
    EXPECT_THAT(simplify(iv | interval_expr::all()), matches(interval_expr::all()));
    EXPECT_THAT(simplify(interval_expr::all() | iv), matches(interval_expr::all()));
    EXPECT_THAT(simplify(iv | iv), matches(simplify(iv)));
    EXPECT_THAT(simplify(iv & iv), matches(simplify(iv)));
  }

  // Agreement with bounds_of for scaling/dividing unbounded and half-unbounded intervals.
  EXPECT_THAT(simplify(interval_expr::all() + y), matches(interval_expr::all()));
  EXPECT_THAT(simplify(-interval_expr::all()), matches(interval_expr::all()));
  EXPECT_THAT(simplify(-bounds(inf, 5)), matches(bounds(-5, inf)));
  EXPECT_THAT(simplify(-bounds(5, inf)), matches(bounds(inf, -5)));
  EXPECT_THAT(simplify(interval_expr::all() * 2), matches(interval_expr::all()));
  EXPECT_THAT(simplify(interval_expr::all() * -2), matches(interval_expr::all()));
  EXPECT_THAT(simplify(interval_expr::all() * 0), matches(point(0)));
  EXPECT_THAT(simplify(bounds(0, inf) * -2), matches(bounds(inf, 0)));
  EXPECT_THAT(simplify(bounds(inf, 5) * -2), matches(bounds(-10, inf)));
  EXPECT_THAT(simplify(bounds(0, inf) / -2), matches(bounds(inf, 0)));
  EXPECT_THAT(simplify(interval_expr::all() * y),
      matches(bounds_of(static_cast<const mul*>(nullptr), interval_expr::all(), point(y))));
  EXPECT_THAT(simplify(interval_expr::all() / y),
      matches(bounds_of(static_cast<const div*>(nullptr), interval_expr::all(), point(y))));
  EXPECT_THAT(simplify(interval_expr::all() * abs(y)), matches(interval_expr::all()));
  EXPECT_THAT(simplify(interval_expr::all() / abs(y)), matches(interval_expr::all()));
  EXPECT_THAT(simplify(bounds(0, inf) * abs(y)), matches(bounds(0, inf)));
  EXPECT_THAT(simplify(bounds(0, inf) / abs(y)), matches(bounds(0, inf)));
  EXPECT_THAT(simplify(align(interval_expr::all(), 4)), matches(interval_expr::all()));
  EXPECT_THAT(simplify(align(bounds(3, inf), 4)), matches(bounds(0, inf)));
  EXPECT_THAT(simplify(align(bounds(inf, 3), 4)), matches(bounds(inf, 3)));

  // select(c, t, f) with unbounded arms widens that side to unbounded; constant conditions pick the active arm.
  EXPECT_THAT(select(x, bounds(inf, 5), bounds(0, 10)), matches(bounds(inf, select(x, 5, 10))));
  EXPECT_THAT(select(x, bounds(0, 10), bounds(5, inf)), matches(bounds(select(x, 0, 5), inf)));
  EXPECT_THAT(select(x, interval_expr::all(), bounds(0, 10)), matches(interval_expr::all()));
  EXPECT_THAT(select(expr(1), bounds(0, 10), interval_expr::all()), matches(bounds(0, 10)));
  EXPECT_THAT(select(expr(0), interval_expr::all(), bounds(0, 10)), matches(bounds(0, 10)));

  // clamp(x, interval_expr) with half-unbounded and unbounded intervals.
  EXPECT_THAT(simplify(clamp(x, interval_expr::all())), matches(x));
  EXPECT_THAT(simplify(clamp(x, bounds(0, inf))), matches(max(x, 0)));
  EXPECT_THAT(simplify(clamp(x, bounds(inf, 10))), matches(min(x, 10)));

  for (auto _ : fuzz_test(fuzz_duration)) {
    interval_expr iv1 = gen.random_finite_interval();
    interval_expr iv2 = gen.random_finite_interval();
    expr s = gen.random_expr(1);
    expr p = gen.random_expr(1);
    expr c = gen.random_expr(1);
    expr align_a = (rng() % 4 == 0) ? expr(1) : (abs(gen.random_expr(1)) + 1);

    interval_expr u = iv1 | iv2;
    interval_expr inter = iv1 & iv2;
    interval_expr sum = iv1 + s;
    interval_expr sum_rev = s + iv1;
    interval_expr diff = iv1 - s;
    interval_expr diff_rev = s - iv1;
    interval_expr neg = -iv1;
    interval_expr scaled = iv1 * s;
    interval_expr scaled_rev = s * iv1;
    interval_expr divided = iv1 / s;
    interval_expr sel = select(c, iv1, iv2);
    interval_expr aligned = align(iv1, align_a);

    interval_expr u_s = simplify(u);
    interval_expr inter_s = simplify(inter);
    interval_expr scaled_s = simplify(scaled);
    interval_expr divided_s = simplify(divided);

    for (int i = 0; i < eval_checks; ++i) {
      gen.init_context(ctx);
      const index_t min1 = evaluate(iv1.min, ctx);
      const index_t max1 = evaluate(iv1.max, ctx);
      const index_t min2 = evaluate(iv2.min, ctx);
      const index_t max2 = evaluate(iv2.max, ctx);
      const index_t vs = evaluate(s, ctx);
      const index_t vp = evaluate(p, ctx);
      const index_t vc = evaluate(c, ctx);
      const index_t va = evaluate(align_a, ctx);

      // Accessors & range / min_extent / point.
      ASSERT_EQ(evaluate(iv1.begin(), ctx), min1);
      ASSERT_EQ(evaluate(iv1.end(), ctx), max1 + 1);
      ASSERT_EQ(evaluate(iv1.extent(), ctx), max1 - min1 + 1);
      ASSERT_EQ(evaluate(iv1.empty(), ctx), (min1 > max1) ? 1 : 0);
      ASSERT_EQ(evaluate(iv1.contains(p), ctx), (min1 <= vp && vp <= max1) ? 1 : 0);

      interval_expr r = range(iv1.min, iv1.max);
      ASSERT_EQ(evaluate(r.begin(), ctx), min1);
      ASSERT_EQ(evaluate(r.end(), ctx), max1);
      ASSERT_EQ(evaluate(r.extent(), ctx), max1 - min1);

      interval_expr me = min_extent(iv1.min, s);
      ASSERT_EQ(evaluate(me.begin(), ctx), min1);
      ASSERT_EQ(evaluate(me.extent(), ctx), vs);

      // Union / intersection semantics & simplify agreement.
      const bool in1 = min1 <= vp && vp <= max1;
      const bool in2 = min2 <= vp && vp <= max2;
      ASSERT_EQ(evaluate(inter.contains(p), ctx), (in1 && in2) ? 1 : 0);
      if (in1 || in2) ASSERT_EQ(evaluate(u.contains(p), ctx), 1);
      ASSERT_EQ(evaluate(u.min, ctx), evaluate(u_s.min, ctx));
      ASSERT_EQ(evaluate(u.max, ctx), evaluate(u_s.max, ctx));
      ASSERT_EQ(evaluate(inter.min, ctx), evaluate(inter_s.min, ctx));
      ASSERT_EQ(evaluate(inter.max, ctx), evaluate(inter_s.max, ctx));
      ASSERT_EQ(evaluate(scaled.min, ctx), evaluate(scaled_s.min, ctx));
      ASSERT_EQ(evaluate(scaled.max, ctx), evaluate(scaled_s.max, ctx));
      ASSERT_EQ(evaluate(divided.min, ctx), evaluate(divided_s.min, ctx));
      ASSERT_EQ(evaluate(divided.max, ctx), evaluate(divided_s.max, ctx));

      // select(c, iv1, iv2)
      ASSERT_EQ(evaluate(sel.min, ctx), vc != 0 ? min1 : min2);
      ASSERT_EQ(evaluate(sel.max, ctx), vc != 0 ? max1 : max2);

      // For non-empty iv1, arithmetic on any contained point stays in the transformed interval.
      if (min1 <= max1) {
        const index_t q = std::clamp(vp, min1, max1);
        ASSERT_EQ(evaluate(iv1.contains(expr(q)), ctx), 1);
        ASSERT_EQ(evaluate(sum.contains(expr(q + vs)), ctx), 1);
        ASSERT_EQ(evaluate(sum_rev.contains(expr(vs + q)), ctx), 1);
        ASSERT_EQ(evaluate(diff.contains(expr(q - vs)), ctx), 1);
        ASSERT_EQ(evaluate(diff_rev.contains(expr(vs - q)), ctx), 1);
        ASSERT_EQ(evaluate(neg.contains(expr(-q)), ctx), 1);
        ASSERT_EQ(evaluate(scaled.contains(expr(q * vs)), ctx), 1);
        ASSERT_EQ(evaluate(scaled_rev.contains(expr(vs * q)), ctx), 1);
        ASSERT_EQ(evaluate(divided.contains(expr(euclidean_div(q, vs))), ctx), 1);

        // align(iv1, a) expands iv1 to aligned min and extent.
        const index_t amin = evaluate(aligned.min, ctx);
        const index_t amax = evaluate(aligned.max, ctx);
        ASSERT_LE(amin, min1);
        ASSERT_GE(amax, max1);
        ASSERT_EQ(euclidean_mod(amin, va), 0);
        ASSERT_EQ(euclidean_mod(amax - amin + 1, va), 0);
      }
    }
  }
}

// Soundness of bounds_of() on finite, half-unbounded, and fully unbounded intervals.
TEST(expr_helpers, bounds_of_unbounded_soundness) {
  gtest_seeded_mt19937 rng;
  helper_expr_generator<gtest_seeded_mt19937> gen(rng);
  eval_context ctx;

  auto in_interval = [&](const interval_expr& iv, index_t v) {
    if (iv.min.defined() && v < evaluate(iv.min, ctx)) return false;
    if (iv.max.defined() && v > evaluate(iv.max, ctx)) return false;
    return true;
  };

  auto sample_in = [&](const interval_expr& iv) -> std::optional<index_t> {
    index_t lo = iv.min.defined() ? evaluate(iv.min, ctx) : -20;
    index_t hi = iv.max.defined() ? evaluate(iv.max, ctx) : 20;
    if (iv.min.defined() && !iv.max.defined()) hi = lo + 20;
    if (!iv.min.defined() && iv.max.defined()) lo = hi - 20;
    if (lo > hi) return std::nullopt;
    return lo + static_cast<index_t>(rng() % static_cast<std::uint64_t>(hi - lo + 1));
  };

  for (auto _ : fuzz_test(fuzz_duration)) {
    interval_expr a = gen.random_interval();
    interval_expr b = gen.random_interval();
    interval_expr c = gen.random_interval();

    interval_expr b_add = bounds_of(static_cast<const add*>(nullptr), a, b);
    interval_expr b_sub = bounds_of(static_cast<const sub*>(nullptr), a, b);
    interval_expr b_mul = bounds_of(static_cast<const mul*>(nullptr), a, b);
    interval_expr b_div = bounds_of(static_cast<const div*>(nullptr), a, b);
    interval_expr b_mod = bounds_of(static_cast<const mod*>(nullptr), a, b);
    interval_expr b_min = bounds_of(static_cast<const class min*>(nullptr), a, b);
    interval_expr b_max = bounds_of(static_cast<const class max*>(nullptr), a, b);
    interval_expr b_lt = bounds_of(static_cast<const less*>(nullptr), a, b);
    interval_expr b_le = bounds_of(static_cast<const less_equal*>(nullptr), a, b);
    interval_expr b_eq = bounds_of(static_cast<const equal*>(nullptr), a, b);
    interval_expr b_ne = bounds_of(static_cast<const not_equal*>(nullptr), a, b);
    interval_expr b_and = bounds_of(static_cast<const logical_and*>(nullptr), a, b);
    interval_expr b_or = bounds_of(static_cast<const logical_or*>(nullptr), a, b);
    interval_expr b_not = bounds_of(static_cast<const logical_not*>(nullptr), a);
    interval_expr b_abs = bounds_of(call::make(intrinsic::abs, {x}).as<call>(), {a});
    interval_expr b_sel = bounds_of(static_cast<const class select*>(nullptr), c, a, b);

    for (int i = 0; i < eval_checks; ++i) {
      gen.init_context(ctx);
      auto va = sample_in(a);
      auto vb = sample_in(b);
      auto vc = sample_in(c);
      if (!va || !vb || !vc) continue;

      ASSERT_TRUE(in_interval(b_add, *va + *vb)) << "add: a=" << a << " b=" << b << " res=" << b_add;
      ASSERT_TRUE(in_interval(b_sub, *va - *vb)) << "sub: a=" << a << " b=" << b << " res=" << b_sub;
      ASSERT_TRUE(in_interval(b_mul, *va * *vb)) << "mul: a=" << a << " b=" << b << " res=" << b_mul;
      ASSERT_TRUE(in_interval(b_div, euclidean_div(*va, *vb))) << "div: a=" << a << " b=" << b << " res=" << b_div;
      ASSERT_TRUE(in_interval(b_mod, euclidean_mod(*va, *vb))) << "mod: a=" << a << " b=" << b << " res=" << b_mod;
      ASSERT_TRUE(in_interval(b_min, std::min(*va, *vb))) << "min: a=" << a << " b=" << b << " res=" << b_min;
      ASSERT_TRUE(in_interval(b_max, std::max(*va, *vb))) << "max: a=" << a << " b=" << b << " res=" << b_max;
      ASSERT_TRUE(in_interval(b_lt, *va < *vb ? 1 : 0)) << "lt: a=" << a << " b=" << b << " res=" << b_lt;
      ASSERT_TRUE(in_interval(b_le, *va <= *vb ? 1 : 0)) << "le: a=" << a << " b=" << b << " res=" << b_le;
      ASSERT_TRUE(in_interval(b_eq, *va == *vb ? 1 : 0)) << "eq: a=" << a << " b=" << b << " res=" << b_eq;
      ASSERT_TRUE(in_interval(b_ne, *va != *vb ? 1 : 0)) << "ne: a=" << a << " b=" << b << " res=" << b_ne;
      ASSERT_TRUE(in_interval(b_and, (*va != 0 && *vb != 0) ? 1 : 0)) << "and: a=" << a << " b=" << b << " res=" << b_and;
      ASSERT_TRUE(in_interval(b_or, (*va != 0 || *vb != 0) ? 1 : 0)) << "or: a=" << a << " b=" << b << " res=" << b_or;
      ASSERT_TRUE(in_interval(b_not, *va == 0 ? 1 : 0)) << "not: a=" << a << " res=" << b_not;
      ASSERT_TRUE(in_interval(b_abs, std::abs(*va))) << "abs: a=" << a << " res=" << b_abs;
      ASSERT_TRUE(in_interval(b_sel, *vc != 0 ? *va : *vb))
          << "select: c=" << c << " a=" << a << " b=" << b << " res=" << b_sel;
    }
  }
}

// Algebraic properties of box_expr union/intersection.
TEST(expr_helpers, box_expr_algebra) {
  // Union and intersection of boxes should be commutative and produce exact expected bounds, even when ranks differ.
  box_expr a = {bounds(0, 5)};
  box_expr b = {bounds(1, 4), bounds(2, 3)};

  for (const box_expr& u : {a | b, b | a}) {
    ASSERT_EQ(u.size(), 2u);
    EXPECT_THAT(simplify(u[0]), matches(bounds(0, 5)));
    EXPECT_FALSE(u[1].min.defined());
    EXPECT_FALSE(u[1].max.defined());
  }

  for (const box_expr& i : {a & b, b & a}) {
    ASSERT_EQ(i.size(), 2u);
    EXPECT_THAT(simplify(i[0]), matches(bounds(1, 4)));
    EXPECT_THAT(simplify(i[1]), matches(bounds(2, 3)));
  }

  // Empty box_expr{} acts as unconstrained (unbounded) on all dimensions.
  for (const box_expr& u : {b | box_expr{}, box_expr{} | b}) {
    ASSERT_EQ(u.size(), 2u);
    EXPECT_FALSE(u[0].min.defined());
    EXPECT_FALSE(u[0].max.defined());
    EXPECT_FALSE(u[1].min.defined());
    EXPECT_FALSE(u[1].max.defined());
  }
  for (const box_expr& i : {b & box_expr{}, box_expr{} & b}) {
    ASSERT_EQ(i.size(), 2u);
    EXPECT_THAT(simplify(i[0]), matches(bounds(1, 4)));
    EXPECT_THAT(simplify(i[1]), matches(bounds(2, 3)));
  }
}

}  // namespace slinky
