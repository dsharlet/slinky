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

// Wraps expr_generator to also produce helper-specific constructs (abs, boolean, non-boolean logical ops)
// as well as non-finite literals (+oo, -oo, indeterminate).
template <typename Rng>
class helper_expr_generator {
  Rng& rng_;
  expr_generator<Rng> gen_;

public:
  explicit helper_expr_generator(Rng& rng, int var_count = 4) : rng_(rng), gen_(rng, var_count) {}

  void init_context(eval_context& ctx) { gen_.init_context(ctx); }
  index_t random_constant(int max = 100) { return gen_.random_constant(max); }

  expr random_finite_expr(int depth) {
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
    case 0: return abs(random_finite_expr(depth - 1));
    case 1: return boolean(random_finite_expr(depth - 1));
    case 2: return !random_finite_expr(depth - 1);
    case 3: return and_then(random_finite_expr(depth - 1), random_finite_expr(depth - 1));
    case 4: return or_else(random_finite_expr(depth - 1), random_finite_expr(depth - 1));
    default: return gen_.random_expr(depth);
    }
  }

  // May produce finite expressions, literal infinities/indeterminate, or compound expressions containing them.
  expr random_extended_expr(int depth) {
    if (depth <= 0) {
      switch (rng_() % 9) {
      case 0: return positive_infinity();
      case 1: return negative_infinity();
      case 2: return indeterminate();
      case 3: return expr(0);
      case 4: return expr(1);
      case 5: return expr(-1);
      case 6: return constant::make(gen_.random_constant());
      default: return gen_.random_variable();
      }
    }
    switch (rng_() % 8) {
    case 0: return positive_infinity();
    case 1: return negative_infinity();
    case 2: return indeterminate();
    case 3: return abs(random_extended_expr(depth - 1));
    case 4: return random_extended_expr(depth - 1) + random_extended_expr(depth - 1);
    case 5: return random_extended_expr(depth - 1) * random_extended_expr(depth - 1);
    case 6: return !random_extended_expr(depth - 1);
    default: return random_finite_expr(depth);
    }
  }

  // Produces an already-simplified operand (either finite, +oo, -oo, or indeterminate), suitable for passing
  // directly to single-op simplify(const Op*, ...).
  expr random_simplified_operand() {
    switch (rng_() % 10) {
    case 0: return positive_infinity();
    case 1: return negative_infinity();
    case 2: return indeterminate();
    case 3: return expr(0);
    case 4: return expr(1);
    case 5: return expr(-1);
    case 6: return constant::make(gen_.random_constant());
    default: return simplify(random_finite_expr(1));
    }
  }

  interval_expr random_finite_interval() {
    expr a = random_finite_expr(1);
    expr b = random_finite_expr(1);
    if (rng_() % 4 == 0) return point(a);
    return bounds(a, b);
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
      positive_infinity(),
      negative_infinity(),
      indeterminate(),
      abs(positive_infinity()),
      abs(negative_infinity()),
      abs(indeterminate()),
      abs(expr(x) + indeterminate()),
      expr(x) + positive_infinity(),
      expr(x) + indeterminate(),
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
      positive_infinity(),
      negative_infinity(),
      indeterminate(),
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

  auto check_expr = [&](const expr& e, bool fatal) {
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
        if (fatal) {
          ADD_FAILURE() << msg << " (e=" << e << ", boolean(e)=" << b << ")";
        } else {
          EXPECT_TRUE(cond) << msg << " (e=" << e << ", boolean(e)=" << b << ")";
        }
      }
    };

    // Idempotence of boolean().
    check(match(boolean(b), b), "boolean(boolean(e)) != boolean(e)");

    // Algebraic agreement of boolean(e) with simplify.
    check(match(sb, sne), "simplify(boolean(e)) != simplify(e != 0)");
    check(match(sb, snotnot), "simplify(boolean(e)) != simplify(!!e)");
    check(match(sb, ssel), "simplify(boolean(e)) != simplify(select(e, 1, 0))");

    // On constants and non-finite literals, boolean(e) should fold directly to the simplified value
    // (1 for +/-oo and nonzero constants, 0 for 0, indeterminate for indeterminate), without leaving
    // a non-finite literal nested inside a compound node.
    if (as_constant(e) || is_infinity(e) || is_indeterminate(e)) {
      const expr expected = simplify(static_cast<const not_equal*>(nullptr), e, 0);
      check(match(b, expected), "boolean(literal) did not fold to simplify(literal != 0)");
    }
    if (is_indeterminate(e)) {
      check(is_indeterminate(b), "boolean(indeterminate) should be indeterminate");
    } else if (!contains_non_finite(e) || is_infinity(e)) {
      check(is_boolean(b), "is_boolean(boolean(e)) should hold");
      check(!contains_non_finite(b), "boolean(e) left a non-finite subexpression");
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

    // Infinities and provably nonzero expressions (positive/negative) are truthy.
    if (is_infinity(e) != 0) {
      check(is_true(e), "infinities are truthy, so is_true(e) should hold");
    }
    if (is_positive(e) || is_negative(e)) {
      check(is_true(e), "is_positive(e) || is_negative(e) implies is_true(e)");
    }

    // Agreement with evaluate on finite expressions.
    if (!contains_non_finite(e)) {
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
    }
    return ok;
  };

  for (const expr& e : edge_case_exprs()) {
    check_expr(e, /*fatal=*/false);
  }
  for (auto _ : fuzz_test(fuzz_duration)) {
    if (!check_expr(gen.random_extended_expr(2), /*fatal=*/true)) break;
  }
}

// Properties of is_positive(), is_non_negative(), is_negative(), is_non_positive(), and constant predicates.
TEST(expr_helpers, sign_predicates) {
  gtest_seeded_mt19937 rng;
  helper_expr_generator<gtest_seeded_mt19937> gen(rng);
  eval_context ctx;

  auto check_expr = [&](const expr& e, bool fatal) {
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
        if (fatal) {
          ADD_FAILURE() << msg << " (e=" << e << ")";
        } else {
          EXPECT_TRUE(cond) << msg << " (e=" << e << ")";
        }
      }
    };

    if (!e.defined() || is_indeterminate(e)) {
      check(!pos && !non_neg && !neg && !non_pos && !zero && !one,
          "undefined/indeterminate must not satisfy any sign or constant predicate");
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
    if (!contains_non_finite(e) || is_infinity(e)) {
      check(is_non_negative(ae), "abs of non-indeterminate expression should satisfy is_non_negative");
    }

    // Soundness against evaluate on finite expressions.
    if (!contains_non_finite(e)) {
      for (int i = 0; i < eval_checks; ++i) {
        gen.init_context(ctx);
        const index_t v = evaluate(e, ctx);
        if (pos) check(v > 0, "is_positive(e) evaluated <= 0 at " + format_ctx(ctx));
        if (non_neg) check(v >= 0, "is_non_negative(e) evaluated < 0 at " + format_ctx(ctx));
        if (neg) check(v < 0, "is_negative(e) evaluated >= 0 at " + format_ctx(ctx));
        if (non_pos) check(v <= 0, "is_non_positive(e) evaluated > 0 at " + format_ctx(ctx));
      }
    }
    return ok;
  };

  for (const expr& e : edge_case_exprs()) {
    check_expr(e, /*fatal=*/false);
  }
  for (auto _ : fuzz_test(fuzz_duration)) {
    if (!check_expr(gen.random_extended_expr(2), /*fatal=*/true)) break;
  }
}

// Properties of is_infinity(), is_positive_infinity(), is_negative_infinity(), is_indeterminate(),
// is_finite(), and contains_non_finite().
TEST(expr_helpers, finiteness_predicates) {
  EXPECT_TRUE(infinity(1).same_as(positive_infinity()));
  EXPECT_TRUE(infinity(-1).same_as(negative_infinity()));
  EXPECT_TRUE(is_positive_infinity(positive_infinity()));
  EXPECT_TRUE(is_negative_infinity(negative_infinity()));
  EXPECT_TRUE(is_indeterminate(indeterminate()));
  EXPECT_FALSE(contains_non_finite(expr()));
  EXPECT_FALSE(is_finite(expr()));

  gtest_seeded_mt19937 rng;
  helper_expr_generator<gtest_seeded_mt19937> gen(rng);

  auto check_expr = [&](const expr& e, bool fatal) {
    bool ok = true;
    auto check = [&](bool cond, const std::string& msg) {
      if (!cond) {
        ok = false;
        if (fatal) {
          ADD_FAILURE() << msg << " (e=" << e << ")";
        } else {
          EXPECT_TRUE(cond) << msg << " (e=" << e << ")";
        }
      }
    };

    const int inf = is_infinity(e);
    check((inf == 1) == is_positive_infinity(e), "is_infinity == 1 <=> is_positive_infinity");
    check((inf == -1) == is_negative_infinity(e), "is_infinity == -1 <=> is_negative_infinity");
    check((inf == 0) == (!is_positive_infinity(e) && !is_negative_infinity(e)),
        "is_infinity == 0 <=> neither +/-infinity");
    check(!(inf != 0 && is_indeterminate(e)), "cannot be both infinity and indeterminate");

    if (inf != 0 || is_indeterminate(e)) {
      check(contains_non_finite(e), "non-finite literal must satisfy contains_non_finite");
      check(!is_finite(e), "non-finite literal must not satisfy is_finite");
    }
    if (is_finite(e)) {
      check(e.defined(), "is_finite implies defined");
      check(!contains_non_finite(e), "is_finite implies !contains_non_finite");
    }
    if (e.defined() && !contains_non_finite(e)) {
      check(!contains_non_finite(simplify(e)), "simplifying a finite expr must stay finite");
    }
    return ok;
  };

  for (const expr& e : edge_case_exprs()) {
    check_expr(e, /*fatal=*/false);
  }
  for (auto _ : fuzz_test(fuzz_duration)) {
    if (!check_expr(gen.random_extended_expr(2), /*fatal=*/true)) break;
  }
}

// Constructors (::make / operators) perform eager constant/identity folding; their simplified result must agree
// algebraically with calling simplify(const Op*, ...) directly on the operands (including +/-oo and indeterminate).
TEST(expr_helpers, constructors_agree_with_simplify) {
  gtest_seeded_mt19937 rng;
  helper_expr_generator<gtest_seeded_mt19937> gen(rng);
  eval_context ctx;

  auto check_binary_pair = [&](const expr& a, const expr& b, bool fatal) {
    bool ok = true;
    auto check_op = [&](auto tag, const char* name, expr via_make) {
      using Op = std::remove_pointer_t<decltype(tag)>;
      expr lhs = simplify(via_make);
      expr rhs = simplify(simplify(static_cast<const Op*>(nullptr), a, b));
      if (!match(lhs, rhs)) {
        ok = false;
        if (fatal) {
          ADD_FAILURE() << name << "::make(" << a << ", " << b << ") simplified to " << lhs
                        << ", but simplify(" << name << ", ...) gave " << rhs;
        } else {
          EXPECT_TRUE(false) << name << "::make(" << a << ", " << b << ") simplified to " << lhs
                             << ", but simplify(" << name << ", ...) gave " << rhs;
        }
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
      if (!match(lhs, rhs)) {
        ok = false;
        if (fatal) {
          ADD_FAILURE() << name << "(" << a << ", " << b << ") simplified to " << lhs
                        << ", but simplify(" << name << ", ...) gave " << rhs;
        } else {
          EXPECT_TRUE(false) << name << "(" << a << ", " << b << ") simplified to " << lhs
                             << ", but simplify(" << name << ", ...) gave " << rhs;
        }
      }
    };
    check_call(intrinsic::and_then, "and_then", and_then(a, b));
    check_call(intrinsic::or_else, "or_else", or_else(a, b));

    // Derived comparison & arithmetic operator helpers.
    auto check_match = [&](const expr& lhs, const expr& rhs, const char* label) {
      if (!match(lhs, rhs)) {
        ok = false;
        if (fatal) {
          ADD_FAILURE() << label << " mismatch for (" << a << ", " << b << "): " << lhs << " vs " << rhs;
        } else {
          EXPECT_TRUE(false) << label << " mismatch for (" << a << ", " << b << "): " << lhs << " vs " << rhs;
        }
      }
    };
    check_match(simplify(a > b), simplify(b < a), "a > b vs b < a");
    check_match(simplify(a >= b), simplify(b <= a), "a >= b vs b <= a");
    check_match(simplify(euclidean_div(a, b)), simplify(a / b), "euclidean_div vs /");
    check_match(simplify(euclidean_mod(a, b)), simplify(a % b), "euclidean_mod vs %");

    return ok;
  };

  auto check_unary = [&](const expr& a, bool fatal) {
    bool ok = true;
    auto check_match = [&](const expr& lhs, const expr& rhs, const char* label) {
      if (!match(lhs, rhs)) {
        ok = false;
        if (fatal) {
          ADD_FAILURE() << label << " mismatch for " << a << ": " << lhs << " vs " << rhs;
        } else {
          EXPECT_TRUE(false) << label << " mismatch for " << a << ": " << lhs << " vs " << rhs;
        }
      }
    };
    check_match(simplify(logical_not::make(a)), simplify(simplify(static_cast<const logical_not*>(nullptr), a)),
        "logical_not");
    check_match(simplify(-a), simplify(simplify(static_cast<const sub*>(nullptr), 0, a)), "unary -");
    check_match(simplify(abs(a)), simplify(simplify(static_cast<const call*>(nullptr), intrinsic::abs, {a})), "abs");
    return ok;
  };

  auto check_select = [&](const expr& c, const expr& t, const expr& f, bool fatal) {
    expr lhs = simplify(select::make(c, t, f));
    expr rhs = simplify(simplify(static_cast<const class select*>(nullptr), c, t, f));
    if (!match(lhs, rhs)) {
      if (fatal) {
        ADD_FAILURE() << "select::make(" << c << ", " << t << ", " << f << ") simplified to " << lhs
                      << ", but simplify(select, ...) gave " << rhs;
      } else {
        EXPECT_TRUE(false) << "select::make(" << c << ", " << t << ", " << f << ") simplified to " << lhs
                           << ", but simplify(select, ...) gave " << rhs;
      }
      return false;
    }
    return true;
  };

  const std::vector<expr> operands = simplified_edge_operands();
  for (const expr& a : operands) {
    check_unary(a, /*fatal=*/false);
    for (const expr& b : operands) {
      check_binary_pair(a, b, /*fatal=*/false);
    }
  }

  // Annihilating ops with indeterminate (an unknown index_t value) or infinities should fold in simplify()
  // just as they do in ::make().
  for (const expr& non_finite : {positive_infinity(), negative_infinity(), indeterminate()}) {
    EXPECT_THAT(simplify(static_cast<const mul*>(nullptr), expr(0), non_finite), matches(0)) << "0 * " << non_finite;
    EXPECT_THAT(simplify(static_cast<const mul*>(nullptr), non_finite, expr(0)), matches(0)) << non_finite << " * 0";
    EXPECT_THAT(simplify(static_cast<const div*>(nullptr), expr(0), non_finite), matches(0)) << "0 / " << non_finite;
    EXPECT_THAT(simplify(static_cast<const div*>(nullptr), non_finite, expr(0)), matches(0)) << non_finite << " / 0";
    EXPECT_THAT(simplify(static_cast<const mod*>(nullptr), expr(0), non_finite), matches(0)) << "0 % " << non_finite;
    EXPECT_THAT(simplify(static_cast<const mod*>(nullptr), non_finite, expr(0)), matches(0)) << non_finite << " % 0";
    EXPECT_THAT(simplify(static_cast<const mod*>(nullptr), non_finite, expr(1)), matches(0)) << non_finite << " % 1";
    EXPECT_THAT(simplify(static_cast<const mod*>(nullptr), non_finite, expr(-1)), matches(0)) << non_finite << " % -1";
    EXPECT_THAT(simplify(static_cast<const logical_and*>(nullptr), expr(0), non_finite), matches(0))
        << "0 && " << non_finite;
    EXPECT_THAT(simplify(static_cast<const logical_and*>(nullptr), non_finite, expr(0)), matches(0))
        << non_finite << " && 0";
    EXPECT_THAT(simplify(static_cast<const logical_or*>(nullptr), expr(1), non_finite), matches(1))
        << "1 || " << non_finite;
    EXPECT_THAT(simplify(static_cast<const logical_or*>(nullptr), non_finite, expr(1)), matches(1))
        << non_finite << " || 1";
  }
  for (const expr& c :
      {expr(0), expr(1), expr(-2), expr(x), positive_infinity(), negative_infinity(), indeterminate()}) {
    for (const expr& t : {expr(0), expr(x), positive_infinity(), indeterminate()}) {
      for (const expr& f : {expr(1), expr(y), negative_infinity(), indeterminate()}) {
        check_select(c, t, f, /*fatal=*/false);
      }
    }
  }

  for (auto _ : fuzz_test(fuzz_duration)) {
    expr a = gen.random_simplified_operand();
    expr b = gen.random_simplified_operand();
    expr c = gen.random_simplified_operand();
    if (!check_unary(a, /*fatal=*/true)) break;
    if (!check_binary_pair(a, b, /*fatal=*/true)) break;
    if (!check_select(c, a, b, /*fatal=*/true)) break;

    if (!contains_non_finite(a) && !contains_non_finite(b)) {
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
}

// Non-annihilating operations propagate undefined operands, while annihilating operations (0 * x, 0 / x, x / 0,
// 0 % x, x % 0, x % 1, false && x, true || x) fold to the annihilator in both constructors and simplify().
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
  EXPECT_THAT(logical_and::make(undef, 0), matches(0));
  EXPECT_THAT(logical_and::make(0, undef), matches(0));
  for (const expr& truthy : {expr(1), expr(-1), expr(2), positive_infinity(), negative_infinity()}) {
    EXPECT_THAT(logical_or::make(undef, truthy), matches(1)) << "undef || " << truthy;
    EXPECT_THAT(logical_or::make(truthy, undef), matches(1)) << truthy << " || undef";
  }

  // Annihilating instances inside simplify() when an operand simplifies to undefined
  // (e.g. select(y == y, undef, 2) simplifies to undef, while y - y simplifies to 0).
  expr undef_expr = select(y == y, undef, 2);
  expr zero_expr = y - y;
  expr one_expr = y - y + 1;
  EXPECT_THAT(simplify(undef_expr * zero_expr), matches(0));
  EXPECT_THAT(simplify(zero_expr * undef_expr), matches(0));
  EXPECT_THAT(simplify(undef_expr / zero_expr), matches(0));
  EXPECT_THAT(simplify(zero_expr / undef_expr), matches(0));
  EXPECT_THAT(simplify(undef_expr % zero_expr), matches(0));
  EXPECT_THAT(simplify(zero_expr % undef_expr), matches(0));
  EXPECT_THAT(simplify(undef_expr % one_expr), matches(0));
  EXPECT_THAT(simplify(undef_expr && zero_expr), matches(0));
  EXPECT_THAT(simplify(zero_expr && undef_expr), matches(0));
  EXPECT_THAT(simplify(undef_expr || one_expr), matches(1));
  EXPECT_THAT(simplify(one_expr || undef_expr), matches(1));

  // Non-annihilating instances must propagate undefined.
  for (const expr& a : {expr(0), expr(1), expr(-1), expr(2), expr(x), positive_infinity(), indeterminate()}) {
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
    if (!is_zero(a) && !is_one(a)) {
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
    }
    if (!is_true(a) && !is_infinity(a)) {
      EXPECT_FALSE(logical_or::make(undef, a).defined()) << "logical_or(undef, " << a << ")";
      EXPECT_FALSE(logical_or::make(a, undef).defined()) << "logical_or(" << a << ", undef)";
    }
    EXPECT_FALSE(select::make(undef, a, a).defined()) << "select(undef, " << a << ", " << a << ")";
  }
  EXPECT_FALSE(logical_not::make(undef).defined()) << "logical_not(undef)";
  EXPECT_FALSE(select::make(x, undef, undef).defined()) << "select(x, undef, undef)";
  EXPECT_FALSE(boolean(undef).defined()) << "boolean(undef)";
}

// bounds_of(const select*, c, t, f) must handle arbitrary integer intervals for c (not just [0, 1]),
// consistent with evaluate() and is_true()'s truthiness semantics.
TEST(expr_helpers, bounds_of_select) {
  const interval_expr t = bounds(10, 20);
  const interval_expr f = bounds(100, 200);
  const interval_expr both = bounds(10, 200);

  // Condition intervals containing both 0 and nonzero values must include both arms.
  EXPECT_THAT(bounds_of(static_cast<const class select*>(nullptr), bounds(-1, 1), t, f), matches(both));
  EXPECT_THAT(bounds_of(static_cast<const class select*>(nullptr), bounds(-5, 0), t, f), matches(both));
  EXPECT_THAT(bounds_of(static_cast<const class select*>(nullptr), bounds(0, 5), t, f), matches(both));
  EXPECT_THAT(bounds_of(static_cast<const class select*>(nullptr), interval_expr::all(), t, f), matches(both));

  // Condition intervals excluding 0 select t; [0, 0] selects f.
  EXPECT_THAT(
      bounds_of(static_cast<const class select*>(nullptr), bounds(1, positive_infinity()), t, f), matches(t));
  EXPECT_THAT(
      bounds_of(static_cast<const class select*>(nullptr), bounds(negative_infinity(), -1), t, f), matches(t));
  EXPECT_THAT(bounds_of(static_cast<const class select*>(nullptr), bounds(2, 5), t, f), matches(t));
  EXPECT_THAT(bounds_of(static_cast<const class select*>(nullptr), bounds(-5, -2), t, f), matches(t));
  EXPECT_THAT(bounds_of(static_cast<const class select*>(nullptr), bounds(0, 0), t, f), matches(f));

  // Exhaustive soundness check over small integer condition intervals.
  for (int c_min = -3; c_min <= 3; ++c_min) {
    for (int c_max = c_min; c_max <= 3; ++c_max) {
      interval_expr b = bounds_of(static_cast<const class select*>(nullptr), bounds(c_min, c_max), t, f);
      index_t bmin = evaluate(b.min);
      index_t bmax = evaluate(b.max);
      for (int c_val = c_min; c_val <= c_max; ++c_val) {
        index_t arm_min = c_val != 0 ? 10 : 100;
        index_t arm_max = c_val != 0 ? 20 : 200;
        EXPECT_LE(bmin, arm_min) << "c=[" << c_min << ", " << c_max << "], c_val=" << c_val << ", b=" << b;
        EXPECT_GE(bmax, arm_max) << "c=[" << c_min << ", " << c_max << "], c_val=" << c_val << ", b=" << b;
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

  // Alignment with non-finite values and positive alignment.
  for (int a : {1, 2, 4, 7}) {
    EXPECT_THAT(simplify(align_down(positive_infinity(), a)), matches(positive_infinity()));
    EXPECT_THAT(simplify(align_down(negative_infinity(), a)), matches(negative_infinity()));
    EXPECT_THAT(simplify(align_down(indeterminate(), a)), matches(indeterminate()));
    EXPECT_THAT(simplify(align_up(positive_infinity(), a)), matches(positive_infinity()));
    EXPECT_THAT(simplify(align_up(negative_infinity(), a)), matches(negative_infinity()));
    EXPECT_THAT(simplify(align_up(indeterminate(), a)), matches(indeterminate()));
  }

  // Short-circuiting with indeterminate and infinities.
  EXPECT_THAT(simplify(and_then(expr(0), indeterminate())), matches(0));
  EXPECT_THAT(simplify(or_else(expr(1), indeterminate())), matches(1));
  EXPECT_THAT(simplify(or_else(positive_infinity(), indeterminate())), matches(1));
  EXPECT_THAT(simplify(or_else(negative_infinity(), indeterminate())), matches(1));

  for (auto _ : fuzz_test(fuzz_duration)) {
    expr e = gen.random_finite_expr(1);
    expr lo = gen.random_finite_expr(1);
    expr hi = gen.random_finite_expr(1);
    // Positive alignment (sometimes 1 to hit the fast path).
    expr align_expr = (rng() % 4 == 0) ? expr(1) : (abs(gen.random_finite_expr(1)) + 1);

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

  // Identity laws for union_identity() / intersection_identity().
  for (const interval_expr& iv : {bounds(x, y), point(x), bounds(0, 10), interval_expr::all(), interval_expr::none()}) {
    EXPECT_THAT(simplify(iv | interval_expr::union_identity()), matches(simplify(iv)));
    EXPECT_THAT(simplify(interval_expr::union_identity() | iv), matches(simplify(iv)));
    EXPECT_THAT(simplify(iv & interval_expr::intersection_identity()), matches(simplify(iv)));
    EXPECT_THAT(simplify(interval_expr::intersection_identity() & iv), matches(simplify(iv)));
    EXPECT_THAT(simplify(iv | iv), matches(simplify(iv)));
    EXPECT_THAT(simplify(iv & iv), matches(simplify(iv)));
  }

  // Agreement with bounds_of for scaling/dividing unbounded intervals by unknown scalars.
  EXPECT_THAT(simplify(interval_expr::all() + y), matches(interval_expr::all()));
  EXPECT_THAT(simplify(-interval_expr::all()), matches(interval_expr::all()));
  EXPECT_THAT(simplify(interval_expr::all() * 2), matches(interval_expr::all()));
  EXPECT_THAT(simplify(interval_expr::all() * -2), matches(interval_expr::all()));
  EXPECT_THAT(simplify(interval_expr::all() * 0), matches(point(0)));
  EXPECT_THAT(simplify(interval_expr::all() * y),
      matches(bounds_of(static_cast<const mul*>(nullptr), interval_expr::all(), point(y))));
  EXPECT_THAT(simplify(interval_expr::all() / y),
      matches(bounds_of(static_cast<const div*>(nullptr), interval_expr::all(), point(y))));

  // select(c, t, f) with infinite/empty arms must not produce a select node containing infinities.
  for (const interval_expr& t : {bounds(x, y), point(x), interval_expr::all(), interval_expr::none()}) {
    for (const interval_expr& f : {bounds(z, w), point(z), interval_expr::all(), interval_expr::none()}) {
      interval_expr s = select(x, t, f);
      EXPECT_TRUE(!contains_non_finite(s.min) || is_infinity(s.min)) << "select.min=" << s.min;
      EXPECT_TRUE(!contains_non_finite(s.max) || is_infinity(s.max)) << "select.max=" << s.max;
    }
  }
  EXPECT_THAT(select(x, bounds(y, z), interval_expr::none()), matches(bounds(y, z)));
  EXPECT_THAT(select(x, interval_expr::none(), bounds(y, z)), matches(bounds(y, z)));

  for (auto _ : fuzz_test(fuzz_duration)) {
    interval_expr iv1 = gen.random_finite_interval();
    interval_expr iv2 = gen.random_finite_interval();
    expr s = gen.random_finite_expr(1);
    expr p = gen.random_finite_expr(1);
    expr c = gen.random_finite_expr(1);
    expr align_a = (rng() % 4 == 0) ? expr(1) : (abs(gen.random_finite_expr(1)) + 1);

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

// Algebraic properties of box_expr union/intersection.
TEST(expr_helpers, box_expr_algebra) {
  // Union and intersection of boxes should be commutative, even when ranks differ.
  box_expr a = {bounds(0, 5)};
  box_expr b = {bounds(1, 4), bounds(2, 3)};

  box_expr u_ab = a | b;
  box_expr u_ba = b | a;
  ASSERT_EQ(u_ab.size(), u_ba.size());
  for (std::size_t d = 0; d < u_ab.size(); ++d) {
    EXPECT_THAT(simplify(u_ab[d]), matches(simplify(u_ba[d]))) << "box union not commutative at dim " << d;
  }

  box_expr i_ab = a & b;
  box_expr i_ba = b & a;
  ASSERT_EQ(i_ab.size(), i_ba.size());
  for (std::size_t d = 0; d < i_ab.size(); ++d) {
    EXPECT_THAT(simplify(i_ab[d]), matches(simplify(i_ba[d]))) << "box intersection not commutative at dim " << d;
  }
}

}  // namespace slinky
