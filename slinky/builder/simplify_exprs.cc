#include "slinky/builder/simplify.h"

#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>

#include "slinky/builder/rewrite.h"
#include "slinky/builder/simplify_rules.h"
#include "slinky/runtime/evaluate.h"
#include "slinky/runtime/expr.h"

// The rules in simplify_rules.h assume that every wildcard matches a finite value. Infinities and indeterminate are
// only meaningful as literal operands (e.g. the endpoints of an unbounded interval), so the functions below eliminate
// them before applying the rules: the result is an infinity, a constant, or indeterminate when the result can't be
// represented. Every other expression denotes a finite value. In debug builds, rewrite::pattern_expr and
// rewrite::rewriter assert that the expressions they match contain no infinity or indeterminate.

namespace slinky {

using namespace rewrite;

expr simplify(const class min* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }

  auto ca = as_constant(a);
  auto cb = as_constant(b);
  if (ca && cb) {
    return std::min(*ca, *cb);
  }

  if (is_indeterminate(a)) return a;
  if (is_indeterminate(b)) return b;
  int inf_a = is_infinity(a);
  int inf_b = is_infinity(b);
  if (inf_a || inf_b) return inf_a <= inf_b ? a : b;

  auto r = make_rewriter(min(pattern_expr{a}, pattern_expr{b}));
  if (apply_min_rules(r)) {
    return r.result;
  } else if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return expr(op);
  } else {
    return min::make(std::move(a), std::move(b));
  }
}

expr simplify(const class max* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }

  auto ca = as_constant(a);
  auto cb = as_constant(b);
  if (ca && cb) {
    return std::max(*ca, *cb);
  }

  if (is_indeterminate(a)) return a;
  if (is_indeterminate(b)) return b;
  int inf_a = is_infinity(a);
  int inf_b = is_infinity(b);
  if (inf_a || inf_b) return inf_a >= inf_b ? a : b;

  auto r = make_rewriter(max(pattern_expr{a}, pattern_expr{b}));
  if (apply_max_rules(r)) {
    return r.result;
  } else if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return expr(op);
  } else {
    return max::make(std::move(a), std::move(b));
  }
}

expr simplify(const add* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }

  auto ca = as_constant(a);
  auto cb = as_constant(b);
  if (ca && cb && !add_overflows(*ca, *cb)) {
    return *ca + *cb;
  }

  if (is_indeterminate(a)) return a;
  if (is_indeterminate(b)) return b;
  int inf_a = is_infinity(a);
  int inf_b = is_infinity(b);
  if (inf_a && inf_b) return inf_a == inf_b ? a : slinky::indeterminate();
  if (inf_a) return a;
  if (inf_b) return b;

  auto r = make_rewriter(pattern_expr{a} + pattern_expr{b});
  if (apply_add_rules(r)) {
    return r.result;
  } else if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return expr(op);
  } else {
    return add::make(std::move(a), std::move(b));
  }
}

expr simplify(const sub* op, expr a, expr b) {
  auto ca = as_constant(a);
  auto cb = as_constant(b);
  if (ca && cb && !sub_overflows(*ca, *cb)) {
    return *ca - *cb;
  } else if (cb && !sub_overflows<index_t>(0, *cb)) {
    // Canonicalize to addition with constants.
    return simplify(static_cast<add*>(nullptr), a, -*cb);
  }

  if (is_indeterminate(a)) return a;
  if (is_indeterminate(b)) return b;
  int inf_a = is_infinity(a);
  int inf_b = is_infinity(b);
  if (inf_a && inf_b) return inf_a == inf_b ? slinky::indeterminate() : a;
  if (inf_a) return a;
  if (inf_b) return infinity(-inf_b);

  auto r = make_rewriter(pattern_expr{a} - pattern_expr{b});
  if (apply_sub_rules(r)) {
    return r.result;
  } else if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return expr(op);
  } else {
    return sub::make(std::move(a), std::move(b));
  }
}

expr simplify(const mul* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }

  auto ca = as_constant(a);
  auto cb = as_constant(b);
  if (ca && cb && !mul_overflows(*ca, *cb)) {
    return *ca * *cb;
  }

  if (is_indeterminate(a)) return a;
  if (is_indeterminate(b)) return b;
  int inf_a = is_infinity(a);
  int inf_b = is_infinity(b);
  if (inf_a && inf_b) return infinity(inf_a * inf_b);
  if (inf_a || inf_b) {
    // As in mul::make, x * 0 is 0 for any x. Otherwise, the sign of the result depends on the sign of the finite
    // operand.
    const expr& other = inf_a ? b : a;
    const int sign = inf_a + inf_b;
    if (is_zero(other)) return 0;
    if (is_positive(other)) return infinity(sign);
    if (is_negative(other)) return infinity(-sign);
    return slinky::indeterminate();
  }

  auto r = make_rewriter(pattern_expr{a} * pattern_expr{b});
  if (apply_mul_rules(r)) {
    return r.result;
  } else if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return expr(op);
  } else {
    return mul::make(std::move(a), std::move(b));
  }
}

expr simplify(const div* op, expr a, expr b) {
  auto ca = as_constant(a);
  auto cb = as_constant(b);
  if (ca && cb && !div_overflows(*ca, *cb)) {
    return euclidean_div(*ca, *cb);
  }

  if (is_indeterminate(a)) return a;
  if (is_indeterminate(b)) return b;
  int inf_a = is_infinity(a);
  int inf_b = is_infinity(b);
  if (inf_a && inf_b) return slinky::indeterminate();
  if (inf_b) return 0;
  if (inf_a) {
    // As for mul, x / 0 is 0 for any x, and otherwise we need to know the sign of b.
    if (is_zero(b)) return 0;
    if (is_positive(b)) return infinity(inf_a);
    if (is_negative(b)) return infinity(-inf_a);
    return slinky::indeterminate();
  }

  auto r = make_rewriter(pattern_expr{a} / pattern_expr{b});
  if (apply_div_rules(r)) {
    return r.result;
  } else if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return expr(op);
  } else {
    return div::make(std::move(a), std::move(b));
  }
}

expr simplify(const mod* op, expr a, expr b) {
  auto ca = as_constant(a);
  auto cb = as_constant(b);
  if (ca && cb) {
    return euclidean_mod(*ca, *cb);
  }

  if (is_indeterminate(a)) return a;
  if (is_indeterminate(b)) return b;
  if (is_infinity(a) || is_infinity(b)) return slinky::indeterminate();

  auto r = make_rewriter(pattern_expr{a} % pattern_expr{b});
  if (apply_mod_rules(r)) {
    return r.result;
  } else if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return expr(op);
  } else {
    return mod::make(std::move(a), std::move(b));
  }
}

expr simplify(const less* op, expr a, expr b) {
  auto ca = as_constant(a);
  auto cb = as_constant(b);
  if (ca && cb) {
    return *ca < *cb;
  }

  if (is_indeterminate(a)) return a;
  if (is_indeterminate(b)) return b;
  // Compare using the total order -oo < (finite values) < +oo.
  int inf_a = is_infinity(a);
  int inf_b = is_infinity(b);
  if (inf_a || inf_b) return inf_a < inf_b;

  auto r = make_rewriter(pattern_expr{a} < pattern_expr{b});
  if (apply_less_rules(r)) {
    return r.result;
  } else if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return expr(op);
  } else {
    return less::make(std::move(a), std::move(b));
  }
}

expr simplify(const less_equal* op, expr a, expr b) {
  if (is_indeterminate(a)) return a;
  if (is_indeterminate(b)) return b;
  int inf_a = is_infinity(a);
  int inf_b = is_infinity(b);
  if (inf_a || inf_b) return inf_a <= inf_b;

  // Do a few quick checks before rewriting to less.
  auto r = make_rewriter(pattern_expr{a} <= pattern_expr{b});
  // clang-format off
  if (r(x <= x, true) ||
      r(x <= y - 1, x < y) || 
      r(x - 1 <= y, x < y)) {
    return r.result;
  }
  // clang-format on

  // Rewrite to !(b < a) and simplify that instead.
  expr result = simplify(static_cast<const logical_not*>(nullptr),
      simplify(static_cast<const less*>(nullptr), std::move(b), std::move(a)));
  if (op) {
    if (const less_equal* le = result.as<less_equal>()) {
      if (le->a.same_as(op->a) && le->b.same_as(op->b)) {
        return expr(op);
      }
    }
  }
  return result;
}

expr simplify(const equal* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }

  auto ca = as_constant(a);
  auto cb = as_constant(b);
  if (ca && cb) {
    return *ca == *cb;
  }

  if (is_indeterminate(a)) return a;
  if (is_indeterminate(b)) return b;
  int inf_a = is_infinity(a);
  int inf_b = is_infinity(b);
  if (inf_a || inf_b) return inf_a == inf_b;

  auto r = make_rewriter(pattern_expr{a} == pattern_expr{b});
  if (apply_equal_rules(r)) {
    return r.result;
  } else if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return expr(op);
  } else {
    return equal::make(std::move(a), std::move(b));
  }
}

expr simplify(const not_equal* op, expr a, expr b) {
  // Rewrite to !(a == b) and simplify that instead.
  expr result = simplify(static_cast<const logical_not*>(nullptr),
      simplify(static_cast<const equal*>(nullptr), std::move(a), std::move(b)));
  if (op) {
    if (const not_equal* ne = result.as<not_equal>()) {
      if (ne->a.same_as(op->a) && ne->b.same_as(op->b)) {
        return expr(op);
      }
    }
  }
  return result;
}

expr simplify(const logical_and* op, expr a, expr b) {
  // Infinities are truthy.
  if (is_infinity(a)) a = true;
  if (is_infinity(b)) b = true;
  if (should_commute(a, b)) {
    std::swap(a, b);
  }
  auto ca = as_constant(a);
  auto cb = as_constant(b);

  if (ca && cb) {
    return *ca != 0 && *cb != 0;
  } else if (cb && *cb == 0) {
    return false;
  }

  if (is_indeterminate(a)) return a;
  if (is_indeterminate(b)) return b;
  if (cb) return boolean(a);

  auto r = make_rewriter(pattern_expr{a} && pattern_expr{b});
  if (apply_logical_and_rules(r)) {
    return r.result;
  } else if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return expr(op);
  } else {
    return logical_and::make(std::move(a), std::move(b));
  }
}

expr simplify(const logical_or* op, expr a, expr b) {
  // Infinities are truthy.
  if (is_infinity(a)) a = true;
  if (is_infinity(b)) b = true;
  if (should_commute(a, b)) {
    std::swap(a, b);
  }
  auto ca = as_constant(a);
  auto cb = as_constant(b);

  if (ca && cb) {
    return *ca != 0 || *cb != 0;
  } else if (cb && *cb != 0) {
    return true;
  }

  if (is_indeterminate(a)) return a;
  if (is_indeterminate(b)) return b;
  if (cb) return boolean(a);

  auto r = make_rewriter(pattern_expr{a} || pattern_expr{b});
  if (apply_logical_or_rules(r)) {
    return r.result;
  } else if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return expr(op);
  } else {
    return logical_or::make(std::move(a), std::move(b));
  }
}

expr simplify(const class logical_not* op, expr a) {
  auto cv = as_constant(a);
  if (cv) {
    return *cv == 0;
  }

  if (is_indeterminate(a)) return a;
  // Infinities are truthy.
  if (is_infinity(a)) return false;

  auto r = make_rewriter(!pattern_expr{a});
  if (apply_logical_not_rules(r)) {
    return r.result;
  } else if (op && a.same_as(op->a)) {
    return expr(op);
  } else {
    return logical_not::make(std::move(a));
  }
}

expr simplify(const class select* op, expr c, expr t, expr f) {
  // Infinities are truthy.
  if (is_true(c) || is_infinity(c)) {
    return t;
  } else if (is_false(c)) {
    return f;
  }

  if (is_indeterminate(c)) return c;
  if (is_indeterminate(t)) return t;
  if (is_indeterminate(f)) return f;
  int inf_t = is_infinity(t);
  int inf_f = is_infinity(f);
  if (inf_t || inf_f) {
    // We can't represent a select with an infinite value. Code that computes bounds should avoid producing these.
    return inf_t == inf_f ? t : slinky::indeterminate();
  }

  auto r = make_rewriter(select(pattern_expr{c}, pattern_expr{t}, pattern_expr{f}));
  if (apply_select_rules(r)) {
    return r.result;
  } else if (op && c.same_as(op->condition) && t.same_as(op->true_value) && f.same_as(op->false_value)) {
    return expr(op);
  } else {
    return select::make(std::move(c), std::move(t), std::move(f));
  }
}

expr simplify(const call* op, intrinsic fn, const call::callable& target, std::vector<expr> args) {
  if (fn == intrinsic::and_then || fn == intrinsic::or_else) {
    assert(args.size() == 2);
    for (expr& i : args) {
      // Infinities are truthy.
      if (is_infinity(i)) i = true;
    }
    const bool is_and = fn == intrinsic::and_then;
    if (auto ca = as_constant(args[0])) {
      // If a is false (for and_then) or true (for or_else), b isn't evaluated. Otherwise, the result is boolean(b).
      if ((*ca != 0) != is_and) return !is_and;
      return is_indeterminate(args[1]) ? args[1] : boolean(args[1]);
    }
    if (is_indeterminate(args[0])) return args[0];
    if (is_indeterminate(args[1])) return args[1];
  }

  bool constant = true;
  bool changed = op == nullptr || op->args.size() != args.size();
  for (std::size_t i = 0; i < args.size(); ++i) {
    constant = constant && as_constant(args[i]);
    changed = changed || !args[i].same_as(op->args[i]);
  }

  if (fn == intrinsic::semaphore_init || fn == intrinsic::semaphore_wait || fn == intrinsic::semaphore_signal) {
    assert(args.size() % 2 == 0);
    for (std::size_t i = 0; i < args.size();) {
      // Remove calls to undefined semaphores.
      if (!args[i].defined()) {
        args.erase(args.begin() + i, args.begin() + i + 2);
        changed = true;
      } else {
        i += 2;
      }
    }
    if (args.empty()) {
      return expr();
    }
  } else if (fn == intrinsic::buffer_at) {
    for (index_t d = 1; d < static_cast<index_t>(args.size()); ++d) {
      auto buf = as_variable(args[0]);
      assert(buf);
      // buffer_at(b, buffer_min(b, 0)) is equivalent to buffer_at(b, <>)
      if (args[d].defined() && is_variable(args[d], *buf, buffer_field::min, d - 1)) {
        args[d] = expr();
        changed = true;
      }
    }
    // Trailing undefined args have no effect.
    while (args.size() > 1 && !args.back().defined()) {
      args.pop_back();
      changed = true;
    }
  } else if (fn == intrinsic::abs) {
    assert(args.size() == 1);
    if (is_indeterminate(args[0])) {
      return args[0];
    } else if (is_infinity(args[0])) {
      return positive_infinity();
    } else if (is_non_negative(args[0])) {
      return args[0];
    } else if (is_non_positive(args[0])) {
      return simplify(static_cast<const sub*>(nullptr), 0, std::move(args[0]));
    }
  }

  expr e;
  if (!changed) {
    assert(op);
    e = expr(op);
  } else {
    e = call::make(fn, target, std::move(args));
  }

  if (constant && can_evaluate(fn)) {
    return evaluate(e);
  }

  // We can't apply the rules to an infinity or indeterminate itself, or to a call with an indeterminate argument.
  const call* c = e.as<call>();
  assert(c);
  if (!can_evaluate(fn) ||
      std::any_of(c->args.begin(), c->args.end(), [](const expr& i) { return is_indeterminate(i); })) {
    return e;
  }

  rewriter r(e);
  if (apply_call_rules(r)) {
    return r.result;
  } else {
    return e;
  }
}

interval_expr simplify_intersection(interval_expr a, interval_expr b) {
  // An undefined or indeterminate bound doesn't tell us anything, so it doesn't constrain the intersection.
  auto unbounded = [](const expr& x) { return !x.defined() || is_indeterminate(x); };
  if (!unbounded(a.min) && !unbounded(b.min)) {
    a.min = simplify(static_cast<const class max*>(nullptr), std::move(a.min), std::move(b.min));
  } else if (unbounded(a.min)) {
    a.min = std::move(b.min);
  }
  if (!unbounded(a.max) && !unbounded(b.max)) {
    a.max = simplify(static_cast<const class min*>(nullptr), std::move(a.max), std::move(b.max));
  } else if (unbounded(a.max)) {
    a.max = std::move(b.max);
  }
  if (match(a.min, a.max)) {
    // Ensure that is_point is true if the result is a point.
    a.max = a.min;
  }
  return a;
}

}  // namespace slinky