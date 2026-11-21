#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "slinky/base/arithmetic.h"
#include "slinky/builder/rewrite.h"
#include "slinky/builder/simplify.h"
#include "slinky/builder/substitute.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"

namespace slinky {

using namespace rewrite;

namespace {

pattern_wildcard<0> x;

pattern_constant<0> c0;
pattern_constant<1> c1;
pattern_constant<2> c2;

using op_simplified = std::true_type;

template <typename T>
expr simplify(op_simplified, const T* op, expr a, expr b) {
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return expr(op);
  } else {
    return simplify(op, std::move(a), std::move(b));
  }
}

// The min or max of two lower bounds or two upper bounds, where an undefined lower bound is -infinity and an undefined
// upper bound is +infinity.
expr min_lower(expr a, expr b, const class min* op = nullptr) {
  if (!a.defined() || !b.defined()) return expr();
  return simplify(op_simplified(), op, std::move(a), std::move(b));
}
expr max_upper(expr a, expr b, const class max* op = nullptr) {
  if (!a.defined() || !b.defined()) return expr();
  return simplify(op_simplified(), op, std::move(a), std::move(b));
}
expr max_lower(expr a, expr b, const class max* op = nullptr) {
  if (!a.defined()) return b;
  if (!b.defined()) return a;
  return simplify(op_simplified(), op, std::move(a), std::move(b));
}
expr min_upper(expr a, expr b, const class min* op = nullptr) {
  if (!a.defined()) return b;
  if (!b.defined()) return a;
  return simplify(op_simplified(), op, std::move(a), std::move(b));
}

template <typename T>
interval_expr bounds_of_linear(const T* op, interval_expr a, interval_expr b) {
  if (a.is_point() && b.is_point()) {
    return point(simplify(op_simplified(), op, std::move(a.min), std::move(b.min)));
  } else {
    return {
        a.min.defined() && b.min.defined() ? simplify(op_simplified(), op, std::move(a.min), std::move(b.min)) : expr(),
        a.max.defined() && b.max.defined() ? simplify(op_simplified(), op, std::move(a.max), std::move(b.max)) : expr(),
    };
  }
}

template <typename T>
interval_expr bounds_of_less(const T* op, interval_expr a, interval_expr b) {
  if (a.is_point() && b.is_point()) {
    return point(simplify(op_simplified(), op, std::move(a.min), std::move(b.min)));
  } else {
    // This bit of genius comes from
    // https://github.com/halide/Halide/blob/61b8d384b2b799cd47634e4a3b67aa7c7f580a46/src/Bounds.cpp#L829
    return {
        a.max.defined() && b.min.defined() ? simplify(op_simplified(), op, std::move(a.max), std::move(b.min))
                                           : expr(0),
        a.min.defined() && b.max.defined() ? simplify(op_simplified(), op, std::move(a.min), std::move(b.max))
                                           : expr(1),
    };
  }
}

// Attempts to tighten the bounds for correlated expressions a +/- b, where the expressions are "stair step" functions
// like ((x + c0) / c1) * c2.
void tighten_correlated_bounds_stairs(interval_expr& bounds, const expr& a, const expr& b, int sign_b) {
  match_context lhs, rhs;
  if (!match(lhs, staircase(x, c0, c1, c2), a) || !match(rhs, staircase(x, c0, c1, c2), b)) {
    return;
  }
  if (!match(lhs.matched(x), rhs.matched(x))) {
    // The x in the above expressions doesn't match, expressions may not be correlated.
    return;
  }

  // We have a sum of two such rational expressions.
  index_t la = lhs.matched(c0);
  index_t lb = lhs.matched(c1);
  index_t lc = lhs.matched(c2);
  index_t ra = rhs.matched(c0);
  index_t rb = rhs.matched(c1);
  index_t rc = rhs.matched(c2) * sign_b;

  interval<std::int64_t> sb = staircase_sum_bounds(la, lb, lc, ra, rb, rc);
  // The bounds may not fit in index_t (when it is 32 bits), in which case they don't tighten anything.
  auto fits = [](std::int64_t x) {
    return std::numeric_limits<index_t>::min() <= x && x <= std::numeric_limits<index_t>::max();
  };
  if (sb.min && fits(*sb.min)) {
    bounds.min = max_lower(std::move(bounds.min), static_cast<index_t>(*sb.min));
  }
  if (sb.max && fits(*sb.max)) {
    bounds.max = min_upper(std::move(bounds.max), static_cast<index_t>(*sb.max));
  }
}

// We can tighten the upper bounds of expressions like min(x, y) - max(z, w) when x or y is correlated to z or w in a
// way we can understand the bounds of.
// TODO: We could also do a better lower bound for max - min.
void tighten_correlated_bounds_min_max(interval_expr& bounds, const expr& a, const expr& b, int sign_b) {
  if (sign_b != -1) return;

  const class min* min_a = a.as<class min>();
  const class max* max_b = b.as<class max>();
  if (!min_a || !max_b) return;

  // min(aa, ab) - max(ba, bb) is bounded in a way that our interval arithmetic below will miss.
  expr aa_ba = min_a->a - max_b->a;
  expr ab_bb = min_a->b - max_b->b;
  expr aa_bb = min_a->a - max_b->b;
  expr ab_ba = min_a->b - max_b->a;

  expr bounds_max = simplify(min(min(aa_ba, ab_bb), min(aa_bb, ab_ba)));

  // TODO: This might be blowing expressions up ridiculously... we might only want to do this in `constant_upper_bound`.
  bounds.max = min_upper(std::move(bounds_max), std::move(bounds.max));
}

// Some correlated expressions are very hard to simplify, but we can get some bounds for them relatively easily.
// These functions take existing bounds from interval arithmetic, and tighten them (via clamps) when we can do so.
void tighten_correlated_bounds(interval_expr& bounds, const expr& a, const expr& b, int sign_b) {
  tighten_correlated_bounds_stairs(bounds, a, b, sign_b);
  tighten_correlated_bounds_min_max(bounds, a, b, sign_b);
}

}  // namespace

interval_expr bounds_of(const add* op, interval_expr a, interval_expr b) {
  interval_expr result = bounds_of_linear(op, std::move(a), std::move(b));
  if (op) {
    tighten_correlated_bounds(result, op->a, op->b, 1);
  }
  return result;
}
interval_expr bounds_of(const sub* op, interval_expr a, interval_expr b) {
  // -b => bounds are {-max, -min}.
  std::swap(b.min, b.max);
  interval_expr result = bounds_of_linear(op, std::move(a), std::move(b));
  if (op) {
    tighten_correlated_bounds(result, op->a, op->b, -1);
  }
  return result;
}

namespace {

// Some `bounds_of` implementations can produce exponential amounts of expr nodes. To avoid problems with this,
// we can "give up" on producing tight bounds if the expressions aren't simple.
// Currently, we only use this for `mul` ops, because it is less likely to encounter large sequences of other ops that
// produce exponential bounds, and it is also unlikely we'll be able to prove useful things about the bounds of mul ops,
// but this is a pretty lame heuristic.
interval_expr simple_or_unbounded(expr min, expr max) {
  auto is_simple = [](const expr& x) { return x.as<variable>() || x.as<constant>(); };
  if (!is_simple(min)) min = expr();
  if (!is_simple(max)) max = expr();
  return {min, max};
}

struct constant_interval {
  std::optional<index_t> min;
  std::optional<index_t> max;

  constant_interval(const interval_expr& x) {
    min = evaluate_constant_lower_bound(x.min);
    max = evaluate_constant_upper_bound(x.max);
  }

  bool is_positive() const { return min && *min > 0; }
  bool is_negative() const { return max && *max < 0; }
  bool is_non_negative() const { return min && *min >= 0; }
  bool is_non_positive() const { return max && *max <= 0; }
};

expr mul_or_unbounded(const mul* op, const expr& a, const expr& b) {
  if (!a.defined() || !b.defined()) return expr();
  return simplify(op_simplified(), op, a, b);
}

expr negate(expr a) { return a.defined() ? simplify(static_cast<const sub*>(nullptr), 0, std::move(a)) : expr(); }
interval_expr negate(interval_expr x) {
  if (x.is_point()) {
    return point(negate(x.min));
  } else {
    return {negate(x.max), negate(x.min)};
  }
}

interval_expr union_x_negate_x(interval_expr x) {
  return {
      min_lower(x.min, negate(x.max)),
      max_upper(x.max, negate(x.min)),
  };
}

expr div_or_unbounded(const div* op, const expr& a, const expr& b) {
  if (!a.defined()) return expr();
  if (!b.defined()) return expr(0);
  return simplify(op_simplified(), op, a, b);
}

expr abs_or_unbounded(const call* op, expr a) {
  return a.defined() ? simplify(op, intrinsic::abs, {std::move(a)}) : expr();
}

}  // namespace

interval_expr bounds_of(const mul* op, interval_expr a, interval_expr b) {
  // TODO: I'm pretty sure there are cases missing here that would produce simpler bounds than the fallback cases.
  if (is_zero(a.as_point()) || is_zero(b.as_point())) {
    return point(0);
  }
  if (a.is_point() && b.is_point()) {
    return point(simplify(op_simplified(), op, std::move(a.min), std::move(b.min)));
  }

  constant_interval a_const(a);
  constant_interval b_const(b);

  if (a_const.is_non_negative() && b_const.is_non_negative()) {
    // Both are >= 0, neither intervals flip.
    return {mul_or_unbounded(op, a.min, b.min), mul_or_unbounded(op, a.max, b.max)};
  } else if (a_const.is_non_positive() && b_const.is_non_positive()) {
    // Both are <= 0, both intervals flip.
    return {mul_or_unbounded(op, a.max, b.max), mul_or_unbounded(op, a.min, b.min)};
  } else if (a_const.is_non_negative() && b_const.is_non_positive()) {
    return {mul_or_unbounded(op, a.max, b.min), mul_or_unbounded(op, a.min, b.max)};
  } else if (a_const.is_non_positive() && b_const.is_non_negative()) {
    return {mul_or_unbounded(op, a.min, b.max), mul_or_unbounded(op, a.max, b.min)};
  } else if (b.is_point()) {
    if (b_const.is_non_negative()) {
      return {mul_or_unbounded(op, a.min, b.min), mul_or_unbounded(op, a.max, b.min)};
    } else if (b_const.is_non_positive()) {
      return {mul_or_unbounded(op, a.max, b.min), mul_or_unbounded(op, a.min, b.min)};
    } else if (!a.min.defined() || !a.max.defined()) {
      return interval_expr::all();
    } else {
      expr corners[] = {
          simplify(op_simplified(), op, a.min, b.min),
          simplify(op_simplified(), op, a.max, b.min),
      };
      return simple_or_unbounded(simplify(static_cast<const class min*>(nullptr), corners[0], corners[1]),
          simplify(static_cast<const class max*>(nullptr), corners[0], corners[1]));
    }
  } else if (a.is_point()) {
    if (a_const.is_non_negative()) {
      return {mul_or_unbounded(op, a.min, b.min), mul_or_unbounded(op, a.min, b.max)};
    } else if (a_const.is_non_positive()) {
      return {mul_or_unbounded(op, a.min, b.max), mul_or_unbounded(op, a.min, b.min)};
    } else if (!b.min.defined() || !b.max.defined()) {
      return interval_expr::all();
    } else {
      expr corners[] = {
          simplify(op_simplified(), op, a.min, b.min),
          simplify(op_simplified(), op, a.min, b.max),
      };
      return simple_or_unbounded(simplify(static_cast<const class min*>(nullptr), corners[0], corners[1]),
          simplify(static_cast<const class max*>(nullptr), corners[0], corners[1]));
    }
  } else if (!a.min.defined() || !a.max.defined() || !b.min.defined() || !b.max.defined()) {
    return interval_expr::all();
  } else {
    // We don't know anything. The results is the union of all 4 possible intervals.
    expr corners[] = {
        simplify(op_simplified(), op, a.min, b.min),
        simplify(op_simplified(), op, a.min, b.max),
        simplify(op_simplified(), op, a.max, b.min),
        simplify(op_simplified(), op, a.max, b.max),
    };
    return simple_or_unbounded(simplify(static_cast<const class min*>(nullptr),
                                   simplify(static_cast<const class min*>(nullptr), corners[0], corners[1]),
                                   simplify(static_cast<const class min*>(nullptr), corners[2], corners[3])),
        simplify(static_cast<const class max*>(nullptr),
            simplify(static_cast<const class max*>(nullptr), corners[0], corners[1]),
            simplify(static_cast<const class max*>(nullptr), corners[2], corners[3])));
  }
}

interval_expr bounds_of(const div* op, interval_expr a, interval_expr b) {
  // Because b is an integer, the bounds of a will only be shrunk
  // (we define division by 0 to be 0). The absolute value of the
  // bounds are maximized when b is 1 or -1.
  constant_interval a_const(a);
  constant_interval b_const(b);

  if (b.is_point()) {
    if (is_zero(b.min)) {
      return {0, 0};
    } else if (a.is_point()) {
      return point(simplify(op_simplified(), op, a.min, b.min));
    } else if (b_const.is_non_negative()) {
      return {div_or_unbounded(op, a.min, b.min), div_or_unbounded(op, a.max, b.min)};
    } else if (b_const.is_non_positive()) {
      return {div_or_unbounded(op, a.max, b.min), div_or_unbounded(op, a.min, b.min)};
    }
  } else if (b_const.is_positive()) {
    // b > 0 => the biggest result in absolute value occurs at the min of b.
    if (a_const.is_non_negative()) {
      return {div_or_unbounded(op, a.min, b.max), div_or_unbounded(op, a.max, b.min)};
    } else if (a_const.is_non_positive()) {
      return {div_or_unbounded(op, a.min, b.min), div_or_unbounded(op, a.max, b.max)};
    } else {
      a = union_x_negate_x(std::move(a));
      return {div_or_unbounded(op, a.min, b.min), div_or_unbounded(op, a.max, b.min)};
    }
  } else if (b_const.is_negative()) {
    // b < 0 => the biggest result in absolute value occurs at the max of b.
    if (a_const.is_non_negative()) {
      return {div_or_unbounded(op, a.max, b.max), div_or_unbounded(op, a.min, b.min)};
    } else if (a_const.is_non_positive()) {
      return {div_or_unbounded(op, a.max, b.min), div_or_unbounded(op, a.min, b.max)};
    } else {
      a = union_x_negate_x(std::move(a));
      return {div_or_unbounded(op, a.max, b.max), div_or_unbounded(op, a.min, b.max)};
    }
  }
  return union_x_negate_x(std::move(a));
}

interval_expr bounds_of(const mod* op, interval_expr a, interval_expr b) {
  if (a.is_point() && b.is_point()) {
    return point(simplify(op_simplified(), op, std::move(a.min), std::move(b.min)));
  }
  expr max_b;
  constant_interval b_const(b);
  if (b_const.is_non_negative()) {
    max_b = b.max;
  } else if (b.is_point()) {
    assert(!b_const.is_non_negative());
    max_b = abs_or_unbounded(nullptr, b.max);
  } else {
    max_b = max_upper(abs_or_unbounded(nullptr, b.min), abs_or_unbounded(nullptr, b.max));
  }
  if (!max_b.defined()) {
    return {0, expr()};
  }
  // The bounds here are weird. The bounds of a % b are [0, max(abs(b))), note the open interval at the max.
  // So, we need to subtract 1 from the max value of b, but we need a special case for b = 0, where we define a % 0 to
  // be 0.
  return {0, simplify(static_cast<const class max*>(nullptr),
                 simplify(static_cast<const sub*>(nullptr), std::move(max_b), 1), 0)};
}

interval_expr bounds_of(const class min* op, interval_expr a, interval_expr b) {
  if (a.is_point() && b.is_point()) {
    return point(simplify(op_simplified(), op, std::move(a.min), std::move(b.min)));
  } else {
    return {
        min_lower(std::move(a.min), std::move(b.min), op),
        min_upper(std::move(a.max), std::move(b.max), op),
    };
  }
}
interval_expr bounds_of(const class max* op, interval_expr a, interval_expr b) {
  if (a.is_point() && b.is_point()) {
    return point(simplify(op_simplified(), op, std::move(a.min), std::move(b.min)));
  } else {
    return {
        max_lower(std::move(a.min), std::move(b.min), op),
        max_upper(std::move(a.max), std::move(b.max), op),
    };
  }
}

interval_expr bounds_of(const less* op, interval_expr a, interval_expr b) {
  return bounds_of_less(op, std::move(a), std::move(b));
}
interval_expr bounds_of(const less_equal* op, interval_expr a, interval_expr b) {
  return bounds_of_less(op, std::move(a), std::move(b));
}
interval_expr bounds_of(const equal* op, interval_expr a, interval_expr b) {
  if (a.is_point() && b.is_point()) {
    return point(simplify(op_simplified(), op, std::move(a.min), std::move(b.min)));
  } else {
    // This is can only be true if the intervals a and b overlap:
    //
    //   max(a.min, b.min) <= min(a.max, b.max)
    //   a.min <= a.max && b.min <= a.max && b.min <= a.max && b.min <= b.max
    //   true && b.min <= a.max && b.min <= a.max && true
    expr le1 = a.min.defined() && b.max.defined()
                   ? simplify(static_cast<const less_equal*>(nullptr), a.min, b.max)
                   : expr(1);
    expr le2 = b.min.defined() && a.max.defined()
                   ? simplify(static_cast<const less_equal*>(nullptr), b.min, a.max)
                   : expr(1);
    return {0, simplify(static_cast<const logical_and*>(nullptr), std::move(le1), std::move(le2))};
  }
}
interval_expr bounds_of(const not_equal* op, interval_expr a, interval_expr b) {
  if (a.is_point() && b.is_point()) {
    return point(simplify(op_simplified(), op, std::move(a.min), std::move(b.min)));
  } else {
    // This can only be false if the intervals a and b do not overlap.
    expr lt1 =
        a.max.defined() && b.min.defined() ? simplify(static_cast<const less*>(nullptr), a.max, b.min) : expr(0);
    expr lt2 =
        b.max.defined() && a.min.defined() ? simplify(static_cast<const less*>(nullptr), b.max, a.min) : expr(0);
    return {simplify(static_cast<const logical_or*>(nullptr), std::move(lt1), std::move(lt2)), 1};
  }
}

interval_expr bounds_of(const logical_and* op, interval_expr a, interval_expr b) {
  if (!a.min.defined()) a.min = 0;
  if (!a.max.defined()) a.max = 1;
  if (!b.min.defined()) b.min = 0;
  if (!b.max.defined()) b.max = 1;
  return bounds_of_linear(op, std::move(a), std::move(b));
}
interval_expr bounds_of(const logical_or* op, interval_expr a, interval_expr b) {
  if (!a.min.defined()) a.min = 0;
  if (!a.max.defined()) a.max = 1;
  if (!b.min.defined()) b.min = 0;
  if (!b.max.defined()) b.max = 1;
  return bounds_of_linear(op, std::move(a), std::move(b));
}
interval_expr bounds_of(const logical_not* op, interval_expr a) {
  if (a.is_point(op)) {
    return point(expr(op));
  } else if (a.is_point()) {
    return point(simplify(op, std::move(a.min)));
  } else {
    return {
        a.max.defined() ? simplify(op, std::move(a.max)) : expr(0),
        a.min.defined() ? simplify(op, std::move(a.min)) : expr(1),
    };
  }
}

interval_expr bounds_of(const class select* op, interval_expr c, interval_expr t, interval_expr f) {
  if (c.is_point() && t.is_point() && f.is_point()) {
    if (op && c.min.same_as(op->condition) && t.min.same_as(op->true_value) && f.min.same_as(op->false_value)) {
      return point(expr(op));
    } else {
      return point(simplify(op, std::move(c.min), std::move(t.min), std::move(f.min)));
    }
  } else if (is_true(c.min)) {
    return t;
  } else if (is_false(c.max)) {
    return f;
  } else if (c.is_point()) {
    return select(c.min, std::move(t), std::move(f));
  } else {
    return {
        min_lower(std::move(t.min), std::move(f.min)),
        max_upper(std::move(t.max), std::move(f.max)),
    };
  }
}

interval_expr bounds_of(const call* op, std::vector<interval_expr> args) {
  switch (op->intrinsic) {
  case intrinsic::abs:
    assert(args.size() == 1);
    if (is_non_negative(args[0].min)) {
      return {args[0].min, args[0].max};
    } else if (is_non_positive(args[0].max)) {
      return negate(args[0]);
    } else if (args[0].is_point()) {
      return {0, simplify(op, intrinsic::abs, {std::move(args[0].min)})};
    } else {
      return {0, max_upper(abs_or_unbounded(op, args[0].min), abs_or_unbounded(op, args[0].max))};
    }
  default: return point(expr(op));
  }
}

}  // namespace slinky
