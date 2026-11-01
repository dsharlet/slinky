#include "slinky/builder/simplify.h"

#include <algorithm>

#include "slinky/builder/rewrite.h"
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

template <typename T>
interval_expr bounds_of_linear(const T* op, interval_expr a, interval_expr b) {
  if (a.is_point() && b.is_point()) {
    return point(simplify(op_simplified(), op, std::move(a.min), std::move(b.min)));
  } else {
    return {
        simplify(op_simplified(), op, std::move(a.min), std::move(b.min)),
        simplify(op_simplified(), op, std::move(a.max), std::move(b.max)),
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
        simplify(op_simplified(), op, std::move(a.max), std::move(b.min)),
        simplify(op_simplified(), op, std::move(a.min), std::move(b.max)),
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

  interval<int> sb = staircase_sum_bounds(la, lb, lc, ra, rb, rc);
  if (sb.min) bounds.min = simplify(static_cast<const class max*>(nullptr), bounds.min, *sb.min);
  if (sb.max) bounds.max = simplify(static_cast<const class min*>(nullptr), bounds.max, *sb.max);
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
  bounds.max = simplify(static_cast<const class min*>(nullptr), bounds_max, bounds.max);
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
  if (!is_simple(min)) min = slinky::negative_infinity();
  if (!is_simple(max)) max = slinky::positive_infinity();
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

}  // namespace

interval_expr bounds_of(const mul* op, interval_expr a, interval_expr b) {
  // TODO: I'm pretty sure there are cases missing here that would produce simpler bounds than the fallback cases.
  if (a.is_point() && b.is_point()) {
    return point(simplify(op_simplified(), op, std::move(a.min), std::move(b.min)));
  }

  constant_interval a_const(a);
  constant_interval b_const(b);

  if (a_const.is_non_negative() && b_const.is_non_negative()) {
    // Both are >= 0, neither intervals flip.
    return {simplify(op_simplified(), op, a.min, b.min), simplify(op_simplified(), op, a.max, b.max)};
  } else if (a_const.is_non_positive() && b_const.is_non_positive()) {
    // Both are <= 0, both intervals flip.
    return {simplify(op_simplified(), op, a.max, b.max), simplify(op_simplified(), op, a.min, b.min)};
  } else if (b.is_point()) {
    if (b_const.is_non_negative()) {
      return {simplify(op_simplified(), op, a.min, b.min), simplify(op_simplified(), op, a.max, b.min)};
    } else if (b_const.is_non_positive()) {
      return {simplify(op_simplified(), op, a.max, b.min), simplify(op_simplified(), op, a.min, b.min)};
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
      return {simplify(op_simplified(), op, a.min, b.min), simplify(op_simplified(), op, a.min, b.max)};
    } else if (a_const.is_non_positive()) {
      return {simplify(op_simplified(), op, a.min, b.max), simplify(op_simplified(), op, a.min, b.min)};
    } else {
      expr corners[] = {
          simplify(op_simplified(), op, a.min, b.min),
          simplify(op_simplified(), op, a.min, b.max),
      };
      return simple_or_unbounded(simplify(static_cast<const class min*>(nullptr), corners[0], corners[1]),
          simplify(static_cast<const class max*>(nullptr), corners[0], corners[1]));
    }
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

namespace {

expr negate(expr a) { return simplify(static_cast<const sub*>(nullptr), 0, std::move(a)); }
interval_expr negate(interval_expr x) { return bounds_of(static_cast<const sub*>(nullptr), point(0), std::move(x)); }

interval_expr union_x_negate_x(interval_expr x) {
  return {
      simplify(static_cast<const class min*>(nullptr), x.min, negate(x.max)),
      simplify(static_cast<const class max*>(nullptr), x.max, negate(x.min)),
  };
}

}  // namespace

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
      return {simplify(op_simplified(), op, a.min, b.min), simplify(op_simplified(), op, a.max, b.min)};
    } else if (b_const.is_non_positive()) {
      return {simplify(op_simplified(), op, a.max, b.min), simplify(op_simplified(), op, a.min, b.min)};
    }
  } else if (b_const.is_positive()) {
    // b > 0 => the biggest result in absolute value occurs at the min of b.
    if (a_const.is_non_negative()) {
      return {simplify(op_simplified(), op, a.min, b.max), simplify(op_simplified(), op, a.max, b.min)};
    } else if (a_const.is_non_positive()) {
      return {simplify(op_simplified(), op, a.min, b.min), simplify(op_simplified(), op, a.max, b.max)};
    } else {
      a = union_x_negate_x(std::move(a));
      return {simplify(op_simplified(), op, a.min, b.min), simplify(op_simplified(), op, a.max, b.min)};
    }
  } else if (b_const.is_negative()) {
    // b < 0 => the biggest result in absolute value occurs at the max of b.
    if (a_const.is_non_negative()) {
      return {simplify(op_simplified(), op, a.max, b.max), simplify(op_simplified(), op, a.min, b.min)};
    } else if (a_const.is_non_positive()) {
      return {simplify(op_simplified(), op, a.max, b.min), simplify(op_simplified(), op, a.min, b.max)};
    } else {
      a = union_x_negate_x(std::move(a));
      return {simplify(op_simplified(), op, a.max, b.max), simplify(op_simplified(), op, a.min, b.max)};
    }
  }
  return union_x_negate_x(std::move(a));
}
interval_expr bounds_of(const mod* op, interval_expr a, interval_expr b) {
  if (a.is_point() && b.is_point()) {
    return point(simplify(op_simplified(), op, std::move(a.min), std::move(b.min)));
  }
  // The bounds here are weird. The bounds of a % b are [0, max(abs(b))), note the open interval at the max.
  // So, we need to subtract 1 from the max value of b, but we need a special case for b = 0, where we define a % 0 to
  // be 0.
  interval_expr abs_b = bounds_of(static_cast<const call*>(nullptr), intrinsic::abs, {std::move(b)});
  return {0, simplify(static_cast<const class max*>(nullptr),
                 simplify(static_cast<const sub*>(nullptr), std::move(abs_b.max), 1), 0)};
}

interval_expr bounds_of(const class min* op, interval_expr a, interval_expr b) {
  return bounds_of_linear(op, std::move(a), std::move(b));
}
interval_expr bounds_of(const class max* op, interval_expr a, interval_expr b) {
  return bounds_of_linear(op, std::move(a), std::move(b));
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
    return {0, simplify(static_cast<const logical_and*>(nullptr),
                   simplify(static_cast<const less_equal*>(nullptr), a.min, b.max),
                   simplify(static_cast<const less_equal*>(nullptr), b.min, a.max))};
  }
}
interval_expr bounds_of(const not_equal* op, interval_expr a, interval_expr b) {
  if (a.is_point() && b.is_point()) {
    return point(simplify(op_simplified(), op, std::move(a.min), std::move(b.min)));
  } else {
    // This can only be false if the intervals a and b do not overlap.
    return {simplify(static_cast<const logical_or*>(nullptr), simplify(static_cast<const less*>(nullptr), a.max, b.min),
                simplify(static_cast<const less*>(nullptr), b.max, a.min)),
        1};
  }
}

interval_expr bounds_of(const logical_and* op, interval_expr a, interval_expr b) {
  return bounds_of_linear(op, std::move(a), std::move(b));
}
interval_expr bounds_of(const logical_or* op, interval_expr a, interval_expr b) {
  return bounds_of_linear(op, std::move(a), std::move(b));
}
interval_expr bounds_of(const logical_not* op, interval_expr a) {
  if (a.is_point(op)) {
    return point(expr(op));
  } else if (a.is_point()) {
    return point(simplify(op, std::move(a.min)));
  } else {
    return {simplify(op, std::move(a.max)), simplify(op, std::move(a.min))};
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
        simplify(static_cast<const class min*>(nullptr), std::move(t.min), std::move(f.min)),
        simplify(static_cast<const class max*>(nullptr), std::move(t.max), std::move(f.max)),
    };
  }
}

interval_expr bounds_of(const call* op, intrinsic fn, std::vector<interval_expr> args) {
  switch (fn) {
  case intrinsic::abs: {
    assert(args.size() == 1);
    constant_interval a_const(args[0]);
    if (a_const.is_non_negative()) {
      return args[0];
    } else if (a_const.is_non_positive()) {
      return negate(args[0]);
    } else if (args[0].is_point()) {
      return {0, simplify(op, intrinsic::abs, {std::move(args[0].min)})};
    } else {
      expr abs_min = simplify(op, intrinsic::abs, {args[0].min});
      expr abs_max = simplify(op, intrinsic::abs, {args[0].max});
      return {0, simplify(static_cast<const class max*>(nullptr), std::move(abs_min), std::move(abs_max))};
    }
  }
  case intrinsic::ravel: {
    assert(args.size() % 2 == 1);
    if (std::all_of(args.begin(), args.end(), [](const interval_expr& i) { return i.is_point(); })) {
      std::vector<expr> points;
      points.reserve(args.size());
      for (interval_expr& i : args) points.push_back(std::move(i.min));
      return point(simplify(op, intrinsic::ravel, std::move(points)));
    }
    // ravel is a sum of the coordinates multiplied by positive coefficients, so it is monotonically increasing in
    // each coordinate. We need the extents to be points to compute the coefficients.
    std::vector<expr> mins, maxs;
    bool min_infinite = false;
    bool max_infinite = false;
    for (std::size_t i = 0; i < args.size(); ++i) {
      if (i % 2 == 1) {
        if (!args[i].is_point()) return interval_expr::all();
        mins.push_back(args[i].min);
        maxs.push_back(args[i].min);
      } else {
        min_infinite = min_infinite || is_infinity(args[i].min);
        max_infinite = max_infinite || is_infinity(args[i].max);
        mins.push_back(args[i].min);
        maxs.push_back(args[i].max);
      }
    }
    return {
        min_infinite ? slinky::negative_infinity() : simplify(op, intrinsic::ravel, std::move(mins)),
        max_infinite ? slinky::positive_infinity() : simplify(op, intrinsic::ravel, std::move(maxs)),
    };
  }
  case intrinsic::unravel: {
    assert(args.size() >= 2);
    if (std::all_of(args.begin(), args.end(), [](const interval_expr& i) { return i.is_point(); })) {
      std::vector<expr> points;
      points.reserve(args.size());
      for (interval_expr& i : args) points.push_back(std::move(i.min));
      return point(simplify(op, intrinsic::unravel, std::move(points)));
    }
    const std::size_t num_extents = args.size() - 2;
    auto d = as_constant(args.back().min);
    if (!d || !args.back().is_point() || *d < 0 || *d > static_cast<index_t>(num_extents)) {
      return interval_expr::all();
    }
    if (*d < static_cast<index_t>(num_extents)) {
      // This coordinate is a remainder modulo the extent.
      interval_expr extent = bounds_of(static_cast<const class max*>(nullptr), std::move(args[*d + 1]), point(1));
      return bounds_of(static_cast<const mod*>(nullptr), interval_expr::all(), std::move(extent));
    } else {
      // This coordinate is the flat index divided by the product of the extents.
      interval_expr stride = point(1);
      for (std::size_t i = 1; i <= num_extents; ++i) {
        interval_expr extent = bounds_of(static_cast<const class max*>(nullptr), std::move(args[i]), point(1));
        stride = bounds_of(static_cast<const mul*>(nullptr), std::move(stride), std::move(extent));
      }
      return bounds_of(static_cast<const div*>(nullptr), std::move(args[0]), std::move(stride));
    }
  }
  default: return point(expr(op));
  }
}

}  // namespace slinky
