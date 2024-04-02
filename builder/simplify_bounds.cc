#include "builder/simplify.h"

#include <algorithm>

#include "runtime/expr.h"

namespace slinky {

namespace {

template <typename T>
interval_expr bounds_of_linear(const T* op, interval_expr a, interval_expr b) {
  return {simplify(op, std::move(a.min), std::move(b.min)), simplify(op, std::move(a.max), std::move(b.max))};
}

template <typename T>
interval_expr bounds_of_less(const T* op, interval_expr a, interval_expr b) {
  // This bit of genius comes from
  // https://github.com/halide/Halide/blob/61b8d384b2b799cd47634e4a3b67aa7c7f580a46/src/Bounds.cpp#L829
  return {simplify(op, std::move(a.max), std::move(b.min)), simplify(op, std::move(a.min), std::move(b.max))};
}

}  // namespace

interval_expr bounds_of(const add* op, interval_expr a, interval_expr b) {
  return bounds_of_linear(op, std::move(a), std::move(b));
}
interval_expr bounds_of(const sub* op, interval_expr a, interval_expr b) {
  return {simplify(op, std::move(a.min), std::move(b.max)), simplify(op, std::move(a.max), std::move(b.min))};
}
interval_expr bounds_of(const mul* op, interval_expr a, interval_expr b) {
  // TODO: I'm pretty sure there are cases missing here that would produce simpler bounds than the fallback cases.
  if (is_non_negative(a.min) && is_non_negative(b.min)) {
    // Both are >= 0, neither intervals flip.
    return {simplify(op, a.min, b.min), simplify(op, a.max, b.max)};
  } else if (is_non_positive(a.max) && is_non_positive(b.max)) {
    // Both are <= 0, both intervals flip.
    return {simplify(op, a.max, b.max), simplify(op, a.min, b.min)};
  } else if (b.is_point()) {
    if (is_non_negative(b.min)) {
      return {simplify(op, a.min, b.min), simplify(op, a.max, b.min)};
    } else if (is_non_positive(b.min)) {
      return {simplify(op, a.max, b.min), simplify(op, a.min, b.min)};
    } else {
      expr corners[] = {
          simplify(op, a.min, b.min),
          simplify(op, a.max, b.min),
      };
      return {
          simplify(static_cast<const class min*>(nullptr), corners[0], corners[1]),
          simplify(static_cast<const class max*>(nullptr), corners[0], corners[1]),
      };
    }
  } else if (a.is_point()) {
    if (is_non_negative(a.min)) {
      return {simplify(op, a.min, b.min), simplify(op, a.min, b.max)};
    } else if (is_non_positive(a.min)) {
      return {simplify(op, a.min, b.max), simplify(op, a.min, b.min)};
    } else {
      expr corners[] = {
          simplify(op, a.min, b.min),
          simplify(op, a.min, b.max),
      };
      return {
          simplify(static_cast<const class min*>(nullptr), corners[0], corners[1]),
          simplify(static_cast<const class max*>(nullptr), corners[0], corners[1]),
      };
    }
  } else {
    // We don't know anything. The results is the union of all 4 possible intervals.
    expr corners[] = {
        simplify(op, a.min, b.min),
        simplify(op, a.min, b.max),
        simplify(op, a.max, b.min),
        simplify(op, a.max, b.max),
    };
    return {
        simplify(static_cast<const class min*>(nullptr),
            simplify(static_cast<const class min*>(nullptr), corners[0], corners[1]),
            simplify(static_cast<const class min*>(nullptr), corners[2], corners[3])),
        simplify(static_cast<const class max*>(nullptr),
            simplify(static_cast<const class max*>(nullptr), corners[0], corners[1]),
            simplify(static_cast<const class max*>(nullptr), corners[2], corners[3])),
    };
  }
}

namespace {

expr negate(expr a) { return simplify(static_cast<const sub*>(nullptr), 0, std::move(a)); }

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
  if (b.is_point()) {
    if (is_zero(b.min)) {
      return {0, 0};
    } else if (is_positive(b.min)) {
      return {simplify(op, a.min, b.min), simplify(op, a.max, b.min)};
    } else if (is_negative(b.min)) {
      return {simplify(op, a.max, b.min), simplify(op, a.min, b.min)};
    }
  } else if (is_positive(b.min)) {
    // b > 0 => the biggest result in absolute value occurs at the min of b.
    if (is_positive(a.min)) {
      return {simplify(op, a.min, b.max), simplify(op, a.max, b.min)};
    } else if (is_negative(a.max)) {
      return {simplify(op, a.min, b.min), simplify(op, a.max, b.max)};
    } else {
      a = union_x_negate_x(std::move(a));
      return {simplify(op, a.min, b.min), simplify(op, a.max, b.min)};
    }
  } else if (is_negative(b.max)) {
    // b < 0 => the biggest result in absolute value occurs at the max of b.
    if (is_positive(a.min)) {
      return {simplify(op, a.max, b.max), simplify(op, a.min, b.min)};
    } else if (is_negative(a.max)) {
      return {simplify(op, a.max, b.min), simplify(op, a.min, b.max)};
    } else {
      a = union_x_negate_x(std::move(a));
      return {simplify(op, a.max, b.max), simplify(op, a.min, b.max)};
    }
  }
  return union_x_negate_x(std::move(a));
}
interval_expr bounds_of(const mod* op, interval_expr a, interval_expr b) {
  return {0, simplify(static_cast<const class max*>(nullptr),
                 simplify(static_cast<const class call*>(nullptr), intrinsic::abs, {b.min}),
                 simplify(static_cast<const class call*>(nullptr), intrinsic::abs, {b.max}))};
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
  return {0, simplify(static_cast<const logical_and*>(nullptr),
                 simplify(static_cast<const less_equal*>(nullptr), a.min, b.max),
                 simplify(static_cast<const less_equal*>(nullptr), b.min, a.max))};
}
interval_expr bounds_of(const not_equal* op, interval_expr a, interval_expr b) {
  return {simplify(static_cast<const logical_or*>(nullptr), simplify(static_cast<const less*>(nullptr), a.max, b.min),
              simplify(static_cast<const less*>(nullptr), b.max, a.min)),
      1};
}

interval_expr bounds_of(const logical_and* op, interval_expr a, interval_expr b) {
  return bounds_of_linear(op, std::move(a), std::move(b));
}
interval_expr bounds_of(const logical_or* op, interval_expr a, interval_expr b) {
  return bounds_of_linear(op, std::move(a), std::move(b));
}
interval_expr bounds_of(const logical_not* op, interval_expr a) {
  return {simplify(op, std::move(a.max)), simplify(op, std::move(a.min))};
}

interval_expr bounds_of(const class select* op, interval_expr c, interval_expr t, interval_expr f) {
  if (is_true(c.min)) {
    return t;
  } else if (is_false(c.max)) {
    return f;
  } else {
    return f | t;
  }
}

interval_expr bounds_of(const call* op, std::vector<interval_expr> args) {
  switch (op->intrinsic) {
  case intrinsic::abs:
    assert(args.size() == 1);
    if (is_positive(args[0].min)) {
      return {args[0].min, args[0].max};
    } else if (is_negative(args[0].max)) {
      return {negate(args[0].max), negate(args[0].min)};
    } else {
      expr abs_min = simplify(op, intrinsic::abs, {args[0].min});
      expr abs_max = simplify(op, intrinsic::abs, {args[0].max});
      return {0, simplify(static_cast<const class max*>(nullptr), std::move(abs_min), std::move(abs_max))};
    }
  default: return {op, op};
  }
}

}  // namespace slinky
