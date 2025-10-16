#ifndef SLINKY_BUILDER_SIMPLIFY_RULES_H
#define SLINKY_BUILDER_SIMPLIFY_RULES_H

#include "slinky/builder/rewrite.h"

namespace slinky {

namespace {

rewrite::pattern_wildcard<0> x;
rewrite::pattern_wildcard<1> y;
rewrite::pattern_wildcard<2> z;
rewrite::pattern_wildcard<3> w;
rewrite::pattern_wildcard<4> u;
rewrite::pattern_wildcard<5> v;
rewrite::pattern_wildcard<6> t;

rewrite::pattern_constant<0> c0;
rewrite::pattern_constant<1> c1;
rewrite::pattern_constant<2> c2;
rewrite::pattern_constant<3> c3;
rewrite::pattern_constant<4> c4;
rewrite::pattern_constant<5> c5;
rewrite::pattern_constant<6> c6;

using rewrite::may_be;

}  // namespace

// clang-format off

template <typename Fn>
bool apply_min_rules(Fn&& apply) {
  return
      // One side is the min
      apply(min(x, std::numeric_limits<index_t>::max()), x) ||
      apply(min(x, rewrite::positive_infinity()), x) ||
      apply(min(x, std::numeric_limits<index_t>::min()), std::numeric_limits<index_t>::min()) ||
      apply(min(x, rewrite::negative_infinity()), rewrite::negative_infinity()) ||
      apply(min(x, x + may_be<0>(c0)),
        x, c0 >= 0,
        x + c0 /*c0 < 0*/) ||
      apply(min(x, y), x && y, is_boolean(x) && is_boolean(y)) ||

      // These might be the only rules that don't have an analogous max rule.
      apply(min(max(x, c0)/may_be<1>(c1), c2),
        c2, c1 > 1 && c0/c1 == c2,
        max(min(x/c1, c2), eval(c0/c1)), c1 > 1 && c0/c1 < c2) ||
      apply(min(min(y, max(x, c0)/may_be<1>(c1)), c2),
        min(y, c2), c1 > 1 && c0/c1 == c2,
        min(y, max(min(x/c1, c2), eval(c0/c1))), c1 > 1 && c0/c1 < c2) ||

      // Canonicalize trees and find duplicate terms.
      apply(min(min(x, y), min(x, z)), min(x, min(y, z))) ||
      apply(min(min(x, y), min(z, w)), min(x, min(y, min(z, w)))) ||
      apply(min(x, min(x, y)), min(x, y)) ||
      apply(min(x, min(y, min(x, z))), min(x, min(y, z))) ||
      apply(min(x, min(y, min(z, min(x, w)))), min(x, min(y, min(z, w)))) ||
      apply(min(x, min(y, min(z, min(w, min(x, u))))), min(x, min(y, min(z, min(w, u))))) ||
      apply(min(x, min(y, min(z, min(w, min(u, min(x, v)))))), min(x, min(y, min(z, min(w, min(u, v)))))) ||

      // Similar rules but with mixes of min and max.
      apply(min(max(y, x + may_be<0>(c0)) + may_be<0>(c1), max(z, x + may_be<0>(c2))), 
        max(x + c2, min(z, y + c1)), eval(c0 + c1 == c2)) ||
      apply(min(x, min(y, x + may_be<0>(c0)) + may_be<0>(c1)),
        min(x, y + c1), c0 + c1 >= 0,
        min(y, x + c0) + c1 /*c0 + c1 < 0*/) ||
      apply(min(x + may_be<0>(c0), max(y, min(x, z) + may_be<0>(c1))), min(x + c0, max(y, z + c1)), c1 >= c0) ||
      apply(min(max(x, min(y, c0)), c1), min(max(x, y), c1), c0 >= c1) ||
      apply(min(min(x, y), max(x, z)), min(x, y)) ||
      apply(min(x, min(y, max(x, z))), min(x, y)) ||
      apply(min(x, max(y, min(x, z))), min(x, max(y, z))) ||
      apply(min(min(y, max(x, z)), max(w, z)), min(y, max(min(w, x), z))) ||
      apply(min(x, max(x, y) + may_be<0>(c1)), x, c1 >= 0) ||
      apply(min(x, max(y, x + may_be<0>(c0))), x, c0 >= 0) ||

      // Pull common terms out.
      apply(min(x + may_be<0>(z), min(y, x + may_be<0>(w))), min(y, x + min(z, w))) ||
      apply(min(x + may_be<0>(y), x + z), x + min(y, z)) ||
      apply(min(x + may_be<0>(y), x - z),
        x - max(-y, z), is_non_positive(y),
        x + min(y, -z)) ||
      apply(min(y - x, z - x), min(y, z) - x) ||
      apply(min(x - y, x - z), x - max(y, z)) ||
      apply(min(x, -x), -abs(x)) ||

      // Selects
      apply(min(x, select(y, max(z, min(x, u)), w)), select(y, min(x, max(z, u)), min(x, w))) ||
      apply(min(x, select(y, z, max(w, min(x, u)))), select(y, min(x, z), min(x, max(w, u)))) ||
      apply(min(x, select(y, min(x, z), w)), min(x, select(y, z, w))) ||
      apply(min(x, select(y, z, min(x, w))), min(x, select(y, z, w))) ||
      apply(min(x, select(y, max(x, z), w)), select(y, x, min(x, w))) ||
      apply(min(x, select(y, z, max(x, w))), select(y, min(x, z), x)) ||
      apply(min(y, select(x, y, w)), select(x, y, min(y, w))) ||
      apply(min(z, select(x, w, z)), select(x, min(z, w), z)) ||
      apply(min(may_be<0>(w) + select(x, y, z), select(x, u, v)), select(x, min(u, w + y), min(v, w + z))) ||
      apply(min(min(v, select(x, y, z)), select(x, w, u)), min(v, select(x, min(y, w), min(z, u)))) ||
      apply(min(max(v, select(x, y, z)), select(x, w, u)), select(x, min(w, max(v, y)), min(u, max(v, z)))) ||
      apply(min(w - select(x, y, z), select(x, u, v)), select(x, min(u, w - y), min(v, w - z))) ||
      apply(min(select(x, y, z) - w, select(x, u, v)), select(x, min(u, y - w), min(v, z - w))) ||

      apply(min(select(x, y, select(z, w, u)), select(z, v, t)), select(z, min(v, select(x, y, w)), min(t, select(x, y, u)))) ||
      apply(min(select(x, select(z, w, u), y), select(z, v, t)), select(z, min(v, select(x, w, y)), min(t, select(x, u, y)))) ||

      apply(min(x + may_be<0>(c2), select(c0 < x, y, c1)), select(c0 < x, min(x, y - c2), x) + c2, c1 >= c0 + c2) ||
      apply(min(x + may_be<0>(c2), select(c0 < x, c1, y)), select(c0 < x, c1, min(y, x + c2)), c1 <= c0 + c2) ||
      apply(min(x + may_be<0>(c2), select(x < c0, y, c1)), select(x < c0, min(y, x + c2), c1), c1 <= c0 + c2) ||
      apply(min(x + may_be<0>(c2), select(x < c0, c1, y)), select(x < c0, x, min(x, y - c2)) + c2, c1 >= c0 + c2) ||

      // Move constants out.
      apply(min(min(x, c0), c1), min(x, eval(min(c0, c1)))) ||
      apply(min(x, min(y, c0)), min(min(x, y), c0)) ||
      apply(min(x + c0, (y + c1)/c2), min(x, (y + eval(c1 - c0*c2))/c2) + c0, c2 != 0) ||
      apply(min(x + c0, y + c1),
        min(x, y + eval(c1 - c0)) + c0, c0 >= c1,  // Canonicalize to pulling bigger constants out.
        min(y, x + eval(c0 - c1)) + c1) ||
      apply(min(x + c0, c1 - may_be<0>(y)), 
        min(x, eval(c1 - c0)) + c0, is_zero(y),
        c1 - max(y, eval(c1 - c0) - x)) ||
      apply(min(c0 - x, c1 - may_be<0>(y)), c0 - max(x, y + eval(c0 - c1))) ||
      apply(min(min(x, c0) + may_be<0>(c1), min(y, c2)), min(min(y, x + c1), eval(min(c0 + c1, c2)))) ||
      apply(min(max(x, c0) + may_be<0>(c1), max(y, c2)),
        max(min(y, max(x, c0) + c1), c2), c2 < c0 + c1,
        max(min(x + c1, max(y, c2)), eval(c0 + c1)) /*c2 >= c0 + c1*/) ||
      apply(min(x + c0, select(y, may_be<0>(z) + c1, may_be<0>(w) + c2)), min(x, select(y, z + eval(c1 - c0), w + eval(c2 - c0))) + c0) ||
      apply(min(select(y, c1, c2), c0), select(y, eval(min(c0, c1)), eval(min(c0, c2)))) ||

      // https://github.com/halide/Halide/blob/7994e7030976f9fcd321a4d1d5f76f4582e01905/src/Simplify_Min.cpp#L276-L311
      apply(min(x*c0, c1),
        min(x, eval(c1/c0))*c0, c0 > 0 && c1%c0 == 0,
        max(x, eval(c1/c0))*c0, c0 < 0 && c1%c0 == 0) ||

      apply(min(x*c0, y*c1),
        min(x, y*eval(c1/c0))*c0, c0 > 0 && c1%c0 == 0,
        max(x, y*eval(c1/c0))*c0, c0 < 0 && c1%c0 == 0,
        min(y, x*eval(c0/c1))*c1, c1 > 0 && c0%c1 == 0,
        max(y, x*eval(c0/c1))*c1, c1 < 0 && c0%c1 == 0) ||
      apply(min(y*c0 + c1, x*c0),
        min(x, y + eval(c1/c0))*c0, c0 > 0 && c1%c0 == 0,
        max(x, y + eval(c1/c0))*c0, c0 < 0 && c1%c0 == 0) ||

      apply(min(y/c0 + may_be<0>(c1), x/c0),
        min(x, y + eval(c1*c0))/c0, c0 > 0,
        max(x, y + eval(c1*c0))/c0, c0 < 0) ||

      apply(min(x/c0, c1),
        min(x, eval(c1*c0))/c0, c0 > 0,
        max(x, eval(c1*c0))/c0, c0 < 0) ||

      apply(min(staircase(x, c0, c1, c2), staircase(x, c3, c4, c5) + may_be<0>(c6)),
        staircase(x, c0, c1, c2), 0 <= staircase_sum_min(c0, c1, -c2, c3, c4, c5) + c6,
        staircase(x, c3, c4, c5) + c6, 0 >= staircase_sum_max(c0, c1, -c2, c3, c4, c5) + c6) ||

      apply(min(x, abs(x)), x) ||

      false;
}

template <typename Fn>
bool apply_max_rules(Fn&& apply) {
  return
      // One side is the max.
      apply(max(x, std::numeric_limits<index_t>::min()), x) ||
      apply(max(x, rewrite::negative_infinity()), x) ||
      apply(max(x, std::numeric_limits<index_t>::max()), std::numeric_limits<index_t>::max()) ||
      apply(max(x, rewrite::positive_infinity()), rewrite::positive_infinity()) ||
      apply(max(x, x + may_be<0>(c0)),
        x + c0, c0 > 0,
        x /*c0 <= 0*/) ||
      apply(max(x, y), x || y, is_boolean(x) && is_boolean(y)) ||

      // Canonicalize trees and find duplicate terms.
      apply(max(max(x, y), max(x, z)), max(x, max(y, z))) ||
      apply(max(max(x, y), max(z, w)), max(x, max(y, max(z, w)))) ||
      apply(max(x, max(x, y)), max(x, y)) ||
      apply(max(x, max(y, max(x, z))), max(x, max(y, z))) ||
      apply(max(x, max(y, max(z, max(x, w)))), max(x, max(y, max(z, w)))) ||
      apply(max(x, max(y, max(z, max(w, max(x, u))))), max(x, max(y, max(z, max(w, u))))) ||
      apply(max(x, max(y, max(z, max(w, max(u, max(x, v)))))), max(x, max(y, max(z, max(w, max(u, v)))))) ||

      // Similar rules but with mixes of min and max.
      apply(max(min(y, x + may_be<0>(c0)) + may_be<0>(c1), min(z, x + may_be<0>(c2))),
        min(x + c2, max(z, y + c1)), eval(c0 + c1 == c2)) ||
      apply(max(x, max(y, x + may_be<0>(c0)) + may_be<0>(c1)),
        max(x, y + c1), c0 + c1 <= 0,
        max(y, x + c0) + c1 /*c0 + c1 > 0)*/) ||
      apply(max(x + may_be<0>(c0), min(y, max(x, z) + may_be<0>(c1))), max(x + c0, min(y, z + c1)), c1 < c0) ||
      apply(max(min(x, max(y, c0)), c1), max(min(x, y), c1), c0 <= c1) ||
      apply(max(min(x, y), max(x, z)), max(x, z)) ||
      apply(max(x, max(y, min(x, z))), max(x, y)) ||
      apply(max(x, min(y, max(x, z))), max(x, min(y, z))) ||
      apply(max(min(w, z), max(y, min(x, z))), max(y, min(max(w, x), z))) ||
      apply(max(x, min(y, x + may_be<0>(c0))), x, c0 <= 0) ||
      apply(max(x, min(x, y) + may_be<0>(c1)), x, c1 <= 0) ||

      // Pull common terms out.
      apply(max(x + may_be<0>(z), max(y, x + may_be<0>(w))), max(y, x + max(z, w))) ||
      apply(max(x + may_be<0>(y), x + z), x + max(y, z)) ||
      apply(max(x + may_be<0>(y), x - z),
        x - min(-y, z), is_non_positive(y),
        x + max(y, -z)) ||
      apply(max(y - x, z - x), max(y, z) - x) ||
      apply(max(x - y, x - z), x - min(y, z)) ||
      apply(max(x, -x), abs(x)) ||

      // Selects
      apply(max(x, select(y, max(z, min(x, u)), w)), select(y, max(z, x), max(x, w))) ||
      apply(max(x, select(y, z, max(w, min(x, u)))), select(y, max(x, z), max(x, w))) ||
      apply(max(x, select(y, max(x, z), w)), max(x, select(y, z, w))) ||
      apply(max(x, select(y, z, max(x, w))), max(x, select(y, z, w))) ||
      apply(max(x, select(y, min(x, z), w)), select(y, x, max(x, w))) ||
      apply(max(x, select(y, z, min(x, w))), select(y, max(x, z), x)) ||
      apply(max(y, select(x, y, w)), select(x, y, max(y, w))) ||
      apply(max(z, select(x, w, z)), select(x, max(z, w), z)) ||
      apply(max(may_be<0>(w) + select(x, y, z), select(x, u, v)), select(x, max(u, w + y), max(v, w + z))) ||
      apply(max(max(v, select(x, y, z)), select(x, w, u)), max(v, select(x, max(y, w), max(z, u)))) ||
      apply(max(min(v, select(x, y, z)), select(x, w, u)), select(x, max(w, min(v, y)), max(u, min(v, z)))) ||
      apply(max(w - select(x, y, z), select(x, u, v)), select(x, max(u, w - y), max(v, w - z))) ||
      apply(max(select(x, y, z) - w, select(x, u, v)), select(x, max(u, y - w), max(v, z - w))) ||

      apply(max(select(x, y, select(z, w, u)), select(z, v, t)), select(z, max(v, select(x, y, w)), max(t, select(x, y, u)))) ||
      apply(max(select(x, select(z, w, u), y), select(z, v, t)), select(z, max(v, select(x, w, y)), max(t, select(x, u, y)))) ||

      apply(max(x + may_be<0>(c2), select(c0 < x, y, c1)), select(c0 < x, max(y, x + c2), c1), c1 >= c0 + c2) ||
      apply(max(x + may_be<0>(c2), select(c0 < x, c1, y)), select(c0 < x, x, max(x, y - c2)) + c2, c1 <= c0 + c2) ||
      apply(max(x + may_be<0>(c2), select(x < c0, y, c1)), select(x < c0, max(x, y - c2), x) + c2, c1 <= c0 + c2) ||
      apply(max(x + may_be<0>(c2), select(x < c0, c1, y)), select(x < c0, c1, max(y, x + c2)), c1 >= c0 + c2) ||

      // Move constants out.
      apply(max(max(x, c0), c1), max(x, eval(max(c0, c1)))) ||
      apply(max(x, max(y, c0)), max(max(x, y), c0)) ||
      apply(max(x + c0, (y + c1)/c2), max(x, (y + eval(c1 - c0*c2))/c2) + c0, c2 != 0) ||
      apply(max(x + c0, y + c1),
        max(x, y + eval(c1 - c0)) + c0, c0 >= c1,  // Canonicalize to pulling bigger constants out.
        max(y, x + eval(c0 - c1)) + c1) ||
      apply(max(x + c0, c1 - may_be<0>(y)), 
        max(x, eval(c1 - c0)) + c0, is_zero(y),
        c1 - min(y, eval(c1 - c0) - x)) ||
      apply(max(c0 - x, c1 - may_be<0>(y)), c0 - min(x, y + eval(c0 - c1))) ||
      apply(max(max(x, c0) + may_be<0>(c1), max(y, c2)), max(max(y, x + c1), eval(max(c0 + c1, c2)))) ||
      apply(max(min(x, c0) + may_be<0>(c1), min(y, c2)),
        min(max(y, min(x, c0) + c1), c2), c2 > c0 + c1,
        min(max(x + c1, min(y, c2)), eval(c0 + c1)) /*c2 <= c0 + c1*/) ||
      apply(max(x + c0, select(y, may_be<0>(z) + c1, may_be<0>(w) + c2)), max(x, select(y, z + eval(c1 - c0), w + eval(c2 - c0))) + c0) ||
      apply(max(select(y, c1, c2), c0), select(y, eval(max(c0, c1)), eval(max(c0, c2)))) ||

      // https://github.com/halide/Halide/blob/7994e7030976f9fcd321a4d1d5f76f4582e01905/src/Simplify_Max.cpp#L271-L300
      apply(max(x*c0, c1),
        max(x, eval(c1/c0))*c0, c0 > 0 && c1%c0 == 0,
        min(x, eval(c1/c0))*c0, c0 < 0 && c1%c0 == 0) ||

      apply(max(x*c0, y*c1),
        max(x, y*eval(c1/c0))*c0, c0 > 0 && c1%c0 == 0,
        min(x, y*eval(c1/c0))*c0, c0 < 0 && c1%c0 == 0,
        max(y, x*eval(c0/c1))*c1, c1 > 0 && c0%c1 == 0,
        min(y, x*eval(c0/c1))*c1, c1 < 0 && c0%c1 == 0) ||
      apply(max(y*c0 + c1, x*c0),
        max(x, y + eval(c1/c0))*c0, c0 > 0 && c1%c0 == 0,
        min(x, y + eval(c1/c0))*c0, c0 < 0 && c1%c0 == 0) ||

      apply(max(y/c0 + may_be<0>(c1), x/c0),
        max(x, y + eval(c1*c0))/c0, c0 > 0,
        min(x, y + eval(c1*c0))/c0, c0 < 0) ||

      apply(max(x/c0, c1),
        max(x, eval(c1*c0))/c0, c0 > 0,
        min(x, eval(c1*c0))/c0, c0 < 0) ||

      apply(max(staircase(x, c0, c1, c2), staircase(x, c3, c4, c5) + may_be<0>(c6)),
        staircase(x, c0, c1, c2), 0 >= staircase_sum_max(c0, c1, -c2, c3, c4, c5) + c6,
        staircase(x, c3, c4, c5) + c6, 0 <= staircase_sum_min(c0, c1, -c2, c3, c4, c5) + c6) ||

      apply(max(x, abs(x)), abs(x)) ||

      false;
}

template <typename Fn>
bool apply_add_rules(Fn&& apply) {
  return
      apply(x + rewrite::positive_infinity(), rewrite::positive_infinity(), is_finite(x)) ||
      apply(x + rewrite::negative_infinity(), rewrite::negative_infinity(), is_finite(x)) ||
      apply(x + 0, x) ||
      apply(x + (x - y), x*2 - y) ||
      apply(x + (y - x), y) ||
      apply(x + x*y, x*(y + 1), !is_constant(x)) ||
      apply(x*y + x*z, x*(y + z)) ||
      apply((x + may_be<0>(y)) + (x + may_be<0>(z)), (y + z) + x*2) ||
      apply((x + y) + (x - z), (y - z) + x*2) ||
      apply((x + y) + (z - x), y + z) ||
      apply((x - y) + (z - x), z - y) ||
      apply((x - y) + (x - z), x*2 - (y + z)) ||
      apply((y - x) + (z - x), (y + z) + x*-2) ||

      apply(x + (c0 - y), (x - y) + c0) ||
      apply((x + may_be<0>(c0)) + (y + may_be<0>(c1)), (x + y) + eval(c0 + c1), c0 != 0 || c1 != 0) ||

      apply(staircase(x, c0, c1, c2) + c3, ((x + eval((c3/c2)*c1 + c0))/c1)*c2, c0 != 0 && c1 != 0 && c2 != 0 && c3%c2 == 0) ||
      apply(staircase(x, 0, c0, c0) + x%c0, x) ||

      apply(min(x, y + c1) + c2, min(y, x + c2), c1 == -c2) ||
      apply(max(x, y + c1) + c2, max(y, x + c2), c1 == -c2) ||
      apply(z + min(x, y - (z + may_be<0>(w))), min(x + z, y - w)) ||
      apply(z + max(x, y - (z + may_be<0>(w))), max(x + z, y - w)) ||
      apply(z + min(x, y - (z - w)), min(x + z, y + w)) ||
      apply(z + max(x, y - (z - w)), max(x + z, y + w)) ||

      apply(w + select(x, y, z - w), select(x, y + w, z)) ||
      apply(w + select(x, y - w, z), select(x, y, z + w)) ||

      false;
}

template <typename Fn>
bool apply_sub_rules(Fn&& apply) {
  return
      apply(x - rewrite::positive_infinity(), rewrite::negative_infinity(), is_finite(x)) ||
      apply(x - rewrite::negative_infinity(), rewrite::positive_infinity(), is_finite(x)) ||
      apply(x - 0, x) ||
      apply(x - y*c0, x + y*eval(-c0)) ||
      apply(x - y/c0, x + y/eval(-c0), c0 <= 0) ||
      apply(x - (c0 - y), (x + y) + eval(-c0)) ||
      apply(c0 - (x - y), (y - x) + c0) ||
      apply((c0 - x) - (y - z), ((z - x) - y) + c0) ||
      apply((c0 - x) - y, c0 - (x + y)) ||
      apply((x + may_be<0>(y)) - x, y) ||
      apply((x + may_be<0>(y)) - (x + z), y - z) ||
      apply((x - y) - (z - y), x - z) ||
      apply((x - y) - (x - may_be<0>(z)), z - y) ||
      apply((x + may_be<0>(c0)) - (y + may_be<0>(c1)), (x - y) + eval(c0 - c1), c0 != 0 || c1 != 0) ||

      // These rules taken from
      // https://github.com/halide/Halide/blob/e3d3c8cacfe6d664a8994166d0998f362bf55ce8/src/Simplify_Sub.cpp#L411-L421
      apply((x + y)/c0 - (x + may_be<0>(c1))/c0, ((y - c1) + ((x + eval(c1%c0))%c0))/c0, c0 > 0) ||
      apply((x + may_be<0>(c1))/c0 - (x + y)/c0, ((eval(c0 + c1 - 1) - y) - ((x + eval(c1%c0))%c0))/c0, c0 > 0) ||
      apply((x - y)/c0 - (x + may_be<0>(c1))/c0, (((x + eval(c1%c0))%c0) - y - c1)/c0, c0 > 0) ||
      apply((x + may_be<0>(c1))/c0 - (x - y)/c0, ((y + eval(c0 + c1 - 1)) - ((x + eval(c1%c0))%c0))/c0, c0 > 0) ||
      apply(x - (x/c0)*c0, x%c0, c0 > 0) ||
      apply((x/c0)*c0 - x, -(x%c0), c0 > 0) ||

      apply(c2 - min(x + c0, y + c1), max(eval(c2 - c0) - x, eval(c2 - c1) - y)) ||
      apply(c2 - max(x + c0, y + c1), min(eval(c2 - c0) - x, eval(c2 - c1) - y)) ||
      apply(c2 - min(x + c0, c1 - y), max(y + eval(c2 - c1), eval(c2 - c0) - x)) ||
      apply(c2 - max(x + c0, c1 - y), min(y + eval(c2 - c1), eval(c2 - c0) - x)) ||
      apply(z - min(x, z - y), max(y, z - x)) ||
      apply(z - max(x, z - y), min(y, z - x)) ||

      apply(min(x, y + z) - z, min(y, x - z)) ||
      apply(max(x, y + z) - z, max(y, x - z)) ||
      apply(min(x, y) - x, min(y - x, 0), !is_constant(x)) ||
      apply(max(x, y) - x, max(y - x, 0), !is_constant(x)) ||
      apply(x - min(x, y), max(x - y, 0), !is_constant(x)) ||
      apply(x - max(x, y), min(x - y, 0), !is_constant(x)) ||

      apply(c2 - select(x, may_be<0>(y) + c0, may_be<0>(z) + c1), select(x, eval(c2 - c0) - y, eval(c2 - c1) - z)) ||
      apply(c2 - select(x, may_be<0>(y) + c0, c1 - z), select(x, eval(c2 - c0) - y, z + eval(c2 - c1))) ||
      apply(c2 - select(x, c0 - y, may_be<0>(z) + c1), select(x, y + eval(c2 - c0), eval(c2 - c1) - z)) ||
      apply(c2 - select(x, c0 - y, c1 - z), select(x, y + eval(c2 - c0), z + eval(c2 - c1))) ||

      apply(max(x, y)/c0 - min(x, y)/c0, abs(x/c0 - y/c0), c0 > 0) ||
      apply(min(x, y)/c0 - max(x, y)/c0, -abs(x/c0 - y/c0), c0 > 0) ||
      apply(max(x, y) - min(x, y), abs(x - y)) ||
      apply(min(x, y) - max(x, y), -abs(x - y)) ||
      apply(select(x, y, z) - select(x, w, u), select(x, y - w, z - u)) ||

      apply(select(x, y, may_be<0>(z) + w) - w, select(x, y - w, z)) ||
      apply(select(x, may_be<0>(y) + w, z) - w, select(x, y, z - w)) ||
      apply(select(x, y, w - z) - w, select(x, y - w, -z)) ||
      apply(select(x, w - y, z) - w, select(x, -y, z - w)) ||
      apply(w - select(x, y, may_be<0>(z) + w), select(x, w - y, -z)) ||
      apply(w - select(x, may_be<0>(y) + w, z), select(x, -y, w - z)) ||
      apply(w - select(x, y, w - z), select(x, w - y, z)) ||
      apply(w - select(x, w - y, z), select(x, y, w - z)) ||

      false;
}

template <typename Fn>
bool apply_mul_rules(Fn&& apply) {
  return
      apply(rewrite::positive_infinity()*c0,
        rewrite::positive_infinity(), c0 > 0,
        rewrite::negative_infinity(), c0 < 0) ||
      apply(rewrite::negative_infinity()*c0,
        rewrite::negative_infinity(), c0 > 0,
        rewrite::positive_infinity(), c0 < 0) ||
      apply(x*0, 0) ||
      apply(x*1, x) ||
      apply((x*c0)*c1, x*eval(c0*c1)) ||
      apply((x + c0)*c1, x*c1 + eval(c0*c1)) ||
      apply((c0 - x)*c1, x*eval(-c1) + eval(c0*c1)) ||
      apply(y*(x*c0), (x*y)*c0) ||

      apply(select(x, c0, c1)*c2, select(x, eval(c0*c2), eval(c1*c2))) ||
      apply(select(x, y, c1)*c2, select(x, y*c2, eval(c1*c2))) ||
      apply(select(x, c0, y)*c2, select(x, eval(c0*c2), y*c2)) ||
      false;
}

template <typename Fn>
bool apply_div_rules(Fn&& apply) {
  return
      apply(x/rewrite::positive_infinity(), 0, is_finite(x)) ||
      apply(x/rewrite::negative_infinity(), 0, is_finite(x)) ||
      apply(rewrite::positive_infinity()/c0,
        rewrite::positive_infinity(), c0 > 0,
        rewrite::negative_infinity(), c0 < 0) ||
      apply(rewrite::negative_infinity()/c0,
        rewrite::negative_infinity(), c0 > 0,
        rewrite::positive_infinity(), c0 < 0) ||
      apply(x/0, 0) ||
      apply(0/x, 0) ||
      apply(x/1, x) ||
      apply(x/-1, -x) ||
      apply(x/x, x != 0) ||

      apply((y + x/c0)/c1, (x + y*c0)/eval(c0*c1)) ||
      apply((y - x/c0)/c1, (y*c0 - x + eval(c0 - 1))/eval(c0*c1)) ||
      apply((x*c0)/(y*c1), (x*eval(c0/c1))/y, c1 > 0 && c0%c1 == 0) ||
      apply((x*c0)/(y*c1), x/(y*eval(c1/c0)), c0 > 0 && c1%c0 == 0) ||
      apply((x*c0)/c1, x*eval(c0/c1), c0%c1 == 0) ||
      apply((x*c0)/c1, x/eval(c1/c0), c0 > 0 && c1%c0 == 0) ||
      apply(c0/(y*c1), eval(c0/c1)/y, c1 > 0 && c0%c1 == 0) ||
      apply((x/c0)/c1, x/eval(c0*c1), c0 > 0) ||
      apply((x*y)/x, y*(x != 0)) ||

      apply((x + may_be<1>(y)*c0)/c1, y*eval(c0/c1) + x/c1, c0%c1 == 0) ||
      apply((may_be<1>(y)*c0 - x)/c1, y*eval(c0/c1) + (-x)/c1, c0%c1 == 0 && c0 != 0) ||

      apply(min(y, may_be<0>(z) + x*c0)/c1, min(x*eval(c0/c1) + z/c1, y/c1), c1 > 0 && c0%c1 == 0) ||
      apply(max(y, may_be<0>(z) + x*c0)/c1, max(x*eval(c0/c1) + z/c1, y/c1), c1 > 0 && c0%c1 == 0) ||
      apply(min(y, x*c0 - z)/c1, min(x*eval(c0/c1) + (-z)/c1, y/c1), c1 > 0 && c0%c1 == 0) ||
      apply(max(y, x*c0 - z)/c1, max(x*eval(c0/c1) + (-z)/c1, y/c1), c1 > 0 && c0%c1 == 0) ||

      apply(select(x, c0, c1)/c2, select(x, eval(c0/c2), eval(c1/c2))) ||
      apply(select(x, y, c1)/c2, select(x, y/c2, eval(c1/c2))) ||
      apply(select(x, c0, y)/c2, select(x, eval(c0/c2), y/c2)) ||

      // Is there some generalization of this rule? 
      apply(((x%2) + c0)/2, x%2 + eval(c0/2), c0%2 == 1) ||

      false;
}

template <typename Fn>
bool apply_mod_rules(Fn&& apply) {
  return
      apply(x%1, 0) ||
      apply(x%0, 0) ||
      apply(x%x, 0) ||

      apply((x + c0)%c1, (x + eval(c0%c1))%c1, c0%c1 != c0) ||

      apply(select(x, c0, c1)%c2, select(x, eval(c0%c2), eval(c1%c2))) ||
      apply(select(x, y, c1)%c2, select(x, y%c2, eval(c1%c2))) ||
      apply(select(x, c0, y)%c2, select(x, eval(c0%c2), y%c2)) ||
      false;
}

template <typename Fn>
bool apply_less_rules(Fn&& apply) {
  return
      apply(rewrite::positive_infinity() < x, false, is_finite(x)) ||
      apply(rewrite::negative_infinity() < x, true, is_finite(x)) ||
      apply(x < rewrite::positive_infinity(), true, is_finite(x)) ||
      apply(x < rewrite::negative_infinity(), false, is_finite(x)) ||
      apply(x < x, false) ||
      apply(x < y + 1, x <= y) ||
      apply(x + -1 < y, x <= y) ||

      // These rules taken from:
      // https://github.com/halide/Halide/blob/e9f8b041f63a1a337ce3be0b07de5a1cfa6f2f65/src/Simplify_LT.cpp#L87-L169
      // with adjustments for the simplifier implementation here.

      // Normalize subtractions to additions to cut down on cases to consider
      apply(may_be<0>(u) + (may_be<0>(z) + (x - y)) < w, x + (z + u) < w + y) ||
      apply(w < may_be<0>(u) + (may_be<0>(z) + (x - y)), w + y < x + (z + u)) ||

      // Cancellations in linear expressions
      apply(may_be<0>(w) + (x + may_be<0>(y)) < may_be<0>(u) + (x + may_be<0>(z)), y + w < z + u) ||

      apply(x + c0 < may_be<0>(y) + c1, x < y + eval(c1 - c0)) ||
      apply(x + c0 < y, x < y + eval(-c0)) ||
      apply(c0 < x + c1, eval(c0 - c1) < x) ||

      // These rules taken from:
      // https://github.com/halide/Halide/blob/e3d3c8cacfe6d664a8994166d0998f362bf55ce8/src/Simplify_LT.cpp#L340-L397
      apply(may_be<0>(w) + staircase(x, c0, c1, c1) < x + may_be<0>(z), w + c0 < z + (x + c0)%c1, c1 > 1) ||
      apply(x + may_be<0>(z) < may_be<0>(w) + staircase(x, c0, c1, c1), z + (x + c0)%c1 < w + c0, c1 > 1) ||
      apply(staircase(x, c0, c1, c1) < x,
        c0 < (x + c0)%c1, c1 > 1 && c0 != 0,
        x%c1 != 0, c1 > 0 /*&& c0 == 0*/) ||
      apply(x < staircase(x, c0, c1, c1),
        (x + c0)%c1 < c0, c1 > 1 && c0 != 0,
        false, c1 > 0 /*&& c0 == 0*/) ||

      apply(x%c0 < c1,
        true, c0 > 0 && c0 <= c1,
        false, c0 > 0 && c1 <= 0) ||
      apply(c0 < x%c1,
        true, c1 > 0 && c0 < 0,
        false, c1 > 0 && c0 >= c1 - 1,
        boolean(x%c1), c1 > 0 && c0 == 0) ||

      apply(staircase(x, c0, c1, c2) < staircase(x, c3, c4, c5) + may_be<0>(c6),
        true, 0 < staircase_sum_min(c0, c1, -c2, c3, c4, c5) + c6,
        false, 0 >= staircase_sum_max(c0, c1, -c2, c3, c4, c5) + c6) ||

      apply(x*c0 < y*c0,
        x < y, c0 > 0,
        y < x, c0 < 0) ||
      apply(x/c0 < c1, 
        x < c0*c1, c0 > 0,
        x >= c0*(c1 - 1), c0 < 0) ||
      apply(c0 < x/c1, 
        (c0 + 1)*c1 <= x, c1 > 0,
        c0*c1 > x, c1 < 0) ||

      apply((max(x, c0)/may_be<1>(c3)) < max(x, c1)/may_be<1>(c3) + c2,
        true, c2 > 0 && c3 > 0 && c1/c3 + c2 > c0/c3,
        false, c2 < 0 && c3 > 0 && c1/c3 + c2 <= c0/c3) ||
      apply((min(x, c0)/may_be<1>(c3)) < min(x, c1)/may_be<1>(c3) + c2,
        true, c2 > 0 && c3 > 0 && c1/c3 + c2 > c0/c3,
        false, c2 < 0 && c3 > 0 && c1/c3 + c2 <= c0/c3) ||

      // The following rules are taken from
      // https://github.com/halide/Halide/blob/7636c44acc2954a7c20275618093973da6767359/src/Simplify_LT.cpp#L186-L263
      // with adjustments for the simplifier implementation here.

      // We want to break max(x, y) < z into x < z && y <
      // z in cases where one of those two terms is going
      // to eval.
      apply(min(y, x + may_be<0>(c0)) < x + may_be<0>(c1), y < x + c1 || eval(c0 < c1)) ||
      apply(max(y, x + may_be<0>(c0)) < x + may_be<0>(c1), y < x + c1 && eval(c0 < c1)) ||
      apply(x < min(y, x + may_be<0>(c0)) + may_be<0>(c1), x < y + c1 && eval(0 < c0 + c1)) ||
      apply(x < max(y, x + may_be<0>(c0)) + may_be<0>(c1), x < y + c1 || eval(0 < c0 + c1)) ||

      // Special case where x is constant
      apply(min(y, c0) < c1, y < c1 || eval(c0 < c1)) ||
      apply(max(y, c0) < c1, y < c1 && eval(c0 < c1)) ||
      apply(c1 < min(y, c0), c1 < y && eval(c1 < c0)) ||
      apply(c1 < max(y, c0), c1 < y || eval(c1 < c0)) ||

      // Cases where we can remove a min on one side because
      // one term dominates another. These rules were
      // synthesized then extended by hand.
      apply(min(z, y) < min(x, y), z < min(x, y)) ||
      apply(min(z, y) < min(x, y + c0), min(z, y) < x, c0 > 0) ||
      apply(min(z, y + c0) < min(x, y), min(z, y + c0) < x, c0 < 0) ||

      // Equivalents with max
      apply(max(z, y) < max(x, y), max(z, y) < x) ||
      apply(max(z, y) < max(x, y + c0), max(z, y) < x, c0 < 0) ||
      apply(max(z, y + c0) < max(x, y), max(z, y + c0) < x, c0 > 0) ||

      apply(min(x, min(y, z)) < y, min(x, z) < y) ||
      apply(min(x, y) < max(x, y), x != y) ||
      apply(max(x, y) < min(x, y), false) ||

      apply(min(x, y + may_be<0>(c0)) < max(z, y + may_be<0>(c1)), true, c0 < c1) ||

      // Subtract terms from both sides within a min/max.
      // These are only enabled for non-constants because they loop with rules that pull constants out of min/max.
      apply(may_be<0>(z) + min(y, x + may_be<0>(u)) < x + may_be<0>(w), min(y - x, u) + z < w, !is_constant(x)) ||
      apply(may_be<0>(z) + max(y, x + may_be<0>(u)) < x + may_be<0>(w), max(y - x, u) + z < w, !is_constant(x)) ||
      apply(may_be<0>(z) + min(y, x - u) < x + may_be<0>(w), z < w + max(x - y, u), !is_constant(x)) ||
      apply(may_be<0>(z) + max(y, x - u) < x + may_be<0>(w), z < w + min(x - y, u), !is_constant(x)) ||

      apply(x + may_be<0>(z) < may_be<0>(w) + min(y, x + may_be<0>(u)), z < min(y - x, u) + w, !is_constant(x)) ||
      apply(x + may_be<0>(z) < may_be<0>(w) + max(y, x + may_be<0>(u)), z < max(y - x, u) + w, !is_constant(x)) ||
      apply(x + may_be<0>(z) < may_be<0>(w) + min(y, x - u), z + max(x - y, u) < w, !is_constant(x)) ||
      apply(x + may_be<0>(z) < may_be<0>(w) + max(y, x - u), z + min(x - y, u) < w, !is_constant(x)) ||

      // Selects
      apply(select(x, c0, y) < c1, select(x, eval(c0 < c1), y < c1)) ||
      apply(select(x, y, c0) < c1, select(x, y < c1, eval(c0 < c1))) ||
      apply(c1 < select(x, c0, y), select(x, eval(c1 < c0), c1 < y)) ||
      apply(c1 < select(x, y, c0), select(x, c1 < y, eval(c1 < c0))) ||

      apply(select(x, y, z) < y, select(x, false, z < y)) ||
      apply(select(x, y, z) < z, select(x, y < z, false)) ||
      apply(y < select(x, y, w), select(x, false, y < w)) ||
      apply(w < select(x, y, w), select(x, w < y, false)) ||
      apply(may_be<0>(t) + select(x, y, z) < may_be<0>(v) + select(x, w, u), select(x, y + t < v + w, z + t < v + u)) ||

      // Nested logicals
      apply(x < y, y && !x, is_boolean(x) && is_boolean(y)) ||
      apply(x < 1, !x, is_boolean(x)) ||
      apply(0 < x, boolean(x), is_boolean(x)) ||

      false;
}

template <typename Fn>
bool apply_equal_rules(Fn&& apply) {
  return
      apply(x == x, true) ||
      apply(x*y == x*z, y == z || x == 0) ||
      apply(x == x*y, y == 1 || x == 0) ||

      // Normalize subtractions to additions to cut down on cases to consider
      apply(w == may_be<0>(u) + (may_be<0>(z) + (x - y)), w + y == x + (z + u)) ||

      // Cancellations in linear expressions
      apply(may_be<0>(w) + (x + may_be<0>(y)) == may_be<0>(u) + (x + may_be<0>(z)), y + w == z + u) ||

      apply(x*c0 == y*c1,
        x == y*eval(c1/c0), c0 != 0 && c1%c0 == 0,
        y == x*eval(c0/c1), c1 != 0 && c0%c1 == 0) ||
      apply(x*c0 == c1, x == eval(c1/c0), c0 != 0 && c1%c0 == 0) ||
      apply(x + c0 == may_be<0>(y) + c1, x == y + eval(c1 - c0)) ||
      apply(may_be<0>(x) + c0 == c1 - y, x + y == eval(c1 - c0)) ||
      apply(x == 0, !x, is_boolean(x)) ||
      apply(x == 1, boolean(x), is_boolean(x)) ||

      apply(x + may_be<0>(y) == may_be<0>(z) + (x/c0)*c0, z == y + x%c0, c0 > 0) ||

      apply(x%c0 == c1, false, c0 > 0 && (c1 >= c0 || c1 < 0)) ||

      apply(may_be<0>(v) + select(x, y, z) == select(x, w, u), select(x, w == v + y, u == v + z)) ||
      apply(select(x, c0, y) == c1, select(x, eval(c0 == c1), y == c1)) ||
      apply(select(x, y, c0) == c1, select(x, y == c1, eval(c0 == c1))) ||
      apply(y == select(x == y, x, z), x == y || y == z) ||
      apply(y == select(x == y, z, x), x == y && y == z) ||
      apply(y == select(x, y, z), select(x, true, y == z)) ||
      apply(z == select(x, y, z), select(x, z == y, true)) ||

      apply(y == max(x, y), x <= y) ||
      apply(y == min(x, y), y <= x) ||
    
      apply(max(x, c0)/c1 == c2,
        x/c1 == c2, c1 > 0 && c0/c1 < c2,
        x < (c2 + 1)*c1, c1 > 0 && c0/c1 == c2,
        false) ||
      apply(min(x, c0)/c1 == c2,
        x/c1 == c2, c1 > 0 && c0/c1 > c2,
        x > (c2 + 1)*c1, c1 > 0 && c0/c1 == c2,
        false) ||

      apply(max(x, c0) == max(x, c1), x >= eval(max(c0, c1)), c0 != c1) ||
      apply(min(x, c0) == min(x, c1), x <= eval(min(c0, c1)), c0 != c1) ||

      apply(max(x, c0) == c1,
        false, c0 > c1,
        x == c1, c0 < c1) ||
      apply(min(x, c0) == c1,
        false, c0 < c1,
        x == c1, c0 > c1) ||

      false;
}

template <typename Fn>
bool apply_logical_and_rules(Fn&& apply) {
  return
      apply(x && c0,
        boolean(x), c0 != 0,
        false) ||
      apply(x && x, boolean(x)) ||

      // Canonicalize trees and find redundant terms.
      apply((x && y) && (z && w), x && (y && (z && w))) ||
      apply(x && (may_be<true>(y) && (may_be<true>(z) && (may_be<true>(w) && x))), x && (y && (z && w))) ||

      apply(x && (x || y), boolean(x)) ||
      apply(x && (y || (x && z)), x && (y || z)) ||
      apply(x && (y && (x || z)), x && y) ||
      apply((x || y) && (x || z), x || (y && z)) ||

      apply(x && !x, false) ||
      apply(x == y && x != y, false) ||
      apply(x == y && x < y, false) ||
      apply(x == y && x <= y, x == y) ||
      apply(x == y && (z && x != y), false) ||
      apply(x != y && x < y, x < y) ||
      apply(x != y && x <= y, x < y) ||
      apply(x != y && (z && x == y), false) ||
      apply(x == c1 && x != c0, x == c1, c0 != c1) ||
      apply(x == c0 && x == c1, false, c0 != c1) ||

      // These rules taken from:
      // https://github.com/halide/Halide/blob/e9f8b041f63a1a337ce3be0b07de5a1cfa6f2f65/src/Simplify_And.cpp#L67-L76
      apply(c0 <= x && x <= c1, false, c1 < c0) ||
      apply(c0 <= x && c1 <= x, eval(max(c0, c1)) <= x) ||
      apply(c0 < x && x < c1, false, c1 <= c0 + 1) ||
      apply(c0 < x && x <= c1, false, c1 <= c0) ||
      apply(c0 < x && c1 < x, eval(max(c0, c1)) < x) ||
      apply(x < c1 && c0 <= x, false, c1 <= c0) ||
      apply(x < c0 && x < c1, x < eval(min(c0, c1))) ||
      apply(x <= c0 && x <= c1, x <= eval(min(c0, c1))) ||

      // The way we implement <= and < means that constants will be on the LHS for <=, and on the RHS for <
      apply(x <= y && y <= x, x == y) ||
      apply(x + may_be<0>(c0) <= y && y + may_be<0>(c1) <= x, false, -c1 < c0) ||
      apply(x < y + may_be<0>(c0) && y + may_be<0>(c1) <= x, false, c0 < c1 + 1) ||
      apply(x < y + may_be<0>(c0) && y < x + may_be<0>(c1), false, c1 + c0 < 2) ||

      // Above, we have rules for combinations of < and <=, or == and !=. Here, we have a mix of both.
      apply(x != c0 && x <= c1, x <= c1, c0 > c1) ||
      apply(x != c0 && x < c1, x < c1, c0 > c1) ||
      apply(x == c0 && x <= c1,
        x == c0, c0 <= c1,
        false /*c0 > c1*/) ||
      apply(x == c0 && x < c1,
        x == c0, c0 < c1,
        false /*c0 >= c1*/) ||

      apply(x != c0 && c1 <= x, c1 <= x, c0 < c1) ||
      apply(x != c0 && c1 < x, c1 < x, c0 < c1) ||
      apply(x == c0 && c1 <= x,
        x == c0, c0 >= c1,
        false /*c0 < c1*/) ||
      apply(x == c0 && c1 < x,
        x == c0, c0 > c1,
        false /*c0 <= c1*/) ||

      false;
}


template <typename Fn>
bool apply_logical_or_rules(Fn&& apply) {
  return
      apply(x || c0,
        boolean(x), c0 == 0,
        true) ||
      apply(x || x, boolean(x)) ||

      // Canonicalize trees and find redundant terms.
      apply((x || y) || (z || w), x || (y || (z || w))) ||
      apply(x || (may_be<false>(y) || (may_be<false>(z) || (may_be<false>(w) || x))), x || (y || (z || w))) ||

      apply(x || (x && y), boolean(x)) ||
      apply(x || (y && (x || z)), x || (y && z)) ||
      apply(x || (y || (x && z)), x || y) ||
      apply((x && y) || (x && z), x && (y || z)) ||

      // These rules taken from:
      // https://github.com/halide/Halide/blob/e9f8b041f63a1a337ce3be0b07de5a1cfa6f2f65/src/Simplify_Or.cpp#L59-L68
      apply(x || !x, true) ||
      apply(x == y || x != y, true) ||
      apply(x == y || x < y, x <= y) ||
      apply(x == y || x <= y, x <= y) ||
      apply(x == y || (z || x != y), true) ||
      apply(x != y || (z || x == y), true) ||
      apply(x == c1 || x != c0, x != c0, c0 != c1) ||

      apply(x <= c0 || c1 <= x, true, c1 <= c0 + 1) ||
      apply(x <= c0 || x <= c1, x <= eval(max(c0, c1))) ||
      apply(x < c0 || x < c1, x < eval(max(c0, c1))) ||
      apply(x < c0 || c1 <= x, true, c1 <= c0) ||
      apply(x < c0 || c1 < x, true, c1 < c0) ||
      apply(c0 < x || c1 < x, eval(min(c0, c1)) < x) ||
      apply(c0 <= x || c1 <= x, eval(min(c0, c1)) <= x) ||
      apply(c1 < x || x <= c0, true, c1 <= c0) ||

      // The way we implement <= and < means that constants will be on the LHS for <=, and on the RHS for <
      apply(x + may_be<0>(c0) <= y || y + may_be<0>(c1) <= x, true, c0 + c1 < 1) ||
      apply(x < y + may_be<0>(c0) || y + may_be<0>(c1) <= x, true, c1 < c0) ||
      apply(x < y + may_be<0>(c0) || y < x + may_be<0>(c1), true, 1 < c1 + c0) ||
      apply(x < y || y < x, x != y) ||

      // Above, we have rules for combinations of < and <=, or == and !=. Here, we have a mix of both.
      apply(x != c0 || x <= c1,
        true, c0 <= c1,
        x != c0 /*c0 > c1*/) ||
      apply(x != c0 || x < c1,
        true, c0 < c1,
        x != c0 /*c0 >= c1*/) ||
      apply(x == c0 || x <= c1, x <= c1, c0 <= c1) ||
      apply(x == c0 || x < c1, x < c1, c0 < c1) ||

      apply(x != c0 || c1 <= x,
        true, c0 >= c1,
        x != c0 /*c0 < c1*/) ||
      apply(x != c0 || c1 < x,
        true, c0 > c1,
        x != c0 /*c0 <= c1*/) ||
      apply(x == c0 || c1 <= x, c1 <= x, c0 >= c1) ||
      apply(x == c0 || c1 < x, c1 < x, c0 > c1) ||

      // TODO: These rules are just a few of many similar possible rules. We should find a way to get at these
      // some other way.
      apply(y || x < y, y || x < 0, is_boolean(y)) ||
      apply(y || y < x, y || 0 < x, is_boolean(y)) ||

      false;
}

template <typename Fn>
bool apply_logical_not_rules(Fn&& apply) {
  return
      apply(!!x, boolean(x)) ||
      apply(!(x == y), x != y) ||
      apply(!(x != y), x == y) ||
      apply(!(x < y), y <= x) ||
      apply(!(x <= y), y < x) ||

      apply(!(x && !y), y || !x) ||
      apply(!(x || !y), y && !x) ||
      apply(!select(x, y, c0), select(x, !y, eval(!c0))) ||
      apply(!select(x, c0, z), select(x, eval(!c0), !z)) ||
      false;
}

template <typename Fn>
bool apply_select_rules(Fn&& apply) {
  return
      apply(select(x, y, y), y) ||
      apply(select(!x, y, z), select(x, z, y)) ||

      // Pull common expressions out
      apply(select(x, y + may_be<0>(z), y + may_be<0>(w)), y + select(x, z, w)) ||
      apply(select(x, z - y, w - y), select(x, z, w) - y) ||
      apply(select(x, w - y, w - z), w - select(x, y, z)) ||
      apply(select(x, max(y, w), max(z, w)), max(w, select(x, y, z))) ||
      apply(select(x, min(y, w), min(z, w)), min(w, select(x, y, z))) ||
      apply(select(x, y + c0, c1), select(x, y, eval(c1 - c0)) + c0) ||
      apply(select(x, c0, y + c1), select(x, eval(c0 - c1), y) + c1) ||

      apply(select(x, select(y, z, w), select(y, u, w + may_be<0>(c0)) + may_be<0>(c1)), select(y, select(x, z, u + c1), w), c0 + c1 == 0) ||
      apply(select(x, select(y, z, w), select(y, z + may_be<0>(c0), u) + may_be<0>(c1)), select(y, z, select(x, w, u + c1)), c0 + c1 == 0) ||
      apply(select(x, select(y, z, w + may_be<0>(c0)) + may_be<0>(c1), select(y, u, w)), select(y, select(x, z + c1, u), w), c0 + c1 == 0) ||
      apply(select(x, select(y, z + may_be<0>(c0), w) + may_be<0>(c1), select(y, z, u)), select(y, z, select(x, w + c1, u)), c0 + c1 == 0) ||
      apply(select(x, select(y, z, w), w), select(x && y, z, w)) ||
      apply(select(x, select(y, z, w), z), select(x && !y, w, z)) ||
      apply(select(x, z, select(y, z, w)), select(x || y, z, w)) ||
      apply(select(x, w, select(y, z, w)), select(y && !x, z, w)) ||
      apply(select(x, false, true), !x) ||
      apply(select(x, true, false), boolean(x)) ||
      apply(select(x, y, true), y || !x, is_boolean(y)) ||
      apply(select(x, y, false), x && y, is_boolean(y)) ||
      apply(select(x, true, y), x || y, is_boolean(y)) ||
      apply(select(x, false, y), y && !x, is_boolean(y)) ||

      // Simplifications of min/max
      apply(select(x < y, min(x, y), z), select(x < y, x, z)) ||
      apply(select(x < y, max(x, y), z), select(x < y, y, z)) ||
      apply(select(x < y, z, min(x, y)), select(x < y, z, y)) ||
      apply(select(x < y, z, max(x, y)), select(x < y, z, x)) ||
      apply(select(x <= y, min(x, y), z), select(x <= y, x, z)) ||
      apply(select(x <= y, max(x, y), z), select(x <= y, y, z)) ||
      apply(select(x <= y, z, min(x, y)), select(x <= y, z, y)) ||
      apply(select(x <= y, z, max(x, y)), select(x <= y, z, x)) ||

      // Equivalents with min/max
      apply(select(x <= y, x, y), min(x, y)) ||
      apply(select(x <= y, y, x), max(x, y)) ||
      apply(select(x < y, x, y), min(x, y)) ||
      apply(select(x < y, y, x), max(x, y)) ||
      apply(select(x < c0, c1, x), max(x, c1), c0 == c1 + 1) ||
      apply(select(x < c0, x, c1), min(x, c1), c0 == c1 + 1) ||
      apply(select(c0 < x, x, c1), max(x, c1), c0 + 1 == c1) ||
      apply(select(c0 < x, c1, x), min(x, c1), c0 + 1 == c1) ||

      false;
}

template <typename Fn>
bool apply_call_rules(Fn&& apply) {
  return
      apply(and_then(x, c0), boolean(x), c0 != 0) ||
      apply(and_then(c0, x), boolean(x), c0 != 0) ||
      apply(and_then(false, x), false) ||
      apply(or_else(x, false), boolean(x)) ||
      apply(or_else(c0, x), true, c0 != 0) ||
      apply(or_else(false, x), boolean(x)) ||

      apply(abs(x*c0), abs(x)*c0, c0 > 0) ||
      apply(abs(x*c0), abs(x)*eval(-c0), c0 < 0) ||
      apply(abs(c0 - x), abs(x + eval(-c0))) ||

      apply(abs(select(x, y, c1)), select(x, abs(y), abs(c1))) ||
      apply(abs(select(x, c0, y)), select(x, abs(c0), abs(y))) ||

      false;
}

}  // namespace slinky

#endif  // SLINKY_BUILDER_SIMPLIFY_RULES_H
