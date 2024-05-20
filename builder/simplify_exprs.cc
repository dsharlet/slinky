#include "builder/simplify.h"

#include <algorithm>
#include <cassert>

#include "builder/rewrite.h"
#include "runtime/evaluate.h"

namespace slinky {

using namespace rewrite;

namespace {

pattern_wildcard<0> x;
pattern_wildcard<1> y;
pattern_wildcard<2> z;
pattern_wildcard<3> w;
pattern_wildcard<4> u;

pattern_constant<0> c0;
pattern_constant<1> c1;
pattern_constant<2> c2;
pattern_constant<3> c3;
pattern_constant<4> c4;

}  // namespace

expr simplify(const class min* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }

  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return std::min(*ca, *cb);
  }

  if (is_indeterminate(a)) return a;
  if (is_indeterminate(b)) return b;

  auto r = make_rewriter(min(pattern_expr{a}, pattern_expr{b}));
  // clang-format off
  if (// One side is the min.
      r.rewrite(min(x, std::numeric_limits<index_t>::max()), x) ||
      r.rewrite(min(x, rewrite::positive_infinity()), x) ||
      r.rewrite(min(x, std::numeric_limits<index_t>::min()), std::numeric_limits<index_t>::min()) ||
      r.rewrite(min(x, rewrite::negative_infinity()), rewrite::negative_infinity()) ||
      r.rewrite(min(x, x + c0), x, eval(c0 > 0)) ||
      r.rewrite(min(x, x + c0), x + c0, eval(c0 < 0)) ||
      r.rewrite(min(x, x), x) ||

      // Canonicalize trees and find duplicate terms.
      r.rewrite(min(min(x, y), min(x, z)), min(x, min(y, z))) ||
      r.rewrite(min(min(x, y), min(z, w)), min(x, min(y, min(z, w)))) ||
      r.rewrite(min(x, min(x, y)), min(x, y)) ||
      r.rewrite(min(x, min(y, min(x, z))), min(x, min(y, z))) ||
      r.rewrite(min(x, min(y, min(z, min(x, w)))), min(x, min(y, min(z, w)))) ||

      // Similar rules but with mixes of min and max.
      r.rewrite(min(max(x, y), max(x, z)), max(x, min(y, z))) ||
      r.rewrite(min(min(x, y), max(x, z)), min(x, y)) ||
      r.rewrite(min(x, min(y, max(x, z))), min(x, y)) ||
      r.rewrite(min(x, max(y, min(x, z))), min(x, max(y, z))) ||
      r.rewrite(min(x, max(x, y)), x) ||

      // Pull common terms out.
      r.rewrite(min(y + z, min(x, y)), min(x, y + min(z, 0))) ||
      r.rewrite(min(y - z, min(x, y)), min(x, y - max(z, 0))) ||
      r.rewrite(min(y, min(x, y + z)), min(x, y + min(z, 0))) ||
      r.rewrite(min(y, min(x, y - z)), min(x, y - max(z, 0))) ||
      r.rewrite(min(x, min(y, x + z)), min(y, min(x, x + z))) ||
      r.rewrite(min(x, min(y, x - z)), min(y, min(x, x - z))) ||
      r.rewrite(min((y + w), min(x, (y + z))), min(x, min(y + z, y + w))) ||
      r.rewrite(min(x + z, y + z), z + min(x, y)) ||
      r.rewrite(min(x - z, y - z), min(x, y) - z) ||
      r.rewrite(min(z - x, z - y), z - max(x, y)) ||
      r.rewrite(min(x + z, z - y), z + min(x, -y)) ||
      r.rewrite(min(x, x + z), x + min(z, 0)) ||
      r.rewrite(min(x, x - z), x - max(z, 0)) ||
      r.rewrite(min(x, -x), -abs(x)) ||

      // Selects
      r.rewrite(min(x, select(y, min(x, z), w)), min(x, select(y, z, w))) ||
      r.rewrite(min(x, select(y, z, min(x, w))), min(x, select(y, z, w))) ||
      r.rewrite(min(x, select(y, max(x, z), w)), select(y, x, min(x, w))) ||
      r.rewrite(min(x, select(y, z, max(x, w))), select(y, min(x, z), x)) ||
      r.rewrite(min(y, select(x, y, w)), select(x, y, min(y, w))) ||
      r.rewrite(min(z, select(x, w, z)), select(x, min(z, w), z)) ||
      r.rewrite(min(select(x, y, z), select(x, w, u)), select(x, min(y, w), min(z, u))) ||

      // Move constants out.
      r.rewrite(min(min(x, c0), c1), min(x, eval(min(c0, c1)))) ||
      r.rewrite(min(x + c0, (y + c1) / c2), min(x, (y + eval(c1 - c0 * c2)) / c2) + c0) ||
      r.rewrite(min(x + c0, y + c1), min(x, y + eval(c1 - c0)) + c0) ||
      r.rewrite(min(x + c0, c1 - y), c1 - max(y, eval(c1 - c0) - x)) ||
      r.rewrite(min(x + c0, c1), min(x, eval(c1 - c0)) + c0) ||
      r.rewrite(min(c0 - x, c1 - y), c0 - max(x, y + eval(c0 - c1))) ||
      r.rewrite(min(c0 - x, c1), c0 - max(x, eval(c0 - c1))) ||
    
      // https://github.com/halide/Halide/blob/7994e7030976f9fcd321a4d1d5f76f4582e01905/src/Simplify_Min.cpp#L276-L311
      r.rewrite(min(x * c0, c1), min(x, eval(c1 / c0)) * c0, eval(c0 > 0 && c1 % c0 == 0)) ||
      r.rewrite(min(x * c0, c1), max(x, eval(c1 / c0)) * c0, eval(c0 < 0 && c1 % c0 == 0)) ||

      r.rewrite(min(x * c0, y * c1), min(x, y * eval(c1 / c0)) * c0, eval(c0 > 0 && c1 % c0 == 0)) ||
      r.rewrite(min(x * c0, y * c1), max(x, y * eval(c1 / c0)) * c0, eval(c0 < 0 && c1 % c0 == 0)) ||
      r.rewrite(min(x * c0, y * c1), min(y, x * eval(c0 / c1)) * c1, eval(c1 > 0 && c0 % c1 == 0)) ||
      r.rewrite(min(x * c0, y * c1), max(y, x * eval(c0 / c1)) * c1, eval(c1 < 0 && c0 % c1 == 0)) ||
      r.rewrite(min(y * c0 + c1, x * c0), min(x, y + eval(c1 / c0)) * c0, eval(c0 > 0 && c1 % c0 == 0)) ||
      r.rewrite(min(y * c0 + c1, x * c0), max(x, y + eval(c1 / c0)) * c0, eval(c0 < 0 && c1 % c0 == 0)) ||

      r.rewrite(min(x / c0, y / c0), min(x, y) / c0, eval(c0 > 0)) ||
      r.rewrite(min(x / c0, y / c0), max(x, y) / c0, eval(c0 < 0)) ||

      r.rewrite(min(x / c0, c1), min(x, eval(c1 * c0)) / c0, eval(c0 > 0)) ||
      r.rewrite(min(x / c0, c1), max(x, eval(c1 * c0)) / c0, eval(c0 < 0)) ||

      r.rewrite(min(y / c0 + c1, x / c0), min(x, y + eval(c1 * c0)) / c0, eval(c0 > 0)) ||
      r.rewrite(min(y / c0 + c1, x / c0), max(x, y + eval(c1 * c0)) / c0, eval(c0 < 0)) ||

      r.rewrite(min(((x + c2) / c3) * c4, (x + c0) / c1), (x + c0) / c1, eval(c0 + c3 - c1 <= c2 && c1 > 0 && c3 > 0 && c1 * c4 == c3)) ||
      r.rewrite(min(((x + c2) / c3) * c4, (x + c0) / c1), ((x + c2) / c3) * c4, eval(c2 <= c0 && c1 > 0 && c3 > 0 && c1 * c4 == c3)) ||
      r.rewrite(min(((x + c2) / c3) * c4, x / c1), x/c1, eval(c3 - c1 <= c2 && c1 > 0 && c3 > 0 && c1 * c4 == c3)) ||
      r.rewrite(min(((x + c2) / c3) * c4, x / c1), ((x + c2) / c3) * c4, eval(c2 <= 0 && c1 > 0 && c3 > 0 && c1 * c4 == c3)) ||
      r.rewrite(min((x / c3) * c4, (x + c0) / c1), (x + c0) / c1, eval(c0 + c3 - c1 <= 0 && c1 > 0 && c3 > 0 && c1 * c4 == c3)) ||
      r.rewrite(min((x / c3) * c4, (x + c0) / c1), (x / c3) * c4, eval(0 <= c0 && c1 > 0 && c3 > 0 && c1 * c4 == c3)) ||
      r.rewrite(min(x / c1 + c0, (x / c3) * c4), (x / c3) * c4, eval(c0 > 0 && c1 > 0 && c3 > 0 && c1 * c4 == c3)) ||
      r.rewrite(min((x / c3) * c4, x / c1), (x / c3) * c4, eval(c1 > 0 && c3 > 0 && c1 * c4 == c3)) ||

      // https://github.com/halide/Halide/blob/f4c78317887b6df4d2486e1f81e81f9012943f0f/src/Simplify_Min.cpp#L115-L129
      // Compare x to a stair-step function in x
      r.rewrite(min(x, ((x + c0) / c1) * c1 + c2), x, eval(c1 > 0 && c0 + c2 >= c1 - 1)) ||
      r.rewrite(min(x, ((x + c0) / c1) * c1 + c2), ((x + c0) / c1) * c1 + c2, eval(c1 > 0 && c0 + c2 <= 0)) ||
      r.rewrite(min((x / c1) * c1 + c2, (x / c0) * c0), (x / c0) * c0, eval(c2 >= c1 && c1 > 0 && c0 != 0)) ||
      // Special cases where c0 or c2 is zero
      r.rewrite(min(x, (x / c1) * c1 + c2), x, eval(c1 > 0 && c2 >= c1 - 1)) ||
      r.rewrite(min(x, ((x + c0) / c1) * c1), x, eval(c1 > 0 && c0 >= c1 - 1)) ||
      r.rewrite(min(x, (x / c1) * c1 + c2), (x / c1) * c1 + c2, eval(c1 > 0 && c2 <= 0)) ||
      r.rewrite(min(x, ((x + c0) / c1) * c1), ((x + c0) / c1) * c1, eval(c1 > 0 && c0 <= 0)) ||

      r.rewrite(min(x, (x / c0) * c0), (x / c0) * c0, eval(c0 > 0)) ||

      false) {
    return r.result;
  }
  // clang-format on
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return op;
  } else {
    return min::make(std::move(a), std::move(b));
  }
}

expr simplify(const class max* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }

  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return std::max(*ca, *cb);
  }

  if (is_indeterminate(a)) return a;
  if (is_indeterminate(b)) return b;

  auto r = make_rewriter(max(pattern_expr{a}, pattern_expr{b}));
  // clang-format off
  if (// One side is the max.
      r.rewrite(max(x, std::numeric_limits<index_t>::min()), x) ||
      r.rewrite(max(x, rewrite::negative_infinity()), x) ||
      r.rewrite(max(x, std::numeric_limits<index_t>::max()), std::numeric_limits<index_t>::max()) ||
      r.rewrite(max(x, rewrite::positive_infinity()), rewrite::positive_infinity()) ||
      r.rewrite(max(x, x + c0), x + c0, eval(c0 > 0)) ||
      r.rewrite(max(x, x + c0), x, eval(c0 < 0)) ||
      r.rewrite(max(x, x), x) ||
    
      // Canonicalize trees and find duplicate terms.
      r.rewrite(max(max(x, y), max(x, z)), max(x, max(y, z))) ||
      r.rewrite(max(max(x, y), max(z, w)), max(x, max(y, max(z, w)))) ||
      r.rewrite(max(x, max(x, y)), max(x, y)) ||
      r.rewrite(max(x, max(y, max(x, z))), max(x, max(y, z))) ||
      r.rewrite(max(x, max(y, max(z, max(x, w)))), max(x, max(y, max(z, w)))) ||
    
      // Similar rules but with mixes of min and max.
      r.rewrite(max(min(x, y), max(x, z)), max(x, z)) ||
      r.rewrite(max(x, max(y, min(x, z))), max(x, y)) ||
      r.rewrite(max(min(x, y), min(x, z)), min(x, max(y, z))) ||
      r.rewrite(max(x, min(y, max(x, z))), max(x, min(y, z))) ||
      r.rewrite(max(x, min(x, y)), x) ||

      // Pull common terms out.
      r.rewrite(max(y + z, max(x, y)), max(x, y + max(z, 0))) ||
      r.rewrite(max(y - z, max(x, y)), max(x, y - min(z, 0))) ||
      r.rewrite(max(y, max(x, y + z)), max(x, y + max(z, 0))) ||
      r.rewrite(max(y, max(x, y - z)), max(x, y - min(z, 0))) ||
      r.rewrite(max(x, max(y, x + z)), max(y, max(x, x + z))) ||
      r.rewrite(max(x, max(y, x - z)), max(y, max(x, x - z))) ||
      r.rewrite(max(x + z, y + z), z + max(x, y)) ||
      r.rewrite(max(x - z, y - z), max(x, y) - z) ||
      r.rewrite(max(z - x, z - y), z - min(x, y)) ||
      r.rewrite(max(x, x + z), x + max(z, 0)) ||
      r.rewrite(max(x, x - z), x - min(z, 0)) ||
      r.rewrite(max(x, -x), abs(x)) ||

      // Selects
      r.rewrite(max(x, select(y, max(x, z), w)), max(x, select(y, z, w))) ||
      r.rewrite(max(x, select(y, z, max(x, w))), max(x, select(y, z, w))) ||
      r.rewrite(max(x, select(y, min(x, z), w)), select(y, x, max(x, w))) ||
      r.rewrite(max(x, select(y, z, min(x, w))), select(y, max(x, z), x)) ||
      r.rewrite(max(y, select(x, y, w)), select(x, y, max(y, w))) ||
      r.rewrite(max(z, select(x, w, z)), select(x, max(z, w), z)) ||
      r.rewrite(max(select(x, y, z), select(x, w, u)), select(x, max(y, w), max(z, u))) ||

      // Move constants out.
      r.rewrite(max(max(x, c0), c1), max(x, eval(max(c0, c1)))) ||
      r.rewrite(max(x + c0, (y + c1) / c2), max(x, (y + eval(c1 - c0 * c2)) / c2) + c0) ||
      r.rewrite(max(x + c0, y + c1), max(x, y + eval(c1 - c0)) + c0) ||
      r.rewrite(max(x + c0, c1 - y), c1 - min(y, eval(c1 - c0) - x)) ||
      r.rewrite(max(x + c0, c1), max(x, eval(c1 - c0)) + c0) ||
      r.rewrite(max(c0 - x, c1 - y), c0 - min(x, y + eval(c0 - c1))) ||
      r.rewrite(max(c0 - x, c1), c0 - min(x, eval(c0 - c1))) ||

      // https://github.com/halide/Halide/blob/7994e7030976f9fcd321a4d1d5f76f4582e01905/src/Simplify_Max.cpp#L271-L300
      r.rewrite(max(x * c0, c1), max(x, eval(c1 / c0)) * c0, eval(c0 > 0 && c1 % c0 == 0)) ||
      r.rewrite(max(x * c0, c1), min(x, eval(c1 / c0)) * c0, eval(c0 < 0 && c1 % c0 == 0)) ||

      r.rewrite(max(x * c0, y * c1), max(x, y * eval(c1 / c0)) * c0, eval(c0 > 0 && c1 % c0 == 0)) ||
      r.rewrite(max(x * c0, y * c1), min(x, y * eval(c1 / c0)) * c0, eval(c0 < 0 && c1 % c0 == 0)) ||
      r.rewrite(max(x * c0, y * c1), max(y, x * eval(c0 / c1)) * c1, eval(c1 > 0 && c0 % c1 == 0)) ||
      r.rewrite(max(x * c0, y * c1), min(y, x * eval(c0 / c1)) * c1, eval(c1 < 0 && c0 % c1 == 0)) ||
      r.rewrite(max(y * c0 + c1, x * c0), max(x, y + eval(c1 / c0)) * c0, eval(c0 > 0 && c1 % c0 == 0)) ||
      r.rewrite(max(y * c0 + c1, x * c0), min(x, y + eval(c1 / c0)) * c0, eval(c0 < 0 && c1 % c0 == 0)) ||

      r.rewrite(max(x / c0, y / c0), max(x, y) / c0, eval(c0 > 0)) ||
      r.rewrite(max(x / c0, y / c0), min(x, y) / c0, eval(c0 < 0)) ||

      r.rewrite(max(x / c0, c1), max(x, eval(c1 * c0)) / c0, eval(c0 > 0)) ||
      r.rewrite(max(x / c0, c1), min(x, eval(c1 * c0)) / c0, eval(c0 < 0)) ||

      r.rewrite(max(y / c0 + c1, x / c0), max(x, y + eval(c1 * c0)) / c0, eval(c0 > 0)) ||
      r.rewrite(max(y / c0 + c1, x / c0), min(x, y + eval(c1 * c0)) / c0, eval(c0 < 0)) ||
 
      r.rewrite(max(((x + c2) / c3) * c4, (x + c0) / c1), (x + c0) / c1, eval(c2 <= c0 && c1 > 0 && c3 > 0 && c1 * c4 == c3)) ||
      r.rewrite(max(((x + c2) / c3) * c4, (x + c0) / c1), ((x + c2) / c3) * c4, eval(c0 + c3 - c1 <= c2 && c1 > 0 && c3 > 0 && c1 * c4 == c3)) ||
      r.rewrite(max(((x + c2) / c3) * c4, x / c1), x/c1, eval(c2 <= 0 && c1 > 0 && c3 > 0 && c1 * c4 == c3)) ||
      r.rewrite(max(((x + c2) / c3) * c4, x / c1), ((x + c2) / c3) * c4, eval(c3 - c1 <= c2 && c1 > 0 && c3 > 0 && c1 * c4 == c3)) ||
      r.rewrite(max((x / c3) * c4, (x + c0) / c1), (x + c0) / c1, eval(0 <= c0 && c1 > 0 && c3 > 0 && c1 * c4 == c3)) ||
      r.rewrite(max((x / c3) * c4, (x + c0) / c1), (x / c3) * c4, eval(c0 + c3 - c1 <= 0 && c1 > 0 && c3 > 0 && c1 * c4 == c3)) ||
      r.rewrite(max(x / c1 + c0, (x / c3) * c4), x / c1 + c0, eval(c0 > 0 && c1 > 0 && c3 > 0 && c1 * c4 == c3)) ||
      r.rewrite(max((x / c3) * c4, x / c1), x / c1, eval(c1 > 0 && c3 > 0 && c1 * c4 == c3)) ||
    
      // https://github.com/halide/Halide/blob/f4c78317887b6df4d2486e1f81e81f9012943f0f/src/Simplify_Max.cpp#L115-L129
      // Compare x to a stair-step function in x
      r.rewrite(max(x, ((x + c0) / c1) * c1 + c2), ((x + c0) / c1) * c1 + c2, eval(c1 > 0 && c0 + c2 >= c1 - 1)) ||
      r.rewrite(max(x, ((x + c0) / c1) * c1 + c2), x, eval(c1 > 0 && c0 + c2 <= 0)) ||
      r.rewrite(max((x / c1) * c1 + c2, (x / c0) * c0), (x / c1) * c1 + c2, eval(c2 >= c1 && c1 > 0 && c0 != 0)) ||
      // Special cases where c0 or c2 is zero
      r.rewrite(max(x, (x / c1) * c1 + c2), (x / c1) * c1 + c2, eval(c1 > 0 && c2 >= c1 - 1)) ||
      r.rewrite(max(x, ((x + c0) / c1) * c1), ((x + c0) / c1) * c1, eval(c1 > 0 && c0 >= c1 - 1)) ||
      r.rewrite(max(x, (x / c1) * c1 + c2), x, eval(c1 > 0 && c2 <= 0)) ||
      r.rewrite(max(x, ((x + c0) / c1) * c1), x, eval(c1 > 0 && c0 <= 0)) ||

      r.rewrite(max(x, (x / c0) * c0), x, eval(c0 > 0)) ||
    
      false) {
    return r.result;
  }
  // clang-format on
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return op;
  } else {
    return max::make(std::move(a), std::move(b));
  }
}

expr simplify(const add* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }

  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return *ca + *cb;
  }

  if (is_indeterminate(a)) return a;
  if (is_indeterminate(b)) return b;
  int inf_a = is_infinity(a);
  int inf_b = is_infinity(b);
  if (inf_a && inf_b) return inf_a == inf_b ? a : slinky::indeterminate();

  auto r = make_rewriter(pattern_expr{a} + pattern_expr{b});
  // clang-format off
  if (r.rewrite(x + rewrite::positive_infinity(), rewrite::positive_infinity(), is_finite(x)) ||
      r.rewrite(x + rewrite::negative_infinity(), rewrite::negative_infinity(), is_finite(x)) ||
      r.rewrite(x + 0, x) ||
      r.rewrite(x + x, x * 2) ||
      r.rewrite(x + (x + y), y + x * 2) ||
      r.rewrite(x + (x - y), x * 2 - y) ||
      r.rewrite(x + (y - x), y) ||
      r.rewrite(x + x * y, x * (y + 1), eval(!is_constant(x))) ||
      r.rewrite(x * y + x * z, x * (y + z)) ||
      r.rewrite((x + y) + (x + z), (y + z) + x * 2) ||
      r.rewrite((x + z) + (x - y), (z - y) + x * 2) ||
      r.rewrite((x + z) + (y - x), y + z) ||
      r.rewrite((x + y) + (x - z), (y - z) + x * 2) ||
      r.rewrite((x + y) + (z - x), y + z) ||
      r.rewrite((x - y) + (x - z), x * 2 - (y + z)) ||
      r.rewrite((y - x) + (x - z), y - z) ||
      r.rewrite((x - y) + (z - x), z - y) ||
      r.rewrite((y - x) + (z - x), (y + z) + x * -2) ||

      r.rewrite((x + c0) + c1, x + eval(c0 + c1)) ||
      r.rewrite((c0 - x) + c1, eval(c0 + c1) - x) ||
      r.rewrite(x + (c0 - y), (x - y) + c0) ||
      r.rewrite(x + (y + c0), (x + y) + c0) ||
      r.rewrite((x + c0) + (y + c1), (x + y) + eval(c0 + c1)) ||
    
      r.rewrite(((x + c0) / c1) * c2 + c3, ((x + eval((c3 / c2) * c1 + c0)) / c1) * c2, eval(c3 % c2 == 0)) ||
      r.rewrite((x + c0) * c2 + c3, (x + eval(c3 / c2 + c0)) * c2, eval(c3 % c2 == 0)) ||
      r.rewrite((x + c0) / c1 + c3, (x + eval(c3 * c1 + c0)) / c1) ||

      r.rewrite(z + min(x, y - (z - w)), min(x + z, y + w)) ||
      r.rewrite(z + max(x, y - (z - w)), max(x + z, y + w)) ||
      r.rewrite(z + min(x, y - z), min(y, x + z)) ||
      r.rewrite(z + max(x, y - z), max(y, x + z)) ||

      r.rewrite(select(x, c0, c1) + c2, select(x, eval(c0 + c2), eval(c1 + c2))) ||
      r.rewrite(select(x, y + c0, c1) + c2, select(x, y + eval(c0 + c2), eval(c1 + c2))) ||
      r.rewrite(select(x, c0 - y, c1) + c2, select(x, eval(c0 + c2) - y, eval(c1 + c2))) ||
      r.rewrite(select(x, c0, y + c1) + c2, select(x, eval(c0 + c2), y + eval(c1 + c2))) ||
      r.rewrite(select(x, c0, c1 - y) + c2, select(x, eval(c0 + c2), eval(c1 + c2) - y)) ||
      r.rewrite(select(x, y + c0, z + c1) + c2, select(x, y + eval(c0 + c2), z + eval(c1 + c2))) ||
      r.rewrite(select(x, c0 - y, z + c1) + c2, select(x, eval(c0 + c2) - y, z + eval(c1 + c2))) ||
      r.rewrite(select(x, y + c0, c1 - z) + c2, select(x, y + eval(c0 + c2), eval(c1 + c2) - z)) ||
      r.rewrite(select(x, c0 - y, c1 - z) + c2, select(x, eval(c0 + c2) - y, eval(c1 + c2) - z)) ||
      false) {
    return r.result;
  }
  // clang-format on
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return op;
  } else {
    return add::make(std::move(a), std::move(b));
  }
}

expr simplify(const sub* op, expr a, expr b) {
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return *ca - *cb;
  } else if (cb) {
    // Canonicalize to addition with constants.
    return simplify(static_cast<add*>(nullptr), a, -*cb);
  }

  if (is_indeterminate(a)) return a;
  if (is_indeterminate(b)) return b;
  int inf_a = is_infinity(a);
  int inf_b = is_infinity(b);
  if (inf_a && inf_b) return inf_a == inf_b ? slinky::indeterminate() : a;

  auto r = make_rewriter(pattern_expr{a} - pattern_expr{b});
  // clang-format off
  if (r.rewrite(x - rewrite::positive_infinity(), rewrite::negative_infinity(), is_finite(x)) ||
      r.rewrite(x - rewrite::negative_infinity(), rewrite::positive_infinity(), is_finite(x)) ||
      r.rewrite(x - x, 0) ||
      r.rewrite(x - 0, x) ||
      r.rewrite(x - y * c0, x + y * (-c0)) ||
      r.rewrite(x - (c0 - y), (x + y) - c0) ||
      r.rewrite(c0 - (x - y), (y - x) + c0) ||
      r.rewrite(x - (y + c0), (x - y) - c0) ||
      r.rewrite((c0 - x) - y, c0 - (x + y)) ||
      r.rewrite((x + c0) - y, (x - y) + c0) ||
      r.rewrite((x + y) - x, y) ||
      r.rewrite((x - y) - x, -y) ||
      r.rewrite(x - (x + y), -y) ||
      r.rewrite(x - (x - y), y) ||
      r.rewrite((x + y) - (x + z), y - z) ||
      r.rewrite((x - y) - (z - y), x - z) ||
      r.rewrite((x - y) - (x - z), z - y) ||
      r.rewrite((c0 - x) - (y - z), ((z - x) - y) + c0) ||
      r.rewrite((x + c0) - (y + c1), (x - y) + eval(c0 - c1)) ||
    
      // These rules taken from https://github.com/halide/Halide/blob/e3d3c8cacfe6d664a8994166d0998f362bf55ce8/src/Simplify_Sub.cpp#L411-L421
      r.rewrite((x + y)/c0 - (x + c1)/c0, ((y - c1) + ((x + eval(c1 % c0)) % c0))/c0, eval(c0 > 0)) ||
      r.rewrite((x + c1)/c0 - (x + y)/c0, ((eval(c0 + c1 - 1) - y) - ((x + eval(c1 % c0)) % c0))/c0, eval(c0 > 0)) ||
      r.rewrite((x - y)/c0 - (x + c1)/c0, (((x + eval(c1 % c0)) % c0) - y - c1)/c0, eval(c0 > 0)) ||
      r.rewrite((x + c1)/c0 - (x - y)/c0, ((y + eval(c0 + c1 - 1)) - ((x + eval(c1 % c0)) % c0))/c0, eval(c0 > 0)) ||
      r.rewrite(x/c0 - (x + y)/c0, ((eval(c0 - 1) - y) - (x % c0))/c0, eval(c0 > 0)) ||
      r.rewrite((x + y)/c0 - x/c0, (y + (x % c0))/c0, eval(c0 > 0)) ||
      r.rewrite(x/c0 - (x - y)/c0, ((y + eval(c0 - 1)) - (x % c0))/c0, eval(c0 > 0)) ||
      r.rewrite((x - y)/c0 - x/c0, ((x % c0) - y)/c0, eval(c0 > 0)) ||
      r.rewrite((x + y) / c0 - x / c0, (y + (x % c0)) / c0, eval(eval(c0 > 0))) ||

      r.rewrite(min(x, y + z) - z, min(y, x - z)) ||
      r.rewrite(max(x, y + z) - z, max(y, x - z)) ||

      r.rewrite(c2 - select(x, c0, c1), select(x, eval(c2 - c0), eval(c2 - c1))) ||
      r.rewrite(c2 - select(x, y + c0, c1), select(x, eval(c2 - c0) - y, eval(c2 - c1))) ||
      r.rewrite(c2 - select(x, c0 - y, c1), select(x, y + eval(c2 - c0), eval(c2 - c1))) ||
      r.rewrite(c2 - select(x, c0, y + c1), select(x, eval(c2 - c0), eval(c2 - c1) - y)) ||
      r.rewrite(c2 - select(x, c0, c1 - y), select(x, eval(c2 - c0), y + eval(c2 - c1))) ||
      r.rewrite(c2 - select(x, y + c0, z + c1), select(x, eval(c2 - c0) - y, eval(c2 - c1) - z)) ||
      r.rewrite(c2 - select(x, c0 - y, z + c1), select(x, y + eval(c2 - c0), eval(c2 - c1) - z)) ||
      r.rewrite(c2 - select(x, y + c0, c1 - z), select(x, eval(c2 - c0) - y, z + eval(c2 - c1))) ||
      r.rewrite(c2 - select(x, c0 - y, c1 - z), select(x, y + eval(c2 - c0), z + eval(c2 - c1))) ||
    
      r.rewrite(max(x, y) / c0 - min(x, y) / c0, abs(x / c0 - y / c0), eval(c0 > 0)) ||
      r.rewrite(min(x, y) / c0 - max(x, y) / c0, -abs(x / c0 - y / c0), eval(c0 > 0)) ||
      r.rewrite(max(x, y) - min(x, y), abs(x - y)) ||
      r.rewrite(min(x, y) - max(x, y), -abs(x - y)) ||
      false) {
    return r.result;
  }
  // clang-format on
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return op;
  } else {
    return sub::make(std::move(a), std::move(b));
  }
}

expr simplify(const mul* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }

  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return *ca * *cb;
  }

  if (is_indeterminate(a)) return a;
  if (is_indeterminate(b)) return b;
  int inf_a = is_infinity(a);
  int inf_b = is_infinity(b);
  if (inf_a && inf_b) return infinity(inf_a * inf_b);

  auto r = make_rewriter(pattern_expr{a} * pattern_expr{b});
  // clang-format off
  if (r.rewrite(rewrite::positive_infinity() * c0, rewrite::positive_infinity(), eval(c0 > 0)) ||
      r.rewrite(rewrite::negative_infinity() * c0, rewrite::negative_infinity(), eval(c0 > 0)) ||
      r.rewrite(rewrite::positive_infinity() * c0, rewrite::negative_infinity(), eval(c0 < 0)) ||
      r.rewrite(rewrite::negative_infinity() * c0, rewrite::positive_infinity(), eval(c0 < 0)) ||
      r.rewrite(x * 0, 0) ||
      r.rewrite(x * 1, x) ||
      r.rewrite((x * c0) * c1, x * eval(c0 * c1)) ||
      r.rewrite((x + c0) * c1, x * c1 + eval(c0 * c1)) ||
      r.rewrite((c0 - x) * c1, x * eval(-c1) + eval(c0 * c1)) ||
      false) {
    return r.result;
  }
  // clang-format on
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return op;
  } else {
    return mul::make(std::move(a), std::move(b));
  }
}

expr simplify(const div* op, expr a, expr b) {
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return euclidean_div(*ca, *cb);
  }

  if (is_indeterminate(a)) return a;
  if (is_indeterminate(b)) return b;
  if (is_infinity(a) && is_infinity(b)) return slinky::indeterminate();

  auto r = make_rewriter(pattern_expr{a} / pattern_expr{b});
  // clang-format off
  if (r.rewrite(x / rewrite::positive_infinity(), 0, is_finite(x)) ||
      r.rewrite(x / rewrite::negative_infinity(), 0, is_finite(x)) ||
      r.rewrite(rewrite::positive_infinity() / c0, rewrite::positive_infinity(), eval(c0 > 0)) ||
      r.rewrite(rewrite::negative_infinity() / c0, rewrite::negative_infinity(), eval(c0 > 0)) ||
      r.rewrite(rewrite::positive_infinity() / c0, rewrite::negative_infinity(), eval(c0 < 0)) ||
      r.rewrite(rewrite::negative_infinity() / c0, rewrite::positive_infinity(), eval(c0 < 0)) ||
      r.rewrite(x / 0, 0) ||
      r.rewrite(0 / x, 0) ||
      r.rewrite(x / 1, x) ||
      r.rewrite(x / -1, -x) ||
      r.rewrite(x / x, x != 0) ||

      r.rewrite((y + x / c0) / c1, (x + y * c0) / eval(c0 * c1), eval(c0 > 0 && c1 > 0)) ||
      r.rewrite((x / c0) / c1, x / eval(c0 * c1), eval(c0 > 0 && c1 > 0)) ||
      r.rewrite((x * c0) / c1, x * eval(c0 / c1), eval(c1 > 0 && c0 % c1 == 0)) ||

      r.rewrite((x + y * c0) / c1, y * eval(c0 / c1) + x / c1, eval(c0 % c1 == 0)) ||
      r.rewrite((x + c0) / c1, x / c1 + eval(c0 / c1), eval(c0 % c1 == 0)) ||
      r.rewrite((y * c0 - x) / c1, y * eval(c0 / c1) + (-x / c1), eval(c0 != 0 && c0 % c1 == 0)) ||
      r.rewrite((c0 - x) / c1, (-x / c1) + eval(c0 / c1), eval(c0 != 0 && c0 % c1 == 0)) ||
      false) {
    return r.result;
  }
  // clang-format on
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return op;
  } else {
    return div::make(std::move(a), std::move(b));
  }
}

expr simplify(const mod* op, expr a, expr b) {
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return euclidean_mod(*ca, *cb);
  }

  auto r = make_rewriter(pattern_expr{a} % pattern_expr{b});
  // clang-format off
  if (r.rewrite(x % 1, 0) || 
      r.rewrite(x % 0, 0) || 
      r.rewrite(x % x, 0) ||
      false) {
    return r.result;
  }
  // clang-format on
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return op;
  } else {
    return mod::make(std::move(a), std::move(b));
  }
}

expr simplify(const less* op, expr a, expr b) {
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return *ca < *cb;
  }

  auto r = make_rewriter(pattern_expr{a} < pattern_expr{b});
  // clang-format off
  if (r.rewrite(rewrite::positive_infinity() < x, false, is_finite(x)) ||
      r.rewrite(rewrite::negative_infinity() < x, true, is_finite(x)) ||
      r.rewrite(x < rewrite::positive_infinity(), true, is_finite(x)) ||
      r.rewrite(x < rewrite::negative_infinity(), false, is_finite(x)) ||
      r.rewrite(x < x, false) ||
    
      // These rules taken from:
      // https://github.com/halide/Halide/blob/e9f8b041f63a1a337ce3be0b07de5a1cfa6f2f65/src/Simplify_LT.cpp#L87-L169
      // with adjustments for the simplifier implementation here.
 
      // Normalize subtractions to additions to cut down on cases to consider
      r.rewrite(x - y < z, x < z + y) ||
      r.rewrite(z < x - y, z + y < x) ||
      r.rewrite(z + (x - y) < w, x + z < y + w) ||
      r.rewrite(w < z + (x - y), w + y < x + z) ||
      r.rewrite(u + (z + (x - y)) < w, x + (z + u) < w + y) ||
      r.rewrite(w < u + (z + (x - y)), w + y < x + (z + u)) ||

      // Cancellations in linear expressions
      r.rewrite(x < x + y, 0 < y) ||
      r.rewrite(x + y < x, y < 0) ||
      r.rewrite(x < z + (x + y), 0 < z + y) ||
      r.rewrite(z + (x + y) < x, z + y < 0) ||
      r.rewrite(x + y < x + z, y < z) ||
      r.rewrite(w + (x + y) < x + z, y + w < z) ||
      r.rewrite(x + z < w + (x + y), z < y + w) ||
      r.rewrite(w + (x + y) < u + (x + z), y + w < z + u) ||

      r.rewrite(x + c0 < y + c1, x < y + eval(c1 - c0)) ||
      r.rewrite(x + c0 < c1, x < eval(c1 - c0)) ||
      r.rewrite(x + c0 < y, x < y + eval(-c0)) ||
      r.rewrite(c0 < x + c1, eval(c0 - c1) < x) ||

      r.rewrite(x < (x / c0) * c0 + c1, true, eval(c0 > 0 && c1 >= c0 - 1)) ||
      r.rewrite(x < (x / c0) * c0 + c1, false, eval(c0 > 0 && c1 <= 0)) ||
      r.rewrite(x + c1 < (x / c0) * c0, true, eval(c0 > 0 && c1 <= -c0 + 1)) ||
      r.rewrite(x + c1 < (x / c0) * c0, false, eval(c0 > 0 && c1 >= 0)) ||
      r.rewrite((x / c0) * c0 < x + c1, true, eval(c0 > 0 && c1 > 0)) ||
      r.rewrite((x / c0) * c0 < x + c1, false, eval(c0 > 0 && -c1 >= c0 - 1)) ||
      r.rewrite((x / c0) * c0 + c1 < x, true, eval(c0 > 0 && c1 < 0)) ||
      r.rewrite((x / c0) * c0 + c1 < x, false, eval(c0 > 0 && c1 >= c0 - 1)) ||
      r.rewrite(x < (x / c0) * c0, false, eval(c0 > 0)) ||
      r.rewrite((x / c0) * c0 < x, x % c0 != 0, eval(c0 > 0)) ||

      r.rewrite((x + c0) / c2 < (x + c1) / c2, eval(c0 < c1), eval(c2 > 0)) ||
      r.rewrite(x / c2 < (x + c1) / c2, eval(0 < c1), eval(c2 > 0)) ||
      r.rewrite((x + c0) / c2 < x / c2, eval(c0 < 0), eval(c2 > 0)) ||

      // TODO: These aren't fully simplified, the above rules can be applied to the rewritten result.
      // If we ever added a c2 < 0 version of the above, these would need to be duplicated as well.
      r.rewrite((x + c0) / c1 < x / c1 + c2, (x + eval(c0 - c2 * c1)) / c1 < x / c1) ||
      r.rewrite(x / c1 < x / c1 + c2, (x - eval(c2 * c1)) / c1 < x / c1) ||
      r.rewrite(x / c1 + c2 < (x + c0) / c1, x / c1 < (x + eval(c0 - c2 * c1)) / c1) ||
      r.rewrite(x / c1 + c2 < x / c1, x / c1 < (x - eval(c2 * c1)) / c1) ||

      r.rewrite(x * c0 < y * c0, x < y, eval(c0 > 0)) ||
      r.rewrite(x * c0 < y * c0, y < x, eval(c0 < 0)) ||
        
      // The following rules are taken from
      // https://github.com/halide/Halide/blob/7636c44acc2954a7c20275618093973da6767359/src/Simplify_LT.cpp#L186-L263
      // with adjustments for the simplifier implementation here.

      // We want to break max(x, y) < z into x < z && y <
      // z in cases where one of those two terms is going
      // to eval.
      r.rewrite(min(y, x + c0) < x + c1, y < x + c1 || eval(c0 < c1)) ||
      r.rewrite(max(y, x + c0) < x + c1, y < x + c1 && eval(c0 < c1)) ||
      r.rewrite(x < min(y, x + c0) + c1, x < y + c1 && eval(0 < c0 + c1)) ||
      r.rewrite(x < max(y, x + c0) + c1, x < y + c1 || eval(0 < c0 + c1)) ||

      // Special cases where c0 == 0
      r.rewrite(min(x, y) < x + c1, y < x + c1 || eval(0 < c1)) ||
      r.rewrite(max(x, y) < x + c1, y < x + c1 && eval(0 < c1)) ||
      r.rewrite(x < min(x, y) + c1, x < y + c1 && eval(0 < c1)) ||
      r.rewrite(x < max(x, y) + c1, x < y + c1 || eval(0 < c1)) ||

      // Special cases where c1 == 0
      r.rewrite(min(y, x + c0) < x, y < x || eval(c0 < 0)) ||
      r.rewrite(max(y, x + c0) < x, y < x && eval(c0 < 0)) ||
      r.rewrite(x < min(y, x + c0), x < y && eval(0 < c0)) ||
      r.rewrite(x < max(y, x + c0), x < y || eval(0 < c0)) ||

      // Special cases where c0 == c1 == 0
      r.rewrite(min(x, y) < x, y < x) ||
      r.rewrite(max(x, y) < x, false) ||
      r.rewrite(x < max(x, y), x < y) ||
      r.rewrite(x < min(x, y), false) ||

      // Special case where x is constant
      r.rewrite(min(y, c0) < c1, y < c1 || eval(c0 < c1)) ||
      r.rewrite(max(y, c0) < c1, y < c1 && eval(c0 < c1)) ||
      r.rewrite(c1 < min(y, c0), c1 < y && eval(c1 < c0)) ||
      r.rewrite(c1 < max(y, c0), c1 < y || eval(c1 < c0)) ||
    
      // TODO: This rule seems a bit specialized, but it's hard to find a more general rule.
      r.rewrite(min(x, y + c0) < min(x, y) + c1, eval(0 < c1 || c0 < c1)) ||

      // Cases where we can remove a min on one side because
      // one term dominates another. These rules were
      // synthesized then extended by hand.
      r.rewrite(min(z, y) < min(x, y), z < min(x, y)) ||
      r.rewrite(min(z, y) < min(x, y + c0), min(z, y) < x, eval(c0 > 0)) ||
      r.rewrite(min(z, y + c0) < min(x, y), min(z, y + c0) < x, eval(c0 < 0)) ||

      // Equivalents with max
      r.rewrite(max(z, y) < max(x, y), max(z, y) < x) ||
      r.rewrite(max(z, y) < max(x, y + c0), max(z, y) < x, eval(c0 < 0)) ||
      r.rewrite(max(z, y + c0) < max(x, y), max(z, y + c0) < x, eval(c0 > 0)) ||

      r.rewrite(min(x, min(y, z)) < y, min(x, z) < y) ||
      r.rewrite(min(x, y) < max(x, y), x != y) ||
      r.rewrite(max(x, y) < min(x, y), false) ||
        
      // Subtract terms from both sides within a min/max.
      // These are only enabled for non-constants because they loop with rules that pull constants out of min/max.
      r.rewrite(min(x, y) < x + z, min(y - x, 0) < z, !is_constant(x)) ||
      r.rewrite(max(x, y) < x + z, max(y - x, 0) < z, !is_constant(x)) ||

      r.rewrite(x + z < min(x, y), z < min(y - x, 0), !is_constant(x)) ||
      r.rewrite(x + z < max(x, y), z < max(y - x, 0), !is_constant(x)) ||

      r.rewrite(min(z, x + y) < x + w, min(y, z - x) < w, !is_constant(x)) ||
      r.rewrite(min(z, x - y) < x + w, min(-y, z - x) < w, !is_constant(x)) ||
      r.rewrite(max(z, x + y) < x + w, max(y, z - x) < w, !is_constant(x)) ||
      r.rewrite(max(z, x - y) < x + w, max(-y, z - x) < w, !is_constant(x)) ||

      r.rewrite(x + y < max(w, x + z), y < max(z, w - y), !is_constant(x)) ||
      r.rewrite(x + y < max(w, x - z), y < max(-z, w - x), !is_constant(x)) ||
      r.rewrite(x + y < min(w, x + z), y < min(z, w - y), !is_constant(x)) ||
      r.rewrite(x + y < min(w, x - z), y < min(-z, w - x), !is_constant(x)) ||

      // Selects
      r.rewrite(select(x, y, z) < select(x, y, w), select(x, false, z < w)) ||
      r.rewrite(select(x, y, z) < select(x, w, z), select(x, y < w, false)) ||
      r.rewrite(select(x, y, z) < y, select(x, false, z < y)) ||
      r.rewrite(select(x, y, z) < z, select(x, y < z, false)) ||
      r.rewrite(y < select(x, y, w), select(x, false, y < w)) ||
      r.rewrite(w < select(x, y, w), select(x, w < y, false)) ||

      false) {
    return r.result;
  }
  // clang-format on
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return op;
  } else {
    return less::make(std::move(a), std::move(b));
  }
}

expr simplify(const less_equal* op, expr a, expr b) {
  // Rewrite to !(b < a) and simplify that instead.
  expr result = simplify(static_cast<const logical_not*>(nullptr), simplify(static_cast<const less*>(nullptr), b, a));
  if (op) {
    if (const less_equal* le = result.as<less_equal>()) {
      if (le->a.same_as(op->a) && le->b.same_as(op->b)) {
        return op;
      }
    }
  }
  return result;
}

expr simplify(const equal* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }

  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return *ca == *cb;
  }

  auto r = make_rewriter(pattern_expr{a} == pattern_expr{b});
  // clang-format off
  if (r.rewrite(x == x, true) ||
      r.rewrite(x - y == 0, x == y) ||
      r.rewrite(x * y == x * z, y == z || x == 0) ||
      r.rewrite(x == x * y, y == 1 || x == 0) ||
      r.rewrite(x + y == z + y, x == z) ||
      r.rewrite(x - y == z - y, x == z) ||
      r.rewrite(x - y == x - z, y == z) ||
      r.rewrite(x * c0 == y * c1, x == y * eval(c1 / c0), eval(c1 % c0 == 0)) ||
      r.rewrite(x * c0 == y * c1, y == x * eval(c0 / c1), eval(c0 % c1 == 0)) ||
      r.rewrite(x * c0 == c1, x == eval(c1 / c0), eval(c1 % c0 == 0)) ||
      r.rewrite(x + c0 == y + c1, x == y + eval(c1 - c0)) ||
      r.rewrite(x + c0 == c1, x == eval(c1 - c0)) ||
      r.rewrite(x + c0 == c1 - y, x + y == eval(c1 - c0)) ||
      r.rewrite(c0 - x == c1, x == eval(c0 - c1)) ||
      false) {
    return r.result;
  }
  // clang-format on
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return op;
  } else {
    return equal::make(std::move(a), std::move(b));
  }
}

expr simplify(const not_equal* op, expr a, expr b) {
  // Rewrite to !(a == b) and simplify that instead.
  expr result = simplify(static_cast<const logical_not*>(nullptr), simplify(static_cast<const equal*>(nullptr), a, b));
  if (op) {
    if (const not_equal* ne = result.as<not_equal>()) {
      if (ne->a.same_as(op->a) && ne->b.same_as(op->b)) {
        return op;
      }
    }
  }
  return result;
}

expr simplify(const logical_and* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);

  if (ca && cb) {
    return *ca != 0 && *cb != 0;
  } else if (cb) {
    return *cb ? a : b;
  }

  auto r = make_rewriter(pattern_expr{a} && pattern_expr{b});
  // clang-format off
  if (r.rewrite(x && x, x) ||
      r.rewrite(x && !x, false) ||
      r.rewrite(!x && x, false) ||
      r.rewrite(!x && !y, !(x || y)) ||
      r.rewrite(x && (x && y), x && y) ||
      r.rewrite((x && y) && x, x && y) ||
      r.rewrite(x && (x || y), x) ||
      r.rewrite((x || y) && x, x) ||
      false) {
    return r.result;
  }
  // clang-format on
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return op;
  } else {
    return logical_and::make(std::move(a), std::move(b));
  }
}

expr simplify(const logical_or* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);

  if (ca && cb) {
    return *ca != 0 || *cb != 0;
  } else if (cb) {
    return *cb ? b : a;
  }

  auto r = make_rewriter(pattern_expr{a} || pattern_expr{b});
  // clang-format off
  if (r.rewrite(x || x, x) ||
      r.rewrite(x || !x, true) ||
      r.rewrite(!x || x, true) ||
      r.rewrite(!x || !y, !(x && y)) ||
      r.rewrite(x || (x && y), x) ||
      r.rewrite((x && y) || x, x) ||
      r.rewrite(x || (x || y), x || y) ||
      r.rewrite((x || y) || x, x || y) ||
      false) {
    return r.result;
  }
  // clang-format on
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return op;
  } else {
    return logical_or::make(std::move(a), std::move(b));
  }
}

expr simplify(const class logical_not* op, expr a) {
  const index_t* cv = as_constant(a);
  if (cv) {
    return *cv == 0;
  }

  auto r = make_rewriter(!pattern_expr{a});
  // clang-format off
  if (r.rewrite(!!x, x) ||
      r.rewrite(!(x == y), x != y) ||
      r.rewrite(!(x != y), x == y) ||
      r.rewrite(!(x < y), y <= x) ||
      r.rewrite(!(x <= y), y < x) ||
      false) {
    return r.result;
  }
  // clang-format on
  if (op && a.same_as(op->a)) {
    return op;
  } else {
    return logical_not::make(std::move(a));
  }
}

expr simplify(const class select* op, expr c, expr t, expr f) {
  std::optional<bool> const_c = attempt_to_prove(c);
  if (const_c) {
    if (*const_c) {
      return op->true_value;
    } else {
      return op->false_value;
    }
  }

  auto r = make_rewriter(select(pattern_expr{c}, pattern_expr{t}, pattern_expr{f}));
  // clang-format off
  if (r.rewrite(select(x, y, y), y) ||
      r.rewrite(select(!x, y, z), select(x, z, y)) ||

      // Pull common expressions out
      r.rewrite(select(x, y, y + z), y + select(x, 0, z)) ||
      r.rewrite(select(x, y + z, y), y + select(x, z, 0)) ||
      r.rewrite(select(x, y + z, y + w), y + select(x, z, w)) ||
      r.rewrite(select(x, z - y, w - y), select(x, z, w) - y) ||
      
      r.rewrite(select(x, select(y, z, w), select(y, u, w)), select(y, select(x, z, u), w)) ||
      r.rewrite(select(x, select(y, z, w), select(y, z, u)), select(y, z, select(x, w, u))) ||

    false) {
    return r.result;
  }
  // clang-format on
  if (op && c.same_as(op->condition) && t.same_as(op->true_value) && f.same_as(op->false_value)) {
    return op;
  } else {
    return select::make(std::move(c), std::move(t), std::move(f));
  }
}

expr simplify(const call* op, intrinsic fn, std::vector<expr> args) {
  bool constant = true;
  bool changed = op == nullptr;
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
      // buffer_at(b, buffer_min(b, 0)) is equivalent to buffer_at(b, <>)
      if (args[d].defined() && match(args[d], buffer_min(args[0], d - 1))) {
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
    if (is_non_negative(args[0])) {
      return args[0];
    } else if (is_non_positive(args[0])) {
      return simplify(static_cast<const sub*>(nullptr), 0, std::move(args[0]));
    }
  }

  expr e;
  if (!changed) {
    assert(op);
    e = op;
  } else {
    e = call::make(fn, std::move(args));
  }

  if (can_evaluate(fn) && constant) {
    return evaluate(e);
  }

  rewriter r(e);
  // clang-format off
  if (r.rewrite(abs(x * c0), abs(x) * c0, c0 > 0) ||
      r.rewrite(abs(x * c0), abs(x) * eval(-c0), c0 < 0) ||
      r.rewrite(abs(c0 - x), abs(x + eval(-c0))) ||
      false) {
    return r.result;
  }
  // clang-format on
  return e;
}

}  // namespace slinky