#include "builder/simplify.h"

#include <algorithm>
#include <cassert>

#include "builder/rewrite.h"
#include "builder/substitute.h"
#include "runtime/evaluate.h"

namespace slinky {

using namespace rewrite;

namespace {

pattern_wildcard x{0};
pattern_wildcard y{1};
pattern_wildcard z{2};
pattern_wildcard w{3};

pattern_constant c0{0};
pattern_constant c1{1};
pattern_constant c2{2};

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
  if (// Constant simplifications
      r.rewrite(min(x, std::numeric_limits<index_t>::max()), x) ||
      r.rewrite(min(x, rewrite::positive_infinity()), x) ||
      r.rewrite(min(x, std::numeric_limits<index_t>::min()), std::numeric_limits<index_t>::min()) ||
      r.rewrite(min(x, rewrite::negative_infinity()), rewrite::negative_infinity()) ||
      r.rewrite(min(min(x, c0), c1), min(x, eval(min(c0, c1)))) ||
      r.rewrite(min(x, x + c0), x, eval(c0 > 0)) ||
      r.rewrite(min(x, x + c0), x + c0, eval(c0 < 0)) ||
      r.rewrite(min(x + c0, c1), min(x, eval(c1 - c0)) + c0) ||
      r.rewrite(min(x, -x), -abs(x)) ||
      r.rewrite(min(x + c0, c0 - x), c0 - abs(x)) ||
      r.rewrite(min(x + c0, y + c1), min(x, y + eval(c1 - c0)) + c0) ||
      r.rewrite(min(c0 - x, c1 - y), c0 - max(x, y + eval(c0 - c1))) ||
      r.rewrite(min(abs(x), c0), c0, c0 <= 0) ||
      r.rewrite(min(c1 - abs(x), c0), c1 - abs(x), c1 <= c0) ||

      // Algebraic simplifications
      r.rewrite(min(x, x), x) ||
      r.rewrite(min(x, max(x, y)), x) ||
      r.rewrite(min(x, min(x, y)), min(x, y)) ||
      r.rewrite(min(y + c0, min(x, y)), min(x, min(y, y + c0))) ||
      r.rewrite(min(y, min(x, y + c0)), min(x, min(y, y + c0))) ||
      r.rewrite(min(min(x, y), max(x, z)), min(x, y)) ||
      r.rewrite(min(min(x, y), min(x, z)), min(x, min(y, z))) ||
      r.rewrite(min(max(x, y), max(x, z)), max(x, min(y, z))) ||
      r.rewrite(min(x, min(y, x + z)), min(y, min(x, x + z))) ||
      r.rewrite(min(x, min(y, x - z)), min(y, min(x, x - z))) ||
      r.rewrite(min((y + w), min(x, (y + z))), min(x, min(y + z, y + w))) ||
      r.rewrite(min(x / c0, y / c0), min(x, y) / c0, eval(c0 > 0)) ||
      r.rewrite(min(x / c0, y / c0), max(x, y) / c0, eval(c0 < 0)) ||
      r.rewrite(min(x * c0, y * c0), min(x, y) * c0, eval(c0 > 0)) ||
      r.rewrite(min(x * c0, y * c0), max(x, y) * c0, eval(c0 < 0)) ||
      r.rewrite(min(x + z, y + z), z + min(x, y)) ||
      r.rewrite(min(x - z, y - z), min(x, y) - z) ||
      r.rewrite(min(z - x, z - y), z - max(x, y)) ||
      r.rewrite(min(x + z, z - y), z + min(x, -y)) ||

      // Buffer meta simplifications
      // TODO: These rules are sketchy, they assume buffer_max(x, y) > buffer_min(x, y), which
      // is true if we disallow empty buffers...
      r.rewrite(min(buffer_min(x, y), buffer_max(x, y)), buffer_min(x, y)) ||
      r.rewrite(min(buffer_max(x, y) + c0, buffer_min(x, y)), buffer_min(x, y), eval(c0 > 0)) ||
      r.rewrite(min(buffer_min(x, y) + c0, buffer_max(x, y)), buffer_min(x, y) + c0, eval(c0 < 0)) ||
      r.rewrite(min(buffer_max(x, y) + c0, buffer_min(x, y) + c1), buffer_min(x, y) + c1, eval(c0 > c1)) || 
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
  if (// Constant simplifications
      r.rewrite(max(x, std::numeric_limits<index_t>::min()), x) ||
      r.rewrite(max(x, rewrite::negative_infinity()), x) ||
      r.rewrite(max(x, std::numeric_limits<index_t>::max()), std::numeric_limits<index_t>::max()) ||
      r.rewrite(max(x, rewrite::positive_infinity()), rewrite::positive_infinity()) ||
      r.rewrite(max(max(x, c0), c1), max(x, eval(max(c0, c1)))) ||
      r.rewrite(max(x, x + c0), x + c0, eval(c0 > 0)) ||
      r.rewrite(max(x, x + c0), x, eval(c0 < 0)) ||
      r.rewrite(max(x + c0, c1), max(x, eval(c1 - c0)) + c0) ||
      r.rewrite(max(x, -x), abs(x)) ||
      r.rewrite(max(x + c0, c0 - x), abs(x) + c0) ||
      r.rewrite(max(x + c0, y + c1), max(x, y + eval(c1 - c0)) + c0) ||
      r.rewrite(max(c0 - x, c1 - y), c0 - min(x, y + eval(c0 - c1))) ||
      r.rewrite(max(abs(x), c0), abs(x), c0 <= 0) ||
      r.rewrite(max(c1 - abs(x), c0), c0, c1 <= c0) ||

      // Algebraic simplifications
      r.rewrite(max(x, x), x) ||
      r.rewrite(max(x, min(x, y)), x) ||
      r.rewrite(max(x, max(x, y)), max(x, y)) ||
      r.rewrite(max(y + c0, max(x, y)), max(x, max(y, y + c0))) ||
      r.rewrite(max(y, max(x, y + c0)), max(x, max(y, y + c0))) ||
      r.rewrite(max(min(x, y), max(x, z)), max(x, z)) ||
      r.rewrite(max(max(x, y), max(x, z)), max(x, max(y, z))) ||
      r.rewrite(max(min(x, y), min(x, z)), min(x, max(y, z))) ||
      r.rewrite(max(x, max(y, x + z)), max(y, max(x, x + z))) ||
      r.rewrite(max(x, max(y, x - z)), max(y, max(x, x - z))) ||
      r.rewrite(max(x / c0, y / c0), max(x, y) / c0, eval(c0 > 0)) ||
      r.rewrite(max(x / c0, y / c0), min(x, y) / c0, eval(c0 < 0)) ||
      r.rewrite(max(x * c0, y * c0), max(x, y) * c0, eval(c0 > 0)) ||
      r.rewrite(max(x * c0, y * c0), min(x, y) * c0, eval(c0 < 0)) ||
      r.rewrite(max(x + z, y + z), z + max(x, y)) ||
      r.rewrite(max(x - z, y - z), max(x, y) - z) ||
      r.rewrite(max(z - x, z - y), z - min(x, y)) ||

      // Buffer meta simplifications
      r.rewrite(max(buffer_min(x, y), buffer_max(x, y)), buffer_max(x, y)) ||
      r.rewrite(max(buffer_max(x, y) + c0, buffer_min(x, y)), buffer_max(x, y) + c0, eval(c0 > 0)) ||
      r.rewrite(max(buffer_min(x, y) + c0, buffer_max(x, y)), buffer_max(x, y), eval(c0 < 0)) ||
      r.rewrite(max(buffer_max(x, y) + c0, buffer_min(x, y) + c1), buffer_max(x, y) + c0, eval(c0 > c1)) || 
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

      r.rewrite(buffer_min(x, y) + buffer_extent(x, y), buffer_max(x, y) + 1) ||
      r.rewrite((z - buffer_max(x, y)) + buffer_min(x, y), (z - buffer_extent(x, y)) + 1) ||
      r.rewrite((z - buffer_min(x, y)) + buffer_max(x, y), (z + buffer_extent(x, y)) + -1) || 
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

      r.rewrite((x + y) / c0 - x / c0, (y + (x % c0)) / c0, eval(c0 > 0)) ||

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
    
      r.rewrite(max(x, y) - min(x, y), abs(x - y)) ||
      r.rewrite(min(x, y) - max(x, y), -abs(x - y)) ||

      r.rewrite(buffer_max(x, y) - buffer_min(x, y), buffer_extent(x, y) + -1) ||
      r.rewrite(buffer_max(x, y) - (z + buffer_min(x, y)), (buffer_extent(x, y) - z) + -1) ||
      r.rewrite((z + buffer_max(x, y)) - buffer_min(x, y), (z + buffer_extent(x, y)) + -1) ||
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

      r.rewrite((x / c0) / c1, x / eval(c0 * c1), eval(c0 > 0 && c1 > 0)) ||
      r.rewrite((x / c0 + c1) / c2, (x + eval(c1 * c0)) / eval(c0 * c2), eval(c0 > 0 && c2 > 0)) ||
      r.rewrite((x * c0) / c1, x * eval(c0 / c1), eval(c1 > 0 && c0 % c1 == 0)) ||

      r.rewrite((x + c0) / c1, x / c1 + eval(c0 / c1), eval(c0 % c1 == 0)) ||
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
      r.rewrite(x < x + y, 0 < y) ||
      r.rewrite(x + y < x, y < 0) ||
      r.rewrite(x - y < x, 0 < y) ||

      r.rewrite(x + c0 < c1, x < eval(c1 - c0)) ||
      r.rewrite(c0 - x < c1, eval(c0 - c1) < x) ||
      r.rewrite(c0 < c1 - x, x < eval(c1 - c0)) ||
      r.rewrite(c0 < x + c1, eval(c0 - c1) < x) ||

      r.rewrite((x + c0) / c1 < x / c1, eval(c0 < 0), eval(c1 > 0)) ||
      r.rewrite(x / c1 < (x + c0) / c1, eval(c1 <= c0), eval(c1 > 0)) ||
    
      r.rewrite(x < x + y, 0 < y) ||
      r.rewrite(x + y < x, y < 0) ||
      r.rewrite(x - y < x, 0 < y) ||
      r.rewrite(x < x - y, y < 0) ||
      r.rewrite(x - y < y, x < y * 2) ||
      r.rewrite(y < x - y, y * 2 < x) ||

      r.rewrite(x + y < x + z, y < z) ||
      r.rewrite(x - y < x - z, z < y) ||
      r.rewrite(x - y < z - y, x < z) ||

      r.rewrite(min(x, y) < x, y < x) ||
      r.rewrite(min(x, min(y, z)) < y, min(x, z) < y) ||
      r.rewrite(max(x, y) < x, false) ||
      r.rewrite(x < max(x, y), x < y) ||
      r.rewrite(x < min(x, y), false) ||
      r.rewrite(min(x, y) < max(x, y), x != y) ||
      r.rewrite(max(x, y) < min(x, y), false) ||
      r.rewrite(min(x, y) < min(x, z), y < min(x, z)) ||
    
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
      r.rewrite(min(y, x) < x, y < x) ||
      r.rewrite(x < max(x, y), x < y) ||
      r.rewrite(x < max(y, x), x < y) ||

      // Special case where x is constant
      r.rewrite(min(y, c0) < c1, y < c1 || eval(c0 < c1)) ||
      r.rewrite(max(y, c0) < c1, y < c1 && eval(c0 < c1)) ||
      r.rewrite(c1 < min(y, c0), c1 < y && eval(c1 < c0)) ||
      r.rewrite(c1 < max(y, c0), c1 < y || eval(c1 < c0)) ||

      // Cases where we can remove a min on one side because
      // one term dominates another. These rules were
      // synthesized then extended by hand.
      r.rewrite(min(z, y) < min(x, y), z < min(x, y)) ||
      r.rewrite(min(z, y) < min(x, y + c0), min(z, y) < x, c0 > 0) ||

      // Equivalents with max
      r.rewrite(max(z, y) < max(x, y), max(z, y) < x) ||
      r.rewrite(max(y, z) < max(x, y), max(z, y) < x) ||

      r.rewrite(buffer_extent(x, y) < c0, false, eval(c0 <= 0)) ||
      r.rewrite(c0 < buffer_extent(x, y), true, eval(c0 < 0)) ||
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
  if (const less_equal* le = result.as<less_equal>()) {
    if (le->a.same_as(a) && le->b.same_as(b)) {
      return op;
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
      r.rewrite(x + c0 == c1, x == eval(c1 - c0)) ||
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
  if (const not_equal* ne = result.as<not_equal>()) {
    if (ne->a.same_as(a) && ne->b.same_as(b)) {
      return op;
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

  if (fn == intrinsic::buffer_at) {
    // Trailing undefined indices can be removed.
    for (index_t d = 1; d < static_cast<index_t>(args.size()); ++d) {
      // buffer_at(b, buffer_min(b, 0)) is equivalent to buffer_at(b)
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