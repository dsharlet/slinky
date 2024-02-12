#include "builder/simplify.h"

#include <algorithm>
#include <cassert>
#include <iostream>

#include "builder/rewrite.h"
#include "builder/substitute.h"
#include "runtime/evaluate.h"

namespace slinky {

using namespace rewrite;

expr simplify(const class min* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return std::min(*ca, *cb);
  }
  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = min::make(std::move(a), std::move(b));
  }

  rewriter r(e);
  if (// Constant simplifications
      r.rewrite(min(x, indeterminate()), indeterminate()) ||
      r.rewrite(min(x, std::numeric_limits<index_t>::max()), x) ||
      r.rewrite(min(x, positive_infinity()), x) ||
      r.rewrite(min(x, std::numeric_limits<index_t>::min()), std::numeric_limits<index_t>::min()) ||
      r.rewrite(min(x, negative_infinity()), negative_infinity()) ||
      r.rewrite(min(min(x, c0), c1), min(x, eval(min(c0, c1)))) ||
      r.rewrite(min(x, x + c0), x, eval(c0 > 0)) ||
      r.rewrite(min(x, x + c0), x + c0, eval(c0 < 0)) ||
      r.rewrite(min(x + c0, c1), min(x, eval(c1 - c0)) + c0) ||
      r.rewrite(min(c0 - x, c0 - y), c0 - max(x, y)) ||
      r.rewrite(min(x, -x), -abs(x)) ||
      r.rewrite(min(x + c0, c0 - x), c0 - abs(x)) ||

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
  return e;
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
  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = max::make(std::move(a), std::move(b));
  }

  rewriter r(e);
  if (// Constant simplifications
      r.rewrite(max(x, indeterminate()), indeterminate()) ||
      r.rewrite(max(x, std::numeric_limits<index_t>::min()), x) ||
      r.rewrite(max(x, negative_infinity()), x) ||
      r.rewrite(max(x, std::numeric_limits<index_t>::max()), std::numeric_limits<index_t>::max()) ||
      r.rewrite(max(x, positive_infinity()), positive_infinity()) ||
      r.rewrite(max(max(x, c0), c1), max(x, eval(max(c0, c1)))) ||
      r.rewrite(max(x, x + c0), x + c0, eval(c0 > 0)) ||
      r.rewrite(max(x, x + c0), x, eval(c0 < 0)) ||
      r.rewrite(max(x + c0, c1), max(x, eval(c1 - c0)) + c0) ||
      r.rewrite(max(c0 - x, c0 - y), c0 - min(x, y)) ||
      r.rewrite(max(x, -x), abs(x)) ||
      r.rewrite(max(x + c0, c0 - x), abs(x) + c0) ||

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
  return e;
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
  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = add::make(std::move(a), std::move(b));
  }

  rewriter r(e);
  if (r.rewrite(x + indeterminate(), indeterminate()) ||
      r.rewrite(positive_infinity() + indeterminate(), indeterminate()) ||
      r.rewrite(negative_infinity() + positive_infinity(), indeterminate()) ||
      r.rewrite(x + positive_infinity(), positive_infinity(), is_finite(x)) ||
      r.rewrite(x + negative_infinity(), negative_infinity(), is_finite(x)) ||
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

      r.rewrite(min(x + c0, y + c1) + c2, min(x + eval(c0 + c2), y + eval(c1 + c2))) ||
      r.rewrite(max(x + c0, y + c1) + c2, max(x + eval(c0 + c2), y + eval(c1 + c2))) ||
      r.rewrite(min(y + c1, c0 - x) + c2, min(y + eval(c1 + c2), eval(c0 + c2) - x)) ||
      r.rewrite(max(y + c1, c0 - x) + c2, max(y + eval(c1 + c2), eval(c0 + c2) - x)) ||
      r.rewrite(min(c0 - x, c1 - y) + c2, min(eval(c0 + c2) - x, eval(c1 + c2) - y)) ||
      r.rewrite(max(c0 - x, c1 - y) + c2, max(eval(c0 + c2) - x, eval(c1 + c2) - y)) ||
      r.rewrite(min(x, y + c0) + c1, min(x + c1, y + eval(c0 + c1))) ||
      r.rewrite(max(x, y + c0) + c1, max(x + c1, y + eval(c0 + c1))) ||

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
  return e;
}

expr simplify(const sub* op, expr a, expr b) {
  assert(a.defined());
  assert(b.defined());
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return *ca - *cb;
  } else if (cb) {
    // Canonicalize to addition with constants.
    return simplify(static_cast<add*>(nullptr), a, -*cb);
  }

  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = sub::make(std::move(a), std::move(b));
  }

  rewriter r(e);
  if (r.rewrite(x - indeterminate(), indeterminate()) ||
      r.rewrite(indeterminate() - x, indeterminate()) ||
      r.rewrite(positive_infinity() - positive_infinity(), indeterminate()) ||
      r.rewrite(positive_infinity() - negative_infinity(), positive_infinity()) ||
      r.rewrite(negative_infinity() - negative_infinity(), indeterminate()) ||
      r.rewrite(negative_infinity() - positive_infinity(), negative_infinity()) ||
      r.rewrite(x - positive_infinity(), negative_infinity(), is_finite(x)) ||
      r.rewrite(x - negative_infinity(), positive_infinity(), is_finite(x)) ||
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

      r.rewrite(buffer_max(x, y) - buffer_min(x, y), buffer_extent(x, y) + -1) ||
      r.rewrite(buffer_max(x, y) - (z + buffer_min(x, y)), (buffer_extent(x, y) - z) + -1) ||
      r.rewrite((z + buffer_max(x, y)) - buffer_min(x, y), (z + buffer_extent(x, y)) + -1) ||
      false) {
    return r.result;
  }
  return e;
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
  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = mul::make(std::move(a), std::move(b));
  }

  rewriter r(e);
  if (r.rewrite(x * indeterminate(), indeterminate()) ||
      r.rewrite(positive_infinity() * positive_infinity(), positive_infinity()) ||
      r.rewrite(negative_infinity() * positive_infinity(), negative_infinity()) ||
      r.rewrite(negative_infinity() * negative_infinity(), positive_infinity()) ||
      r.rewrite(positive_infinity() * c0, positive_infinity(), eval(c0 > 0)) ||
      r.rewrite(negative_infinity() * c0, negative_infinity(), eval(c0 > 0)) ||
      r.rewrite(positive_infinity() * c0, negative_infinity(), eval(c0 < 0)) ||
      r.rewrite(negative_infinity() * c0, positive_infinity(), eval(c0 < 0)) ||
      r.rewrite(x * 0, 0) ||
      r.rewrite(x * 1, x) ||
      r.rewrite((x * c0) * c1, x * eval(c0 * c1)) ||
      r.rewrite((x + c0) * c1, x * c1 + eval(c0 * c1)) ||
      r.rewrite((0 - x) * c1, x * eval(-c1)) ||
      r.rewrite((c0 - x) * c1, eval(c0 * c1) - x * c1) ||
      false) {
    return r.result;
  }
  return e;
}

expr simplify(const div* op, expr a, expr b) {
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return euclidean_div(*ca, *cb);
  }
  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = div::make(std::move(a), std::move(b));
  }

  rewriter r(e);
  if (r.rewrite(x / indeterminate(), indeterminate()) ||
      r.rewrite(indeterminate() / x, indeterminate()) ||
      r.rewrite(positive_infinity() / positive_infinity(), indeterminate()) ||
      r.rewrite(positive_infinity() / negative_infinity(), indeterminate()) ||
      r.rewrite(negative_infinity() / positive_infinity(), indeterminate()) ||
      r.rewrite(negative_infinity() / negative_infinity(), indeterminate()) ||
      r.rewrite(x / positive_infinity(), 0, is_finite(x)) ||
      r.rewrite(x / negative_infinity(), 0, is_finite(x)) ||
      r.rewrite(positive_infinity() / c0, positive_infinity(), eval(c0 > 0)) ||
      r.rewrite(negative_infinity() / c0, negative_infinity(), eval(c0 > 0)) ||
      r.rewrite(positive_infinity() / c0, negative_infinity(), eval(c0 < 0)) ||
      r.rewrite(negative_infinity() / c0, positive_infinity(), eval(c0 < 0)) ||
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
  return e;
}

expr simplify(const mod* op, expr a, expr b) {
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return euclidean_mod(*ca, *cb);
  }
  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = mod::make(std::move(a), std::move(b));
  }

  rewriter r(e);
  if (r.rewrite(x % 1, 0) || 
      r.rewrite(x % 0, 0) || 
      r.rewrite(x % x, 0) ||
      false) {
    return r.result;
  }
  return e;
}

expr simplify(const less* op, expr a, expr b) {
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return *ca < *cb;
  }

  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = less::make(std::move(a), std::move(b));
  }

  rewriter r(e);
  if (r.rewrite(positive_infinity() < x, false, is_finite(x)) ||
      r.rewrite(negative_infinity() < x, true, is_finite(x)) ||
      r.rewrite(x < positive_infinity(), true, is_finite(x)) ||
      r.rewrite(x < negative_infinity(), false, is_finite(x)) ||
      r.rewrite(x < x, false) ||
      r.rewrite(x + c0 < c1, x < eval(c1 - c0)) ||
      r.rewrite(x < x + y, 0 < y) ||
      r.rewrite(x + y < x, y < 0) ||
      r.rewrite(x - y < x, 0 < y) ||
      r.rewrite(0 - x < c0, -c0 < x) ||
      r.rewrite(c0 - x < c1, eval(c0 - c1) < x) ||
      r.rewrite(c0 < c1 - x, x < eval(c1 - c0)) ||

      r.rewrite(x < x + y, 0 < y) ||
      r.rewrite(x + y < x, y < 0) ||
      r.rewrite(x < x - y, y < 0) ||
      r.rewrite(x - y < x, 0 < y) ||
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

      r.rewrite(c0 < max(x, c1), c0 < x || eval(c0 < c1)) ||
      r.rewrite(c0 < min(x, c1), c0 < x && eval(c0 < c1)) ||
      r.rewrite(max(x, c0) < c1, x < c1 && eval(c0 < c1)) ||
      r.rewrite(min(x, c0) < c1, x < c1 || eval(c0 < c1)) ||

      r.rewrite(buffer_extent(x, y) < c0, false, eval(c0 < 0)) ||
      r.rewrite(c0 < buffer_extent(x, y), true, eval(c0 < 0)) ||
      false) {
    return r.result;
  }
  return e;
}

expr simplify(const less_equal* op, expr a, expr b) {
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return *ca <= *cb;
  }

  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = less_equal::make(std::move(a), std::move(b));
  }

  rewriter r(e);
  if (r.rewrite(positive_infinity() <= x, false, is_finite(x)) ||
      r.rewrite(negative_infinity() <= x, true, is_finite(x)) ||
      r.rewrite(x <= positive_infinity(), true, is_finite(x)) ||
      r.rewrite(x <= negative_infinity(), false, is_finite(x)) ||
      r.rewrite(x <= x, true) ||
      r.rewrite(x <= x + y, 0 <= y) ||
      r.rewrite(x + y <= x, y <= 0) ||
      r.rewrite(x - y <= x, 0 <= y) ||
      r.rewrite(0 - x <= c0, -c0 <= x) ||
      r.rewrite(c0 - x <= y, c0 <= y + x) ||
      r.rewrite(x <= c1 - y, x + y <= c1) ||
      r.rewrite(x + c0 <= y + c1, x - y <= eval(c1 - c0)) ||

      r.rewrite((x + c0) / c1 <= x / c1, eval(c0 <= 0)) ||
      r.rewrite(x / c1 <= (x + c0) / c1, eval(0 <= c0)) ||

      r.rewrite(x <= x + y, 0 <= y) ||
      r.rewrite(x + y <= x, y <= 0) ||
      r.rewrite(x <= x - y, y <= 0) ||
      r.rewrite(x - y <= x, 0 <= y) ||
      r.rewrite(x + y <= x + z, y <= z) ||
      r.rewrite(x - y <= x - z, z <= y) ||
      r.rewrite(x - y <= z - y, x <= z) ||

      r.rewrite(min(x, y) <= x, true) ||
      r.rewrite(min(x, min(y, z)) <= y, true) ||
      r.rewrite(max(x, y) <= x, y <= x) ||
      r.rewrite(x <= max(x, y), true) ||
      r.rewrite(x <= min(x, y), x <= y) ||
      r.rewrite(min(x, y) <= max(x, y), true) ||
      r.rewrite(max(x, y) <= min(x, y), x == y) ||

      r.rewrite(c0 <= max(x, c1), c0 <= x || eval(c0 <= c1)) ||
      r.rewrite(c0 <= min(x, c1), c0 <= x && eval(c0 <= c1)) ||
      r.rewrite(max(x, c0) <= c1, x <= c1 && eval(c0 <= c1)) ||
      r.rewrite(min(x, c0) <= c1, x <= c1 || eval(c0 <= c1)) ||

      r.rewrite(buffer_extent(x, y) <= c0, false, eval(c0 <= 0)) ||
      r.rewrite(c0 <= buffer_extent(x, y), true, eval(c0 <= 0)) ||
      false) {
    return r.result;
  }
  return e;
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

  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = equal::make(std::move(a), std::move(b));
  }
  
  rewriter r(e);
  if (r.rewrite(x == x, true) ||
      r.rewrite(x + c0 == c1, x == eval(c1 - c0)) ||
      r.rewrite(c0 - x == c1, -x == eval(c1 - c0), eval(c0 != 0)) ||
      false) {
    return r.result;
  }
  return e;
}

expr simplify(const not_equal* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return *ca != *cb;
  }

  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = not_equal::make(std::move(a), std::move(b));
  }

  rewriter r(e);
  if (r.rewrite(x != x, false) ||
      r.rewrite(x + c0 != c1, x != eval(c1 - c0)) ||
      r.rewrite(c0 - x != c1, -x != eval(c1 - c0), eval(c0 != 0)) ||
      false) {
    return r.result;
  }
  return e;
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

  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = logical_and::make(std::move(a), std::move(b));
  }

  rewriter r(e);
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
  return e;
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

  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = logical_or::make(std::move(a), std::move(b));
  }

  rewriter r(e);
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
  };
  return e;
}

expr simplify(const logical_not* op, expr a) {
  const index_t* cv = as_constant(a);
  if (cv) {
    return *cv == 0;
  }

  expr e;
  if (op && a.same_as(op->a)) {
    e = op;
  } else {
    e = logical_not::make(std::move(a));
  }

  rewriter r(e);
  if (r.rewrite(!!x, x) ||
      r.rewrite(!(x == y), x != y) ||
      r.rewrite(!(x != y), x == y) ||
      r.rewrite(!(x < y), y <= x) ||
      r.rewrite(!(x <= y), y < x) ||
      false) {
    return r.result;
  }
  return e;
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

  expr e;
  if (match(t, f)) {
    return t;
  } else if (op && c.same_as(op->condition) && t.same_as(op->true_value) && f.same_as(op->false_value)) {
    e = op;
  } else {
    e = select::make(std::move(c), std::move(t), std::move(f));
  }

  rewriter r(e);
  if (r.rewrite(select(!x, y, z), select(x, z, y)) ||

      // Pull common expressions out
      r.rewrite(select(x, y, y + z), y + select(x, 0, z)) ||
      r.rewrite(select(x, y + z, y), y + select(x, z, 0)) ||
      r.rewrite(select(x, y + z, y + w), y + select(x, z, w)) ||
      r.rewrite(select(x, z - y, w - y), select(x, z, w) - y) ||
      false) {
    return r.result;
  }
  return e;
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
      // buffer_at(b, buffer_min(b, 0)) is equivalent to buffer_base(b)
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

    if (args.size() == 1) {
      return call::make(intrinsic::buffer_base, std::move(args));
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
  if (r.rewrite(abs(negative_infinity()), positive_infinity()) || 
      r.rewrite(abs(-x), abs(x)) ||
      r.rewrite(abs(abs(x)), abs(x)) ||
      false) {
    return r.result;
  }
  return e;
}

}  // namespace slinky