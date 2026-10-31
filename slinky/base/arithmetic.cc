#include <algorithm>
#include <cstdint>
#include <limits>

#include "slinky/base/arithmetic.h"

namespace slinky {

namespace {

// Computes x1*y1 + x2*y2, returning true on overflow.
bool dot_with_overflow(std::int64_t x1, std::int64_t y1, std::int64_t x2, std::int64_t y2, std::int64_t& result) {
  std::int64_t xy1, xy2;
  return mul_with_overflow(x1, y1, xy1) ||  //
         mul_with_overflow(x2, y2, xy2) ||  //
         add_with_overflow(xy1, xy2, result);
}

// Rewrites staircase(x, a, b, c) as staircase(x, a, -b, -c) if b is negative, returning true on overflow.
bool make_divisor_positive(std::int64_t& b, std::int64_t& c) {
  if (b >= 0) return false;
  return sub_with_overflow<std::int64_t>(0, b, b) ||  //
         sub_with_overflow<std::int64_t>(0, c, c);
}

}  // namespace

interval<std::int64_t> staircase_sum_bounds(
    std::int64_t a1, std::int64_t b1, std::int64_t c1, std::int64_t a2, std::int64_t b2, std::int64_t c2) {
  if (b1 == 0 && b2 == 0) return {0, 0};
  if (b1 == 0 || b2 == 0) return {};
  if (make_divisor_positive(b1, c1) || make_divisor_positive(b2, c2)) return {};

  // We search one period, lcm(b1, b2), so it must be short.
  if (b1 > 1024 || b2 > 1024) return {};

  // The sum is periodic only if the slopes c1/b1 and c2/b2 cancel.
  std::int64_t slope;
  if (dot_with_overflow(c1, b2, c2, b1, slope) || slope != 0) return {};

  // With a = q*b + r and 0 <= r < b, (x + a)/b = q + (x + r)/b, so we can search with the offsets reduced to r1 and r2,
  // and add the q terms afterwards.
  const std::int64_t q1 = euclidean_div_positive_divisor(a1, b1);
  const std::int64_t q2 = euclidean_div_positive_divisor(a2, b2);
  const std::int64_t r1 = euclidean_mod_positive_modulus(a1, b1);
  const std::int64_t r2 = euclidean_mod_positive_modulus(a2, b2);
  std::int64_t offset;
  if (dot_with_overflow(q1, c1, q2, c2, offset)) return {};

  std::int64_t min = std::numeric_limits<std::int64_t>::max();
  std::int64_t max = std::numeric_limits<std::int64_t>::min();
  const std::int64_t period = lcm(b1, b2);
  for (std::int64_t x = 0; x < period; ++x) {
    std::int64_t y;
    if (dot_with_overflow((x + r1) / b1, c1, (x + r2) / b2, c2, y)) return {};
    min = std::min(min, y);
    max = std::max(max, y);
  }

  if (add_with_overflow(min, offset, min) ||  //
      add_with_overflow(max, offset, max))
    return {};

  return {min, max};
}

}  // namespace slinky
