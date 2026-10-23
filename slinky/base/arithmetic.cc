#include <algorithm>
#include <cstdint>
#include <limits>

#include "slinky/base/arithmetic.h"

namespace slinky {

interval<std::int64_t> staircase_sum_bounds(
    std::int64_t a1, std::int64_t b1, std::int64_t c1, std::int64_t a2, std::int64_t b2, std::int64_t c2) {
  // Unbounded is always a safe answer, and is what we return if any intermediate value overflows an int64.
  //
  // The ratios of the two sides being equal means the value of this expression is a periodic pattern. We need to search
  // the period for the min and max, so we limit |b| to keep the period (at most |b1| * |b2| <= 2^20) small.
  for (std::int64_t i : {b1, b2}) {
    if (i < -1024 || i > 1024) return {};
  }

  if (b1 == 0 && b2 == 0) {
    return {0, 0};
  }
  std::int64_t c1b2, c2b1, neg_c2b1;
  if (mul_with_overflow(c1, b2, c1b2) || mul_with_overflow(c2, b1, c2b1) ||
      sub_with_overflow<std::int64_t>(0, c2b1, neg_c2b1)) {
    return {};
  }
  if (c1b2 != neg_c2b1 || b1 == 0 || b2 == 0) {
    return {};
  }

  // Write a = q * b + r with 0 <= r < |b|. Then euclidean_div(x + a, b) = q + euclidean_div(x + r, b), so the sum is
  // q1 * c1 + q2 * c2 plus the same staircase sum with the offsets reduced to r1 and r2. This keeps the loop below
  // small no matter how large a1 and a2 are.
  if (div_overflows(a1, b1) || div_overflows(a2, b2)) return {};
  const std::int64_t q1 = euclidean_div(a1, b1);
  const std::int64_t q2 = euclidean_div(a2, b2);
  const std::int64_t r1 = euclidean_mod(a1, b1);
  const std::int64_t r2 = euclidean_mod(a2, b2);
  std::int64_t q1c1, q2c2, offset;
  if (mul_with_overflow(q1, c1, q1c1) || mul_with_overflow(q2, c2, q2c2) || add_with_overflow(q1c1, q2c2, offset)) {
    return {};
  }

  std::int64_t min = std::numeric_limits<std::int64_t>::max();
  std::int64_t max = std::numeric_limits<std::int64_t>::min();
  const std::int64_t period = lcm<std::int64_t>(b1 < 0 ? -b1 : b1, b2 < 0 ? -b2 : b2);
  for (std::int64_t x = 0; x < period; ++x) {
    // x < 2^20 and 0 <= r < 2^10, so x + r cannot overflow, and the quotients are at most 2^20 in magnitude.
    std::int64_t t1, t2, y;
    if (mul_with_overflow(euclidean_div(x + r1, b1), c1, t1) || mul_with_overflow(euclidean_div(x + r2, b2), c2, t2) ||
        add_with_overflow(t1, t2, y)) {
      return {};
    }
    min = std::min(min, y);
    max = std::max(max, y);
  }
  if (add_with_overflow(min, offset, min) || add_with_overflow(max, offset, max)) return {};
  return {min, max};
}

}  // namespace slinky
