#include <algorithm>
#include <limits>

#include "base/arithmetic.h"

namespace slinky {

std::optional<std::pair<int, int>> staircase_sum_bounds(int a1, int b1, int c1, int a2, int b2, int c2) {
  if (b1 == 0 && b2 == 0) {
    return {{0, 0}};
  } else if (c1 * b2 != -c2 * b1) {
    return std::nullopt;
  } else if (b1 == 0 || b2 == 0) {
    return std::nullopt;
  }

  // The ratios of the two sides are equal. The value of this expression is a periodic pattern.
  // We need to search the period for the min and max.
  // If these constants get so big, we need to revisit this algorithm.
  if (abs(b1) > 1024 || abs(b2) > 1024) {
    return std::nullopt;
  }
  int min = std::numeric_limits<int>::max();
  int max = std::numeric_limits<int>::min();
  const int period = lcm(abs(b1), abs(b2));
  for (int x = 0; x < period; ++x) {
    const int y = euclidean_div(x + a1, b1) * c1 + euclidean_div(x + a2, b2) * c2;
    min = std::min(min, y);
    max = std::max(max, y);
  }
  return {{min, max}};
}

}  // namespace slinky
