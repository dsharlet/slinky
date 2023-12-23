#include "simplify.h"
#include "expr.h"
#include "print.h"
#include "substitute.h"
#include "test.h"

#include <cassert>

using namespace slinky;

expr x = variable::make(0);
expr y = variable::make(1);
expr z = variable::make(1);

void test_simplify(const expr& test, const expr& expected) {
  expr result = simplify(test);
  if (!match(result, expected)) {
    std::cout << "simplify(" << test << ") -> " << result << " != " << expected << std::endl;
    ASSERT(false);
  }
}

TEST(simplify) {
  test_simplify(expr(1) + 2, 3);
  test_simplify(expr(1) - 2, -1);
  test_simplify(expr(1) < 2, 1);
  test_simplify(expr(1) > 2, 0);

  test_simplify(min(1, 2), 1);
  test_simplify(max(1, 2), 2);
  test_simplify(min(x, y), min(x, y));
  test_simplify(max(x, y), max(x, y));
  test_simplify(min(x, x), x);
  test_simplify(max(x, x), x);
  test_simplify(min(x / 2, y / 2), min(x, y) / 2);
  test_simplify(max(x / 2, y / 2), max(x, y) / 2);

  test_simplify(x + 0, x);
  test_simplify(x - 0, x);
  test_simplify(0 + x + 0, x);
  test_simplify(x - 0, x);
  test_simplify(1 * x * 1, x);
  test_simplify(x * 0, 0);
  test_simplify(0 * x, 0);
  test_simplify(x / 1, x);

  test_simplify(x / x, x / x);  // Not simplified due to possible division by zero.
  test_simplify(0 / x, 0 / x);  // Not simplified due to possible division by zero.

  test_simplify(((x + 1) - (y - 1)) + 1, x - y + 3);
}

TEST(simplify_let) {
  // lets that should be removed
  test_simplify(let::make(0, y, z), z);          // Dead let
  test_simplify(let::make(0, y * 2, x), y * 2);  // Single use, substitute
  test_simplify(let::make(0, y, x / x), y / y);  // Trivial value, substitute
  test_simplify(let::make(0, 10, x / x), 1);     // Trivial value, substitute

  // lets that should be kept
  test_simplify(let::make(0, y * 2, x / x), let::make(0, y * 2, x / x));  // Non-trivial, used more than once.
}