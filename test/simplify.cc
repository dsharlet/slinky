#include "test.h"
#include "expr.h"
#include "simplify.h"
#include "print.h"
#include "substitute.h"

#include <cassert>

using namespace slinky;

expr x = variable::make(0);
expr y = variable::make(1);

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
}
