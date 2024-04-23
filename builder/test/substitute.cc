#include <gtest/gtest.h>

#include <cassert>
#include <iostream>

#include "builder/substitute.h"
#include "runtime/expr.h"
#include "runtime/print.h"

namespace slinky {

namespace {

node_context symbols;

var x(symbols, "x");
var y(symbols, "y");
var z(symbols, "z");
var w(symbols, "w");
var u(symbols, "u");

}  // namespace

template <typename T>
void test_substitute(const expr& test, T target, const expr& replacement, const expr& expected) {
  expr result = substitute(test, target, replacement);
  if (!match(result, expected)) {
    std::cout << "substitute failed" << std::endl;
    std::cout << test << std::endl;
    std::cout << "got: " << std::endl;
    std::cout << result << std::endl;
    std::cout << "expected: " << std::endl;
    std::cout << expected << std::endl;
    ASSERT_TRUE(false);
  }
}

template <typename T>
void test_substitute(const stmt& test, T target, const expr& replacement, const stmt& expected) {
  stmt result = substitute(test, target, replacement);
  if (!match(result, expected)) {
    std::cout << "substitute failed" << std::endl;
    std::cout << test << std::endl;
    std::cout << "got: " << std::endl;
    std::cout << result << std::endl;
    std::cout << "expected: " << std::endl;
    std::cout << expected << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST(substitute, basic) {
  test_substitute(x + y, x, z, z + y);
  test_substitute(check::make(y == buffer_min(x, 3)), buffer_min(x, 3), z, check::make(expr(y) == expr(z)));
}

TEST(substitute, shadowed) {
  test_substitute(let::make(x, y, x + z), x, w, let::make(x, y, x + z));
  test_substitute(let::make({{x, 1}, {y, 2}}, z + 1), z, z + w, let::make({{x, 1}, {y, 2}}, z + w + 1));

  test_substitute(crop_dim::make(x, 1, {y, z}, check::make(0 < buffer_min(x, 1))), buffer_min(x, 1), w,
      crop_dim::make(x, 1, {max(y, w), z}, check::make(0 < buffer_min(x, 1))));

  test_substitute(slice_dim::make(x, 2, 0, check::make(buffer_min(x, 3) == 0)), buffer_min(x, 3), 1,
      slice_dim::make(x, 2, 0, check::make(buffer_min(x, 3) == 0)));
  test_substitute(slice_dim::make(x, 2, 0, check::make(y == buffer_min(x, 3))), y, buffer_max(x, 3),
      slice_dim::make(x, 2, 0, check::make(buffer_max(x, 2) == buffer_min(x, 3))));
  test_substitute(slice_dim::make(x, 2, 0, check::make(y == buffer_min(x, 3))), y, buffer_max(x, 2),
      slice_dim::make(x, 2, 0, check::make(expr() == buffer_min(x, 3))));
  test_substitute(slice_dim::make(x, 2, 0, check::make(y == buffer_min(x, 3))), y, buffer_max(x, 1),
      slice_dim::make(x, 2, 0, check::make(buffer_max(x, 1) == buffer_min(x, 3))));

  test_substitute(copy_stmt::make(x, {y, z}, w, {y, z}, {}), y, z, copy_stmt::make(x, {y, z}, w, {y, z}, {}));
  test_substitute(copy_stmt::make(x, {y}, w, {y}, {}), y, z, copy_stmt::make(x, {y}, w, {y}, {}));
  test_substitute(copy_stmt::make(x, {y}, w, {z}, {}), y, u, copy_stmt::make(x, {u}, w, {z}, {}));
}

TEST(substitute, implicit_bounds) {
  test_substitute(crop_dim::make(x, 0, bounds(y, z), check::make(x)), buffer_min(x, 0), w,
      crop_dim::make(x, 0, bounds(max(y, w), z), check::make(x)));
  test_substitute(crop_dim::make(x, 0, bounds(y, z), check::make(x)), buffer_max(x, 0), w,
      crop_dim::make(x, 0, bounds(y, min(z, w)), check::make(x)));
  test_substitute(buffer_at(x), buffer_min(x, 2), y, buffer_at(x, std::vector<expr>{expr(), expr(), expr(y)}));
}

TEST(match, basic) {
  ASSERT_TRUE(match(x, x));
  ASSERT_FALSE(match(x, y));
  ASSERT_FALSE(match(x, 2));
  ASSERT_TRUE(match(x * 2, x * 2));
  ASSERT_FALSE(match(x, x * 2));
  ASSERT_FALSE(match(x + y, x - y));
  ASSERT_TRUE(match(x + y, x + y));
}

}  // namespace slinky
