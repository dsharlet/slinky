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
  test_substitute(x + y, x.sym(), z, z + y);
  test_substitute(check::make(y == buffer_min(x, 3)), buffer_min(x, 3), z, check::make(y == z));
}

TEST(substitute, shadowed) {
  test_substitute(let::make(x.sym(), y, x + z), x.sym(), w, let::make(x.sym(), y, x + z));
  test_substitute(let::make({{x.sym(), 1}, {y.sym(), 2}}, z + 1), z.sym(), z + w,
      let::make({{x.sym(), 1}, {y.sym(), 2}}, z + w + 1));

  test_substitute(crop_dim::make(x.sym(), 1, {y, z}, check::make(0 < buffer_min(x, 1))), buffer_min(x, 1), w,
      crop_dim::make(x.sym(), 1, {max(y, w), z}, check::make(0 < buffer_min(x, 1))));

  test_substitute(slice_dim::make(x.sym(), 2, 0, check::make(buffer_min(x, 3) == 0)), buffer_min(x, 3), 1,
      slice_dim::make(x.sym(), 2, 0, check::make(buffer_min(x, 3) == 0)));
  test_substitute(slice_dim::make(x.sym(), 2, 0, check::make(y == buffer_min(x, 3))), y, buffer_max(x, 3),
      slice_dim::make(x.sym(), 2, 0, check::make(buffer_max(x, 2) == buffer_min(x, 3))));
  test_substitute(slice_dim::make(x.sym(), 2, 0, check::make(y == buffer_min(x, 3))), y, buffer_max(x, 2),
      slice_dim::make(x.sym(), 2, 0, check::make(expr() == buffer_min(x, 3))));
  test_substitute(slice_dim::make(x.sym(), 2, 0, check::make(y == buffer_min(x, 3))), y, buffer_max(x, 1),
      slice_dim::make(x.sym(), 2, 0, check::make(buffer_max(x, 1) == buffer_min(x, 3))));

  test_substitute(copy_stmt::make(x.sym(), {y, z}, w.sym(), {y.sym(), z.sym()}, {}), y, z,
      copy_stmt::make(x.sym(), {y, z}, w.sym(), {y.sym(), z.sym()}, {}));
  test_substitute(copy_stmt::make(x.sym(), {y}, w.sym(), {y.sym()}, {}), y, z,
      copy_stmt::make(x.sym(), {y}, w.sym(), {y.sym()}, {}));
  test_substitute(copy_stmt::make(x.sym(), {y}, w.sym(), {z.sym()}, {}), y, u,
      copy_stmt::make(x.sym(), {u}, w.sym(), {z.sym()}, {}));
}

TEST(substitute, implicit_bounds) {
  test_substitute(crop_dim::make(x.sym(), 0, bounds(y, z), check::make(x)), buffer_min(x, 0), w,
      crop_dim::make(x.sym(), 0, bounds(max(y, w), z), check::make(x)));
  test_substitute(crop_dim::make(x.sym(), 0, bounds(y, z), check::make(x)), buffer_max(x, 0), w,
      crop_dim::make(x.sym(), 0, bounds(y, min(z, w)), check::make(x)));
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

void test_wildcards(const expr& pattern, const expr& target, const expr& replacement, const expr& expected) {
  symbol_map<expr> matches;
  if (!match(pattern, target, matches)) {
    std::cout << "match failed" << std::endl;
    std::cout << "pattern: " << pattern << std::endl;
    std::cout << "target: " << target << std::endl;
    ASSERT_TRUE(false);
  }
  expr result = substitute(replacement, matches);
  if (!match(result, expected)) {
    std::cout << "match failed" << std::endl;
    std::cout << "pattern: " << pattern << std::endl;
    std::cout << "target: " << target << std::endl;
    std::cout << "result: " << result << std::endl;
    std::cout << "expected: " << expected << std::endl;
  }
}

TEST(match, wildcards) {
  test_wildcards(x, y, x * 2, y * 2);
  test_wildcards(x - y, z - 2, x + y, z + 2);
  test_wildcards(x - y, x * 2 - y * 3, x + y, x * 2 + y * 3);
}

}  // namespace slinky
