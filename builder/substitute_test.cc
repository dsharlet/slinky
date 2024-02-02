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
    std::cout << "simplify failed" << std::endl;
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
  test_substitute(check::make(buffer_min(x, 3) == y), buffer_min(x, 3), z, check::make(z == y));
}

TEST(substitute, shadowed) {
  test_substitute(let::make(x.sym(), y, x + z), x.sym(), w, let::make(x.sym(), y, x + z));
  test_substitute(slice_dim::make(x.sym(), 2, 0, check::make(buffer_min(x, 3) == 0)), buffer_min(x, 3), 1,
      slice_dim::make(x.sym(), 2, 0, check::make(buffer_min(x, 3) == 0)));
}

}  // namespace slinky
