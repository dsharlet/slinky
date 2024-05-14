#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cassert>

#include "builder/optimizations.h"
#include "builder/substitute.h"
#include "runtime/expr.h"
#include "runtime/print.h"

namespace slinky {

namespace {

node_context symbols;

var x(symbols, "x");
var y(symbols, "y");
var z(symbols, "z");

MATCHER_P(matches, expected, "") { return match(arg, expected); }

}  // namespace

TEST(optimizations, optimize_symbols) {
  {
    node_context ctx = symbols;
    ASSERT_THAT(optimize_symbols(crop_dim::make(y, x, 0, {0, 0}, check::make(y)), ctx),
        matches(crop_dim::make(x, x, 0, {0, 0}, check::make(x))));
  }

  {
    node_context ctx = symbols;
    ASSERT_THAT(optimize_symbols(crop_dim::make(y, x, 0, {0, 0}, crop_dim::make(z, y, 0, {0, 0}, check::make(z))), ctx),
        matches(crop_dim::make(x, x, 0, {0, 0}, crop_dim::make(x, x, 0, {0, 0}, check::make(x)))));
  }

  {
    node_context ctx = symbols;
    ASSERT_THAT(optimize_symbols(crop_dim::make(x, y, 0, {0, 0}, check::make(y)), ctx),
        matches(crop_dim::make(x, y, 0, {0, 0}, check::make(y))));
  }
}

}  // namespace slinky
