#include <gmock/gmock.h>
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

MATCHER_P(matches, x, "") { return match(arg, x); }

TEST(substitute, basic) {
  ASSERT_THAT(substitute(x + y, x, z), matches(z + y));
  ASSERT_THAT(
      substitute(check::make(y == buffer_min(x, 3)), buffer_min(x, 3), z), matches(check::make(expr(y) == expr(z))));
}

TEST(substitute, shadowed) {
  ASSERT_THAT(substitute(let::make(x, y, x + z), x, w), matches(let::make(x, y, x + z)));
  ASSERT_THAT(
      substitute(let::make({{x, 1}, {y, 2}}, z + 1), z, z + w), matches(let::make({{x, 1}, {y, 2}}, z + w + 1)));

  ASSERT_THAT(substitute(crop_dim::make(x, 1, {y, z}, check::make(0 < buffer_min(x, 1))), buffer_min(x, 1), w),
      matches(crop_dim::make(x, 1, {max(y, w), z}, check::make(0 < buffer_min(x, 1)))));

  ASSERT_THAT(substitute(slice_dim::make(x, 2, 0, check::make(buffer_min(x, 3) == 0)), buffer_min(x, 3), 1),
      matches(slice_dim::make(x, 2, 0, check::make(buffer_min(x, 3) == 0))));
  ASSERT_THAT(substitute(slice_dim::make(x, 2, 0, check::make(y == buffer_min(x, 3))), y, buffer_max(x, 3)),
      matches(slice_dim::make(x, 2, 0, check::make(buffer_max(x, 2) == buffer_min(x, 3)))));
  ASSERT_THAT(substitute(slice_dim::make(x, 2, 0, check::make(y == buffer_min(x, 3))), y, buffer_max(x, 2)),
      matches(slice_dim::make(x, 2, 0, check::make(expr() == buffer_min(x, 3)))));
  ASSERT_THAT(substitute(slice_dim::make(x, 2, 0, check::make(y == buffer_min(x, 3))), y, buffer_max(x, 1)),
      matches(slice_dim::make(x, 2, 0, check::make(buffer_max(x, 1) == buffer_min(x, 3)))));

  ASSERT_THAT(
      substitute(copy_stmt::make(x, {y, z}, w, {y, z}, {}), y, z), matches(copy_stmt::make(x, {y, z}, w, {y, z}, {})));
  ASSERT_THAT(substitute(copy_stmt::make(x, {y}, w, {y}, {}), y, z), matches(copy_stmt::make(x, {y}, w, {y}, {})));
  ASSERT_THAT(substitute(copy_stmt::make(x, {y}, w, {z}, {}), y, u), matches(copy_stmt::make(x, {u}, w, {z}, {})));
}

TEST(substitute, implicit_bounds) {
  ASSERT_THAT(substitute(crop_dim::make(x, 0, bounds(y, z), check::make(x)), buffer_min(x, 0), w),
      matches(crop_dim::make(x, 0, bounds(max(y, w), z), check::make(x))));
  ASSERT_THAT(substitute(crop_dim::make(x, 0, bounds(y, z), check::make(x)), buffer_max(x, 0), w),
      matches(crop_dim::make(x, 0, bounds(y, min(z, w)), check::make(x))));
  ASSERT_THAT(
      substitute(buffer_at(x), buffer_min(x, 2), y), matches(buffer_at(x, std::vector<expr>{expr(), expr(), expr(y)})));
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
