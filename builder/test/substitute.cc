#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cassert>

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

MATCHER_P(matches, expected, "") { return match(arg, expected); }

}  // namespace

TEST(substitute, basic) {
  ASSERT_THAT(substitute(x + y, x, z), matches(z + y));
  ASSERT_THAT(substitute(crop_dim::make(x, y, 0, {0, 0}, call_stmt::make(nullptr, {}, {x}, {})), y, z),
      matches(crop_dim::make(x, z, 0, interval_expr{0, 0}, call_stmt::make(nullptr, {}, {x}, {}))));
  ASSERT_THAT(substitute(crop_dim::make(y, z, 0, {0, 0}, call_stmt::make(nullptr, {x}, {y}, {})), x, w),
      matches(crop_dim::make(y, z, 0, {0, 0}, call_stmt::make(nullptr, {w}, {y}, {}))));
  ASSERT_THAT(substitute(crop_dim::make(
                             y, y, 0, {0, 0}, crop_dim::make(y, y, 0, {0, 0}, call_stmt::make(nullptr, {x}, {y}, {}))),
                  x, w),
      matches(
          crop_dim::make(y, y, 0, {0, 0}, crop_dim::make(y, y, 0, {0, 0}, call_stmt::make(nullptr, {w}, {y}, {})))));
  ASSERT_THAT(substitute_buffer(buffer_stride(x, 0), x, expr(), {}), matches(buffer_stride(x, 0)));
  ASSERT_THAT(substitute_buffer(buffer_stride(x, 0), x, expr(), {{{0, 1}, 2, 3}}), matches(2));
  ASSERT_THAT(substitute_buffer(buffer_stride(x, 0), x, expr(), {dim_expr()}), matches(expr()));
}

TEST(substitute, shadowed) {
  ASSERT_THAT(substitute(let::make(x, y, x + z), x, w), matches(let::make(x, y, x + z)));

  ASSERT_THAT(
      substitute(let::make({{x, 1}, {y, 2}}, z + 1), z, z + w), matches(let::make({{x, 1}, {y, 2}}, z + w + 1)));

  ASSERT_THAT(substitute(slice_dim::make(x, x, 2, 0, check::make(y == buffer_min(x, 3))), y, buffer_max(x, 3)),
      matches(slice_dim::make(x, x, 2, 0, check::make(y == buffer_min(x, 3)))));
  ASSERT_THAT(substitute(slice_dim::make(x, u, 2, 0, check::make(y == buffer_min(x, 3))), y, buffer_max(x, 3)),
      matches(slice_dim::make(x, u, 2, 0, check::make(y == buffer_min(x, 3)))));
  ASSERT_THAT(substitute(slice_dim::make(x, u, 2, 0, check::make(y == buffer_min(x, 3))), y, buffer_max(u, 3)),
      matches(slice_dim::make(x, u, 2, 0, check::make(buffer_max(u, 3) == buffer_min(x, 3)))));

  ASSERT_THAT(
      substitute(copy_stmt::make(x, {y, z}, w, {y, z}, {}), y, z), matches(copy_stmt::make(x, {y, z}, w, {y, z}, {})));
  ASSERT_THAT(substitute(copy_stmt::make(x, {y}, w, {y}, {}), y, z), matches(copy_stmt::make(x, {y}, w, {y}, {})));
  ASSERT_THAT(substitute(copy_stmt::make(x, {y}, w, {z}, {}), y, u), matches(copy_stmt::make(x, {u}, w, {z}, {})));
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
