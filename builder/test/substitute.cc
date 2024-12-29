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
  ASSERT_THAT(
      substitute(check::make(y == buffer_min(x, 3)), buffer_min(x, 3), z), matches(check::make(expr(y) == expr(z))));
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

  ASSERT_THAT(substitute(crop_dim::make(x, x, 1, {y, z}, check::make(0 < buffer_min(x, 1))), buffer_min(x, 1), w),
      matches(crop_dim::make(x, x, 1, {max(y, w), z}, check::make(0 < buffer_min(x, 1)))));
  ASSERT_THAT(substitute_bounds(crop_dim::make(x, x, 1, {y, z}, check::make(0 < buffer_min(x, 1))), x, 1, {w, expr()}),
      matches(crop_dim::make(x, x, 1, {max(y, w), z}, check::make(0 < buffer_min(x, 1)))));

  ASSERT_THAT(substitute(crop_dim::make(x, u, 1, {y, z}, check::make(0 < buffer_min(u, 1))), buffer_min(u, 1), w),
      matches(crop_dim::make(x, u, 1, {max(y, w), z}, check::make(0 < w))));
  ASSERT_THAT(substitute_bounds(crop_dim::make(x, u, 1, {y, z}, check::make(0 < buffer_min(u, 1))), u, 1, {w, expr()}),
      matches(crop_dim::make(x, u, 1, {max(y, w), z}, check::make(0 < w))));

  ASSERT_THAT(substitute(crop_dim::make(x, u, 1, {y, z}, check::make(0 < buffer_min(x, 1))), buffer_min(x, 1), w),
      matches(crop_dim::make(x, u, 1, {y, z}, check::make(0 < buffer_min(x, 1)))));
  ASSERT_THAT(substitute_bounds(crop_dim::make(x, u, 1, {y, z}, check::make(0 < buffer_min(x, 1))), x, 1, {w, expr()}),
      matches(crop_dim::make(x, u, 1, {y, z}, check::make(0 < buffer_min(x, 1)))));

  ASSERT_THAT(substitute(slice_dim::make(x, x, 2, 0, check::make(buffer_min(x, 3) == 0)), buffer_min(x, 3), 1),
      matches(slice_dim::make(x, x, 2, 0, check::make(buffer_min(x, 3) == 0))));
  ASSERT_THAT(substitute(slice_dim::make(x, u, 2, 0, check::make(buffer_min(x, 3) == 0)), buffer_min(x, 3), 1),
      matches(slice_dim::make(x, u, 2, 0, check::make(buffer_min(x, 3) == 0))));
  ASSERT_THAT(substitute(slice_dim::make(x, u, 2, 0, check::make(buffer_min(u, 3) == 0)), buffer_min(u, 3), 1),
      matches(slice_dim::make(x, u, 2, 0, check::make(expr(1) == 0))));

  ASSERT_THAT(substitute(slice_dim::make(x, x, 2, 0, check::make(y == buffer_min(x, 3))), y, buffer_max(x, 3)),
      matches(slice_dim::make(x, x, 2, 0, check::make(y == buffer_min(x, 3)))));
  ASSERT_THAT(substitute(slice_dim::make(x, u, 2, 0, check::make(y == buffer_min(x, 3))), y, buffer_max(x, 3)),
      matches(slice_dim::make(x, u, 2, 0, check::make(y == buffer_min(x, 3)))));
  ASSERT_THAT(substitute(slice_dim::make(x, u, 2, 0, check::make(y == buffer_min(x, 3))), y, buffer_max(u, 3)),
      matches(slice_dim::make(x, u, 2, 0, check::make(buffer_max(u, 3) == buffer_min(x, 3)))));

  ASSERT_THAT(substitute(slice_dim::make(x, x, 2, 0, check::make(y == buffer_min(x, 3))), y, buffer_max(x, 2)),
      matches(slice_dim::make(x, x, 2, 0, check::make(expr() == buffer_min(x, 3)))));
  ASSERT_THAT(substitute(slice_dim::make(x, x, 2, 0, check::make(y == buffer_min(x, 3))), y, buffer_max(x, 1)),
      matches(slice_dim::make(x, x, 2, 0, check::make(y == buffer_min(x, 3)))));

  ASSERT_THAT(substitute_bounds(slice_dim::make(x, x, 2, 0, check::make(buffer_min(x, 3) == w)), x, 3, {1, expr()}),
      matches(slice_dim::make(x, x, 2, 0, check::make(buffer_min(x, 3) == w))));
  for (int slice_d = 0; slice_d < 4; ++slice_d) {
    for (int check_d = 0; check_d < 4; ++check_d) {
      expr expected;
      if (check_d >= 3) {
        expected = buffer_min(x, check_d);
      } else {
        expected = check_d < slice_d ? check_d : check_d + 1;
      }
      ASSERT_THAT(substitute_bounds(slice_dim::make(x, x, slice_d, 0, check::make(buffer_min(x, check_d) == w)), x,
                      {{0, expr()}, {1, expr()}, {2, expr()}, {3, expr()}}),
          matches(slice_dim::make(x, x, slice_d, 0, check::make(expected == w))))
          << "slice_d=" << slice_d << ", check_d=" << check_d;
      ASSERT_THAT(substitute_bounds(slice_dim::make(x, x, slice_d, 0, check::make(buffer_min(y, check_d) == w)), y,
                      {{0, expr()}, {1, expr()}, {2, expr()}, {3, expr()}}),
          matches(slice_dim::make(x, x, slice_d, 0, check::make(check_d == w))))
          << "slice_d=" << slice_d << ", check_d=" << check_d;
    }
  }

  ASSERT_THAT(substitute(transpose::make(x, x, {2, 1, 0}, check::make(buffer_min(x, 0) == 0)), buffer_min(x, 0), 1),
      matches(transpose::make(x, x, {2, 1, 0}, check::make(buffer_min(x, 0) == 0))));
  ASSERT_THAT(substitute(transpose::make(x, x, {2, 1, 0}, check::make(buffer_min(x, 0) == 0)), buffer_min(x, 1), 1),
      matches(transpose::make(x, x, {2, 1, 0}, check::make(buffer_min(x, 0) == 0))));
  ASSERT_THAT(substitute(transpose::make(x, x, {2, 1, 0}, check::make(buffer_min(x, 0) == 0)), buffer_min(x, 2), 1),
      matches(transpose::make(x, x, {2, 1, 0}, check::make(expr(1) == 0))));
  ASSERT_THAT(substitute(transpose::make(x, x, {2, 1, 0}, check::make(buffer_min(x, 1) == 0)), buffer_min(x, 0), 1),
      matches(transpose::make(x, x, {2, 1, 0}, check::make(buffer_min(x, 1) == 0))));
  ASSERT_THAT(substitute(transpose::make(x, x, {2, 1, 0}, check::make(buffer_min(x, 1) == 0)), buffer_min(x, 1), 1),
      matches(transpose::make(x, x, {2, 1, 0}, check::make(expr(1) == 0))));
  ASSERT_THAT(substitute(transpose::make(x, x, {2, 1, 0}, check::make(buffer_min(x, 1) == 0)), buffer_min(x, 2), 1),
      matches(transpose::make(x, x, {2, 1, 0}, check::make(buffer_min(x, 1) == 0))));
  ASSERT_THAT(substitute(transpose::make(x, x, {2, 1, 0}, check::make(buffer_min(x, 2) == 0)), buffer_min(x, 0), 1),
      matches(transpose::make(x, x, {2, 1, 0}, check::make(expr(1) == 0))));
  ASSERT_THAT(substitute(transpose::make(x, x, {2, 1, 0}, check::make(buffer_min(x, 2) == 0)), buffer_min(x, 1), 1),
      matches(transpose::make(x, x, {2, 1, 0}, check::make(buffer_min(x, 2) == 0))));
  ASSERT_THAT(substitute(transpose::make(x, x, {2, 1, 0}, check::make(buffer_min(x, 2) == 0)), buffer_min(x, 2), 1),
      matches(transpose::make(x, x, {2, 1, 0}, check::make(buffer_min(x, 2) == 0))));

  ASSERT_THAT(
      substitute(copy_stmt::make(x, {y, z}, w, {y, z}, {}), y, z), matches(copy_stmt::make(x, {y, z}, w, {y, z}, {})));
  ASSERT_THAT(substitute(copy_stmt::make(x, {y}, w, {y}, {}), y, z), matches(copy_stmt::make(x, {y}, w, {y}, {})));
  ASSERT_THAT(substitute(copy_stmt::make(x, {y}, w, {z}, {}), y, u), matches(copy_stmt::make(x, {u}, w, {z}, {})));
}

TEST(substitute, implicit_bounds) {
  ASSERT_THAT(substitute(crop_dim::make(x, u, 0, bounds(y, z), check::make(x)), buffer_min(u, 0), w),
      matches(crop_dim::make(x, u, 0, bounds(max(y, w), z), check::make(x))));
  ASSERT_THAT(substitute(crop_dim::make(x, u, 0, bounds(y, z), check::make(x)), buffer_max(u, 0), w),
      matches(crop_dim::make(x, u, 0, bounds(y, min(z, w)), check::make(x))));

  ASSERT_THAT(substitute(buffer_at(x, buffer_min(y, 0)), y, x), matches(buffer_at(x)));
  ASSERT_THAT(substitute(buffer_at(x, buffer_min(y, 0)), y, z), matches(buffer_at(x, buffer_min(z, 0))));
  ASSERT_THAT(substitute(buffer_at(x), buffer_min(x, 2), y), matches(buffer_at(x, expr(), expr(), expr(y))));
  ASSERT_THAT(substitute(buffer_at(x, expr(), expr(), y), y, buffer_min(x, 2)), matches(buffer_at(x)));
  ASSERT_THAT(substitute(buffer_at(x, y, buffer_min(z, 1)), z, x), matches(buffer_at(x, expr(y))));
}

TEST(substitute_buffer, basic) {
  expr elem_size = 100;
  std::vector<dim_expr> dims = {
      {{0, 1}, 2, 3},
      {{10, 11}, 12, 13},
      {{20, 21}, 22, 23},
  };
  ASSERT_THAT(substitute_buffer(make_buffer::make(z, expr(), buffer_elem_size(x),
                                    {buffer_dim(x, 2), buffer_dim(x, 1), buffer_dim(x, 0)}, stmt()),
                  x, elem_size, dims),
      matches(make_buffer::make(z, expr(), elem_size, {dims[2], dims[1], dims[0]}, stmt())));

  ASSERT_THAT(substitute_buffer(
                  slice_dim::make(x, x, 0, w,
                      make_buffer::make(y, expr(), buffer_elem_size(x), {buffer_dim(x, 0), buffer_dim(x, 1)}, stmt())),
                  x, elem_size, dims),
      matches(slice_dim::make(x, x, 0, w, make_buffer::make(y, expr(), elem_size, {dims[1], dims[2]}, stmt()))));
  ASSERT_THAT(substitute_buffer(
                  slice_dim::make(x, x, 1, w,
                      make_buffer::make(y, expr(), buffer_elem_size(x), {buffer_dim(x, 0), buffer_dim(x, 1)}, stmt())),
                  x, elem_size, dims),
      matches(slice_dim::make(x, x, 1, w, make_buffer::make(y, expr(), elem_size, {dims[0], dims[2]}, stmt()))));
  ASSERT_THAT(substitute_buffer(
                  slice_dim::make(x, x, 2, w,
                      make_buffer::make(y, expr(), buffer_elem_size(x), {buffer_dim(x, 0), buffer_dim(x, 1)}, stmt())),
                  x, elem_size, dims),
      matches(slice_dim::make(x, x, 2, w, make_buffer::make(y, expr(), elem_size, {dims[0], dims[1]}, stmt()))));

  ASSERT_THAT(substitute_buffer(transpose::make(x, x, {2, 1, 0},
                                    make_buffer::make(y, expr(), buffer_elem_size(x),
                                        {buffer_dim(x, 0), buffer_dim(x, 1), buffer_dim(x, 2)}, stmt())),
                  x, elem_size, dims),
      matches(transpose::make(
          x, x, {2, 1, 0}, make_buffer::make(y, expr(), elem_size, {dims[2], dims[1], dims[0]}, stmt()))));
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
