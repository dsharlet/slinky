#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cassert>
#include <chrono>

#include "base/test/seeded_test.h"
#include "builder/simplify.h"
#include "builder/substitute.h"
#include "builder/test/simplify/expr_generator.h"
#include "runtime/buffer.h"
#include "runtime/evaluate.h"
#include "runtime/expr.h"
#include "runtime/print.h"

namespace slinky {

// Hackily get at this function in evaluate.cc that we don't want to put in the public API.
void dump_context_for_expr(
    std::ostream&, const eval_context&, const expr& = expr(), const node_context* symbols = nullptr);

// And this one too.
const node_context* set_default_print_context(const node_context* ctx);

namespace {

node_context symbols;

var x(symbols, "x");
var y(symbols, "y");
var z(symbols, "z");
var w(symbols, "w");
var u(symbols, "u");
var v(symbols, "v");
var b0(symbols, "b0");
var b1(symbols, "b1");
var b2(symbols, "b2");
var b3(symbols, "b3");
var b4(symbols, "b4");
var b5(symbols, "b5");
var b6(symbols, "b6");

// Make test failures easier to read.
auto _ = []() {
  set_default_print_context(&symbols);
  return 0;
}();

MATCHER_P(matches, x, "") { return match(arg, x); }

}  // namespace

template <typename T>
void dump_symbol_map(std::ostream& s, const symbol_map<T>& m) {
  for (std::size_t n = 0; n < m.size(); ++n) {
    const std::optional<T>& value = m[n];
    if (value) {
      s << "  " << symbols.name(var(n)) << " = " << *value << std::endl;
    }
  }
}

TEST(simplify, basic) {
  ASSERT_THAT(simplify(expr() == 1), matches(expr()));
  ASSERT_THAT(simplify(expr(1) + 2), matches(3));
  ASSERT_THAT(simplify(expr(1) - 2), matches(-1));
  ASSERT_THAT(simplify(expr(1) < 2), matches(1));
  ASSERT_THAT(simplify(expr(1) > 2), matches(0));
  ASSERT_THAT(simplify(negative_infinity() + 3), matches(negative_infinity()));
  ASSERT_THAT(simplify(3 + negative_infinity()), matches(negative_infinity()));
  ASSERT_THAT(simplify(positive_infinity() + positive_infinity()), matches(positive_infinity()));
  ASSERT_THAT(simplify(positive_infinity() + negative_infinity()), matches(indeterminate()));
  ASSERT_THAT(simplify(positive_infinity() * positive_infinity()), matches(positive_infinity()));
  ASSERT_THAT(simplify(positive_infinity() * negative_infinity()), matches(negative_infinity()));
  ASSERT_THAT(simplify(abs(negative_infinity())), matches(positive_infinity()));

  ASSERT_THAT(simplify(min(1, 2)), matches(1));
  ASSERT_THAT(simplify(max(1, 2)), matches(2));
  ASSERT_THAT(simplify(min(x, y)), matches(min(x, y)));
  ASSERT_THAT(simplify(max(x, y)), matches(max(x, y)));
  ASSERT_THAT(simplify(min(x, x)), matches(x));
  ASSERT_THAT(simplify(max(x, x)), matches(x));
  ASSERT_THAT(simplify(min(x / 2, y / 2)), matches(min(y, x) / 2));
  ASSERT_THAT(simplify(max(x / 2, y / 2)), matches(max(y, x) / 2));
  ASSERT_THAT(simplify(min(negative_infinity(), x)), matches(negative_infinity()));
  ASSERT_THAT(simplify(max(negative_infinity(), x)), matches(x));
  ASSERT_THAT(simplify(min(positive_infinity(), x)), matches(x));
  ASSERT_THAT(simplify(max(positive_infinity(), x)), matches(positive_infinity()));
  ASSERT_THAT(simplify(min(min(x, 7), min(y, 7))), matches(min(min(x, y), 7)));
  ASSERT_THAT(simplify(min(min(x, 7), min(7, y))), matches(min(min(x, y), 7)));
  ASSERT_THAT(simplify(min(min(7, x), min(y, 7))), matches(min(min(x, y), 7)));
  ASSERT_THAT(simplify(min(min(7, x), min(7, y))), matches(min(min(x, y), 7)));

  ASSERT_THAT(simplify(x + 0), matches(x));
  ASSERT_THAT(simplify(x - 0), matches(x));
  ASSERT_THAT(simplify(0 + x + 0), matches(x));
  ASSERT_THAT(simplify(x - 0), matches(x));
  ASSERT_THAT(simplify(1 * x * 1), matches(x));
  ASSERT_THAT(simplify(x * 0), matches(0));
  ASSERT_THAT(simplify(0 * x), matches(0));
  ASSERT_THAT(simplify(x / 1), matches(x));

  ASSERT_THAT(simplify(x / x), matches(x != 0));
  ASSERT_THAT(simplify(0 / x), matches(0));

  ASSERT_THAT(simplify(((x + 1) - (y - 1)) + 1), matches(x - y + 3));

  ASSERT_THAT(simplify(select(x, y, y)), matches(y));
  ASSERT_THAT(simplify(select(x == x, y, z)), matches(y));
  ASSERT_THAT(simplify(select(x != x, y, z)), matches(z));

  ASSERT_THAT(simplify(x && false), matches(false));
  ASSERT_THAT(simplify(x || true), matches(true));
  ASSERT_THAT(simplify(false && x), matches(false));
  ASSERT_THAT(simplify(true || x), matches(true));

  ASSERT_THAT(simplify(x < x + 1), matches(true));
  ASSERT_THAT(simplify(x - 1 < x + 1), matches(true));
  ASSERT_THAT(simplify(min(x + 1, z) < x + 2), matches(true));

  ASSERT_THAT(simplify(abs(abs(x))), matches(abs(x)));
  ASSERT_THAT(simplify(max(abs(x), 0)), matches(abs(x)));
  ASSERT_THAT(simplify(min(abs(x), 0)), matches(0));

  ASSERT_THAT(simplify(select(z == z, x, y)), matches(x));
  ASSERT_THAT(simplify(select(z != z, x, y)), matches(y));

  ASSERT_THAT(simplify(select(x, y + 1, y + 2)), matches(y + select(x, 1, 2)));
  ASSERT_THAT(simplify(select(x, 1, 2) + 1), matches(select(x, 2, 3)));

  ASSERT_THAT(simplify(max(select(x, y, z), select(x, y, w))), matches(select(x, y, max(w, z))));
  ASSERT_THAT(simplify(max(select(x, y, z), select(x, w, z))), matches(select(x, max(w, y), z)));
  ASSERT_THAT(simplify(min(select(x, y, z), select(x, y, w))), matches(select(x, y, min(w, z))));
  ASSERT_THAT(simplify(min(select(x, y, z), select(x, w, z))), matches(select(x, min(w, y), z)));
  ASSERT_THAT(simplify((select(x, y, z) < select(x, y, w))), matches(((expr(z) < expr(w)) && !x)));

  ASSERT_THAT(simplify(select(x == 1, y, select(x == 1, z, w))), matches(select(x == 1, y, w)));
  ASSERT_THAT(simplify(select(x == 1, select(x == 1, y, z), w)), matches(select(x == 1, y, w)));

  ASSERT_THAT(simplify(select(x == y, x, y)), matches(y));

  ASSERT_THAT(simplify(max((x + -1), select((1 < x), (max(min(x, 128), 118) + -1), 0))),
      matches(select(1 < x, max(x, 118), 1) + -1));
  ASSERT_THAT(simplify(min((y + -1), max((x + -1), max(min(x, (((z / 16) * 16) + 16)) + -1, select((1 < x), 117, 0))))),
      matches((min(y, select((1 < x), max(x, 118), 1)) + -1)));

  ASSERT_THAT(simplify(min(y, z) <= y + 1), matches(true));
  ASSERT_THAT(simplify((min(x, y + -1) <= min(x, y))), matches(true));

  ASSERT_THAT(simplify(and_then(expr(false), x)), matches(false));
  ASSERT_THAT(simplify(and_then(expr(true), x)), matches(boolean(x)));
  ASSERT_THAT(simplify(and_then(x, expr(false))), matches(and_then(x, expr(false))));
  ASSERT_THAT(simplify(and_then(x, expr(true))), matches(boolean(x)));
  ASSERT_THAT(simplify(or_else(expr(true), x)), matches(true));
  ASSERT_THAT(simplify(or_else(expr(false), x)), matches(boolean(x)));
  ASSERT_THAT(simplify(or_else(x, expr(false))), matches(boolean(x)));

  ASSERT_THAT(simplify((x != y) < 1), matches(y == x));
  ASSERT_THAT(simplify((x && y) < 2), matches(true));
  ASSERT_THAT(simplify((x && y) < 0), matches(false));
  ASSERT_THAT(simplify(-1 < (x || y)), matches(true));
  ASSERT_THAT(simplify(2 < (x || y)), matches(false));

  ASSERT_THAT(simplify(min(x < y, 0)), matches(0));
  ASSERT_THAT(simplify(max(x && y, 0)), matches(x && y));
  ASSERT_THAT(simplify(min(!x, 1)), matches(!x));
  ASSERT_THAT(simplify(max(x == y, 1)), matches(1));

  ASSERT_THAT(simplify(buffer_fold_factor(x, 2) >= 1), matches(true));

  ASSERT_THAT(simplify(0 <= x % 4), matches(true));
  ASSERT_THAT(simplify(4 <= x % 4), matches(false));
  ASSERT_THAT(simplify((y / 4) * 4 <= y - 4), matches(false));
  ASSERT_THAT(simplify((y / 4) * 4 <= y - 3), matches(3 <= y % 4));
  ASSERT_THAT(simplify((y / 4) * 4 <= y - 1), matches(boolean(y % 4)));
  ASSERT_THAT(simplify((y / 4) * 4 <= y), matches(true));

  ASSERT_THAT(simplify(x % -4 <= 1), matches(x % -4 <= 1));

  ASSERT_THAT(
      simplify(max(((y + 14) / 16) * 2 + 1, (y + 6) / 8) <= max(((y + 15) / 16) * 2 + 1, (y + 7) / 8)), matches(true));

  ASSERT_THAT(simplify((x < 1) != 0), matches(x < 1));

  ASSERT_THAT(simplify(select(x == 3 && y == 2, x == 3 && y == 2, true)), matches(true));
  ASSERT_THAT(simplify(select(x == 3 && y == 2, x == 3, true)), matches(true));
  ASSERT_THAT(simplify(select(x == 3 || y == 2, x == 3, y == 2)), matches(x == 3));
  ASSERT_THAT(simplify(select(x == 3 || y == 2, false, y == 2)), matches(false));
  ASSERT_THAT(simplify(select(!(x == 3) && y, x == 3, false)), matches(false));
  ASSERT_THAT(simplify(select(x != 3 && y, x == 3, false)), matches(false));

  ASSERT_THAT(simplify(select(x == 0 && y == 0, x == 0 && y == 0, true)), matches(true));
  ASSERT_THAT(simplify(select(x == 0 && y == 0, x == 0, true)), matches(true));
  ASSERT_THAT(simplify(select(x == 0 || y == 0, x == 0, y == 0)), matches(x == 0));
  ASSERT_THAT(simplify(select(x == 0 || y == 0, false, y == 0)), matches(false));
  ASSERT_THAT(simplify(select(!(x == 0) && y, x == 0, false)), matches(false));
  ASSERT_THAT(simplify(select(x != 0 && y, x == 0, false)), matches(false));

  ASSERT_THAT(simplify(select(x, expr(), 2) == 1), matches(select(x, expr(), false)));
  ASSERT_THAT(simplify(!select(x, expr(), true)), matches(select(x, expr(), false)));

  ASSERT_THAT(simplify(min(select(x, 0, y) + 4, select(x, expr(), min(y, 113) + 4))),
      matches(select(x, expr(), min(y, 113) + 4)));

  ASSERT_THAT(simplify(min(x + 64, max(min(x, 113) + 5, min(y, 128)))),
      matches(min(min(x + 64, max(y, min(x, 113) + 5)), 128)));

  ASSERT_THAT(simplify(select(x, (y - 4), 2) + 4), matches(select(x, y, 6)));
  ASSERT_THAT(simplify(select(x, y + 3, 5) - 1), matches(select(x, y, 2) + 2));
  ASSERT_THAT(simplify(min(x + 2, select(y, 3, z + 4)) - 1), matches((min(x + -2, select(y, -1, z)) + 3)));

  ASSERT_THAT(simplify(select((y <= 0), select((x <= 0), z, x), z)), matches(select(0 < x && y <= 0, x, z)));

  ASSERT_THAT(simplify(crop_dim::make(y, x, 1, {expr(), expr()}, call_stmt::make(nullptr, {}, {y}, {}))),
      matches(call_stmt::make(nullptr, {}, {x}, {})));
  ASSERT_THAT(simplify(crop_buffer::make(y, x, {}, call_stmt::make(nullptr, {}, {y}, {}))),
      matches(call_stmt::make(nullptr, {}, {x}, {})));
  ASSERT_THAT(simplify(slice_buffer::make(y, x, {}, call_stmt::make(nullptr, {}, {y}, {}))),
      matches(call_stmt::make(nullptr, {}, {x}, {})));

  ASSERT_THAT(simplify(max(select(z <= 0, -1, select(1 <= y, min(x, z + -1), 0)) + 1, select((1 <= y), z, 0))),
      matches(select((1 <= y), max(z, 0), (0 < z))));

  ASSERT_THAT(simplify(min(min(x / 16 + -2, y) + 1, min(y + 1, x / 16)) + 2), matches(min(y, x / 16 + -2) + 3));

  ASSERT_THAT(simplify((x / 2) * 2 + x % 2), matches(x));
  ASSERT_THAT(simplify((x / 5) * 5 + x % 5), matches(x));

  ASSERT_THAT(simplify(((x / 2) * 2 - (x % 2 + -1) / -2)), matches(x + -1));
}

TEST(simplify, staircase) {
  ASSERT_THAT(simplify(min(((x + 7) / 8) * 8, x)), matches(x));
  ASSERT_THAT(simplify(max(((x + 7) / 8) * 8, x)), matches(((x + 7) / 8) * 8));

  ASSERT_THAT(simplify(max((x / 8) * 8, x)), matches(x));
  ASSERT_THAT(simplify(min((x / 8) * 8, x)), matches((x / 8) * 8));
}

TEST(simplify, optional) {
  ASSERT_THAT(simplify(x == x), matches(true));
  ASSERT_THAT(simplify(x + y == x), matches(y == 0));
  ASSERT_THAT(simplify(x + y == x + z), matches(expr(y) == z));
  ASSERT_THAT(simplify(y + (x + w) == x + z), matches(w + y == z));
  ASSERT_THAT(simplify(y + (x + w) == u + (x + z)), matches(w + y == z + u));
  ASSERT_THAT(simplify(x < x + y), matches(0 < y));
  ASSERT_THAT(simplify(x + y < x), matches(y < 0));
  ASSERT_THAT(simplify(x < z + (x + y)), matches(0 < y + z));
  ASSERT_THAT(simplify(z + (x + y) < x), matches(y + z < 0));
  ASSERT_THAT(simplify(x + y < x + z), matches(expr(y) < z));
  ASSERT_THAT(simplify(w + (x + y) < x + z), matches(y + w < z));
  ASSERT_THAT(simplify(x + z < w + (x + y)), matches(z < y + w));
  ASSERT_THAT(simplify(x || (y || x)), matches(x || y));
  ASSERT_THAT(simplify(x || (y || (z || x))), matches(x || (y || z)));
  ASSERT_THAT(simplify(x || (y || (z || (w || x)))), matches(x || (y || (z || w))));
}

TEST(simplify, let) {
  // lets that should be removed
  ASSERT_THAT(simplify(let::make(x, y, z)), matches(z));                      // Dead let
  ASSERT_THAT(simplify(let::make(x, y, (x + 1) / x)), matches((y + 1) / y));  // Trivial value, substitute
  ASSERT_THAT(simplify(let::make(x, 10, x / x)), matches(1));                 // Trivial value, substitute

  ASSERT_THAT(simplify(let_stmt::make(x, y, loop::make(z, loop::serial, bounds(0, 3), 1, check::make(x + z)))),
      matches(loop::make(z, loop::serial, bounds(0, 3), 1, check::make(y + z))));  // Trivial value, substitute

  // lets that should be kept
  ASSERT_THAT(simplify(let::make(x, y * 2, (x + 1) / x)),
      matches(let::make(x, y * 2, (x + 1) / x)));  // Non-trivial, used more than once.

  ASSERT_THAT(simplify(let_stmt::make(x, y * w, loop::make(z, loop::serial, bounds(0, 3), 1, check::make(x + z)))),
      matches(let_stmt::make(
          x, y * w, loop::make(z, loop::serial, bounds(0, 3), 1, check::make(x + z)))));  // Non-trivial, used in loop

  ASSERT_THAT(simplify(let_stmt::make(x, y * w, block::make({check::make(x > 0), check::make(x < 10)}))),
      matches(let_stmt::make(
          x, y * w, block::make({check::make(x > 0), check::make(x < 10)}))));  // Non-trivial, used twice

  // Compound lets with dependencies between let values.
  ASSERT_THAT(simplify(let::make({{x, y}, {z, x}}, z)), matches(y));
  ASSERT_THAT(simplify(let::make({{x, y}, {z, x * 2}}, z)), matches(let::make(z, y * 2, z)));
  ASSERT_THAT(simplify(let::make({{x, y * 2}, {z, x}}, z)), matches(let::make(x, y * 2, x)));
  ASSERT_THAT(simplify(let::make({{x, y * 2}, {z, y}}, z)), matches(y));
  ASSERT_THAT(simplify(let::make({{x, y}, {z, (x + 1) / x}}, (z + 1) / z)),
      matches(let::make({{z, (y + 1) / y}}, (z + 1) / z)));

  // Duplicate lets
  ASSERT_THAT(simplify(let::make({{x, y * 2}, {z, y * 2}}, x + z)), matches(let::make(x, y * 2, x * 2)));

  // Nested lets
  ASSERT_THAT(
      simplify(let::make(x, y * 2, let::make(z, w + 2, x + z))), matches(let::make({{x, y * 2}, {z, w + 2}}, x + z)));
}

TEST(simplify, loop) {
  auto make_call = [](var out) { return call_stmt::make(nullptr, {}, {out}, {}); };

  ASSERT_THAT(simplify(loop::make(
                  x, loop::serial, buffer_bounds(b0, 0), 1, crop_dim::make(b1, b0, 0, point(x), make_call(b1)))),
      matches(make_call(b0)));
  ASSERT_THAT(simplify(loop::make(
                  x, loop::serial, buffer_bounds(b3, 0), 1, crop_dim::make(b1, b0, 0, point(x), make_call(b1)))),
      matches(crop_dim::make(b1, b0, 0, buffer_bounds(b3, 0), make_call(b1))));
  for (expr min : {expr(-1), expr(0), expr(1), expr(3), expr(z)}) {
    ASSERT_THAT(simplify(loop::make(x, loop::serial, bounds(min, buffer_max(b0, 0)), 1,
                    crop_dim::make(b1, b0, 0, point(x), make_call(b1)))),
        matches(crop_dim::make(b1, b0, 0, bounds(min, expr()), make_call(b1))));
  }
  ASSERT_THAT(simplify(loop::make(
                  x, loop::serial, buffer_bounds(b0, 0), y, crop_dim::make(b1, b0, 0, point(x), make_call(b1)))),
      matches(
          loop::make(x, loop::serial, buffer_bounds(b0, 0), y, crop_dim::make(b1, b0, 0, point(x), make_call(b1)))));
  ASSERT_THAT(simplify(loop::make(x, loop::serial, buffer_bounds(b0, 0), y,
                  crop_dim::make(b1, b0, 0, min_extent(x, y), make_call(b1)))),
      matches(make_call(b0)));
  ASSERT_THAT(simplify(loop::make(x, loop::serial, buffer_bounds(b3, 0), y,
                  crop_dim::make(b1, b0, 0, bounds(x, min(x + y - 1, buffer_max(b3, 0))), make_call(b1)))),
      matches(crop_dim::make(b1, b0, 0, buffer_bounds(b3, 0), make_call(b1))));
  ASSERT_THAT(simplify(loop::make(x, loop::serial, bounds(0, buffer_max(b3, 0)), y,
                  crop_dim::make(b1, b0, 0, bounds(x, min(x + y - 1, buffer_max(b3, 0))), make_call(b1)))),
      matches(crop_dim::make(b1, b0, 0, bounds(0, buffer_max(b3, 0)), make_call(b1))));
}

TEST(simplify, siblings) {
  auto make_call = [](var out) { return call_stmt::make(nullptr, {}, {out}, {}); };
  auto make_crop_x = [](var out, var in, int dim, const stmt& body) {
    return crop_dim::make(out, in, dim, point(x), body);
  };

  ASSERT_THAT(simplify(make_crop_x(b1, b0, 0, block::make({make_call(b1), make_call(b0)}))),
      matches(block::make({make_crop_x(b1, b0, 0, make_call(b1)), make_call(b0)})));
  ASSERT_THAT(simplify(make_crop_x(b1, b0, 0, block::make({make_call(b0), make_call(b1)}))),
      matches(block::make({make_call(b0), make_crop_x(b1, b0, 0, make_call(b1))})));
  ASSERT_THAT(
      simplify(make_crop_x(b1, b0, 0, block::make({make_call(b1), make_call(b0), make_call(b1), make_call(b0)}))),
      matches(block::make({make_crop_x(b1, b0, 0, make_call(b1)), make_call(b0), make_crop_x(b1, b0, 0, make_call(b1)),
          make_call(b0)})));
  ASSERT_THAT(
      simplify(make_crop_x(b1, b0, 0, block::make({make_call(b0), make_call(b1), make_call(b1), make_call(b0)}))),
      matches(block::make(
          {make_call(b0), make_crop_x(b1, b0, 0, block::make({make_call(b1), make_call(b1)})), make_call(b0)})));
}

TEST(simplify, licm) {
  // Use parallel loops so loops of one call don't get rewritten to a single call.
  auto make_loop_x = [](const stmt& body) { return loop::make(x, loop::parallel, bounds(0, 10), 1, body); };
  auto make_loop_y = [](const stmt& body) { return loop::make(y, loop::parallel, bounds(0, 10), 1, body); };
  auto make_call = [](var in, var out) { return call_stmt::make(nullptr, {in}, {out}, {}); };
  auto make_crop_x = [](var b, int dim, const stmt& body) { return crop_dim::make(b, b, dim, point(x), body); };
  auto make_crop_y = [](var b, int dim, const stmt& body) { return crop_dim::make(b, b, dim, point(y), body); };

  // One call doesn't depend on the loop.
  ASSERT_THAT(simplify(make_loop_x(make_call(b0, b1))), matches(make_call(b0, b1)));
  // Two calls don't depend on the loop.
  ASSERT_THAT(simplify(make_loop_x(block::make({
                  make_call(b0, b1),
                  make_call(b0, b2),
              }))),
      matches(block::make({
          make_call(b0, b1),
          make_call(b0, b2),
      })));
  // Last call depends on the loop, first call does not.
  ASSERT_THAT(simplify(make_loop_x(block::make({
                  make_call(b0, b1),
                  make_crop_x(b2, 0, make_call(b0, b2)),
              }))),
      matches(block::make({
          make_call(b0, b1),
          make_loop_x(make_crop_x(b2, 0, make_call(b0, b2))),
      })));
  // A call in the middle of the loop depends on the loop.
  ASSERT_THAT(simplify(make_loop_x(block::make({
                  make_call(b0, b1),
                  make_crop_x(b2, 0, make_call(b0, b2)),
                  make_call(b0, b3),
              }))),
      matches(block::make({
          make_call(b0, b1),
          make_call(b0, b3),
          make_loop_x(make_crop_x(b2, 0, make_call(b0, b2))),
      })));
  // A call in the middle of the loop does not depend on the loop, but does depend on the first call.
  ASSERT_THAT(simplify(make_loop_x(block::make({
                  make_crop_x(b1, 0, make_call(b0, b1)),
                  make_call(b1, b2),
                  make_crop_x(b3, 0, make_call(b0, b3)),
              }))),
      matches(make_loop_x(block::make({
          make_crop_x(b1, 0, make_call(b0, b1)),
          make_call(b1, b2),
          make_crop_x(b3, 0, make_call(b0, b3)),
      }))));
  // A call in the middle of the loop does not depend on the loop, but does depend on the first call, and we know that
  // the first call doesn't write a folded buffer.
  ASSERT_THAT(simplify(allocate::make(b1, memory_type::heap, 1, {{{0, 10}, 1, dim::unfolded}},
                  make_loop_x(block::make({
                      make_crop_x(b1, 0, make_call(b0, b1)),
                      make_call(b1, b2),
                      make_call(b1, b3),
                      make_crop_x(b4, 0, make_call(b0, b4)),
                  })))),
      matches(block::make({
          allocate::make(b1, memory_type::heap, 1, {{{0, 10}, 1, expr()}},
              block::make({
                  make_call(b0, b1),
                  make_call(b1, b2),
                  make_call(b1, b3),
              })),
          make_loop_x(make_crop_x(b4, 0, make_call(b0, b4))),
      })));
  // A call at the end of the loop does not depend on the loop, but each call depends on the previous, and the first
  // call writes a folded buffer.
  ASSERT_THAT(simplify(allocate::make(b1, memory_type::heap, 1, {{{0, 10}, 1, dim::unfolded}},
                  make_loop_x(block::make({
                      make_crop_x(b1, 0, make_call(b0, b1)),
                      make_crop_x(b2, 0, make_call(b1, b2)),
                      make_call(b2, b3),
                  })))),
      matches(allocate::make(b1, memory_type::heap, 1, {{{0, 10}, 1, expr()}},
          make_loop_x(block::make({
              make_crop_x(b1, 0, make_call(b0, b1)),
              make_crop_x(b2, 0, make_call(b1, b2)),
              make_call(b2, b3),
          })))));
  // Same as above, but with another loop invariant stmt that does get lifted.
  ASSERT_THAT(simplify(allocate::make(b1, memory_type::heap, 1, {{{0, 10}, 1, dim::unfolded}},
                  make_loop_x(block::make({
                      make_crop_x(b1, 0, make_call(b0, b1)),
                      make_crop_x(b2, 0, make_call(b1, b2)),
                      make_call(b0, b4),
                      make_call(b2, b3),
                  })))),
      matches(block::make({
          make_call(b0, b4),
          allocate::make(b1, memory_type::heap, 1, {{{0, 10}, 1, expr()}},
              make_loop_x(block::make({
                  make_crop_x(b1, 0, make_call(b0, b1)),
                  make_crop_x(b2, 0, make_call(b1, b2)),
                  make_call(b2, b3),
              }))),
      })));
  // A call at the end of the loop that is loop invariant.
  ASSERT_THAT(simplify(make_loop_x(block::make({
                  make_crop_x(b1, 0, make_call(b0, b1)),
                  make_crop_x(b2, 0, make_call(b1, b2)),
                  make_call(b0, b3),
              }))),
      matches(block::make({
          make_call(b0, b3),
          make_loop_x(block::make({
              make_crop_x(b1, 0, make_call(b0, b1)),
              make_crop_x(b2, 0, make_call(b1, b2)),
          })),
      })));
  // A nested loop.
  ASSERT_THAT(simplify(make_loop_y(make_crop_y(b2, 1,
                  make_loop_x(block::make({
                      make_call(b0, b1),
                      make_crop_x(b2, 0, make_call(b0, b2)),
                  }))))),
      matches(block::make({
          make_call(b0, b1),
          make_loop_y(make_crop_y(b2, 1, make_loop_x(make_crop_x(b2, 0, make_call(b0, b2))))),
      })));

  // A call that is loop invariant, but with a transitive dependency on a loop variant.
  ASSERT_THAT(simplify(make_loop_x(block::make({
                  make_crop_x(b1, 0, make_call(b0, b1)),
                  make_call(b1, b2),
                  make_call(b2, b3),
                  make_crop_x(b4, 0, make_call(b3, b4)),
              }))),
      matches(make_loop_x(block::make({
          make_crop_x(b1, 0, make_call(b0, b1)),
          make_call(b1, b2),
          make_call(b2, b3),
          make_crop_x(b4, 0, make_call(b3, b4)),
      }))));

  // A call that is loop invariant, but with a transitive dependency on a loop variant.
  ASSERT_THAT(simplify(make_loop_x(block::make({
                  make_crop_x(b1, 0, make_call(b0, b1)),
                  make_call(b5, b6),
                  make_call(b1, b2),
                  make_call(b2, b3),
                  make_crop_x(b4, 0, make_call(b3, b4)),
              }))),
      matches(block::make({
          make_call(b5, b6),
          make_loop_x(block::make({
              make_crop_x(b1, 0, make_call(b0, b1)),
              make_call(b1, b2),
              make_call(b2, b3),
              make_crop_x(b4, 0, make_call(b3, b4)),
          })),
      })));
}

TEST(simplify, bounds) {
  ASSERT_THAT(simplify(min(x, y), {{x, {y, z}}}), matches(y));
  ASSERT_THAT(simplify(max(x, y), {{x, {y, z}}}), matches(x));
  ASSERT_THAT(simplify(min(x, z), {{x, {y, z}}}), matches(x));
  ASSERT_THAT(simplify(max(x, z), {{x, {y, z}}}), matches(z));
  ASSERT_THAT(simplify(min(x + 1, y), {{x, {y, z}}}), matches(y));
  ASSERT_THAT(simplify(min(x, y - 1), {{x, {y, z}}}), matches(y + -1));
  ASSERT_THAT(simplify(max(x + 2, y), {{x, {y, z}}}), matches(x + 2));
  ASSERT_THAT(simplify(max(x, y - 2), {{x, {y, z}}}), matches(x));
  ASSERT_THAT(simplify(min(x, y), {{x, {y + 1, z}}}), matches(y));
  ASSERT_THAT(simplify(min(x, y), {{x, {y + abs(w), z}}}), matches(y));

  ASSERT_THAT(simplify(expr(x) == y, {{x, {y, y}}}), matches(true));

  ASSERT_THAT(simplify(loop::make(x, loop::serial, bounds(y - 2, z), 2, check::make(y - 2 <= x))), matches(stmt()));
  ASSERT_THAT(simplify(loop::make(x, loop::serial, min_extent(x, z), z, check::make(y))), matches(check::make(y)));

  // Tricky case because want to use the bounds of x but the value of y.
  symbol_map<interval_expr> xy_bounds = {{x, {0, y}}, {y, {z, w}}};
  ASSERT_THAT(simplify(min(x, y + 1), xy_bounds), matches(x));

  ASSERT_THAT(simplify(let::make(x, select(1 < y, y, max(z, 1)), max(x, 0))),
      matches(let::make(x, select(1 < y, y, max(z, 1)), x)));

  ASSERT_THAT(simplify(let::make({{x, select(1 < y, y, max(z, 1))}, {w, select(1 < x, x, max(u, 1))}}, max(w, 0))),
      matches(let::make({{x, select(1 < y, y, max(z, 1))}, {w, select(1 < x, x, max(u, 1))}}, w)));
}

TEST(simplify, buffer_bounds) {
  auto decl_bounds = [](var buf, std::vector<interval_expr> bounds, stmt body) {
    std::vector<dim_expr> dims;
    for (auto i : bounds) {
      dims.push_back({i});
    }
    return allocate::make(buf, memory_type::heap, 1, dims, body);
  };
  auto use_buffer = [](var b) { return call_stmt::make(nullptr, {}, {b}, {}); };
  auto use_buffers = [](std::vector<var> bs) { return call_stmt::make(nullptr, {}, std::move(bs), {}); };
  ASSERT_THAT(simplify(decl_bounds(b0, {buffer_bounds(b1, 0)},
                  crop_dim::make(b2, b0, 0, buffer_bounds(b1, 0) & bounds(x, y), use_buffer(b2)))),
      matches(decl_bounds(b0, {{buffer_bounds(b1, 0)}}, crop_dim::make(b2, b0, 0, bounds(x, y), use_buffer(b2)))));

  ASSERT_THAT(simplify(decl_bounds(x, {bounds(2, 3)}, check::make(buffer_min(x, 0) == 2))), matches(stmt()));
  ASSERT_THAT(simplify(decl_bounds(
                  x, {bounds(2, 3)}, crop_dim::make(x, x, 0, bounds(1, 4), check::make(buffer_min(x, 0) == 2)))),
      matches(stmt()));
  ASSERT_THAT(simplify(decl_bounds(x, {{bounds(y, z)}},
                  crop_dim::make(x, x, 0, bounds(y - 1, z + 1), check::make(buffer_min(x, 0) == y && buffer_at(x))))),
      matches(decl_bounds(x, {{bounds(y, z)}}, check::make(buffer_at(x)))));
  ASSERT_THAT(simplify(decl_bounds(x, {{buffer_bounds(b0, 0) + 2}},
                  crop_dim::make(x, x, 0, bounds(expr(), min(z, buffer_max(b0, 0)) + 2), use_buffer(x)))),
      matches(
          decl_bounds(x, {{buffer_bounds(b0, 0) + 2}}, crop_dim::make(x, x, 0, bounds(expr(), z + 2), use_buffer(x)))));
  ASSERT_THAT(simplify(decl_bounds(x, {{buffer_bounds(b0, 0) + 2}},
                  crop_dim::make(x, x, 0, bounds(expr(), max(z, buffer_max(b0, 0)) + 2), use_buffer(x)))),
      matches(decl_bounds(x, {{buffer_bounds(b0, 0) + 2}}, use_buffer(x))));

  ASSERT_THAT(
      simplify(decl_bounds(b0, {{x, y}}, crop_dim::make(b1, b0, 0, bounds(max(x, w), min(y, z)), use_buffer(b1)))),
      matches(decl_bounds(b0, {{x, y}}, crop_dim::make(b1, b0, 0, bounds(w, z), use_buffer(b1)))));
  ASSERT_THAT(simplify(decl_bounds(x, {{x, y}}, crop_dim::make(x, x, 0, bounds(min(x, w), max(y, z)), use_buffer(x)))),
      matches(decl_bounds(x, {{x, y}}, use_buffer(x))));

  ASSERT_THAT(simplify(decl_bounds(b0, {{0, x + -1}},
                  decl_bounds(b1, {{0, x + -1}},
                      crop_dim::make(b2, b0, 0, {expr(), buffer_max(b1, 0)},
                          crop_dim::make(b3, b0, 0, {expr(), buffer_max(b2, 0)}, use_buffers({b1, b3})))))),
      matches(decl_bounds(b0, {{0, x + -1}}, decl_bounds(b1, {{0, x + -1}}, use_buffers({b1, b0})))));

  ASSERT_THAT(simplify(let_stmt::make(x, select(1 < y, y, max(w, 1)),
                  decl_bounds(b0, {{0, x + -1}},
                      decl_bounds(b1, {{0, x + -1}},
                          crop_dim::make(b2, b0, 0, {expr(), buffer_max(b1, 0)},
                              crop_dim::make(b3, b0, 0, {expr(), buffer_max(b2, 0)}, use_buffers({b1, b3}))))))),
      matches(let_stmt::make(x, select(1 < y, y, max(w, 1)),
          decl_bounds(b0, {{0, x + -1}}, decl_bounds(b1, {{0, x + -1}}, use_buffers({b1, b0}))))));

  ASSERT_THAT(simplify(decl_bounds(b0, {{0, select(1 <= x, ((y + 15) / 16) * 16, 16) + -1}},
                  decl_bounds(b1, {{0, select(1 <= x, y + -1, 0)}},
                      crop_dim::make(b2, b0, 0, {expr(), (buffer_max(b0, 0) / 16) * 16 + 15}, use_buffer(b2))))),
      matches(decl_bounds(b0, {{0, select(1 <= x, ((y + 15) / 16) * 16, 16) + -1}}, use_buffer(b0))));

  ASSERT_THAT(simplify(decl_bounds(b0, {{0, max(x, 0)}},
                  decl_bounds(b1, {{0, ((max(x, 0) / 16) * 16) + 15}, {0, 20}},
                      crop_dim::make(b2, b0, 0, {expr(), buffer_max(b1, 0)}, use_buffers({b2}))))),
      matches(decl_bounds(b0, {{0, max(x, 0)}}, use_buffers({b0}))));

  ASSERT_THAT(simplify(decl_bounds(b0, {{0, max(x, 0)}},
                  decl_bounds(b1, {{0, ((max(x, 0) / 16) * 16) + 15}, {0, 20}},
                      crop_dim::make(b3, b1, 1, {1, 10},
                          crop_dim::make(b2, b0, 0, {expr(), buffer_max(b3, 0)}, use_buffers({b2, b3})))))),
      matches(decl_bounds(b0, {{0, max(x, 0)}},
          decl_bounds(b1, {{0, ((max(x, 0) / 16) * 16) + 15}, {0, 20}},
              crop_dim::make(b3, b1, 1, {1, 10}, use_buffers({b0, b3}))))));

  ASSERT_THAT(simplify(loop::make(x, loop::parallel, {0, 256}, 16,
                  crop_dim::make(b1, b0, 0, {(x / 16) * 16, (x / 16) * 16 + 15}, use_buffer(b1)))),
      matches(loop::make(x, loop::parallel, {0, 256}, 16, crop_dim::make(b1, b0, 0, {x, x + 15}, use_buffer(b1)))));

  ASSERT_THAT(simplify(decl_bounds(b0, {{0, select(1 < z, 127, 15)}},
                  loop::make(x, loop::parallel, {0, select((1 < z), 117, 0)}, y,
                      crop_dim::make(
                          b1, b0, 0, {x, ((min((x + y), select((1 < z), 118, 1)) + 15) / 16) * 16}, use_buffer(b1))))),
      matches(decl_bounds(b0, {{0, select(1 < z, 127, 15)}},
          loop::make(x, loop::parallel, {0, select((1 < z), 117, 0)}, y,
              crop_dim::make(b1, b0, 0, {x, (((x + y) + 15) / 16) * 16}, use_buffer(b1))))));

  ASSERT_THAT(simplify(loop::make(x, loop::serial, {0, y}, z,
                  crop_dim::make(b1, b0, 0, {select(x <= 0, x, expr()), y}, use_buffer(b1)))),
      matches(loop::make(
          x, loop::serial, {0, y}, z, crop_dim::make(b1, b0, 0, {select(x <= 0, 0, expr()), y}, use_buffer(b1)))));
}

TEST(simplify, crop_not_needed) {
  ASSERT_THAT(simplify(crop_dim::make(b0, b0, 1, {x, y}, check::make(b0))), matches(check::make(b0)));
  ASSERT_THAT(simplify(crop_dim::make(b1, b0, 1, {y, z}, check::make(b1))), matches(check::make(b0)));
  ASSERT_THAT(simplify(crop_dim::make(b1, b0, 1, {y, z}, check::make(b0))), matches(check::make(b0)));
}

TEST(simplify, clone) {
  ASSERT_THAT(simplify(clone_buffer::make(b1, b0, check::make(b0))), matches(check::make(b0)));
  ASSERT_THAT(simplify(clone_buffer::make(b1, b0, check::make(b1))), matches(check::make(b0)));
  ASSERT_THAT(simplify(clone_buffer::make(b1, b0, check::make(b0 && b1))), matches(check::make(b0)));
  ASSERT_THAT(
      simplify(clone_buffer::make(b1, b0, clone_buffer::make(b2, b1, check::make(b2)))), matches(check::make(b0)));
  ASSERT_THAT(simplify(clone_buffer::make(b1, b0, clone_buffer::make(b2, b1, check::make(b0 && b2)))),
      matches(check::make(b0)));

  ASSERT_THAT(
      simplify(clone_buffer::make(b1, b0, transpose::make(b2, b1, {1, 0}, call_stmt::make(nullptr, {}, {b0, b2}, {})))),
      matches(transpose::make(b2, b0, {1, 0}, call_stmt::make(nullptr, {}, {b0, b2}, {}))));

  // Clone should be substituted.
  ASSERT_THAT(
      simplify(clone_buffer::make(y, x, crop_dim::make(z, y, 0, {0, 0}, call_stmt::make(nullptr, {w}, {z}, {})))),
      matches(crop_dim::make(z, x, 0, {0, 0}, call_stmt::make(nullptr, {w}, {z}, {}))));

  ASSERT_THAT(simplify(crop_dim::make(x, u, 1, point(10),
                  clone_buffer::make(y, x,
                      make_buffer::make(z, buffer_at(w), buffer_elem_size(w), {buffer_dim(y, 0), buffer_dim(y, 1)},
                          call_stmt::make(nullptr, {}, {x, z}, {}))))),
      matches(crop_dim::make(x, u, 1, point(10),
          make_buffer::make(z, buffer_at(w), buffer_elem_size(w),
              {buffer_dim(u, 0), {buffer_bounds(x, 1), buffer_stride(u, 1), buffer_fold_factor(u, 1)}},
              call_stmt::make(nullptr, {}, {x, z}, {})))));
}

TEST(simplify, allocate) {
  // Pull statements that don't use the buffer out of allocate nodes.
  ASSERT_THAT(simplify(allocate::make(x, memory_type::heap, 1, {{bounds(2, 3), 4, 5}},
                  block::make({check::make(y), check::make(buffer_at(x)), check::make(z)}))),
      matches(block::make({check::make(y),
          allocate::make(x, memory_type::heap, 1, {{bounds(2, 3), 4, expr()}}, check::make(buffer_at(x))),
          check::make(z)})));

  // Make sure clone_buffer doesn't hide uses of buffers or bounds.
  ASSERT_THAT(simplify(allocate::make(x, memory_type::heap, 1, {{bounds(2, 3), 4, 5}},
                  block::make({check::make(y), clone_buffer::make(w, x, check::make(buffer_at(w))), check::make(z)}))),
      matches(block::make({check::make(y),
          allocate::make(x, memory_type::heap, 1, {{bounds(2, 3), 4, expr()}}, check::make(buffer_at(x))),
          check::make(z)})));
}

TEST(simplify, slice_of_crop) {
  stmt body = call_stmt::make(nullptr, {}, {b3}, {});

  // Unchanged.
  ASSERT_THAT(simplify(crop_dim::make(b1, b0, 1, {0, 3}, slice_dim::make(b3, b1, 2, 0, body))),
      matches(crop_dim::make(b1, b0, 1, {0, 3}, slice_dim::make(b3, b1, 2, 0, body))));

  // Unchanged.
  ASSERT_THAT(
      simplify(crop_buffer::make(b1, b0, {{0, 3}, {0, 4}, {0, 5}}, slice_buffer::make(b3, b1, {{1}, {2}}, body))),
      matches(crop_buffer::make(b1, b0, {{0, 3}, {0, 4}, {0, 5}}, slice_buffer::make(b3, b1, {{1}, {2}}, body))));

  // Test support for slice_dim.
  ASSERT_THAT(simplify(crop_dim::make(b1, b0, 1, {0, 3}, slice_dim::make(b3, b1, 1, 0, body))),
      matches(slice_dim::make(b3, b0, 1, 0, body)));

  // Test support for slice_buffer.
  ASSERT_THAT(simplify(crop_dim::make(b1, b0, 1, {0, 3}, slice_buffer::make(b3, b1, {{}, 0}, body))),
      matches(slice_dim::make(b3, b0, 1, 0, body)));

  // Test support for slicing multiple dimensions.
  ASSERT_THAT(simplify(crop_dim::make(b1, b0, 1, {0, 3}, slice_buffer::make(b3, b1, {{1}, {0}}, body))),
      matches(slice_buffer::make(b3, b0, {{1}, {0}}, body)));
}

TEST(simplify, crop) {
  stmt body = call_stmt::make(nullptr, {}, {b2}, {});

  ASSERT_THAT(simplify(crop_dim::make(b2, b0, 0, {buffer_min(b0, 0), x}, body)),
      matches(crop_dim::make(b2, b0, 0, {expr(), x}, body)));
  ASSERT_THAT(simplify(crop_dim::make(b2, b0, 0, {x, buffer_max(b0, 0)}, body)),
      matches(crop_dim::make(b2, b0, 0, {x, expr()}, body)));

  ASSERT_THAT(simplify(crop_dim::make(b1, b0, 0, {x, y}, crop_dim::make(b2, b1, 0, {z, w}, body))),
      matches(crop_dim::make(b2, b0, 0, {max(x, z), min(y, w)}, body)));
  ASSERT_THAT(simplify(crop_dim::make(b1, b0, 0, {x, y}, crop_dim::make(b2, b1, 1, {z, w}, body))),
      matches(crop_buffer::make(b2, b0, {{x, y}, {z, w}}, body)));
  ASSERT_THAT(simplify(crop_dim::make(b1, b0, 0, {x, y}, crop_dim::make(b2, b1, 2, {z, w}, body))),
      matches(crop_buffer::make(b2, b0, {{x, y}, {}, {z, w}}, body)));
  ASSERT_THAT(simplify(crop_dim::make(b1, b0, 0, {x, y}, crop_dim::make(b2, b1, 2, {z, w}, body))),
      matches(crop_buffer::make(b2, b0, {{x, y}, {}, {z, w}}, body)));

  ASSERT_THAT(simplify(crop_buffer::make(b1, b0, {{x, y}}, crop_buffer::make(b2, b1, {{z, w}}, body))),
      matches(crop_dim::make(b2, b0, 0, {max(x, z), min(y, w)}, body)));
  ASSERT_THAT(simplify(crop_buffer::make(b1, b0, {{x, y}}, crop_buffer::make(b2, b1, {{z, w}, {u, v}}, body))),
      matches(crop_buffer::make(b2, b0, {{max(x, z), min(y, w)}, {u, v}}, body)));
  ASSERT_THAT(simplify(crop_buffer::make(b1, b0, {{z, w}, {u, v}}, crop_buffer::make(b2, b1, {{x, y}}, body))),
      matches(crop_buffer::make(b2, b0, {{max(z, x), min(w, y)}, {u, v}}, body)));
  ASSERT_THAT(
      simplify(crop_buffer::make(b1, b0, {{x, y}, {z, w}}, crop_buffer::make(b2, b1, {{}, {z, w}, {u, v}}, body))),
      matches(crop_buffer::make(b2, b0, {{x, y}, {z, w}, {u, v}}, body)));

  // Nested crops of the same buffer.
  ASSERT_THAT(simplify(crop_dim::make(
                  b1, b0, 0, {x, y}, crop_dim::make(b2, b0, 0, {x, y}, call_stmt::make(nullptr, {}, {b1, b2}, {})))),
      matches(crop_dim::make(b1, b0, 0, {x, y}, call_stmt::make(nullptr, {}, {b1, b1}, {}))));
  ASSERT_THAT(simplify(clone_buffer::make(b1, b0,
                  crop_buffer::make(b2, b1, {buffer_bounds(b0, 0)},
                      crop_dim::make(b3, b1, 0, {x, y},
                          crop_dim::make(b4, b2, 0, {x, y}, call_stmt::make(nullptr, {}, {b3, b4}, {})))))),
      matches(crop_dim::make(b3, b0, 0, {x, y}, call_stmt::make(nullptr, {}, {b3, b3}, {}))));

  ASSERT_THAT(simplify(block::make({
                  check::make(buffer_min(b0, 0) == 0),
                  crop_dim::make(b2, b0, 0,
                      {select(buffer_max(b0, 0) < 0, buffer_max(b0, 0) + 32, 0), buffer_max(b0, 0) + 31}, body),
              })),
      matches(block::make({
          check::make(buffer_min(b0, 0) == 0),
          crop_dim::make(b2, b0, 0, {select(buffer_max(b0, 0) < 0, buffer_max(b0, 0), -32) + 32, expr()}, body),
      })));
}

TEST(simplify, make_buffer) {
  stmt body = call_stmt::make(nullptr, {}, {b1}, {});
  auto make_slice = [body](var sym, var src, std::vector<expr> at, std::vector<dim_expr> dims) {
    for (int i = static_cast<int>(at.size()) - 1; i >= 0; --i) {
      if (at[i].defined()) {
        dims.erase(dims.begin() + i);
      }
    }
    return make_buffer::make(sym, buffer_at(src, at), buffer_elem_size(src), dims, body);
  };

  auto make_crop = [body](var sym, var src, std::vector<expr> at, std::vector<interval_expr> bounds,
                       std::vector<dim_expr> dims) {
    for (int d = 0; d < static_cast<int>(bounds.size()); ++d) {
      if (bounds[d].min.defined()) dims[d].bounds.min = bounds[d].min;
      if (bounds[d].max.defined()) dims[d].bounds.max = bounds[d].max;
    }
    return make_buffer::make(sym, buffer_at(src, at), buffer_elem_size(src), dims, body);
  };

  // Slices
  ASSERT_THAT(simplify(make_slice(b1, b0, {}, buffer_dims(b0, 0))), matches(transpose::make_truncate(b1, b0, 0, body)));
  ASSERT_THAT(simplify(make_slice(b1, b0, {}, buffer_dims(b0, 1))), matches(transpose::make_truncate(b1, b0, 1, body)));
  ASSERT_THAT(simplify(make_slice(b1, b0, {}, buffer_dims(b0, 3))), matches(transpose::make_truncate(b1, b0, 3, body)));
  ASSERT_THAT(simplify(make_slice(b1, b0, {x}, buffer_dims(b0, 1))),
      matches(slice_dim::make(b1, b0, 0, x, transpose::make_truncate(b1, b1, 0, body))));
  ASSERT_THAT(simplify(make_slice(b1, b0, {x}, buffer_dims(b0, 2))),
      matches(slice_dim::make(b1, b0, 0, x, transpose::make_truncate(b1, b1, 1, body))));
  ASSERT_THAT(simplify(make_slice(b1, b0, {x, y}, buffer_dims(b0, 2))),
      matches(slice_buffer::make(b1, b0, {x, y}, transpose::make_truncate(b1, b1, 0, body))));
  ASSERT_THAT(simplify(make_slice(b1, b0, {expr(), y}, buffer_dims(b0, 2))),
      matches(slice_dim::make(b1, b0, 1, y, transpose::make_truncate(b1, b1, 1, body))));
  ASSERT_THAT(simplify(make_slice(b1, b0, {expr(), y}, buffer_dims(b0, 3))),
      matches(slice_dim::make(b1, b0, 1, y, transpose::make_truncate(b1, b1, 2, body))));

  // Not slices
  ASSERT_THAT(
      simplify(make_slice(b1, b0, {}, buffer_dims(b1, 1))), matches(make_slice(b1, b0, {}, buffer_dims(b1, 1))));
  ASSERT_THAT(simplify(make_crop(b1, b0, {}, {buffer_bounds(b0, 0)}, buffer_dims(b1, 1))),
      matches(make_crop(b1, b0, {}, {buffer_bounds(b0, 0)}, buffer_dims(b1, 1))));

  // Crops
  ASSERT_THAT(
      simplify(make_crop(b1, b0, {}, {}, buffer_dims(b0, 0))), matches(transpose::make_truncate(b1, b0, 0, body)));
  ASSERT_THAT(
      simplify(make_crop(b1, b0, {}, {}, buffer_dims(b0, 1))), matches(transpose::make_truncate(b1, b0, 1, body)));
  ASSERT_THAT(simplify(make_crop(b1, b0, {x}, {{x, y}}, buffer_dims(b0, 1))),
      matches(crop_dim::make(b1, b0, 0, {x, y}, transpose::make_truncate(b1, b1, 1, body))));
  ASSERT_THAT(simplify(make_crop(b1, b0, {x}, {{x, y}}, buffer_dims(b0, 2))),
      matches(crop_dim::make(b1, b0, 0, {x, y}, transpose::make_truncate(b1, b1, 2, body))));
  ASSERT_THAT(simplify(make_crop(b1, b0, {x, z}, {{x, y}, {z, w}}, buffer_dims(b0, 2))),
      matches(crop_buffer::make(b1, b0, {{x, y}, {z, w}}, transpose::make_truncate(b1, b1, 2, body))));
  ASSERT_THAT(simplify(make_crop(b1, b0, {expr(), z}, {{expr(), expr()}, {z, w}}, buffer_dims(b0, 2))),
      matches(crop_dim::make(b1, b0, 1, {z, w}, transpose::make_truncate(b1, b1, 2, body))));
  ASSERT_THAT(simplify(make_crop(b1, b0, {expr(), z}, {{expr(), expr()}, {z, w}}, buffer_dims(b0, 3))),
      matches(crop_dim::make(b1, b0, 1, {z, w}, transpose::make_truncate(b1, b1, 3, body))));
  ASSERT_THAT(simplify(make_crop(b1, b0, {expr(), z}, {{expr(), expr()}, {z, w}}, buffer_dims(b0, 3))),
      matches(crop_dim::make(b1, b0, 1, {z, w}, transpose::make_truncate(b1, b1, 3, body))));

  // Not crops
  ASSERT_THAT(simplify(make_crop(b1, b0, {}, {{x, y}}, buffer_dims(b0, 1))),
      matches(make_crop(b1, b0, {}, {{x, y}}, buffer_dims(b0, 1))));
  ASSERT_THAT(simplify(make_crop(b1, b0, {y}, {{x, y}}, buffer_dims(b0, 1))),
      matches(make_crop(b1, b0, {y}, {{x, y}}, buffer_dims(b0, 1))));

  // Transpose
  ASSERT_THAT(simplify(make_buffer::make(b1, buffer_at(b0), buffer_elem_size(b0), {buffer_dim(b0, 2)}, body)),
      matches(transpose::make(b1, b0, {2}, body)));
  ASSERT_THAT(simplify(make_buffer::make(
                  b1, buffer_at(b0), buffer_elem_size(b0), {buffer_dim(b0, 0), buffer_dim(b0, 2)}, body)),
      matches(transpose::make(b1, b0, {0, 2}, body)));

  ASSERT_THAT(
      simplify(allocate::make(b0, memory_type::heap, 4, {{{0, 10}, {}, {}}, {{0, 20}, {}, {}}, {{0, 30}, {}, {}}},
          make_buffer::make(b1, buffer_at(b0), buffer_elem_size(b0),
              {{{0, 10}, buffer_stride(b0, 0), {}}, {{0, 20}, buffer_stride(b0, 1), {}}},
              call_stmt::make(nullptr, {}, {b0, b1}, {})))),
      matches(allocate::make(b0, memory_type::heap, 4, {{{0, 10}, {}, {}}, {{0, 20}, {}, {}}, {{0, 30}, {}, {}}},
          transpose::make(b1, b0, {0, 1}, call_stmt::make(nullptr, {}, {b0, b1}, {})))));

  ASSERT_THAT(simplify(transpose::make(b1, b2, {1, 0},
                  make_buffer::make(b0, buffer_at(b1), buffer_elem_size(b1), {{{0, 10}, 2}},
                      call_stmt::make(nullptr, {}, {b0}, {})))),
      matches(make_buffer::make(
          b0, buffer_at(b2), buffer_elem_size(b2), {{{0, 10}, 2}}, call_stmt::make(nullptr, {}, {b0}, {}))));

  // The same buffer
  ASSERT_THAT(simplify(allocate::make(b0, memory_type::heap, 4, {{{0, 255}, {}, {}}, {{0, 0}, {}, {}}},
                  make_buffer::make(b1, buffer_at(b0), buffer_elem_size(b0), {buffer_dim(b0, 0), buffer_dim(b0, 1)},
                      call_stmt::make(nullptr, {}, {b1}, {})))),
      matches(allocate::make(
          b0, memory_type::heap, 4, {{{0, 255}, {}, {}}, {{0, 0}, {}, {}}}, call_stmt::make(nullptr, {}, {b0}, {}))));
  ASSERT_THAT(simplify(allocate::make(b0, memory_type::heap, 4, {{{0, 255}, {}, {}}, {{0, 0}, {}, {}}},
                  make_buffer::make(b1, buffer_at(b0), buffer_elem_size(b0), {buffer_dim(b0, 0), buffer_dim(b0, 1)},
                      call_stmt::make(nullptr, {}, {b1}, {})))),
      matches(allocate::make(
          b0, memory_type::heap, 4, {{{0, 255}, {}, {}}, {{0, 0}, {}, {}}}, call_stmt::make(nullptr, {}, {b0}, {}))));
  ASSERT_THAT(simplify(allocate::make(b0, memory_type::heap, 4, {{{0, 255}, {}, {}}, {{0, 0}, {}, {}}},
                  make_buffer::make(b1, buffer_at(b0), buffer_elem_size(b0), {buffer_dim(b0, 0), {{0, 0}, 0, {}}},
                      call_stmt::make(nullptr, {}, {b1}, {})))),
      matches(allocate::make(
          b0, memory_type::heap, 4, {{{0, 255}, {}, {}}, {{0, 0}, {}, {}}}, call_stmt::make(nullptr, {}, {b0}, {}))));
  ASSERT_THAT(
      simplify(allocate::make(b0, memory_type::heap, 4, {{{0, 255}, {}, {}}, {{0, 0}, {}, {}}, {{0, 0}, {}, {}}},
          make_buffer::make(b1, buffer_at(b0), buffer_elem_size(b0),
              {buffer_dim(b0, 0), {{0, 0}, 0, {}}, {{0, 0}, 0, {}}}, call_stmt::make(nullptr, {}, {b1}, {})))),
      matches(allocate::make(b0, memory_type::heap, 4, {{{0, 255}, {}, {}}, {{0, 0}, {}, {}}, {{0, 0}, {}, {}}},
          call_stmt::make(nullptr, {}, {b0}, {}))));
}

TEST(simplify, transpose) {
  ASSERT_THAT(simplify(transpose::make(
                  b1, b0, {2, 1, 0}, transpose::make(b2, b1, {2, 1, 0}, call_stmt::make(nullptr, {}, {b2}, {})))),
      matches(transpose::make(b2, b0, {0, 1, 2}, call_stmt::make(nullptr, {}, {b2}, {}))));
  ASSERT_THAT(simplify(transpose::make(
                  b1, b0, {3, 2, 1}, transpose::make(b2, b1, {1, 0}, call_stmt::make(nullptr, {}, {b2}, {})))),
      matches(transpose::make(b2, b0, {2, 3}, call_stmt::make(nullptr, {}, {b2}, {}))));

  ASSERT_THAT(simplify(crop_buffer::make(b1, b0, {{x, y}, {z, w}},
                  transpose::make_truncate(b2, b1, 3, call_stmt::make(nullptr, {}, {b2}, {})))),
      matches(crop_buffer::make(
          b1, b0, {{x, y}, {z, w}}, transpose::make_truncate(b2, b1, 3, call_stmt::make(nullptr, {}, {b2}, {})))));

  ASSERT_THAT(simplify(crop_buffer::make(
                  b1, b0, {{x, y}, {z, w}}, transpose::make(b2, b1, {1, 0}, check::make(buffer_max(b2, 0) <= w)))),
      matches(stmt()));
  ASSERT_THAT(simplify(crop_buffer::make(
                  b1, b0, {{x, y}, {z, w}}, transpose::make(b2, b1, {1, 0}, check::make(buffer_max(b2, 1) <= w)))),
      matches(crop_buffer::make(
          b1, b0, {{x, y}, {z, w}}, transpose::make(b2, b1, {1, 0}, check::make(buffer_max(b2, 1) <= w)))));
}

TEST(simplify, knowledge) {
  ASSERT_THAT(simplify(select(expr(x) < y, min(x, y), z)), matches(select(expr(x) < y, x, z)));
  ASSERT_THAT(simplify(select(expr(x) < y, max(x, y), z)), matches(select(expr(x) < y, y, z)));
  ASSERT_THAT(simplify(select(expr(x) < y, z, min(x, y))), matches(select(expr(x) < y, z, y)));
  ASSERT_THAT(simplify(select(expr(x) < y, z, max(x, y))), matches(select(expr(x) < y, z, x)));
  ASSERT_THAT(simplify(select(expr(x) <= y, min(x, y), z)), matches(select(expr(x) <= y, x, z)));
  ASSERT_THAT(simplify(select(expr(x) <= y, max(x, y), z)), matches(select(expr(x) <= y, y, z)));
  ASSERT_THAT(simplify(select(expr(x) <= y, z, min(x, y))), matches(select(expr(x) <= y, z, y)));
  ASSERT_THAT(simplify(select(expr(x) <= y, z, max(x, y))), matches(select(expr(x) <= y, z, x)));

  ASSERT_THAT(simplify(select(expr(x) == y, z, select(expr(x) == y, 2, w))), matches(select(expr(x) == y, z, w)));

  ASSERT_THAT(simplify(select(x <= 1, y, min(x, 1))), matches(select(x <= 1, y, 1)));
  ASSERT_THAT(simplify(select(x > 0 && x < 4, max(x, 1), y)), matches(select(x > 0 && x < 4, x, y)));

  ASSERT_THAT(simplify(select(x < 5, y, abs(x))), matches(select(x < 5, y, x)));
  ASSERT_THAT(simplify(select(x < -3, abs(x), y)), matches(select(x < -3, -x, y)));

  ASSERT_THAT(simplify(let::make(x, (y / 8) * 8, (x / 8) * 8)), matches(let::make(x, (y / 8) * 8, x)));
  ASSERT_THAT(simplify(let::make(x, (y / 8) * 8, (x / 8) * 16)), matches(let::make(x, (y / 8) * 8, x * 2)));
  ASSERT_THAT(simplify(let::make(x, (y / 8) * 8, (x / 8) * 4)), matches(let::make(x, (y / 8) * 8, x / 2)));
  ASSERT_THAT(simplify(let::make(x, (y / 8) * 8, (x / 4) * 4)), matches(let::make(x, (y / 8) * 8, x)));
  ASSERT_THAT(simplify(let::make(x, (y / 8) * 8, (x / 16) * 16)), matches(let::make(x, (y / 8) * 8, (x / 16) * 16)));
  ASSERT_THAT(simplify(let::make(x, (y / 8) * 8, (x / 3) * 3)), matches(let::make(x, (y / 8) * 8, (x / 3) * 3)));

  ASSERT_THAT(simplify(block::make({check::make(x % 8 == 0), check::make((x / 8) * 8 == x)})),
      matches(check::make(x % 8 == 0)));
  ASSERT_THAT(simplify(block::make({check::make(x % 2 == 0), check::make(x % 3 == 0), check::make(x % 6 == 0)})),
      matches(block::make({check::make(x % 2 == 0), check::make(x % 3 == 0)})));
  ASSERT_THAT(simplify(let::make(x, y % 2 == 0, x && y % 2 == 1)), matches(false));

  ASSERT_THAT(simplify(block::make({check::make(x % 2 == 0), check::make((x / 2) * 2 == x)})),
      matches(check::make(x % 2 == 0)));
  ASSERT_THAT(simplify(block::make({check::make(x % 2 == 1), check::make((x / 2) * 2 != x)})),
      matches(check::make(x % 2 == 1)));
  ASSERT_THAT(simplify(block::make({check::make(x % 6 == 4), check::make((x / 2) * 2 == x)})),
      matches(check::make(x % 6 == 4)));
  ASSERT_THAT(simplify(block::make({check::make(x % 6 == 3), check::make((x / 2) * 2 != x)})),
      matches(check::make(x % 6 == 3)));

  ASSERT_THAT(
      simplify(block::make({check::make(3 <= max(x, y)), check::make(3 <= x)})), matches(check::make(3 <= max(x, y))));
  ASSERT_THAT(simplify(block::make({check::make(3 <= min(x, y)), check::make(3 <= x)})),
      matches(block::make({check::make(3 <= min(x, y)), check::make(3 <= x)})));
  ASSERT_THAT(
      simplify(block::make({check::make(3 < max(x, y)), check::make(3 < x)})), matches(check::make(3 < max(x, y))));
  ASSERT_THAT(simplify(block::make({check::make(3 < min(x, y)), check::make(3 < x)})),
      matches(block::make({check::make(3 < min(x, y)), check::make(3 < x)})));

  ASSERT_THAT(
      simplify(block::make({check::make(min(x, y) <= 4), check::make(x <= 4)})), matches(check::make(min(x, y) <= 4)));
  ASSERT_THAT(simplify(block::make({check::make(max(x, y) <= 4), check::make(x <= 4)})),
      matches(block::make({check::make(max(x, y) <= 4), check::make(x <= 4)})));
  ASSERT_THAT(
      simplify(block::make({check::make(min(x, y) < 4), check::make(x < 4)})), matches(check::make(min(x, y) < 4)));
  ASSERT_THAT(simplify(block::make({check::make(max(x, y) < 4), check::make(x < 4)})),
      matches(block::make({check::make(max(x, y) < 4), check::make(x < 4)})));

  ASSERT_THAT(simplify(block::make({check::make(x == 3), check::make(x == 3)})), matches(check::make(x == 3)));
  ASSERT_THAT(simplify(block::make({check::make(x < 3), check::make(x < 4)})), matches(check::make(x < 3)));
  ASSERT_THAT(simplify(block::make({
                  check::make(buffer_min(b0, 0) == 0),
                  check::make(buffer_max(b0, 1) == 10),
                  check::make(buffer_min(b0, 2) == x + 1),
                  crop_buffer::make(b1, b0, {{0, 1}, {0, 20}, {x, 3}}, call_stmt::make(nullptr, {}, {b1}, {})),
              })),
      matches(block::make({
          check::make(buffer_min(b0, 0) == 0),
          check::make(buffer_max(b0, 1) == 10),
          check::make(buffer_min(b0, 2) == x + 1),
          crop_buffer::make(b1, b0, {{expr(), 1}, {0, expr()}, {expr(), 3}}, call_stmt::make(nullptr, {}, {b1}, {})),
      })));

  ASSERT_THAT(simplify(let_stmt::make(x, max((buffer_max(b1, 0) + 1) * (buffer_max(b1, 1) + 1), 10) / 10,
                  make_buffer::make(b0, expr(), expr(), {{{0, max(abs(x), 1) - 1}}},
                      check::make(buffer_max(b0, 0) <= ((buffer_max(b0, 0) + 16) / 16) * 16 - 1)))),
      matches(stmt()));

  ASSERT_THAT(simplify(let::make(x, clamp(y, 0, 10), select(x <= 0, x, 0))), matches(0));

  expr huge_select = 1;
  for (int i = 0; i < 100; ++i) {
    switch (i % 4) {
    case 0: huge_select = select(var(i) < i, huge_select, i); break;
    case 1: huge_select = select(var(i) <= i, huge_select, i); break;
    case 2: huge_select = select(var(i) == i, huge_select, i); break;
    case 3: huge_select = select(var(i) != i, huge_select, i); break;
    }
  }
  simplify(huge_select);
}

TEST(simplify, bounds_of) {
  // Test bounds_of by testing expressions of up to two operands, and setting the
  // bounds of the two operands to all possible cases of overlap. This approach
  // to testing should be great at finding cases where bounds are incorrectly tight,
  // but this test doesn't cover regressions that relax the bounds produced.
  int scale = 3;
  expr exprs[] = {
      x + y,
      x - y,
      x * y,
      x / y,
      x % y,
      slinky::min(x, y),
      slinky::max(x, y),
      x < y,
      x <= y,
      x == y,
      x != y,
      !(x < y),
      x < y && x != y,
      x < y || x == y,
      abs(x),
  };

  for (const expr& e : exprs) {
    for (int x_min_sign : {-2, -1, 0, 1, 2}) {
      for (int x_max_sign : {-2, -1, 0, 1, 2}) {
        if (x_max_sign < x_min_sign) continue;
        int x_min = x_min_sign * scale;
        int x_max = x_max_sign * scale;
        for (int y_min_sign : {-2, -1, 0, 1, 2}) {
          for (int y_max_sign : {-2, -1, 0, 1, 2}) {
            if (y_max_sign < y_min_sign) continue;
            int y_min = y_min_sign * scale;
            int y_max = y_max_sign * scale;

            symbol_map<interval_expr> bounds;
            bounds[x] = slinky::bounds(x_min, x_max);
            bounds[y] = slinky::bounds(y_min, y_max);

            interval_expr bounds_e = bounds_of(e, bounds);

            // These bounds should be fully simplified (constants in this case).
            ASSERT_TRUE(as_constant(bounds_e.min));
            ASSERT_TRUE(as_constant(bounds_e.max));

            eval_context ctx;
            for (int y_val = y_min; y_val <= y_max; ++y_val) {
              for (int x_val = x_min; x_val <= x_max; ++x_val) {
                ctx[x] = x_val;
                ctx[y] = y_val;

                index_t result = evaluate(e, ctx);
                index_t min = evaluate(bounds_e.min);
                index_t max = evaluate(bounds_e.max);

                if (result < min || result > max) {
                  std::cerr << "bounds_of failure: " << e << " -> " << bounds_e << std::endl;
                  std::cerr << result << " not in [" << min << ", " << max << "]" << std::endl;
                  std::cerr << "ctx: " << std::endl;
                  dump_context_for_expr(std::cerr, ctx, e, &symbols);
                  std::cerr << std::endl;
                  std::cerr << "bounds: " << std::endl;
                  dump_symbol_map(std::cerr, bounds);
                  std::cerr << std::endl;
                  std::abort();
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(constant_lower_bound, basic) {
  ASSERT_THAT(constant_lower_bound(min(x, 0) < 0), matches(0));
  ASSERT_THAT(constant_lower_bound(min(x, 0) * 256 < 0), matches(0));
  ASSERT_THAT(constant_lower_bound(max(x, 0) < 0), matches(0));
  ASSERT_THAT(constant_lower_bound(max(x, 0) * 256 < 0), matches(0));
  ASSERT_THAT(constant_lower_bound(x % 4), matches(0));
  ASSERT_THAT(constant_lower_bound(abs(x)), matches(0));
  ASSERT_THAT(constant_lower_bound(abs(min(x, -5))), matches(5));
  ASSERT_THAT(constant_lower_bound(min(1, max(x, 1))), matches(1));
  ASSERT_THAT(constant_lower_bound(clamp(x, -2, 3)), matches(-2));

  ASSERT_THAT(constant_lower_bound(x || false), matches(boolean(x)));
  ASSERT_THAT(constant_lower_bound(false || x), matches(boolean(x)));
  ASSERT_THAT(constant_lower_bound(x || true), matches(true));
  ASSERT_THAT(constant_lower_bound(true || x), matches(true));
  ASSERT_THAT(constant_lower_bound(x && false), matches(false));
  ASSERT_THAT(constant_lower_bound(false && x), matches(false));
  ASSERT_THAT(constant_lower_bound(x && true), matches(boolean(x)));
  ASSERT_THAT(constant_lower_bound(true && x), matches(boolean(x)));
}

TEST(constant_upper_bound, basic) {
  ASSERT_THAT(constant_upper_bound(min(x, 4)), matches(4));
  ASSERT_THAT(constant_upper_bound(max(x, 4)), matches(max(x, 4)));
  ASSERT_THAT(constant_upper_bound(x - min(y, 4)), matches(x - min(y, 4)));
  ASSERT_THAT(constant_upper_bound(x - max(y, 4)), matches(x - 4));
  ASSERT_THAT(constant_upper_bound(x * 3), matches(x * 3));
  ASSERT_THAT(constant_upper_bound(min(x, 4) * 2), matches(8));
  ASSERT_THAT(constant_upper_bound(min(x, 4) * -2), matches(min(x, 4) * -2));
  ASSERT_THAT(constant_upper_bound(max(x, 4) * -2), matches(-8));
  ASSERT_THAT(constant_upper_bound(min(x, 4) / 2), matches(2));
  ASSERT_THAT(constant_upper_bound(max(x, 4) / 2), matches(max(x, 4) / 2));
  ASSERT_THAT(constant_upper_bound(min(x, 4) / -2), matches(min(x, 4) / -2));
  ASSERT_THAT(constant_upper_bound(max(x, 4) / -2), matches(-2));
  ASSERT_THAT(constant_upper_bound(select(x, 3, 1)), matches(3));
  ASSERT_THAT(constant_upper_bound(x % 4), matches(3));
  ASSERT_THAT(constant_upper_bound(clamp(x, -2, 3)), matches(3));

  ASSERT_THAT(constant_upper_bound(min(x, 0) < 0), matches(1));
  ASSERT_THAT(constant_upper_bound(min(x, 0) * 256 < 0), matches(1));
  ASSERT_THAT(constant_upper_bound(max(x, 0) < 0), matches(0));
  ASSERT_THAT(constant_upper_bound(max(x, 0) * 256 < 0), matches(0));

  ASSERT_THAT(constant_upper_bound(x || false), matches(boolean(x)));
  ASSERT_THAT(constant_upper_bound(false || x), matches(boolean(x)));
  ASSERT_THAT(constant_upper_bound(x || true), matches(true));
  ASSERT_THAT(constant_upper_bound(true || x), matches(true));
  ASSERT_THAT(constant_upper_bound(x && false), matches(false));
  ASSERT_THAT(constant_upper_bound(false && x), matches(false));
  ASSERT_THAT(constant_upper_bound(x && true), matches(boolean(x)));
  ASSERT_THAT(constant_upper_bound(true && x), matches(boolean(x)));
}

TEST(evaluate_constant_lower_bound, basic) {
  ASSERT_EQ(evaluate_constant_lower_bound(min(x, 0) < 0), 0);
  ASSERT_EQ(evaluate_constant_lower_bound(min(x, 0) * 256 < 0), 0);
  ASSERT_EQ(evaluate_constant_lower_bound(max(x, 0) < 0), 0);
  ASSERT_EQ(evaluate_constant_lower_bound(max(x, 0) * 256 < 0), 0);
  ASSERT_EQ(evaluate_constant_lower_bound(x % 4), 0);
  ASSERT_EQ(evaluate_constant_lower_bound(abs(x)), 0);
  ASSERT_EQ(evaluate_constant_lower_bound(abs(min(x, -5))), 5);
  ASSERT_EQ(evaluate_constant_lower_bound(min(1, max(x, 1))), 1);
  ASSERT_EQ(evaluate_constant_lower_bound(clamp(x, -2, 3)), -2);

  ASSERT_EQ(evaluate_constant_lower_bound(x || false), std::nullopt);
  ASSERT_EQ(evaluate_constant_lower_bound(false || x), std::nullopt);
  ASSERT_EQ(evaluate_constant_lower_bound(x || true), true);
  ASSERT_EQ(evaluate_constant_lower_bound(true || x), true);
  ASSERT_EQ(evaluate_constant_lower_bound(x && false), false);
  ASSERT_EQ(evaluate_constant_lower_bound(false && x), false);
  ASSERT_EQ(evaluate_constant_lower_bound(x && true), std::nullopt);
  ASSERT_EQ(evaluate_constant_lower_bound(true && x), std::nullopt);
}

TEST(evaluate_constant_upper_bound, basic) {
  ASSERT_EQ(evaluate_constant_upper_bound(min(x, 4)), 4);
  ASSERT_EQ(evaluate_constant_upper_bound(max(x, 4)), std::nullopt);
  ASSERT_EQ(evaluate_constant_upper_bound(x - min(y, 4)), std::nullopt);
  ASSERT_EQ(evaluate_constant_upper_bound(x - max(y, 4)), std::nullopt);
  ASSERT_EQ(evaluate_constant_upper_bound(x * 3), std::nullopt);
  ASSERT_EQ(evaluate_constant_upper_bound(min(x, 4) * 2), 8);
  ASSERT_EQ(evaluate_constant_upper_bound(min(x, 4) * -2), std::nullopt);
  ASSERT_EQ(evaluate_constant_upper_bound(max(x, 4) * -2), -8);
  ASSERT_EQ(evaluate_constant_upper_bound(min(x, 4) / 2), 2);
  ASSERT_EQ(evaluate_constant_upper_bound(max(x, 4) / 2), std::nullopt);
  ASSERT_EQ(evaluate_constant_upper_bound(min(x, 4) / -2), std::nullopt);
  ASSERT_EQ(evaluate_constant_upper_bound(max(x, 4) / -2), -2);
  ASSERT_EQ(evaluate_constant_upper_bound(select(x, 3, 1)), 3);
  ASSERT_EQ(evaluate_constant_upper_bound(x % 4), 3);
  ASSERT_EQ(evaluate_constant_upper_bound(clamp(x, -2, 3)), 3);

  ASSERT_EQ(evaluate_constant_upper_bound(min(x, 0) < 0), 1);
  ASSERT_EQ(evaluate_constant_upper_bound(min(x, 0) * 256 < 0), 1);
  ASSERT_EQ(evaluate_constant_upper_bound(max(x, 0) < 0), 0);
  ASSERT_EQ(evaluate_constant_upper_bound(max(x, 0) * 256 < 0), 0);

  ASSERT_EQ(evaluate_constant_upper_bound(x || false), std::nullopt);
  ASSERT_EQ(evaluate_constant_upper_bound(false || x), std::nullopt);
  ASSERT_EQ(evaluate_constant_upper_bound(x || true), true);
  ASSERT_EQ(evaluate_constant_upper_bound(true || x), true);
  ASSERT_EQ(evaluate_constant_upper_bound(x && false), false);
  ASSERT_EQ(evaluate_constant_upper_bound(false && x), false);
  ASSERT_EQ(evaluate_constant_upper_bound(x && true), std::nullopt);
  ASSERT_EQ(evaluate_constant_upper_bound(true && x), std::nullopt);
}

TEST(evaluate_constant, basic) {
  ASSERT_EQ(evaluate_constant(x || false), std::nullopt);
  ASSERT_EQ(evaluate_constant(false || x), std::nullopt);
  ASSERT_EQ(evaluate_constant(x || true), true);
  ASSERT_EQ(evaluate_constant(true || x), true);
  ASSERT_EQ(evaluate_constant(x && false), false);
  ASSERT_EQ(evaluate_constant(false && x), false);
  ASSERT_EQ(evaluate_constant(x && true), std::nullopt);
  ASSERT_EQ(evaluate_constant(true && x), std::nullopt);
}

TEST(simplify, modulus_remainder) {
  ASSERT_THAT(simplify((x + 15) / 16), matches((x + 15) / 16));
  ASSERT_THAT(simplify((x + 15) / 16, {}, {{x, {16, 0}}}), matches(x / 16));
  ASSERT_THAT(simplify((x + 15) / 16, {}, {{x, {16, 1}}}), matches(x / 16 + 1));
  ASSERT_THAT(simplify((x + 15) / 16, {}, {{x, {32, 0}}}), matches(x / 16));
  ASSERT_THAT(simplify((x + 15) / 16, {}, {{x, {32, 1}}}), matches(x / 16 + 1));
  ASSERT_THAT(simplify((x + 15) / 16, {}, {{x, {32, 2}}}), matches(x / 16 + 1));
  ASSERT_THAT(simplify((x + 15) / 16, {}, {{x, {8, 0}}}), matches((x + 15) / 16));
}

TEST(simplify, fuzz) {
  gtest_seeded_mt19937 rng;
  expr_generator<gtest_seeded_mt19937> gen(rng, 4);

  constexpr int checks = 10;

  eval_context ctx;

  for (auto _ : fuzz_test(std::chrono::seconds(1))) {
    expr test = gen.random_expr(2);
    expr simplified = simplify(test);

    // Also test bounds_of and constant_lower/upper_bound.
    interval_expr bounds = bounds_of(test, gen.var_bounds());
    expr lower_bound = constant_lower_bound(test);
    expr upper_bound = constant_upper_bound(test);
    std::optional<int> evaluated_lower_bound = evaluate_constant_lower_bound(test);
    std::optional<int> evaluated_upper_bound = evaluate_constant_upper_bound(test);

    if (evaluate_constant(lower_bound)) {
      // constant_lower_bound and evaluate_constant_lower_bound should never leave constants to be folded.
      ASSERT_EQ(!as_constant(lower_bound), !evaluated_lower_bound) << test << " -> " << lower_bound;
    }
    if (evaluate_constant(upper_bound)) {
      ASSERT_EQ(!as_constant(upper_bound), !evaluated_upper_bound) << test << " -> " << upper_bound;
    }

    for (int j = 0; j < checks; ++j) {
      gen.init_context(ctx);

      index_t eval_test = evaluate(test, ctx);
      index_t eval_simplified = evaluate(simplified, ctx);
      if (eval_test != eval_simplified) {
        std::cerr << "simplify failure: " << std::endl;
        print(std::cerr, test, &symbols);
        std::cerr << " -> " << eval_test << std::endl;
        print(std::cerr, simplified, &symbols);
        std::cerr << " -> " << eval_simplified << std::endl;
        dump_context_for_expr(std::cerr, ctx, test, &symbols);
        ASSERT_EQ(eval_test, eval_simplified);
      } else {
        index_t min = !is_infinity(bounds.min) ? evaluate(bounds.min, ctx) : std::numeric_limits<index_t>::min();
        index_t max = !is_infinity(bounds.max) ? evaluate(bounds.max, ctx) : std::numeric_limits<index_t>::max();
        index_t constant_min =
            !is_infinity(lower_bound) ? evaluate(lower_bound, ctx) : std::numeric_limits<index_t>::min();
        index_t constant_max =
            !is_infinity(upper_bound) ? evaluate(upper_bound, ctx) : std::numeric_limits<index_t>::max();
        if (eval_test < min) {
          std::cerr << "bounds_of lower bound failure: " << std::endl;
          print(std::cerr, test, &symbols);
          std::cerr << " -> " << eval_test << std::endl;
          print(std::cerr, bounds.min, &symbols);
          std::cerr << " -> " << min << std::endl;
          dump_context_for_expr(std::cerr, ctx, test, &symbols);
          std::cerr << std::endl;
          ASSERT_LE(min, eval_test);
        }
        if (eval_test > max) {
          std::cerr << "bounds_of upper bound failure: " << std::endl;
          print(std::cerr, test, &symbols);
          std::cerr << " -> " << eval_test << std::endl;
          print(std::cerr, bounds.max, &symbols);
          std::cerr << " -> " << max << std::endl;
          dump_context_for_expr(std::cerr, ctx, test, &symbols);
          std::cerr << std::endl;
          ASSERT_LE(eval_test, max);
        }
        if (eval_test > constant_max) {
          std::cerr << "constant_upper_bound failure: " << std::endl;
          print(std::cerr, test, &symbols);
          std::cerr << " -> " << eval_test << std::endl;
          print(std::cerr, upper_bound, &symbols);
          std::cerr << " -> " << constant_max << std::endl;
          dump_context_for_expr(std::cerr, ctx, test, &symbols);
          std::cerr << std::endl;
          ASSERT_LE(eval_test, constant_max);
        }
        if (eval_test < constant_min) {
          std::cerr << "constant_lower_bound failure: " << std::endl;
          print(std::cerr, test, &symbols);
          std::cerr << " -> " << eval_test << std::endl;
          print(std::cerr, lower_bound, &symbols);
          std::cerr << " -> " << constant_min << std::endl;
          dump_context_for_expr(std::cerr, ctx, test, &symbols);
          std::cerr << std::endl;
          ASSERT_LE(constant_min, eval_test);
        }
        if (evaluated_upper_bound && eval_test > *evaluated_upper_bound) {
          std::cerr << "evaluate_constant_upper_bound failure: " << std::endl;
          print(std::cerr, test, &symbols);
          std::cerr << " -> " << eval_test << std::endl;
          std::cerr << *evaluated_upper_bound << std::endl;
          dump_context_for_expr(std::cerr, ctx, test, &symbols);
          std::cerr << std::endl;
          ASSERT_LE(eval_test, *evaluated_upper_bound);
        }
        if (evaluated_lower_bound && eval_test < *evaluated_lower_bound) {
          std::cerr << "evaluate_constant_lower_bound failure: " << std::endl;
          print(std::cerr, test, &symbols);
          std::cerr << " -> " << eval_test << std::endl;
          std::cerr << *evaluated_lower_bound << std::endl;
          dump_context_for_expr(std::cerr, ctx, test, &symbols);
          std::cerr << std::endl;
          ASSERT_LE(constant_min, *evaluated_lower_bound);
        }
      }
    }
  }
}

TEST(simplify, fuzz_correlated_bounds) {
  gtest_seeded_mt19937 rng;
  expr_generator<gtest_seeded_mt19937> gen(rng, 4);

  constexpr int checks = 10;

  eval_context ctx;

  auto random_staircase = [&](index_t a, index_t b) {
    if (rng() & 1) {
      return ((x + gen.random_constant()) / a) * b;
    } else {
      return ((gen.random_constant() - x) / a) * b;
    }
  };

  int finite_bounds_count = 0;
  int total_count = 0;

  for (auto _ : fuzz_test(std::chrono::seconds(1))) {
    index_t a = gen.random_constant(16);
    index_t b = gen.random_constant(16);
    index_t c = gen.random_constant(16);
    index_t d = euclidean_div(b * c, a);
    expr lhs = random_staircase(a, b);
    expr rhs = random_staircase(c, d);
    expr test = rng() & 1 ? lhs + rhs : lhs - rhs;

    interval_expr bounds = bounds_of(test, gen.var_bounds());

    finite_bounds_count += !is_infinity(bounds.min);
    finite_bounds_count += !is_infinity(bounds.max);
    total_count += 2;

    for (int j = 0; j < checks; ++j) {
      gen.init_context(ctx);
      index_t eval_test = evaluate(test, ctx);
      index_t min = !is_infinity(bounds.min) ? evaluate(bounds.min, ctx) : std::numeric_limits<index_t>::min();
      index_t max = !is_infinity(bounds.max) ? evaluate(bounds.max, ctx) : std::numeric_limits<index_t>::max();
      if (eval_test < min) {
        std::cerr << "bounds_of lower bound failure: " << std::endl;
        print(std::cerr, test, &symbols);
        std::cerr << " -> " << eval_test << std::endl;
        print(std::cerr, bounds.min, &symbols);
        std::cerr << " -> " << min << std::endl;
        dump_context_for_expr(std::cerr, ctx, test, &symbols);
        std::cerr << std::endl;
        ASSERT_LE(min, eval_test);
      }
      if (eval_test > max) {
        std::cerr << "bounds_of upper bound failure: " << std::endl;
        print(std::cerr, test, &symbols);
        std::cerr << " -> " << eval_test << std::endl;
        print(std::cerr, bounds.max, &symbols);
        std::cerr << " -> " << max << std::endl;
        dump_context_for_expr(std::cerr, ctx, test, &symbols);
        std::cerr << std::endl;
        ASSERT_LE(eval_test, max);
      }
    }
  }

  // Half the time, the staircases should be correlated, and the other half of the time, they should be anti-correlated,
  // so we should be able to simplify ~half of the cases. However, it's random, so give an extra 50% tolerance.
  ASSERT_GE(finite_bounds_count, total_count / 3);
}

}  // namespace slinky
