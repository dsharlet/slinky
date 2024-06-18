#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cassert>

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
    std::ostream&, const symbol_map<index_t>&, const expr& = expr(), const node_context* symbols = nullptr);

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
  ASSERT_THAT(simplify(min(x / 2, y / 2)), matches(min(x, y) / 2));
  ASSERT_THAT(simplify(max(x / 2, y / 2)), matches(max(x, y) / 2));
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

  ASSERT_THAT(simplify(max(select(x, y, z), select(x, y, w))), matches(select(x, y, max(z, w))));
  ASSERT_THAT(simplify(max(select(x, y, z), select(x, w, z))), matches(select(x, max(y, w), z)));
  ASSERT_THAT(simplify(min(select(x, y, z), select(x, y, w))), matches(select(x, y, min(z, w))));
  ASSERT_THAT(simplify(min(select(x, y, z), select(x, w, z))), matches(select(x, min(y, w), z)));
  ASSERT_THAT(simplify((select(x, y, z) < select(x, y, w))), matches(((expr(z) < expr(w)) && !x)));

  ASSERT_THAT(simplify(select(x == 1, y, select(x == 1, z, w))), matches(select(x == 1, y, w)));
  ASSERT_THAT(simplify(select(x == 1, select(x == 1, y, z), w)), matches(select(x == 1, y, w)));

  ASSERT_THAT(simplify(min(y, z) <= y + 1), matches(true));

  ASSERT_THAT(simplify(and_then({expr(true), expr(true)})), matches(true));
  ASSERT_THAT(simplify(and_then({expr(true), expr(false)})), matches(false));
  ASSERT_THAT(simplify(and_then({expr(false), x})), matches(false));
  ASSERT_THAT(simplify(and_then({expr(true), x})), matches(x));
  ASSERT_THAT(simplify(and_then({expr(true), x, y})), matches(and_then({x, y})));
  ASSERT_THAT(simplify(or_else({expr(true), expr(true)})), matches(true));
  ASSERT_THAT(simplify(or_else({expr(false), expr(true)})), matches(true));
  ASSERT_THAT(simplify(or_else({expr(false), expr(false)})), matches(false));
  ASSERT_THAT(simplify(or_else({expr(true), x})), matches(true));
  ASSERT_THAT(simplify(or_else({expr(false), x})), matches(x));
  ASSERT_THAT(simplify(or_else({expr(false), x, y})), matches(or_else({x, y})));

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
  ASSERT_THAT(simplify((y / 4) * 4 <= y - 3), matches(3 == y % 4));
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
  // TODO: We need stronger learning from conditions for these to work.
  // ASSERT_THAT(simplify(select(!(x == 0) && y, x == 0, false)), matches(false));
  // ASSERT_THAT(simplify(select(x != 0 && y, x == 0, false)), matches(false));

  ASSERT_THAT(simplify(select(x, expr(), 2) == 1), matches(select(x, expr(), false)));
  ASSERT_THAT(simplify(!select(x, expr(), true)), matches(select(x, expr(), false)));

  ASSERT_THAT(simplify(min(select(x, 0, y) + 4, select(x, expr(), min(y, 113) + 4))),
      matches(select(x, expr(), min(y, 113) + 4)));

  ASSERT_THAT(simplify(select(expr(x) == y, z, select(expr(x) == y, 2, w))), matches(select(expr(x) == y, z, w)));
  ASSERT_THAT(simplify(select(x == 1, 0, max(abs(x), 1) + -1)), matches(max(abs(x), 1) + -1));
  ASSERT_THAT(simplify(select(x != 1, max(abs(x), 1), 1)), matches(max(abs(x), 1)));

  ASSERT_THAT(simplify(crop_dim::make(y, x, 1, {expr(), expr()}, call_stmt::make(nullptr, {}, {y}, {}))),
      matches(call_stmt::make(nullptr, {}, {x}, {})));
  ASSERT_THAT(simplify(crop_buffer::make(y, x, {}, call_stmt::make(nullptr, {}, {y}, {}))),
      matches(call_stmt::make(nullptr, {}, {x}, {})));
  ASSERT_THAT(simplify(slice_buffer::make(y, x, {}, call_stmt::make(nullptr, {}, {y}, {}))),
      matches(call_stmt::make(nullptr, {}, {x}, {})));
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
}

TEST(simplify, licm) {
  // Use parallel loops so loops of one call don't get rewritten to a single call.
  auto make_loop_x = [](const stmt& body) { return loop::make(x, loop::parallel, bounds(0, 10), 1, body); };
  auto make_loop_y = [](const stmt& body) { return loop::make(y, loop::parallel, bounds(0, 10), 1, body); };
  auto make_call = [](const var& in, const var& out) { return call_stmt::make(nullptr, {in}, {out}, {}); };
  auto make_crop_x = [](const var& b, int dim, const stmt& body) { return crop_dim::make(b, b, dim, point(x), body); };
  auto make_crop_y = [](const var& b, int dim, const stmt& body) { return crop_dim::make(b, b, dim, point(y), body); };

  // One call doesn't depend on the loop.
  ASSERT_THAT(simplify(make_loop_x(make_call(b0, b1))), matches(make_call(b0, b1)));
  // Two calls don't depend on the loop.
  ASSERT_THAT(simplify(make_loop_x(block::make({make_call(b0, b1), make_call(b0, b2)}))),
      matches(block::make({make_call(b0, b1), make_call(b0, b2)})));
  // Last call depends on the loop, first call does not.
  ASSERT_THAT(simplify(make_loop_x(block::make({make_call(b0, b1), make_crop_x(b2, 0, make_call(b0, b2))}))),
      matches(block::make({make_call(b0, b1), make_loop_x(make_crop_x(b2, 0, make_call(b0, b2)))})));
  // A call in the middle of the loop depends on the loop.
  ASSERT_THAT(
      simplify(make_loop_x(block::make({make_call(b0, b1), make_crop_x(b2, 0, make_call(b0, b2)), make_call(b0, b3)}))),
      matches(block::make({make_call(b0, b1), make_call(b0, b3), make_loop_x(make_crop_x(b2, 0, make_call(b0, b2)))})));
  // A call in the middle of the loop does not depend on the loop, but does depend on the first call.
  ASSERT_THAT(simplify(make_loop_x(block::make(
                  {make_crop_x(b1, 0, make_call(b0, b1)), make_call(b1, b2), make_crop_x(b3, 0, make_call(b0, b3))}))),
      matches(make_loop_x(block::make(
          {make_crop_x(b1, 0, make_call(b0, b1)), make_call(b1, b2), make_crop_x(b3, 0, make_call(b0, b3))}))));
  // A nested loop.
  ASSERT_THAT(simplify(make_loop_y(make_crop_y(
                  b2, 1, make_loop_x(block::make({make_call(b0, b1), make_crop_x(b2, 0, make_call(b0, b2))}))))),
      matches(block::make(
          {make_call(b0, b1), make_loop_y(make_crop_y(b2, 1, make_loop_x(make_crop_x(b2, 0, make_call(b0, b2)))))})));
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
  symbol_map<interval_expr> xy_bounds = {{x, {0, y}}, {y, {0, 5}}};
  ASSERT_THAT(simplify(min(x, y + 1), xy_bounds), matches(x));
}

TEST(simplify, buffer_bounds) {
  ASSERT_THAT(
      simplify(allocate::make(b0, memory_type::heap, 1, {{buffer_bounds(b1, 0)}},
          crop_dim::make(b2, b0, 0, buffer_bounds(b1, 0) & bounds(x, y), call_stmt::make(nullptr, {}, {b2}, {})))),
      matches(allocate::make(b0, memory_type::heap, 1, {{buffer_bounds(b1, 0)}},
          crop_dim::make(b2, b0, 0, bounds(x, y), call_stmt::make(nullptr, {}, {b2}, {})))));

  ASSERT_THAT(simplify(allocate::make(x, memory_type::heap, 1, {{bounds(2, 3)}}, check::make(buffer_min(x, 0) == 2))),
      matches(stmt()));
  ASSERT_THAT(simplify(allocate::make(x, memory_type::heap, 1, {{bounds(2, 3)}},
                  clone_buffer::make(y, x, check::make(buffer_min(y, 0) == 2)))),
      matches(stmt()));
  ASSERT_THAT(simplify(allocate::make(x, memory_type::heap, 1, {{bounds(2, 3)}},
                  crop_dim::make(x, x, 0, bounds(1, 4), check::make(buffer_min(x, 0) == 2)))),
      matches(stmt()));
  ASSERT_THAT(simplify(allocate::make(x, memory_type::heap, 1, {{bounds(y, z)}},
                  crop_dim::make(x, x, 0, bounds(y - 1, z + 1), check::make(buffer_min(x, 0) == y && buffer_at(x))))),
      matches(allocate::make(x, memory_type::heap, 1, {{bounds(y, z)}}, check::make(buffer_at(x)))));
  ASSERT_THAT(simplify(allocate::make(x, memory_type::heap, 1, {{buffer_bounds(b0, 0) + 2}},
                  crop_dim::make(x, x, 0, bounds(expr(), min(z, buffer_max(b0, 0)) + 2), check::make(buffer_at(x))))),
      matches(allocate::make(x, memory_type::heap, 1, {{buffer_bounds(b0, 0) + 2}}, check::make(buffer_at(x)))));
}

TEST(simplify, crop_not_needed) {
  ASSERT_THAT(simplify(crop_dim::make(b0, b0, 1, {x, y}, check::make(b0))), matches(check::make(b0)));
  ASSERT_THAT(simplify(crop_dim::make(b1, b0, 1, {y, z}, check::make(b1))), matches(check::make(b0)));
  ASSERT_THAT(simplify(crop_dim::make(b1, b0, 1, {y, z}, check::make(b0))), matches(check::make(b0)));
}

TEST(simplify, clone) {
  // Clone is shadowed
  ASSERT_THAT(
      simplify(clone_buffer::make(x, y, crop_dim::make(x, y, 0, {0, 0}, call_stmt::make(nullptr, {z}, {x}, {})))),
      matches(crop_dim::make(x, y, 0, {0, 0}, call_stmt::make(nullptr, {z}, {x}, {}))));

  // Clone should be substituted.
  ASSERT_THAT(
      simplify(clone_buffer::make(x, y, crop_dim::make(x, x, 0, {0, 0}, call_stmt::make(nullptr, {z}, {x}, {})))),
      matches(crop_dim::make(x, y, 0, {0, 0}, call_stmt::make(nullptr, {z}, {x}, {}))));
}

TEST(simplify, allocate) {
  // Pull statements that don't use the buffer out of allocate nodes.
  ASSERT_THAT(simplify(allocate::make(x, memory_type::heap, 1, {{bounds(2, 3), 4, 5}},
                  block::make({check::make(y), check::make(buffer_at(x)), check::make(z)}))),
      matches(block::make(
          {check::make(y), allocate::make(x, memory_type::heap, 1, {{bounds(2, 3), 4, 5}}, check::make(buffer_at(x))),
              check::make(z)})));

  // Make sure clone_buffer doesn't hide uses of buffers or bounds.
  ASSERT_THAT(simplify(allocate::make(x, memory_type::heap, 1, {{bounds(2, 3), 4, 5}},
                  block::make({check::make(y), clone_buffer::make(w, x, check::make(buffer_at(w))), check::make(z)}))),
      matches(block::make(
          {check::make(y), allocate::make(x, memory_type::heap, 1, {{bounds(2, 3), 4, 5}}, check::make(buffer_at(x))),
              check::make(z)})));
}

TEST(simplify, crop) {
  stmt body = call_stmt::make(nullptr, {}, {b2}, {});
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
  ASSERT_THAT(
      simplify(crop_buffer::make(b1, b0, {{x, y}, {z, w}}, crop_buffer::make(b2, b1, {{}, {z, w}, {u, v}}, body))),
      matches(crop_buffer::make(b2, b0, {{x, y}, {z, w}, {u, v}}, body)));
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
      matches(transpose::make_truncate(b1, b0, 1, slice_dim::make(b1, b1, 0, x, body))));
  ASSERT_THAT(simplify(make_slice(b1, b0, {x}, buffer_dims(b0, 2))),
      matches(transpose::make_truncate(b1, b0, 2, slice_dim::make(b1, b1, 0, x, body))));
  ASSERT_THAT(simplify(make_slice(b1, b0, {x, y}, buffer_dims(b0, 2))),
      matches(transpose::make_truncate(b1, b0, 2, slice_buffer::make(b1, b1, {x, y}, body))));
  ASSERT_THAT(simplify(make_slice(b1, b0, {expr(), y}, buffer_dims(b0, 2))),
      matches(transpose::make_truncate(b1, b0, 2, slice_dim::make(b1, b1, 1, y, body))));
  ASSERT_THAT(simplify(make_slice(b1, b0, {expr(), y}, buffer_dims(b0, 3))),
      matches(transpose::make_truncate(b1, b0, 3, slice_dim::make(b1, b1, 1, y, body))));

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
      matches(transpose::make_truncate(b1, b0, 1, crop_dim::make(b1, b1, 0, {x, y}, body))));
  ASSERT_THAT(simplify(make_crop(b1, b0, {x}, {{x, y}}, buffer_dims(b0, 2))),
      matches(transpose::make_truncate(b1, b0, 2, crop_dim::make(b1, b1, 0, {x, y}, body))));
  ASSERT_THAT(simplify(make_crop(b1, b0, {x, z}, {{x, y}, {z, w}}, buffer_dims(b0, 2))),
      matches(transpose::make_truncate(b1, b0, 2, crop_buffer::make(b1, b1, {{x, y}, {z, w}}, body))));
  ASSERT_THAT(simplify(make_crop(b1, b0, {expr(), z}, {{expr(), expr()}, {z, w}}, buffer_dims(b0, 2))),
      matches(transpose::make_truncate(b1, b0, 2, crop_dim::make(b1, b1, 1, {z, w}, body))));
  ASSERT_THAT(simplify(make_crop(b1, b0, {expr(), z}, {{expr(), expr()}, {z, w}}, buffer_dims(b0, 3))),
      matches(transpose::make_truncate(b1, b0, 3, crop_dim::make(b1, b1, 1, {z, w}, body))));
  ASSERT_THAT(simplify(make_crop(b1, b0, {expr(), z}, {{expr(), expr()}, {z, w}}, buffer_dims(b0, 3))),
      matches(transpose::make_truncate(b1, b0, 3, crop_dim::make(b1, b1, 1, {z, w}, body))));

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
}

TEST(simplify, transpose) {
  ASSERT_THAT(simplify(transpose::make(
                  b1, b0, {2, 1, 0}, transpose::make(b2, b1, {2, 1, 0}, call_stmt::make(nullptr, {}, {b2}, {})))),
      matches(transpose::make(b2, b0, {0, 1, 2}, call_stmt::make(nullptr, {}, {b2}, {}))));
  ASSERT_THAT(simplify(transpose::make(
                  b1, b0, {3, 2, 1}, transpose::make(b2, b1, {1, 0}, call_stmt::make(nullptr, {}, {b2}, {})))),
      matches(transpose::make(b2, b0, {2, 3}, call_stmt::make(nullptr, {}, {b2}, {}))));
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

TEST(simplify, constant_upper_bound) {
  ASSERT_THAT(constant_upper_bound(min(x, 4)), matches(4));
  ASSERT_THAT(constant_upper_bound(max(x, 4)), matches(max(x, 4)));
  ASSERT_THAT(constant_upper_bound(x - min(y, 4)), matches(x - min(y, 4)));
  ASSERT_THAT(constant_upper_bound(x - max(y, 4)), matches(x - 4));
  ASSERT_THAT(constant_upper_bound(x * 3), matches(x * 3));
  ASSERT_THAT(constant_upper_bound(min(x, 4) * 2), matches(expr(4) * 2));
  ASSERT_THAT(constant_upper_bound(min(x, 4) * -2), matches(min(x, 4) * -2));
  ASSERT_THAT(constant_upper_bound(max(x, 4) * -2), matches(expr(4) * -2));
  ASSERT_THAT(constant_upper_bound(min(x, 4) / 2), matches(expr(4) / 2));
  ASSERT_THAT(constant_upper_bound(max(x, 4) / 2), matches(max(x, 4) / 2));
  ASSERT_THAT(constant_upper_bound(min(x, 4) / -2), matches(min(x, 4) / -2));
  ASSERT_THAT(constant_upper_bound(max(x, 4) / -2), matches(expr(4) / -2));
  ASSERT_THAT(constant_upper_bound(select(x, 3, 1)), matches(3));
}

TEST(simplify, fuzz) {
  gtest_seeded_mt19937 rng;
  expr_generator<gtest_seeded_mt19937> gen(rng, 4);

  constexpr int tests = 10000;
  constexpr int checks = 10;

  eval_context ctx;

  for (int i = 0; i < tests; ++i) {
    expr test = gen.random_expr(2);
    expr simplified = simplify(test);

    // Also test bounds_of and constant_lower/upper_bound.
    interval_expr bounds = bounds_of(test, gen.var_bounds());
    expr lower_bound = constant_lower_bound(test);
    expr upper_bound = constant_upper_bound(test);

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
      }
    }
  }
}

TEST(simplify, fuzz_correlated_bounds) {
  gtest_seeded_mt19937 rng;
  expr_generator<gtest_seeded_mt19937> gen(rng, 4);

  constexpr int tests = 1000;
  constexpr int checks = 10;

  eval_context ctx;

  for (int i = 0; i < tests; ++i) {
    index_t a = gen.random_constant(16);
    index_t b = gen.random_constant(16);
    index_t c = gen.random_constant(16);
    index_t d = euclidean_div(b * c, a);
    expr test = ((x + gen.random_constant()) / a) * b - ((x + gen.random_constant()) / c) * d;

    interval_expr bounds = bounds_of(test, gen.var_bounds());

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
}

}  // namespace slinky

