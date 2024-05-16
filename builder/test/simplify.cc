#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cassert>

#include "builder/simplify.h"
#include "builder/substitute.h"
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
  ASSERT_THAT(simplify(expr() == 1), matches(expr() == 1));
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
  ASSERT_THAT(simplify((select(x, y, z) < select(x, y, w))), matches(select(x, 0, expr(z) < expr(w))));

  ASSERT_THAT(simplify(select(x == 1, y, select(x == 1, z, w))), matches(select(x == 1, y, w)));
  ASSERT_THAT(simplify(select(x == 1, select(x == 1, y, z), w)), matches(select(x == 1, y, w)));
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
      matches(block::make(
          {make_call(b0, b1), make_loop_x(block::make({make_crop_x(b2, 0, make_call(b0, b2))})), make_call(b0, b3)})));
  // A call in the middle of the loop does not depend on the loop, but does depend on the first call.
  ASSERT_THAT(simplify(make_loop_x(block::make(
                  {make_crop_x(b1, 0, make_call(b0, b1)), make_call(b1, b2), make_crop_x(b3, 0, make_call(b0, b3))}))),
      matches(block::make({make_loop_x(make_crop_x(b1, 0, make_call(b0, b1))), make_call(b1, b2),
          make_loop_x(make_crop_x(b3, 0, make_call(b0, b3)))})));
  // A nested loop.
  ASSERT_THAT(simplify(make_loop_y(make_crop_y(
                  b2, 1, make_loop_x(block::make({make_call(b0, b1), make_crop_x(b2, 0, make_call(b0, b2))}))))),
      matches(block::make(
          {make_call(b0, b1), make_loop_y(make_crop_y(b2, 1, make_loop_x(make_crop_x(b2, 0, make_call(b0, b2)))))})));
}

TEST(simplify, buffer_intrinsics) {
  ASSERT_THAT(simplify(max(buffer_max(x, 0) + 1, buffer_min(x, 0) - 1)), matches(buffer_max(x, 0) + 1));
}

TEST(simplify, bounds) {
  ASSERT_THAT(simplify(loop::make(x, loop::serial, bounds(y - 2, z), 2, check::make(y - 2 <= x))), matches(stmt()));
  ASSERT_THAT(simplify(loop::make(x, loop::serial, min_extent(x, z), z, check::make(y))), matches(check::make(y)));

  ASSERT_THAT(
      simplify(allocate::make(x, memory_type::heap, 1, {{bounds(2, 3), 4, 5}}, check::make(buffer_min(x, 0) == 2))),
      matches(stmt()));
  ASSERT_THAT(simplify(allocate::make(x, memory_type::heap, 1, {{bounds(2, 3), 4, 5}},
                  clone_buffer::make(y, x, check::make(buffer_min(y, 0) == 2)))),
      matches(stmt()));
  ASSERT_THAT(simplify(allocate::make(x, memory_type::heap, 1, {{bounds(2, 3), 4, 5}},
                  crop_dim::make(x, x, 0, bounds(1, 4), check::make(buffer_min(x, 0) == 2)))),
      matches(stmt()));
  ASSERT_THAT(simplify(allocate::make(x, memory_type::heap, 1, {{bounds(y, z), 4, 5}},
                  crop_dim::make(x, x, 0, bounds(y - 1, z + 1), check::make(buffer_min(x, 0) == 2)))),
      matches(allocate::make(x, memory_type::heap, 1, {{bounds(y, z), 4, 5}}, check::make(y == 2))));

  ASSERT_THAT(simplify(crop_dim::make(x, x, 1, {buffer_min(y, 1), buffer_max(y, 1)},
                  crop_dim::make(y, y, 1, {1, 3}, check::make(buffer_min(x, 1) == buffer_min(y, 1))))),
      matches(check::make(max(1, buffer_min(y, 1)) == max(buffer_min(y, 1), buffer_min(x, 1)))));
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
                  block::make({check::make(y), check::make(x), check::make(z)}))),
      matches(block::make({check::make(y),
          allocate::make(x, memory_type::heap, 1, {{bounds(2, 3), 4, 5}}, check::make(x)), check::make(z)})));

  // Make sure clone_buffer doesn't hide uses of buffers or bounds.
  ASSERT_THAT(simplify(allocate::make(x, memory_type::heap, 1, {{bounds(2, 3), 4, 5}},
                  block::make({check::make(y), clone_buffer::make(w, x, check::make(w)), check::make(z)}))),
      matches(block::make({check::make(y),
          allocate::make(x, memory_type::heap, 1, {{bounds(2, 3), 4, 5}}, check::make(x)), check::make(z)})));
}

TEST(simplify, make_buffer) {
  stmt body = call_stmt::make(nullptr, {}, {b0}, {});
  auto make_slice = [body](var buf, std::vector<expr> at, std::vector<dim_expr> dims) {
    for (int i = static_cast<int>(at.size()) - 1; i >= 0; --i) {
      if (at[i].defined()) {
        dims.erase(dims.begin() + i);
      }
    }
    return make_buffer::make(buf, buffer_at(buf, at), buffer_elem_size(buf), dims, body);
  };

  auto make_crop = [body](
                       var buf, std::vector<expr> at, std::vector<interval_expr> bounds, std::vector<dim_expr> dims) {
    for (int d = 0; d < static_cast<int>(bounds.size()); ++d) {
      if (bounds[d].min.defined()) dims[d].bounds.min = bounds[d].min;
      if (bounds[d].max.defined()) dims[d].bounds.max = bounds[d].max;
    }
    return make_buffer::make(buf, buffer_at(buf, at), buffer_elem_size(buf), dims, body);
  };

  // Slices
  ASSERT_THAT(simplify(make_slice(b0, {}, buffer_dims(b0, 0))), matches(truncate_rank::make(b0, b0, 0, body)));
  ASSERT_THAT(simplify(make_slice(b0, {}, buffer_dims(b0, 1))), matches(truncate_rank::make(b0, b0, 1, body)));
  ASSERT_THAT(simplify(make_slice(b0, {}, buffer_dims(b0, 3))), matches(truncate_rank::make(b0, b0, 3, body)));
  ASSERT_THAT(simplify(make_slice(b0, {x}, buffer_dims(b0, 1))),
      matches(truncate_rank::make(b0, b0, 1, slice_dim::make(b0, b0, 0, x, body))));
  ASSERT_THAT(simplify(make_slice(b0, {x}, buffer_dims(b0, 2))),
      matches(truncate_rank::make(b0, b0, 2, slice_dim::make(b0, b0, 0, x, body))));
  ASSERT_THAT(simplify(make_slice(b0, {x, y}, buffer_dims(b0, 2))),
      matches(truncate_rank::make(b0, b0, 2, slice_buffer::make(b0, b0, {x, y}, body))));
  ASSERT_THAT(simplify(make_slice(b0, {expr(), y}, buffer_dims(b0, 2))),
      matches(truncate_rank::make(b0, b0, 2, slice_dim::make(b0, b0, 1, y, body))));
  ASSERT_THAT(simplify(make_slice(b0, {expr(), y}, buffer_dims(b0, 3))),
      matches(truncate_rank::make(b0, b0, 3, slice_dim::make(b0, b0, 1, y, body))));

  // Not slices
  ASSERT_THAT(simplify(make_slice(b0, {}, buffer_dims(b1, 1))), matches(make_slice(b0, {}, buffer_dims(b1, 1))));
  ASSERT_THAT(simplify(make_crop(b0, {}, {buffer_bounds(b0, 0)}, buffer_dims(b1, 1))),
      matches(make_crop(b0, {}, {buffer_bounds(b0, 0)}, buffer_dims(b1, 1))));

  // Crops
  ASSERT_THAT(simplify(make_crop(b0, {}, {}, buffer_dims(b0, 0))), matches(truncate_rank::make(b0, b0, 0, body)));
  ASSERT_THAT(simplify(make_crop(b0, {}, {}, buffer_dims(b0, 1))), matches(truncate_rank::make(b0, b0, 1, body)));
  ASSERT_THAT(simplify(make_crop(b0, {x}, {{x, y}}, buffer_dims(b0, 1))),
      matches(truncate_rank::make(b0, b0, 1, crop_dim::make(b0, b0, 0, {x, y}, body))));
  ASSERT_THAT(simplify(make_crop(b0, {x}, {{x, y}}, buffer_dims(b0, 2))),
      matches(truncate_rank::make(b0, b0, 2, crop_dim::make(b0, b0, 0, {x, y}, body))));
  ASSERT_THAT(simplify(make_crop(b0, {x, z}, {{x, y}, {z, w}}, buffer_dims(b0, 2))),
      matches(truncate_rank::make(b0, b0, 2, crop_buffer::make(b0, b0, {{x, y}, {z, w}}, body))));
  ASSERT_THAT(simplify(make_crop(b0, {expr(), z}, {{expr(), expr()}, {z, w}}, buffer_dims(b0, 2))),
      matches(truncate_rank::make(b0, b0, 2, crop_dim::make(b0, b0, 1, {z, w}, body))));
  ASSERT_THAT(simplify(make_crop(b0, {expr(), z}, {{expr(), expr()}, {z, w}}, buffer_dims(b0, 3))),
      matches(truncate_rank::make(b0, b0, 3, crop_dim::make(b0, b0, 1, {z, w}, body))));
  ASSERT_THAT(simplify(make_crop(b0, {expr(), z}, {{expr(), expr()}, {z, w}}, buffer_dims(b0, 3))),
      matches(truncate_rank::make(b0, b0, 3, crop_dim::make(b0, b0, 1, {z, w}, body))));

  // Not crops
  ASSERT_THAT(simplify(make_crop(b0, {}, {{x, y}}, buffer_dims(b0, 1))),
      matches(make_crop(b0, {}, {{x, y}}, buffer_dims(b0, 1))));
  ASSERT_THAT(simplify(make_crop(b0, {y}, {{x, y}}, buffer_dims(b0, 1))),
      matches(make_crop(b0, {y}, {{x, y}}, buffer_dims(b0, 1))));
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

TEST(simplify, where_true) {
  ASSERT_THAT(where_true(x < 5, x), matches(bounds(negative_infinity(), 4)));
  ASSERT_THAT(where_true(x < buffer_min(y, 0), x), matches(bounds(negative_infinity(), buffer_min(y, 0) + -1)));
  ASSERT_THAT(where_true(x / 2 < 7, x), matches(bounds(negative_infinity(), 13)));
  ASSERT_THAT(where_true(min(x, 6) < 7, x), matches(bounds(negative_infinity(), positive_infinity())));
  ASSERT_THAT(where_true(-10 <= x && x < 5, x), matches(bounds(-10, 4)));
  ASSERT_THAT(where_true(-x < 5, x), matches(bounds(-4, positive_infinity())));
  ASSERT_THAT(where_true(3 * x < 5, x), matches(bounds(negative_infinity(), 1)));
  ASSERT_THAT(where_true(3 * (x + 2) < 5, x), matches(bounds(negative_infinity(), -1)));
}

std::vector<var> vars = {x, y, z};
std::vector<var> bufs = {b0, b1};

template <typename T>
T random_pick(const std::vector<T>& from) {
  return from[rand() % from.size()];
}

constexpr int max_rank = 2;

constexpr int max_abs_constant = 256;

index_t random_constant(int max = max_abs_constant) { return (rand() & (2 * max - 1)) - max; }

expr random_buffer_intrinsic() {
  switch (rand() % 2) {
  case 0: return buffer_min(random_pick(bufs), rand() % max_rank);
  case 1: return buffer_max(random_pick(bufs), rand() % max_rank);
  default: return buffer_at(random_pick(bufs));
  }
}

expr make_random_expr(int depth);

expr make_random_condition(int depth) {
  expr a = make_random_expr(depth - 1);
  expr b = make_random_expr(depth - 1);
  switch (rand() % 8) {
  default: return a == b;
  case 1: return a < b;
  case 2: return a <= b;
  case 3: return a != b;
  case 4: return make_random_condition(depth - 1) && make_random_condition(depth - 1);
  case 5: return make_random_condition(depth - 1) || make_random_condition(depth - 1);
  case 6: return !make_random_condition(depth - 1);
  }
}

expr make_random_expr(int depth) {
  if (depth <= 0) {
    switch (rand() % 4) {
    default: return random_pick(vars);
    case 1: return constant::make(random_constant());
    case 2: return random_buffer_intrinsic();
    }
  } else {
    expr a = make_random_expr(depth - 1);
    expr b = make_random_expr(depth - 1);
    switch (rand() % 9) {
    default: return a + b;
    case 1: return a - b;
    case 2: return a * b;
    case 3: return a / b;
    case 4: return a % b;
    case 5: return min(a, b);
    case 6: return max(a, b);
    case 7: return select(make_random_condition(depth - 1), a, b);
    case 8: return random_constant();
    }
  }
}

TEST(simplify, fuzz) {
  const int seed = time(nullptr);
  srand(seed);
  constexpr int tests = 10000;
  constexpr int checks = 10;

  eval_context ctx;

  std::vector<buffer<int, max_rank>> buffers(bufs.size());
  for (int i = 0; i < static_cast<int>(bufs.size()); ++i) {
    ctx[bufs[i]] = reinterpret_cast<index_t>(&buffers[i]);
  }

  symbol_map<interval_expr> var_bounds;
  for (const var& v : vars) {
    var_bounds[v] = {-max_abs_constant, max_abs_constant};
  }

  for (int i = 0; i < tests; ++i) {
    expr test = make_random_expr(3);
    expr simplified = simplify(test);

    // Also test bounds_of.
    interval_expr bounds = bounds_of(test, var_bounds);

    for (int j = 0; j < checks; ++j) {
      for (const var& v : vars) {
        ctx[v] = random_constant();
      }
      for (auto& b : buffers) {
        for (int d = 0; d < max_rank; ++d) {
          // TODO: Add one to extent because the simplifier assumes buffer_max >= buffer_min. This is not
          // correct in the case of empty buffers. But do we need to handle empty buffers...?
          index_t min = random_constant();
          index_t max = std::max(min + 1, random_constant());
          b.dim(d).set_bounds(min, max);
        }
      }
      index_t eval_test = evaluate(test, ctx);
      index_t eval_simplified = evaluate(simplified, ctx);
      if (eval_test != eval_simplified) {
        std::cerr << "simplify failure (seed = " << seed << "): " << std::endl;
        print(std::cerr, test, &symbols);
        std::cerr << " -> " << eval_test << std::endl;
        print(std::cerr, simplified, &symbols);
        std::cerr << " -> " << eval_simplified << std::endl;
        dump_context_for_expr(std::cerr, ctx, test, &symbols);
        ASSERT_EQ(eval_test, eval_simplified);
      } else {
        index_t min = !is_infinity(bounds.min) ? evaluate(bounds.min, ctx) : std::numeric_limits<index_t>::min();
        index_t max = !is_infinity(bounds.max) ? evaluate(bounds.max, ctx) : std::numeric_limits<index_t>::max();
        if (eval_test < min) {
          std::cerr << "bounds_of lower bound failure (seed = " << seed << "): " << std::endl;
          print(std::cerr, test, &symbols);
          std::cerr << " -> " << eval_test << std::endl;
          print(std::cerr, bounds.min, &symbols);
          std::cerr << " -> " << min << std::endl;
          dump_context_for_expr(std::cerr, ctx, test, &symbols);
          std::cerr << std::endl;
          ASSERT_LE(min, eval_test);
        }
        if (eval_test > max) {
          std::cerr << "bounds_of upper bound failure (seed = " << seed << "): " << std::endl;
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
}

TEST(simplify, fuzz_correlated_bounds) {
  const int seed = time(nullptr);
  srand(seed);
  constexpr int tests = 1000;
  constexpr int checks = 10;

  eval_context ctx;

  symbol_map<interval_expr> var_bounds;
  for (const var& v : vars) {
    var_bounds[v] = {-max_abs_constant, max_abs_constant};
  }

  for (int i = 0; i < tests; ++i) {
    index_t a = random_constant(16);
    index_t b = random_constant(16);
    index_t c = random_constant(16);
    index_t d = euclidean_div(b * c, a);
    expr test = ((x + random_constant()) / a) * b - ((x + random_constant()) / c) * d;

    interval_expr bounds = bounds_of(test, var_bounds);

    for (int j = 0; j < checks; ++j) {
      for (const var& v : vars) {
        ctx[v] = random_constant();
      }
      index_t eval_test = evaluate(test, ctx);
      index_t min = !is_infinity(bounds.min) ? evaluate(bounds.min, ctx) : std::numeric_limits<index_t>::min();
      index_t max = !is_infinity(bounds.max) ? evaluate(bounds.max, ctx) : std::numeric_limits<index_t>::max();
      if (eval_test < min) {
        std::cerr << "bounds_of lower bound failure (seed = " << seed << "): " << std::endl;
        print(std::cerr, test, &symbols);
        std::cerr << " -> " << eval_test << std::endl;
        print(std::cerr, bounds.min, &symbols);
        std::cerr << " -> " << min << std::endl;
        dump_context_for_expr(std::cerr, ctx, test, &symbols);
        std::cerr << std::endl;
        ASSERT_LE(min, eval_test);
      }
      if (eval_test > max) {
        std::cerr << "bounds_of upper bound failure (seed = " << seed << "): " << std::endl;
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
