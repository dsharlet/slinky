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
var b0(symbols, "b0");
var b1(symbols, "b1");
var b2(symbols, "b2");

MATCHER_P(matches, expected, "") { return match(arg, expected); }

}  // namespace

TEST(optimizations, optimize_symbols) {
  auto make_dummy_decl = [](var x, stmt body) { return allocate::make(x, memory_type::heap, 1, {}, body); };

  {
    // We don't know about x, we can't mutate it.
    node_context ctx = symbols;
    ASSERT_THAT(optimize_symbols(crop_dim::make(y, x, 0, {0, 0}, check::make(y)), ctx),
        matches(crop_dim::make(y, x, 0, {0, 0}, check::make(y))));
  }

  {
    // We know about x, we can mutate it.
    node_context ctx = symbols;
    ASSERT_THAT(optimize_symbols(make_dummy_decl(x, crop_dim::make(y, x, 0, {0, 0}, check::make(y))), ctx),
        matches(make_dummy_decl(x, crop_dim::make(x, x, 0, {0, 0}, check::make(x)))));
  }

  {
    node_context ctx = symbols;
    ASSERT_THAT(
        optimize_symbols(
            make_dummy_decl(x, crop_dim::make(y, x, 0, {0, 0}, crop_dim::make(z, y, 0, {0, 0}, check::make(z)))), ctx),
        matches(make_dummy_decl(x, crop_dim::make(x, x, 0, {0, 0}, crop_dim::make(x, x, 0, {0, 0}, check::make(x))))));
  }

  {
    node_context ctx = symbols;
    ASSERT_THAT(optimize_symbols(make_dummy_decl(y, crop_dim::make(x, y, 0, {0, 0}, check::make(y))), ctx),
        matches(make_dummy_decl(y, crop_dim::make(x, y, 0, {0, 0}, check::make(y)))));
  }
}

TEST(optimizations, optimize_async) {
  ASSERT_THAT(
      optimize_async(block::make({async::make({x, y, z}, {b0, b2}, {}, {y, expr(), z, 2}, check::make(buffer_rank(b0))),
          async::make({y}, {b0, b1}, {y, expr()}, {}, check::make(buffer_rank(b0) + buffer_rank(b1)))})),
      matches(async::make({z}, {b0, b1}, {}, {},
          block::make({check::make(buffer_rank(b0)), check::make(semaphore_signal(z, 2)),
              check::make(buffer_rank(b0) + buffer_rank(b1))}))));
  ASSERT_THAT(
      optimize_async(block::make({async::make({y, z}, {b0, b2}, {}, {y, expr(), z, 2}, check::make(buffer_rank(b0))),
          async::make({z}, {b0, b1}, {y, expr(), z, 1}, {}, check::make(buffer_rank(b0) + buffer_rank(b1)))})),
      matches(block::make({async::make({y, z}, {b0}, {}, {y, expr(), z, 2}, check::make(buffer_rank(b0))),
          async::make({y, z}, {b0, b1}, {y, expr(), z, 1}, {}, check::make(buffer_rank(b0) + buffer_rank(b1)))})));
}

TEST(optimizations, deshadow_speed) {
  node_context ctx = symbols;
  stmt s = call_stmt::make(nullptr, {x}, {y}, {});
  for (int i = 0; i < 1000; ++i) {
    s = crop_dim::make(y, y, 0, {0, 0}, s);
  }
  stmt s2 = deshadow(s, ctx);
}

}  // namespace slinky
