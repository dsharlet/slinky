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
var w(symbols, "w");

MATCHER_P(matches, expected, "") { return match(arg, expected); }

}  // namespace

TEST(optimizations, fuse_siblings) {
  auto use_buffer = [](var x) { return call_stmt::make(nullptr, {}, {x}, {}); };

  ASSERT_THAT(fuse_siblings(block::make({
                  allocate::make(x, memory_type::heap, 1, {}, use_buffer(x)),
                  allocate::make(y, memory_type::heap, 1, {}, use_buffer(y)),
              })),
      matches(allocate::make(x, memory_type::heap, 1, {}, block::make({use_buffer(x), use_buffer(x)}))));

  ASSERT_THAT(fuse_siblings(block::make({
                  allocate::make(x, memory_type::heap, 1, {}, use_buffer(x)),
                  allocate::make(y, memory_type::heap, 2, {}, use_buffer(y)),
              })),
      matches(fuse_siblings(block::make({
          allocate::make(x, memory_type::heap, 1, {}, use_buffer(x)),
          allocate::make(y, memory_type::heap, 2, {}, use_buffer(y)),
      }))));

  ASSERT_THAT(fuse_siblings(block::make({
                  allocate::make(x, memory_type::heap, 1, {}, use_buffer(x)),
                  allocate::make(y, memory_type::stack, 1, {}, use_buffer(y)),
              })),
      matches(fuse_siblings(block::make({
          allocate::make(x, memory_type::heap, 1, {}, use_buffer(x)),
          allocate::make(y, memory_type::stack, 1, {}, use_buffer(y)),
      }))));

  ASSERT_THAT(fuse_siblings(block::make({
                  allocate::make(x, memory_type::heap, 1, {{}}, use_buffer(x)),
                  allocate::make(y, memory_type::heap, 1, {}, use_buffer(y)),
              })),
      matches(fuse_siblings(block::make({
          allocate::make(x, memory_type::heap, 1, {{}}, use_buffer(x)),
          allocate::make(y, memory_type::heap, 1, {}, use_buffer(y)),
      }))));

  ASSERT_THAT(fuse_siblings(block::make({
                  allocate::make(x, memory_type::heap, 1, {}, use_buffer(x)),
                  use_buffer(z),
                  allocate::make(y, memory_type::heap, 1, {}, use_buffer(y)),
              })),
      matches(block::make({
          allocate::make(x, memory_type::heap, 1, {}, use_buffer(x)),
          use_buffer(z),
          allocate::make(y, memory_type::heap, 1, {}, use_buffer(y)),
      })));
  ASSERT_THAT(fuse_siblings(block::make({
                  crop_dim::make(x, y, 0, {0, 10}, crop_dim::make(z, x, 1, {0, 10}, use_buffer(z))),
                  crop_dim::make(z, y, 0, {0, 10}, crop_dim::make(w, z, 1, {0, 10}, use_buffer(w))),
              })),
      matches(crop_dim::make(
          x, y, 0, {0, 10}, crop_dim::make(z, x, 1, {0, 10}, block::make({use_buffer(z), use_buffer(z)})))));
}

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

TEST(optimizations, deshadow_speed) {
  node_context ctx = symbols;
  stmt s = call_stmt::make(nullptr, {x}, {y}, {});
  for (int i = 0; i < 1000; ++i) {
    s = crop_dim::make(y, y, 0, {0, 0}, s);
  }
  stmt s2 = deshadow(s, {}, ctx);
}

}  // namespace slinky
