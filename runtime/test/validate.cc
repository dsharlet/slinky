#include <gtest/gtest.h>

#include <cassert>

#include "runtime/validate.h"
#include "runtime/expr.h"

namespace slinky {

namespace {

node_context ctx;
var x(ctx, "x");
var y(ctx, "y");
var z(ctx, "z");

}  // namespace

TEST(validate, var) { 
  std::vector<var> vars = {x, y};
  ASSERT_TRUE(is_valid(x, vars, &ctx));
  ASSERT_TRUE(is_valid(y, vars, &ctx));
  ASSERT_FALSE(is_valid(z, vars, &ctx));
}

TEST(validate, buffer) {
  std::vector<var> vars = {x, y};
  ASSERT_FALSE(is_valid(let::make(z, x + y, buffer_max(z, 0)), vars, &ctx));
  ASSERT_TRUE(is_valid(allocate::make(z, memory_type::heap, 1, {}, check::make(buffer_max(z, 0))), vars, &ctx));
}

TEST(validate, out_of_scope) {
  std::vector<var> vars = {x, y};
  ASSERT_FALSE(is_valid(block::make({let_stmt::make(z, x + y, check::make(z)), check::make(z)}), vars, &ctx));
}

}  // namespace slinky
