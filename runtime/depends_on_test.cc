#include <gtest/gtest.h>

#include "runtime/depends_on.h"
#include "runtime/expr.h"

namespace slinky {

namespace {

node_context symbols;

var x(symbols, "x");
var y(symbols, "y");
var z(symbols, "z");
var w(symbols, "w");

}  // namespace

bool operator==(const depends_on_result& l, const depends_on_result& r) {
  if (l.var != r.var) return false;
  if (l.buffer_input != r.buffer_input) return false;
  if (l.buffer_output != r.buffer_output) return false;
  if (l.buffer_src != r.buffer_src) return false;
  if (l.buffer_dst != r.buffer_dst) return false;
  if (l.buffer_meta_read != r.buffer_meta_read) return false;
  if (l.buffer_meta_mutated != r.buffer_meta_mutated) return false;
  if (l.ref_count != r.ref_count) return false;
  if (l.used_in_loop != r.used_in_loop) return false;
  return true;
}

TEST(depends_on, basic) {
  ASSERT_EQ(depends_on(x + y, x.sym()), (depends_on_result{.var = true, .ref_count = 1}));
  ASSERT_EQ(depends_on(x + x, x.sym()), (depends_on_result{.var = true, .ref_count = 2}));

  stmt loop_x = loop::make(x.sym(), loop::serial, {y, z}, 1, check::make(x && z));
  ASSERT_EQ(depends_on(loop_x, x.sym()), depends_on_result{});
  ASSERT_EQ(depends_on(loop_x, y.sym()), (depends_on_result{.var = true, .ref_count = 1}));

  stmt call = call_stmt::make(nullptr, {x.sym()}, {y.sym()}, {});
  ASSERT_EQ(
      depends_on(call, x.sym()), (depends_on_result{.buffer_input = true, .buffer_meta_read = true, .ref_count = 1}));
  ASSERT_EQ(
      depends_on(call, y.sym()), (depends_on_result{.buffer_output = true, .buffer_meta_read = true, .ref_count = 1}));

  stmt crop = crop_dim::make(x.sym(), 1, {y, z}, check::make(y));
  ASSERT_EQ(depends_on(crop, x.sym()),
      (depends_on_result{.buffer_meta_read = true, .buffer_meta_mutated = true, .ref_count = 1}));

  stmt make_buffer = make_buffer::make(x.sym(), 0, 1, {{{y, z}, w}}, check::make(x && z));
  ASSERT_EQ(depends_on(make_buffer, x.sym()), (depends_on_result{}));
  ASSERT_EQ(depends_on(make_buffer, y.sym()), (depends_on_result{.var = true, .ref_count = 1}));
  ASSERT_EQ(depends_on(make_buffer, z.sym()), (depends_on_result{.var = true, .ref_count = 2}));
}

}  // namespace slinky
