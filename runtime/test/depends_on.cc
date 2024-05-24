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
  if (l.buffer_base != r.buffer_base) return false;
  if (l.buffer_meta != r.buffer_meta) return false;
  return true;
}

std::ostream& operator<<(std::ostream& os, const depends_on_result& r) {
  os << "{.var = " << r.var;
  os << ", .buffer_input = " << r.buffer_input;
  os << ", .buffer_output = " << r.buffer_output;
  os << ", .buffer_src = " << r.buffer_src;
  os << ", .buffer_dst = " << r.buffer_dst;
  os << ", .buffer_base = " << r.buffer_base;
  os << ", .buffer_meta = " << r.buffer_meta;
  os << "}";
  return os;
}

TEST(depends_on, basic) {
  ASSERT_EQ(depends_on(x + y, x), (depends_on_result{.var = true}));
  ASSERT_EQ(depends_on(x + x, x), (depends_on_result{.var = true}));
  ASSERT_EQ(depends_on(buffer_at(x), x), (depends_on_result{.buffer_base = true, .buffer_meta = true}));

  stmt loop_x = loop::make(x, loop::serial, {y, z}, 1, check::make(x && z));
  ASSERT_EQ(depends_on(loop_x, x), depends_on_result{});
  ASSERT_EQ(depends_on(loop_x, y), (depends_on_result{.var = true}));

  stmt call = call_stmt::make(nullptr, {x}, {y}, {});
  ASSERT_EQ(depends_on(call, x), (depends_on_result{.buffer_input = true}));
  ASSERT_EQ(depends_on(call, y), (depends_on_result{.buffer_output = true, .buffer_meta = true}));

  stmt crop = crop_dim::make(x, w, 1, {y, z}, check::make(y));
  ASSERT_EQ(depends_on(crop, x), (depends_on_result{}));

  stmt crop_shadowed = crop_dim::make(x, x, 1, {y, z}, check::make(y));
  ASSERT_EQ(depends_on(crop_shadowed, x), (depends_on_result{.buffer_meta = true}));

  stmt make_buffer = make_buffer::make(x, 0, 1, {{{y, z}, w}}, check::make(x && z));
  ASSERT_EQ(depends_on(make_buffer, x), (depends_on_result{}));
  ASSERT_EQ(depends_on(make_buffer, y), (depends_on_result{.var = true}));
  ASSERT_EQ(depends_on(make_buffer, z), (depends_on_result{.var = true}));

  stmt cropped_output = crop_dim::make(y, z, 0, {w, w}, call);
  ASSERT_EQ(depends_on(cropped_output, z), (depends_on_result{.buffer_output = true, .buffer_meta = true}));

  stmt cropped_input = crop_dim::make(x, z, 0, {w, w}, call);
  ASSERT_EQ(depends_on(cropped_input, z), (depends_on_result{.buffer_input = true, .buffer_meta = true}));
}

TEST(depends_on, copy) {
  ASSERT_EQ(
      depends_on(copy_stmt::make(x, {z}, y, {z}, {}), x), (depends_on_result{.buffer_src = true, .buffer_meta = true}));
  ASSERT_EQ(
      depends_on(copy_stmt::make(x, {z}, y, {z}, {}), y), (depends_on_result{.buffer_dst = true, .buffer_meta = true}));
  ASSERT_EQ(depends_on(copy_stmt::make(x, {z + w}, y, {z}, {}), z), (depends_on_result{}));
  ASSERT_EQ(depends_on(copy_stmt::make(x, {z + w}, y, {z}, {}), w), (depends_on_result{.var = true}));
}

}  // namespace slinky
