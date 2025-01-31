#include <gmock/gmock.h>
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
var u(symbols, "u");
var v(symbols, "v");
var xc(symbols, "xc");
var yc(symbols, "yc");

}  // namespace

bool operator==(const depends_on_result& l, const depends_on_result& r) {
  if (l.var != r.var) return false;
  if (l.buffer_input != r.buffer_input) return false;
  if (l.buffer_output != r.buffer_output) return false;
  if (l.buffer_src != r.buffer_src) return false;
  if (l.buffer_dst != r.buffer_dst) return false;
  if (l.buffer_base != r.buffer_base) return false;
  if (l.buffer_dims != r.buffer_dims) return false;
  if (l.buffer_bounds != r.buffer_bounds) return false;
  return true;
}

std::ostream& operator<<(std::ostream& os, const depends_on_result& r) {
  os << "{.var = " << r.var;
  os << ", .buffer_input = " << r.buffer_input;
  os << ", .buffer_output = " << r.buffer_output;
  os << ", .buffer_src = " << r.buffer_src;
  os << ", .buffer_dst = " << r.buffer_dst;
  os << ", .buffer_base = " << r.buffer_base;
  os << ", .buffer_dims = " << r.buffer_dims;
  os << ", .buffer_bounds = " << r.buffer_bounds;
  os << "}";
  return os;
}

TEST(depends_on, basic) {
  ASSERT_EQ(depends_on(x + y, x), (depends_on_result{.var = true}));
  ASSERT_EQ(depends_on(x + x, x), (depends_on_result{.var = true}));
  ASSERT_EQ(depends_on(buffer_at(x), x), (depends_on_result{.buffer_base = true}));

  ASSERT_EQ(depends_on(buffer_min(x, 0), x), (depends_on_result{.buffer_dims = true, .buffer_bounds = true}));
  ASSERT_EQ(depends_on(buffer_stride(x, 0), x), (depends_on_result{.buffer_dims = true}));
  ASSERT_EQ(depends_on(buffer_elem_size(x), x), (depends_on_result{.var = true}));

  stmt loop_x = loop::make(x, loop::serial, {y, z}, 1, check::make(x && z));
  ASSERT_EQ(depends_on(loop_x, x), depends_on_result{});
  ASSERT_EQ(depends_on(loop_x, y), (depends_on_result{.var = true}));

  stmt call = call_stmt::make(nullptr, {xc}, {yc}, {});
  // Everything here should be transparent to clones.
  call = clone_buffer::make(xc, x, clone_buffer::make(yc, y, call));
  ASSERT_EQ(depends_on(call, x), (depends_on_result{.var = true, .buffer_input = true, .buffer_dims = true}));
  ASSERT_EQ(depends_on(call, y),
      (depends_on_result{.var = true, .buffer_output = true, .buffer_dims = true, .buffer_bounds = true}));

  stmt crop = crop_dim::make(x, w, 1, {y, z}, check::make(y));
  ASSERT_EQ(depends_on(crop, x), (depends_on_result{}));

  stmt crop_shadowed = crop_dim::make(x, x, 1, {y, z}, check::make(y));
  ASSERT_EQ(depends_on(crop_shadowed, x), (depends_on_result{.buffer_dims = true, .buffer_bounds = true}));

  stmt make_buffer = make_buffer::make(x, 0, 1, {{{y, z}, w}}, check::make(x && z));
  ASSERT_EQ(depends_on(make_buffer, x), (depends_on_result{}));
  ASSERT_EQ(depends_on(make_buffer, y), (depends_on_result{.var = true}));
  ASSERT_EQ(depends_on(make_buffer, z), (depends_on_result{.var = true}));

  stmt cropped_output = crop_dim::make(y, z, 0, {w, w}, call);
  ASSERT_EQ(depends_on(cropped_output, z),
      (depends_on_result{.var = true, .buffer_output = true, .buffer_dims = true, .buffer_bounds = true}));

  stmt cropped_input = crop_dim::make(x, z, 0, {w, w}, call);
  ASSERT_EQ(depends_on(cropped_input, z),
      (depends_on_result{.var = true, .buffer_input = true, .buffer_dims = true, .buffer_bounds = true}));
}

TEST(depends_on, copy) {
  ASSERT_EQ(depends_on(copy_stmt::make(x, {z}, y, {z}, {}), x),
      (depends_on_result{.var = true, .buffer_src = true, .buffer_dims = true}));
  ASSERT_EQ(depends_on(copy_stmt::make(x, {z}, y, {z}, {{3}}), x),
      (depends_on_result{.var = true, .buffer_src = true, .buffer_dims = true, .buffer_bounds = true}));
  ASSERT_EQ(depends_on(copy_stmt::make(x, {z}, y, {z}, {}), y),
      (depends_on_result{.var = true, .buffer_dst = true, .buffer_dims = true, .buffer_bounds = true}));
  ASSERT_EQ(depends_on(copy_stmt::make(x, {z}, y, {z}, {{3}}), y),
      (depends_on_result{.var = true, .buffer_dst = true, .buffer_dims = true, .buffer_bounds = true}));
  ASSERT_EQ(depends_on(copy_stmt::make(x, {z + w}, y, {z}, {}), z), (depends_on_result{}));
  ASSERT_EQ(depends_on(copy_stmt::make(x, {z + w}, y, {z}, {}), w), (depends_on_result{.var = true}));
}

TEST(depends_on, is_pure) {
  ASSERT_TRUE(is_pure(x + y));
  ASSERT_TRUE(is_pure(abs(x)));
  ASSERT_FALSE(is_pure(buffer_min(x, 0)));
  ASSERT_FALSE(is_pure(y + buffer_min(x, 0)));
}

TEST(find_buffer_dependencies, basic) {
  ASSERT_THAT(find_buffer_dependencies(crop_buffer::make(z, y, {}, call_stmt::make(nullptr, {x}, {z}, {})),
                  /*input=*/false, /*output=*/true),
      testing::ElementsAre(y));
  ASSERT_EQ(find_buffer_data_dependency(buffer_at(x)), x);
  ASSERT_EQ(find_buffer_data_dependency(buffer_at(x, buffer_min(y, 0))), x);
  ASSERT_EQ(find_buffer_data_dependency(buffer_at(x) + buffer_at(y)), var());

  ASSERT_THAT(find_buffer_dependencies(crop_buffer::make(x, y, {}, call_stmt::make(nullptr, {y}, {x}, {}))),
      testing::ElementsAre(y));
  ASSERT_THAT(find_buffer_dependencies(crop_buffer::make(z, y, {}, call_stmt::make(nullptr, {x}, {z}, {})),
                  /*input=*/true, /*output=*/false),
      testing::ElementsAre(x));
  ASSERT_THAT(find_buffer_dependencies(crop_buffer::make(z, y, {}, call_stmt::make(nullptr, {x}, {z}, {})),
                  /*input=*/false, /*output=*/true),
      testing::ElementsAre(y));

  stmt test = block::make({
      crop_buffer::make(z, x, {}, call_stmt::make(nullptr, {y}, {z}, {})),
      slice_buffer::make(z, w, {}, call_stmt::make(nullptr, {y}, {z}, {})),
      make_buffer::make(v, buffer_at(u), buffer_elem_size(u), {}, call_stmt::make(nullptr, {x}, {v}, {})),
  });

  ASSERT_THAT(find_buffer_dependencies(test, /*input=*/true, /*output=*/false), testing::ElementsAre(x, y));
  ASSERT_THAT(find_buffer_dependencies(test, /*input=*/false, /*output=*/true), testing::ElementsAre(x, w));
}

TEST(find_dependencies, basic) {
  ASSERT_THAT(find_dependencies(buffer_at(x)), testing::ElementsAre(x));
  ASSERT_THAT(find_dependencies(x + y), testing::ElementsAre(x, y));
  ASSERT_THAT(find_dependencies(let::make(x, y, x + z)), testing::ElementsAre(y, z));
  ASSERT_THAT(find_dependencies(crop_dim::make(x, y, 0, {z, z}, call_stmt::make(nullptr, {w}, {u}, {}))),
      testing::ElementsAre(y, z, w, u));
  ASSERT_THAT(find_dependencies(block::make({check::make(x), check::make(y)})), testing::ElementsAre(x, y));
}

}  // namespace slinky
