#include <gtest/gtest.h>

#include "builder/pipeline.h"
#include "builder/test/context.h"
#include "builder/test/funcs.h"
#include "builder/test/util.h"
#include "runtime/expr.h"
#include "runtime/pipeline.h"

namespace slinky {

// This set of tests ensures that we don't alias buffers when doing so would violate assumptions that the client code
// asked slinky to maintain.

TEST(cannot_alias_transpose_input, pipeline) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  auto in_t = buffer_expr::make(ctx, "in_t", 2, sizeof(int));

  // Our callback requires the stride to be 1 element.
  in_t->dim(0).stride = static_cast<index_t>(sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");

  func transposed = func::make_copy({in, {point(y), point(x)}}, {in_t, {x, y}});
  func add1 = func::make(
      [](const buffer<const int>& a, const buffer<int>& b) -> index_t {
        if (a.dim(0).stride() != 4) return 1;
        return add_1<int>(a, b);
      },
      {{{in_t, {point(x), point(y)}}}}, {{{out, {x, y}}}}, call_stmt::attributes{.name = "add1"});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 20;
  const int H = 10;
  buffer<int, 2> in_buf({H, W});
  init_random(in_buf);

  buffer<int, 2> out_buf({W, H});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  ASSERT_EQ(0, p.evaluate(inputs, outputs, eval_ctx));

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y), in_buf(y, x) + 1);
    }
  }
}

TEST(cannot_alias_transpose_output, pipeline) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  auto out_t = buffer_expr::make(ctx, "out_t", 2, sizeof(int));

  // Our callback requires the stride to be 1 element.
  out_t->dim(0).stride = static_cast<index_t>(sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");

  func add1 = func::make(
      [](const buffer<const int>& a, const buffer<int>& b) -> index_t {
        if (b.dim(0).stride() != 4) return 1;
        return add_1<int>(a, b);
      },
      {{{in, {point(x), point(y)}}}}, {{{out_t, {x, y}}}}, call_stmt::attributes{.name = "add1"});
  func transposed = func::make_copy({out_t, {point(y), point(x)}}, {out, {x, y}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 20;
  const int H = 10;
  buffer<int, 2> in_buf({H, W});
  init_random(in_buf);

  buffer<int, 2> out_buf({W, H});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  ASSERT_EQ(0, p.evaluate(inputs, outputs, eval_ctx));

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y), in_buf(y, x) + 1);
    }
  }
}

}  // namespace slinky
