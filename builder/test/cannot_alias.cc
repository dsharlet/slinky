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

class may_alias : public testing::TestWithParam<bool> {};

INSTANTIATE_TEST_SUITE_P(constrain, may_alias, testing::Values(false, true));

TEST_P(may_alias, transpose_input) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  auto in_t = buffer_expr::make(ctx, "in_t", 2, sizeof(int));

  const bool may_alias = GetParam();

  if (!may_alias) {
    // Our callback requires the stride to be 1 element.
    in_t->dim(0).stride = static_cast<index_t>(sizeof(int));
  }

  var x(ctx, "x");
  var y(ctx, "y");

  func transposed = func::make_copy({in, {point(y), point(x)}}, {in_t, {x, y}});
  func add1 = func::make(
      [=](const buffer<const int>& a, const buffer<int>& b) -> index_t {
        if (!may_alias && a.dim(0).stride() != 4) return 1;
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

  ASSERT_EQ(eval_ctx.heap.total_count, may_alias ? 0 : 1);
}

TEST_P(may_alias, transpose_output) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  auto out_t = buffer_expr::make(ctx, "out_t", 2, sizeof(int));

  const bool may_alias = GetParam();

  if (!may_alias) {
    // Our callback requires the stride to be 1 element.
    out_t->dim(0).stride = static_cast<index_t>(sizeof(int));
  }

  var x(ctx, "x");
  var y(ctx, "y");

  func add1 = func::make(
      [=](const buffer<const int>& a, const buffer<int>& b) -> index_t {
        if (!may_alias && b.dim(0).stride() != 4) return 1;
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

  ASSERT_EQ(eval_ctx.heap.total_count, may_alias ? 0 : 1);
}


TEST_P(may_alias, aligned) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));
  
  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(short));

  var x(ctx, "x");
  var y(ctx, "y");

  intm->dim(0).bounds = align(intm->dim(0).bounds, 2);

  // In this pipeline, the result is copied to two outputs. We can only alias in this case if we know the two outputs
  // have the same bounds.
  const bool may_alias = GetParam();
  if (may_alias) {
    out->dim(0).bounds = align(out->dim(0).bounds, 2);
  }
  func add = func::make(add_1<short>, {{in, {point(x), point(y)}}}, {{intm, {x, y}}});
  func copied = func::make_copy({intm, {point(x), point(y)}}, {out, {x, y}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 20;
  const int H = 10;
  buffer<short, 2> in_buf({W, H});
  init_random(in_buf);

  buffer<short, 2> out_buf({W, H});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y), in_buf(x, y) + 1);
    }
  }

  ASSERT_EQ(eval_ctx.heap.total_count, may_alias ? 0 : 1);
}

TEST_P(may_alias, same_bounds) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(short));
  auto out1 = buffer_expr::make(ctx, "out1", 2, sizeof(short));
  auto out2 = buffer_expr::make(ctx, "out2", 2, sizeof(short));

  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(short));

  var x(ctx, "x");
  var y(ctx, "y");

  // In this pipeline, the result is copied to two outputs. We can only alias in this case if we know the two outputs
  // have the same bounds.
  const bool may_alias = GetParam();
  if (may_alias) {
    out2->dim(0).bounds = out1->dim(0).bounds;
    out2->dim(1).bounds = out1->dim(1).bounds;
  }
  func add = func::make(add_1<short>, {{in, {point(x), point(y)}}}, {{intm, {x, y}}});
  func copied1 = func::make_copy({intm, {point(x), point(y)}}, {out1, {x, y}});
  func copied2 = func::make_copy({intm, {point(x), point(y)}}, {out2, {x, y}});

  pipeline p = build_pipeline(ctx, {in}, {out1, out2});

  // Run the pipeline.
  const int W = 20;
  const int H = 10;
  buffer<short, 2> in_buf({W, H});
  init_random(in_buf);

  buffer<short, 2> out1_buf({W, H});
  buffer<short, 2> out2_buf({W, H});
  out1_buf.allocate();
  out2_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out1_buf, &out2_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out1_buf(x, y), in_buf(x, y) + 1);
      ASSERT_EQ(out2_buf(x, y), in_buf(x, y) + 1);
    }
  }

  ASSERT_EQ(eval_ctx.heap.total_count, may_alias ? 0 : 1);
}

}  // namespace slinky
