#include <gtest/gtest.h>

#include <vector>

#include "slinky/builder/pipeline.h"
#include "slinky/builder/test/context.h"
#include "slinky/builder/test/funcs.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/pipeline.h"

namespace slinky {

// Splits dimension 0 of the input into dimensions 0 and 1 of the output.
TEST(ravel, one) {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 3, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");
  test_context eval_ctx;

  box_expr bounds = {point(ravel({x, y}, {out->dim(0).extent()})), point(z)};
  func copy = func::make_copy({in, bounds}, {out, {x, y, z}}, eval_ctx.copy);
  pipeline p = build_pipeline(ctx, {in}, {out});

  const int W = 8;
  const int H = 5;
  const int D = 3;
  buffer<int, 2> in_buf({W * H, D});
  init_random(in_buf);
  buffer<int, 3> out_buf({W, H, D});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  ASSERT_EQ(0, p.evaluate(inputs, outputs, eval_ctx));

  for (int z = 0; z < D; ++z) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        ASSERT_EQ(out_buf(x, y, z), in_buf(x + y * W, z));
      }
    }
  }
  // A ravel is affine, so this is a simple copy.
  ASSERT_EQ(eval_ctx.copy_calls, 1);
}

// Splits dimensions 0 and 1 of the input into dimensions 0, 1 and 2, 3 of the output.
TEST(ravel, two) {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 4, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");
  var w(ctx, "w");
  test_context eval_ctx;

  box_expr bounds = {
      point(ravel({x, y}, {out->dim(0).extent()})),
      point(ravel({z, w}, {out->dim(2).extent()})),
  };
  func copy = func::make_copy({in, bounds}, {out, {x, y, z, w}}, eval_ctx.copy);
  pipeline p = build_pipeline(ctx, {in}, {out});

  const int W = 8;
  const int H = 5;
  const int D = 3;
  const int E = 4;
  buffer<int, 2> in_buf({W * H, D * E});
  init_random(in_buf);
  buffer<int, 4> out_buf({W, H, D, E});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  ASSERT_EQ(0, p.evaluate(inputs, outputs, eval_ctx));

  for (int w = 0; w < E; ++w) {
    for (int z = 0; z < D; ++z) {
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          ASSERT_EQ(out_buf(x, y, z, w), in_buf(x + y * W, z + w * D));
        }
      }
    }
  }
  ASSERT_EQ(eval_ctx.copy_calls, 1);
}

// Splits dimension 0 of the input into dimensions 0, 1 and 2 of the output.
TEST(ravel, three_dims) {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 4, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");
  var w(ctx, "w");
  test_context eval_ctx;

  box_expr bounds = {point(ravel({x, y, z}, {out->dim(0).extent(), out->dim(1).extent()})), point(w)};
  func copy = func::make_copy({in, bounds}, {out, {x, y, z, w}}, eval_ctx.copy);
  pipeline p = build_pipeline(ctx, {in}, {out});

  const int W = 8;
  const int H = 5;
  const int D = 3;
  const int E = 4;
  buffer<int, 2> in_buf({W * H * D, E});
  init_random(in_buf);
  buffer<int, 4> out_buf({W, H, D, E});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  ASSERT_EQ(0, p.evaluate(inputs, outputs, eval_ctx));

  for (int w = 0; w < E; ++w) {
    for (int z = 0; z < D; ++z) {
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          ASSERT_EQ(out_buf(x, y, z, w), in_buf(x + (y + z * H) * W, w));
        }
      }
    }
  }
  ASSERT_EQ(eval_ctx.copy_calls, 1);
}

// Computes an intermediate, and splits dimension 0 of the intermediate into dimensions 0 and 1 of the output. The
// intermediate should be an alias of the output.
TEST(ravel, alias_output) {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 3, sizeof(int));
  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(int));

  // The aliaser needs to know the output is not folded, and is dense in the dimensions of the ravel.
  for (int d = 0; d < 3; ++d) out->dim(d).fold_factor = dim::unfolded;
  out->dim(1).stride = out->dim(0).stride * out->dim(0).extent();

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");
  test_context eval_ctx;

  func add = func::make(add_1<int>, {{in, {point(x), point(y)}}}, {{intm, {x, y}}});
  box_expr bounds = {point(ravel({x, y}, {out->dim(0).extent()})), point(z)};
  func copy = func::make_copy({intm, bounds}, {out, {x, y, z}}, eval_ctx.copy);
  pipeline p = build_pipeline(ctx, {in}, {out});

  const int W = 8;
  const int H = 5;
  const int D = 3;
  buffer<int, 2> in_buf({W * H, D});
  init_random(in_buf);
  buffer<int, 3> out_buf({W, H, D});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  ASSERT_EQ(0, p.evaluate(inputs, outputs, eval_ctx));

  for (int z = 0; z < D; ++z) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        ASSERT_EQ(out_buf(x, y, z), in_buf(x + y * W, z) + 1);
      }
    }
  }
  ASSERT_EQ(eval_ctx.heap.allocs.size(), 0);
  ASSERT_EQ(eval_ctx.copy_calls, 0);
}

// Splits dimension 0 of the input into dimensions 0 and 1 of an intermediate, which is then used to compute the
// output. The intermediate should be an alias of the input.
TEST(ravel, alias_input) {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 3, sizeof(int));
  auto intm = buffer_expr::make(ctx, "intm", 3, sizeof(int));

  // The aliaser needs to know the input is not folded.
  for (int d = 0; d < 2; ++d) in->dim(d).fold_factor = dim::unfolded;

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");
  test_context eval_ctx;

  box_expr bounds = {point(ravel({x, y}, {out->dim(0).extent()})), point(z)};
  func copy = func::make_copy({in, bounds}, {intm, {x, y, z}}, eval_ctx.copy);
  func add = func::make(add_1<int>, {{intm, {point(x), point(y), point(z)}}}, {{out, {x, y, z}}});
  pipeline p = build_pipeline(ctx, {in}, {out});

  const int W = 8;
  const int H = 5;
  const int D = 3;
  buffer<int, 2> in_buf({W * H, D});
  init_random(in_buf);
  buffer<int, 3> out_buf({W, H, D});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  ASSERT_EQ(0, p.evaluate(inputs, outputs, eval_ctx));

  for (int z = 0; z < D; ++z) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        ASSERT_EQ(out_buf(x, y, z), in_buf(x + y * W, z) + 1);
      }
    }
  }
  ASSERT_EQ(eval_ctx.heap.allocs.size(), 0);
  ASSERT_EQ(eval_ctx.copy_calls, 0);
}

}  // namespace slinky
