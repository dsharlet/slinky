#include "expr.h"
#include "funcs.h"
#include "pipeline.h"
#include "print.h"
#include "test.h"

#include <cassert>

using namespace slinky;

TEST(copy_trivial) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 2);

  var x(ctx, "x");
  var y(ctx, "y");

  // This copy should be implemented as a single call to copy.
  func copy = func::make_copy({in, {point(x), point(y)}}, {out, {x, y}});

  pipeline p(ctx, {in}, {out});

  // Run the pipeline.
  const int H = 20;
  const int W = 10;
  buffer<int, 2> in_buf({W, H});
  init_random(in_buf);

  buffer<int, 2> out_buf({W, H});
  out_buf.allocate();
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y), in_buf(x, y));
    }
  }
}

TEST(copy_flip_y) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 2);

  var x(ctx, "x");
  var y(ctx, "y");

  // This copy should be implemented with a loop over y, and a call to copy at each y.
  func flip = func::make_copy({in, {point(x), point(-y)}}, {out, {x, y}});

  pipeline p(ctx, {in}, {out});

  // Run the pipeline.
  const int H = 20;
  const int W = 10;
  buffer<int, 2> in_buf({W, H});
  init_random(in_buf);

  buffer<int, 2> out_buf({W, H});
  out_buf.dim(1).translate(-H + 1);
  out_buf.allocate();
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, -y), in_buf(x, y));
    }
  }
}

TEST(copy_upsample_y) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 2);

  var x(ctx, "x");
  var y(ctx, "y");

  // This copy should be implemented with a loop over y, and a call to copy at each y.
  // TODO: It could be implemented as a copy for each two lines, with a broadcast in y!
  func flip = func::make_copy({in, {point(x), point(y/2)}}, {out, {x, y}});

  pipeline p(ctx, {in}, {out});

  // Run the pipeline.
  const int H = 20;
  const int W = 10;
  buffer<int, 2> in_buf({W, H/2});
  init_random(in_buf);

  buffer<int, 2> out_buf({W, H});
  out_buf.allocate();
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y), in_buf(x, y/2));
    }
  }
}

TEST(copy_transpose) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 2);

  var x(ctx, "x");
  var y(ctx, "y");

  // This copy should be implemented with a loop over y, and a call to copy at each y.
  // TODO: It could be implemented as a copy for each two lines, with a broadcast in y!
  func flip = func::make_copy({in, {point(y), point(x)}}, {out, {x, y}});

  pipeline p(ctx, {in}, {out});

  // Run the pipeline.
  const int H = 20;
  const int W = 10;
  buffer<int, 2> in_buf({H, W});
  init_random(in_buf);

  buffer<int, 2> out_buf({W, H});
  out_buf.allocate();
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y), in_buf(y, x));
    }
  }
}

TEST(copy_padded) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 2);
  
  var x(ctx, "x");
  var y(ctx, "y");

  // We could just clamp using the bounds directly below, but that would hardcode the bounds we clamp
  // in the pipeline. This way, the bounds can vary at eval-time.
  var w(ctx, "w");
  var h(ctx, "h");

  // This is elementwise, but with a clamp to limit the bounds required of the input.
  func crop = func::make_copy({in, {point(clamp(x, 0, w - 1)), point(clamp(y, 0, h - 1))}}, {out, {x, y}});

  pipeline p(ctx, {w, h}, {in}, {out});

  const int W = 8;
  const int H = 5;

  // Run the pipeline.
  buffer<int, 2> in_buf({W, H});
  init_random(in_buf);

  // Ask for an output padded in every direction.
  buffer<int, 2> out_buf({W * 3, H * 3});
  out_buf.dim(0).translate(-W);
  out_buf.dim(1).translate(-H);
  out_buf.allocate();

  index_t args[] = {W, H};
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(args, inputs, outputs, eval_ctx);

  for (int y = -H; y < 2 * H; ++y) {
    for (int x = -W; x < 2 * W; ++x) {
      if (0 <= x && x < W && 0 <= y && y < H) {
        ASSERT_EQ(out_buf(x, y), in_buf(x, y));
      } else {
        // TODO: Doesn't pad yet.
        //ASSERT_EQ(out_buf(x, y), 0);
      }
    }
  }
}
