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

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 3);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 3);

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");

  // This copy should be implemented with a loop over y, and a call to copy at each y.
  func flip = func::make_copy({in, {point(x), point(-y), point(z)}}, {out, {x, y, z}});

  pipeline p(ctx, {in}, {out});

  // Run the pipeline.
  const int H = 20;
  const int W = 10;
  const int D = 10;
  buffer<int, 3> in_buf({W, H, D});
  init_random(in_buf);

  buffer<int, 3> out_buf({W, H, D});
  out_buf.dim(1).translate(-H + 1);
  out_buf.allocate();
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int z = 0; z < D; ++z) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        ASSERT_EQ(out_buf(x, -y, z), in_buf(x, y, z));
      }
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
  func flip = func::make_copy({in, {point(x), point(y / 2)}}, {out, {x, y}});

  pipeline p(ctx, {in}, {out});

  // Run the pipeline.
  const int H = 20;
  const int W = 10;
  buffer<int, 2> in_buf({W, H / 2});
  init_random(in_buf);

  buffer<int, 2> out_buf({W, H});
  out_buf.allocate();
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y), in_buf(x, y / 2));
    }
  }
}

TEST(copy_transpose) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 3);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 3);

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");

  // Transpose the first two dimensions.
  // TODO: We should have a special case for transposing 2 dimensions, and this copy should
  // result in a loop plus a call to that.
  func flip = func::make_copy({in, {point(y), point(x), point(z)}}, {out, {x, y, z}});

  pipeline p(ctx, {in}, {out});

  // Run the pipeline.
  const int H = 20;
  const int W = 10;
  const int D = 10;
  buffer<int, 3> in_buf({H, W, D});
  init_random(in_buf);

  buffer<int, 3> out_buf({W, H, D});
  out_buf.allocate();
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int z = 0; z < D; ++z) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        ASSERT_EQ(out_buf(x, y, z), in_buf(y, x, z));
      }
    }
  }
}

TEST(copy_broadcast) {
  for (int dim = 0; dim < 3; ++dim) {
    // Make the pipeline
    node_context ctx;

    auto in = buffer_expr::make(ctx, "in", sizeof(int), 3);
    auto out = buffer_expr::make(ctx, "out", sizeof(int), 3);

    var x(ctx, "x");
    var y(ctx, "y");
    var z(ctx, "z");

    // TODO: It would be nice if we could do this without the broadcast dimension being in the bounds at all.
    box_expr bounds = {point(x), point(y), point(z)};
    bounds[dim] = point(0);
    func crop = func::make_copy({in, bounds}, {out, {x, y, z}});

    pipeline p(ctx, {in}, {out});

    const int W = 8;
    const int H = 5;
    const int D = 3;

    // Run the pipeline.
    buffer<int, 3> in_buf({W, H, D});
    in_buf.dim(dim).set_extent(1);
    init_random(in_buf);

    // Ask for an output padded in every direction.
    buffer<int, 3> out_buf({W, H, D});
    out_buf.allocate();

    const raw_buffer* inputs[] = {&in_buf};
    const raw_buffer* outputs[] = {&out_buf};
    eval_context eval_ctx;
    p.evaluate(inputs, outputs, eval_ctx);

    for (int z = 0; z < D; ++z) {
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          index_t broadcast_xyz[] = {x, y, z};
          broadcast_xyz[dim] = 0;
          ASSERT_EQ(out_buf(x, y, z), in_buf(broadcast_xyz));
        }
      }
    }
  }
}

TEST(copy_padded) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 4);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 4);

  var c(ctx, "c");
  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");

  // We could just clamp using the bounds directly below, but that would hardcode the bounds we clamp
  // in the pipeline. This way, the bounds can vary at eval-time.
  var w(ctx, "w");
  var h(ctx, "h");

  // This is elementwise, but with a clamp to limit the bounds required of the input.
  std::vector<char> padding(sizeof(int), 0);
  func crop = func::make_copy(
      {in, {point(c), point(clamp(x, 0, w - 1)), point(clamp(y, 0, h - 1)), point(z)}}, {out, {c, x, y, z}}, padding);

  pipeline p(ctx, {w, h}, {in}, {out});

  const int C = 4;
  const int W = 8;
  const int H = 5;
  const int D = 3;

  // Run the pipeline.
  buffer<int, 4> in_buf({C, W, H, D});
  init_random(in_buf);

  // Ask for an output padded in every direction.
  buffer<int, 4> out_buf({C, W * 3, H * 3, D});
  out_buf.dim(1).translate(-W);
  out_buf.dim(2).translate(-H);
  out_buf.allocate();

  index_t args[] = {W, H};
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(args, inputs, outputs, eval_ctx);

  for (int z = 0; z < D; ++z) {
    for (int y = -H; y < 2 * H; ++y) {
      for (int x = -W; x < 2 * W; ++x) {
        for (int c = 0; c < C; ++c) {
          if (0 <= x && x < W && 0 <= y && y < H) {
            ASSERT_EQ(out_buf(c, x, y, z), in_buf(c, x, y, z));
          } else {
            ASSERT_EQ(out_buf(c, x, y, z), 0);
          }
        }
      }
    }
  }
}
