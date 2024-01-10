#include "expr.h"
#include "funcs.h"
#include "pipeline.h"
#include "print.h"
#include "test.h"

#include <cassert>

using namespace slinky;

TEST(copy_trivial_1d) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 1);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 1);

  var x(ctx, "x");
  var dx(ctx, "dx");

  // This copy should be implemented as a single call to copy.
  func copy = func::make_copy({in, {point(x + dx)}}, {out, {x}});

  pipeline p(ctx, {dx}, {in}, {out});

  const int W = 10;
  buffer<int, 1> out_buf({W});
  out_buf.allocate();

  for (int offset : {0, 2, -2}) {
    // Run the pipeline.
    buffer<int, 1> in_buf({W});
    in_buf.translate(offset);
    init_random(in_buf);

    const index_t args[] = {offset};
    const raw_buffer* inputs[] = {&in_buf};
    const raw_buffer* outputs[] = {&out_buf};
    eval_context eval_ctx;
    p.evaluate(args, inputs, outputs, eval_ctx);

    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x), in_buf(x + offset));
    }
  }
}

TEST(copy_trivial_2d) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 2);

  var x(ctx, "x");
  var y(ctx, "y");
  var dy(ctx, "dy");

  // This copy should be implemented as a single call to copy.
  func copy = func::make_copy({in, {point(x), point(y + dy)}}, {out, {x, y}});

  pipeline p(ctx, {dy}, {in}, {out});

  // Run the pipeline.
  const int H = 20;
  const int W = 10;
  buffer<int, 2> out_buf({W, H});
  out_buf.allocate();

  for (int offset : {0, -4, 3}) {
    buffer<int, 2> in_buf({W, H});
    in_buf.translate(0, offset);
    init_random(in_buf);

    const index_t args[] = {offset};
    const raw_buffer* inputs[] = {&in_buf};
    const raw_buffer* outputs[] = {&out_buf};
    eval_context eval_ctx;
    p.evaluate(args, inputs, outputs, eval_ctx);

    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        ASSERT_EQ(out_buf(x, y), in_buf(x, y + offset));
      }
    }
  }
}

TEST(copy_trivial_3d) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 3);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 3);

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");

  // This copy should be implemented as a single call to copy.
  func copy = func::make_copy({in, {point(x), point(y), point(z)}}, {out, {x, y, z}});

  pipeline p(ctx, {in}, {out});

  // Run the pipeline.
  const int H = 20;
  const int W = 10;
  const int D = 5;
  buffer<int, 3> in_buf({W, H, D});
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
        ASSERT_EQ(out_buf(x, y, z), in_buf(x, y, z));
      }
    }
  }
}

TEST(copy_flip_x) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 1);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 1);

  var x(ctx, "x");

  func flip = func::make_copy({in, {point(-x)}}, {out, {x}});

  pipeline p(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 10;
  buffer<int, 1> in_buf({W});
  init_random(in_buf);

  buffer<int, 1> out_buf({W});
  out_buf.dim(0).translate(-W + 1);
  out_buf.allocate();
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int x = 0; x < W; ++x) {
    ASSERT_EQ(out_buf(-x), in_buf(x));
  }
}

TEST(copy_flip_y) {
  for (int split : {-1, 1, 2, 3}) {
    // Make the pipeline
    node_context ctx;

    auto in = buffer_expr::make(ctx, "in", sizeof(int), 3);
    auto out = buffer_expr::make(ctx, "out", sizeof(int), 3);

    var x(ctx, "x");
    var y(ctx, "y");
    var z(ctx, "z");

    // This copy should be implemented with a loop over y, and a call to copy at each y.
    func flip = func::make_copy({in, {point(x), point(-y), point(z)}}, {out, {x, y, z}});

    if (split > 0) {
      flip.loops({{y, split}});
    }

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
}

TEST(copy_upsample_y) {
  for (int split : {-1, 1, 2, 4}) {
    // Make the pipeline
    node_context ctx;

    auto in = buffer_expr::make(ctx, "in", sizeof(int), 2);
    auto out = buffer_expr::make(ctx, "out", sizeof(int), 2);

    var x(ctx, "x");
    var y(ctx, "y");

    // This copy should be implemented with a loop over y, and a call to copy at each y.
    // TODO: It could be implemented as a copy for each two lines, with a broadcast in y!
    func upsample = func::make_copy({in, {point(x), point(y / 2)}}, {out, {x, y}});

    if (split > 0) {
      upsample.loops({{y, split}});
    }

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
          index_t i[] = {x, y, z};
          i[dim] = 0;
          ASSERT_EQ(out_buf(x, y, z), in_buf(i));
        }
      }
    }
  }
}

TEST(copy_broadcast_sliced) {
  for (int dim = 0; dim < 3; ++dim) {
    // Make the pipeline
    node_context ctx;

    auto in = buffer_expr::make(ctx, "in", sizeof(int), 2);
    auto out = buffer_expr::make(ctx, "out", sizeof(int), 3);

    var x(ctx, "x");
    var y(ctx, "y");
    var z(ctx, "z");

    box_expr bounds = {point(x), point(y), point(z)};
    bounds.erase(bounds.begin() + dim);
    func crop = func::make_copy({in, bounds}, {out, {x, y, z}});

    pipeline p(ctx, {in}, {out});

    const int W = 8;
    const int H = 5;
    const int D = 3;

    // Run the pipeline.
    std::vector<index_t> in_extents = {W, H, D};
    in_extents.erase(in_extents.begin() + dim);
    buffer<int, 2> in_buf({in_extents[0], in_extents[1]});
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
          std::vector<index_t> i = {x, y, z};
          i.erase(i.begin() + dim);
          ASSERT_EQ(out_buf(x, y, z), in_buf(i));
        }
      }
    }
  }
}

void test_copy_padded_translated(int translate_x, int translate_z, bool clamped) {
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
  var dx(ctx, "dx");
  var dz(ctx, "dz");

  // This is elementwise, but with a clamp to limit the bounds required of the input.
  std::vector<char> padding;
  if (!clamped) {
    padding.assign(sizeof(int), 0);
  }
  expr in_x = clamped ? clamp(x + dx, 0, w - 1) : x + dx;
  expr in_y = clamped ? clamp(y, 0, h - 1) : y;
  func crop = func::make_copy({in, {point(c), point(in_x), point(in_y), point(z + dz)}}, {out, {c, x, y, z}}, padding);

  pipeline p(ctx, {w, h, dx, dz}, {in}, {out});

  const int C = 4;
  const int W = 8;
  const int H = 5;
  const int D = 3;

  // Run the pipeline.
  buffer<int, 4> in_buf({C, W, H, D});
  in_buf.dim(3).translate(translate_z);
  init_random(in_buf);

  // Ask for an output padded in every direction.
  buffer<int, 4> out_buf({C, W * 3, H * 3, D});
  out_buf.translate(0, -W, -H);
  out_buf.allocate();

  index_t args[] = {W, H, translate_x, translate_z};
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(args, inputs, outputs, eval_ctx);

  for (int z = 0; z < D; ++z) {
    int in_z = z + translate_z;
    for (int y = -H; y < 2 * H; ++y) {
      for (int x = -W; x < 2 * W; ++x) {
        int in_x = x + translate_x;
        for (int c = 0; c < C; ++c) {
          if (clamped) {
            ASSERT_EQ(out_buf(c, x, y, z), in_buf(c, std::clamp(in_x, 0, W - 1), std::clamp(y, 0, H - 1), in_z));
          } else if (in_buf.contains(c, in_x, y, in_z)) {
            ASSERT_EQ(out_buf(c, x, y, z), in_buf(c, in_x, y, in_z));
          } else {
            ASSERT_EQ(out_buf(c, x, y, z), 0);
          }
        }
      }
    }
  }
}

//TEST(copy_clamped) { test_copy_padded_translated(0, 0, true); }
//TEST(copy_translated_clamped) { 
//  test_copy_padded_translated(0, 0, true);
//  test_copy_padded_translated(-2, 0, true);
//  test_copy_padded_translated(1, -1, true);
//  test_copy_padded_translated(-3, 3, true);
//}

// TODO: How to represent padding without clamps?
//TEST(copy_padded) { test_copy_padded_translated(0, 0, false); }
