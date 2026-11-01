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

// Fuses dimensions 0 and 1 of the input into dimension 0 of the output.
TEST(unravel, one) {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", 3, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  var f(ctx, "f");
  var z(ctx, "z");
  test_context eval_ctx;

  std::vector<expr> extents = {in->dim(0).extent()};
  box_expr bounds = {point(unravel(f, extents, 0)), point(unravel(f, extents, 1)), point(z)};
  func copy = func::make_copy({in, bounds}, {out, {f, z}}, eval_ctx.copy);
  pipeline p = build_pipeline(ctx, {in}, {out});

  const int W = 8;
  const int H = 5;
  const int D = 3;
  buffer<int, 3> in_buf({W, H, D});
  init_random(in_buf);

  buffer<int, 2> out_buf({W * H, D});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  ASSERT_EQ(0, p.evaluate(inputs, outputs, eval_ctx));

  for (int z = 0; z < D; ++z) {
    for (int i = 0; i < W * H; ++i) {
      ASSERT_EQ(out_buf(i, z), in_buf(i % W, i / W, z));
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

// Fuses dimensions 0, 1 and 2 of the input into dimension 0 of the output.
TEST(unravel, three_dims) {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", 4, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  var f(ctx, "f");
  var w(ctx, "w");
  test_context eval_ctx;

  std::vector<expr> extents = {in->dim(0).extent(), in->dim(1).extent()};
  box_expr bounds = {
      point(unravel(f, extents, 0)),
      point(unravel(f, extents, 1)),
      point(unravel(f, extents, 2)),
      point(w),
  };
  func copy = func::make_copy({in, bounds}, {out, {f, w}}, eval_ctx.copy);
  pipeline p = build_pipeline(ctx, {in}, {out});

  const int W = 8;
  const int H = 5;
  const int D = 3;
  const int E = 4;
  buffer<int, 4> in_buf({W, H, D, E});
  init_random(in_buf);

  buffer<int, 2> out_buf({W * H * D, E});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  ASSERT_EQ(0, p.evaluate(inputs, outputs, eval_ctx));

  for (int w = 0; w < E; ++w) {
    for (int i = 0; i < W * H * D; ++i) {
      ASSERT_EQ(out_buf(i, w), in_buf(i % W, (i / W) % H, i / (W * H), w));
    }
  }
  ASSERT_EQ(eval_ctx.copy_calls, 1);
}

// Fuses dimensions 0, 1 and 2, 3 of the input into dimensions 0 and 1 of the output.
TEST(unravel, two) {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", 4, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  var f(ctx, "f");
  var g(ctx, "g");
  test_context eval_ctx;

  std::vector<expr> f_extents = {in->dim(0).extent()};
  std::vector<expr> g_extents = {in->dim(2).extent()};
  box_expr bounds = {
      point(unravel(f, f_extents, 0)),
      point(unravel(f, f_extents, 1)),
      point(unravel(g, g_extents, 0)),
      point(unravel(g, g_extents, 1)),
  };
  func copy = func::make_copy({in, bounds}, {out, {f, g}}, eval_ctx.copy);
  pipeline p = build_pipeline(ctx, {in}, {out});

  const int W = 8;
  const int H = 5;
  const int D = 3;
  const int E = 4;
  buffer<int, 4> in_buf({W, H, D, E});
  init_random(in_buf);

  buffer<int, 2> out_buf({W * H, D * E});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  ASSERT_EQ(0, p.evaluate(inputs, outputs, eval_ctx));

  for (int j = 0; j < D * E; ++j) {
    for (int i = 0; i < W * H; ++i) {
      ASSERT_EQ(out_buf(i, j), in_buf(i % W, i / W, j % D, j / D));
    }
  }
  ASSERT_EQ(eval_ctx.copy_calls, 1);
}

// Splits dimension 0 of the input into dimensions 0 and 1 of the output, and fuses dimensions 1 and 2 of the input into
// dimension 2 of the output.
TEST(ravel_unravel, one_each) {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", 3, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 3, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");
  var g(ctx, "g");
  test_context eval_ctx;

  std::vector<expr> g_extents = {in->dim(1).extent()};
  box_expr bounds = {
      point(ravel({x, y}, {out->dim(0).extent()})),
      point(unravel(g, g_extents, 0)),
      point(unravel(g, g_extents, 1)),
  };
  func copy = func::make_copy({in, bounds}, {out, {x, y, g}}, eval_ctx.copy);
  pipeline p = build_pipeline(ctx, {in}, {out});

  const int W = 8;
  const int H = 5;
  const int D = 3;
  const int E = 4;
  buffer<int, 3> in_buf({W * H, D, E});
  init_random(in_buf);

  buffer<int, 3> out_buf({W, H, D * E});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  ASSERT_EQ(0, p.evaluate(inputs, outputs, eval_ctx));

  for (int k = 0; k < D * E; ++k) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        ASSERT_EQ(out_buf(x, y, k), in_buf(x + y * W, k % D, k / D));
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

// Computes an intermediate, and fuses dimensions 0 and 1 of the intermediate into dimension 0 of the output. The
// intermediate should be an alias of the output.
TEST(unravel, alias_output) {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", 3, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));
  auto intm = buffer_expr::make(ctx, "intm", 3, sizeof(int));
  // The aliaser needs to know the output is not folded.
  for (int d = 0; d < 2; ++d) out->dim(d).fold_factor = dim::unfolded;

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");
  var f(ctx, "f");
  test_context eval_ctx;

  func add = func::make(add_1<int>, {{in, {point(x), point(y), point(z)}}}, {{intm, {x, y, z}}});
  std::vector<expr> extents = {in->dim(0).extent()};
  box_expr bounds = {point(unravel(f, extents, 0)), point(unravel(f, extents, 1)), point(z)};
  func copy = func::make_copy({intm, bounds}, {out, {f, z}}, eval_ctx.copy);
  pipeline p = build_pipeline(ctx, {in}, {out});

  const int W = 8;
  const int H = 5;
  const int D = 3;
  buffer<int, 3> in_buf({W, H, D});
  init_random(in_buf);
  buffer<int, 2> out_buf({W * H, D});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  ASSERT_EQ(0, p.evaluate(inputs, outputs, eval_ctx));

  for (int z = 0; z < D; ++z) {
    for (int i = 0; i < W * H; ++i) {
      ASSERT_EQ(out_buf(i, z), in_buf(i % W, i / W, z) + 1);
    }
  }
  ASSERT_EQ(eval_ctx.heap.allocs.size(), 0);
  ASSERT_EQ(eval_ctx.copy_calls, 0);
}

// Fuses dimensions 0 and 1 of the input into dimension 0 of an intermediate, which is then used to compute the output.
// The intermediate should be an alias of the input.
TEST(unravel, alias_input) {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", 3, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));
  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(int));
  // The aliaser needs to know the input is not folded, and is dense in the dimensions of the unravel.
  for (int d = 0; d < 3; ++d) in->dim(d).fold_factor = dim::unfolded;
  in->dim(1).stride = in->dim(0).stride * in->dim(0).extent();

  var f(ctx, "f");
  var z(ctx, "z");
  test_context eval_ctx;

  std::vector<expr> extents = {in->dim(0).extent()};
  box_expr bounds = {point(unravel(f, extents, 0)), point(unravel(f, extents, 1)), point(z)};
  func copy = func::make_copy({in, bounds}, {intm, {f, z}}, eval_ctx.copy);
  func add = func::make(add_1<int>, {{intm, {point(f), point(z)}}}, {{out, {f, z}}});
  pipeline p = build_pipeline(ctx, {in}, {out});

  const int W = 8;
  const int H = 5;
  const int D = 3;
  buffer<int, 3> in_buf({W, H, D});
  init_random(in_buf);
  buffer<int, 2> out_buf({W * H, D});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  ASSERT_EQ(0, p.evaluate(inputs, outputs, eval_ctx));

  for (int z = 0; z < D; ++z) {
    for (int i = 0; i < W * H; ++i) {
      ASSERT_EQ(out_buf(i, z), in_buf(i % W, i / W, z) + 1);
    }
  }
  ASSERT_EQ(eval_ctx.heap.allocs.size(), 0);
  ASSERT_EQ(eval_ctx.copy_calls, 0);
}

// Fuses dimensions 0, 1 and 2, 3 of the input into dimensions 0 and 1 of an intermediate, which is then used to compute
// the output. The intermediate should be an alias of the input, which only needs to be dense within each group of
// dimensions.
TEST(unravel, alias_input_two) {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", 4, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));
  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(int));
  // The aliaser needs to know the input is not folded, and is dense in the dimensions of each unravel.
  for (int d = 0; d < 4; ++d) in->dim(d).fold_factor = dim::unfolded;
  in->dim(1).stride = in->dim(0).stride * in->dim(0).extent();
  in->dim(3).stride = in->dim(2).stride * in->dim(2).extent();

  var f(ctx, "f");
  var g(ctx, "g");
  test_context eval_ctx;

  std::vector<expr> f_extents = {in->dim(0).extent()};
  std::vector<expr> g_extents = {in->dim(2).extent()};
  box_expr bounds = {
      point(unravel(f, f_extents, 0)),
      point(unravel(f, f_extents, 1)),
      point(unravel(g, g_extents, 0)),
      point(unravel(g, g_extents, 1)),
  };
  func copy = func::make_copy({in, bounds}, {intm, {f, g}}, eval_ctx.copy);
  func add = func::make(add_1<int>, {{intm, {point(f), point(g)}}}, {{out, {f, g}}});
  pipeline p = build_pipeline(ctx, {in}, {out});

  const int W = 8;
  const int H = 5;
  const int D = 3;
  const int E = 4;
  // The two groups of dimensions are not dense with respect to each other.
  buffer<int, 4> in_buf({W, H, D, E});
  in_buf.mutable_dim(0).set_stride(sizeof(int));
  in_buf.mutable_dim(1).set_stride(sizeof(int) * W);
  in_buf.mutable_dim(2).set_stride(sizeof(int) * (W * H + 3));
  in_buf.mutable_dim(3).set_stride(sizeof(int) * (W * H + 3) * D);
  init_random(in_buf);
  buffer<int, 2> out_buf({W * H, D * E});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  ASSERT_EQ(0, p.evaluate(inputs, outputs, eval_ctx));

  for (int j = 0; j < D * E; ++j) {
    for (int i = 0; i < W * H; ++i) {
      ASSERT_EQ(out_buf(i, j), in_buf(i % W, i / W, j % D, j / D) + 1);
    }
  }
  ASSERT_EQ(eval_ctx.heap.allocs.size(), 0);
  ASSERT_EQ(eval_ctx.copy_calls, 0);
}

// Reshapes dimensions 0 and 1 of the input into dimensions 0 and 1 of the output with different extents. This
// requires two copies: one that fuses the dimensions of the input into a flat intermediate, and one that splits the
// flat intermediate into the dimensions of the output.
TEST(ravel_unravel, reshape) {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", 3, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 3, sizeof(int));
  auto flat = buffer_expr::make(ctx, "flat", 2, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");
  var f(ctx, "f");
  test_context eval_ctx;

  std::vector<expr> in_extents = {in->dim(0).extent()};
  box_expr fuse_bounds = {point(unravel(f, in_extents, 0)), point(unravel(f, in_extents, 1)), point(z)};
  func fuse = func::make_copy({in, fuse_bounds}, {flat, {f, z}}, eval_ctx.copy);
  box_expr split_bounds = {point(ravel({x, y}, {out->dim(0).extent()})), point(z)};
  func split = func::make_copy({flat, split_bounds}, {out, {x, y, z}}, eval_ctx.copy);
  pipeline p = build_pipeline(ctx, {in}, {out});

  const int W = 6;
  const int H = 4;
  const int D = 3;
  buffer<int, 3> in_buf({W, H, D});
  init_random(in_buf);
  buffer<int, 3> out_buf({8, 3, D});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  ASSERT_EQ(0, p.evaluate(inputs, outputs, eval_ctx));

  for (int z = 0; z < D; ++z) {
    for (int y = 0; y < 3; ++y) {
      for (int x = 0; x < 8; ++x) {
        const int i = x + y * 8;
        ASSERT_EQ(out_buf(x, y, z), in_buf(i % W, i / W, z));
      }
    }
  }
  ASSERT_EQ(eval_ctx.copy_calls, 2);
}

}  // namespace slinky
