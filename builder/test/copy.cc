#include <gtest/gtest.h>

#include <cassert>
#include <vector>

#include "builder/pipeline.h"
#include "builder/test/context.h"
#include "builder/test/util.h"
#include "runtime/expr.h"
#include "runtime/pipeline.h"

namespace slinky {

template <typename T, std::size_t N>
void init_random(buffer<T, N>& x) {
  x.allocate();
  for_each_contiguous_slice(x, [&](index_t extent, T* base) {
    for (index_t i = 0; i < extent; ++i) {
      base[i] = (rand() % 20) - 10;
    }
  });
}

TEST(trivial_scalar, copy) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 0, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 0, sizeof(int));

  var x(ctx, "x");

  func copy = func::make_copy({in, {}}, {out, {}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  buffer<int> out_buf;
  out_buf.allocate();

  // Run the pipeline.
  buffer<int> in_buf;
  init_random(in_buf);

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  ASSERT_EQ(out_buf(), in_buf());

  ASSERT_EQ(eval_ctx.copy_calls, 1);
}

TEST(trivial_1d, copy) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 1, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 1, sizeof(int));

  var x(ctx, "x");

  std::vector<char> padding(sizeof(int), 0);

  // Crop the output to the intersection of the input and output buffer.
  box_expr output_crop = in->bounds() & out->bounds();
  func copy = func::make_copy({in, {point(x)}, output_crop}, {out, {x}}, padding);

  pipeline p = build_pipeline(ctx, {in}, {out});

  const int W = 10;
  buffer<int, 1> out_buf({W});
  out_buf.allocate();

  for (int offset : {0, 2, -2}) {
    // Run the pipeline.
    buffer<int, 1> in_buf({W});
    in_buf.translate(offset);
    init_random(in_buf);

    const raw_buffer* inputs[] = {&in_buf};
    const raw_buffer* outputs[] = {&out_buf};
    test_context eval_ctx;
    p.evaluate(inputs, outputs, eval_ctx);

    for (int x = 0; x < W; ++x) {
      if (in_buf.contains(x)) {
        ASSERT_EQ(out_buf(x), in_buf(x));
      } else {
        ASSERT_EQ(out_buf(x), 0);
      }
    }

    ASSERT_EQ(eval_ctx.copy_calls, 1);
  }
}

TEST(trivial_2d, copy) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");

  std::vector<char> padding(sizeof(int), 0);

  // Crop the output to the intersection of the input and output buffer.
  box_expr output_crop = in->bounds() & out->bounds();
  func copy = func::make_copy({in, {point(x), point(y)}, output_crop}, {out, {x, y}}, padding);

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int H = 20;
  const int W = 10;
  buffer<int, 2> out_buf({W, H});
  out_buf.allocate();

  for (int offset : {0, -4, 3}) {
    buffer<int, 2> in_buf({W, H});
    in_buf.translate(0, offset);
    init_random(in_buf);

    const raw_buffer* inputs[] = {&in_buf};
    const raw_buffer* outputs[] = {&out_buf};
    test_context eval_ctx;
    p.evaluate(inputs, outputs, eval_ctx);

    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        if (in_buf.contains(x, y)) {
          ASSERT_EQ(out_buf(x, y), in_buf(x, y));
        } else {
          ASSERT_EQ(out_buf(x, y), 0);
        }
      }
    }

    ASSERT_EQ(eval_ctx.copy_calls, 1);
  }
}

TEST(trivial_3d, copy) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 3, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 3, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");

  func copy = func::make_copy({in, {point(x), point(y), point(z)}}, {out, {x, y, z}});

  pipeline p = build_pipeline(ctx, {in}, {out});

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
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int z = 0; z < D; ++z) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        ASSERT_EQ(out_buf(x, y, z), in_buf(x, y, z));
      }
    }
  }

  ASSERT_EQ(eval_ctx.copy_calls, 1);
}

TEST(padded, copy) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");

  int padding_x = 3;
  int padding_y = 2;

  std::vector<char> padding(sizeof(int), 0);

  func copy = func::make_copy({in, {point(x) - padding_x, point(y) - padding_y}, in->bounds()}, {out, {x, y}}, padding);

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int H = 10;
  const int W = 7;
  buffer<int, 2> out_buf({W + padding_x * 2, H + padding_y * 2});
  out_buf.allocate();

  buffer<int, 2> in_buf({W, H});
  init_random(in_buf);

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H + padding_y * 2; ++y) {
    for (int x = 0; x < W + padding_x * 2; ++x) {
      if (padding_x <= x && x < padding_x + W && padding_y <= y && y < padding_y + H) {
        ASSERT_EQ(out_buf(x, y), in_buf(x - padding_x, y - padding_y));
      } else {
        ASSERT_EQ(out_buf(x, y), 0);
      }
    }
  }

  ASSERT_EQ(eval_ctx.copy_calls, 1);
}

TEST(custom_pad, copy) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");

  std::vector<char> padding(sizeof(int), 0);

  const int x_min = 1;
  const int x_max = 8;
  const int y_min = 2;
  const int y_max = 4;
  func copy = func::make_copy({in, {point(x), point(y)}, {{x_min, x_max}, {y_min, y_max}}}, {out, {x, y}}, padding);

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int H = 10;
  const int W = 7;
  buffer<int, 2> out_buf({W, H});
  out_buf.allocate();

  buffer<int, 2> in_buf({W, H});
  init_random(in_buf);

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      if (x_min <= x && x <= x_max && y_min <= y && y <= y_max) {
        ASSERT_EQ(out_buf(x, y), in_buf(x, y));
      } else {
        ASSERT_EQ(out_buf(x, y), 0);
      }
    }
  }

  ASSERT_EQ(eval_ctx.copy_calls, 1);
}

TEST(flip_x, copy) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 1, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 1, sizeof(int));

  var x(ctx, "x");

  func flip = func::make_copy({in, {point(-x)}}, {out, {x}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 10;
  buffer<int, 1> in_buf({W});
  init_random(in_buf);

  buffer<int, 1> out_buf({W});
  out_buf.dim(0).translate(-W + 1);
  out_buf.allocate();
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int x = 0; x < W; ++x) {
    ASSERT_EQ(out_buf(-x), in_buf(x));
  }

  // TODO: This could be expressed with a single copy with a negative stride.
  ASSERT_EQ(eval_ctx.copy_calls, W);
  ASSERT_EQ(eval_ctx.copy_elements, W);
}

class flip_y : public testing::TestWithParam<int> {};

INSTANTIATE_TEST_SUITE_P(split, flip_y, testing::Range(0, 5));

TEST_P(flip_y, copy) {
  int split = GetParam();

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 3, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 3, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");

  func flip = func::make_copy({in, {point(x), point(-y), point(z)}}, {out, {x, y, z}});

  if (split > 0) {
    flip.loops({{y, split}});
  }

  pipeline p = build_pipeline(ctx, {in}, {out});

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
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int z = 0; z < D; ++z) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        ASSERT_EQ(out_buf(x, -y, z), in_buf(x, y, z));
      }
    }
  }

  // TODO: This could be expressed with a single copy with a negative stride in y.
  ASSERT_EQ(eval_ctx.copy_calls, H);
  ASSERT_EQ(eval_ctx.copy_elements, W * H * D);
}

class upsample_y : public testing::TestWithParam<int> {};

INSTANTIATE_TEST_SUITE_P(split, upsample_y, testing::Range(0, 5));

TEST_P(upsample_y, copy) {
  int split = GetParam();

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");

  func upsample = func::make_copy({in, {point(x), point(y / 2)}}, {out, {x, y}});

  if (split > 0) {
    upsample.loops({{y, split}});
  }

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int H = 20;
  const int W = 10;
  buffer<int, 2> in_buf({W, H / 2});
  init_random(in_buf);

  buffer<int, 2> out_buf({W, H});
  out_buf.allocate();
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y), in_buf(x, y / 2));
    }
  }

  // This copy should be implemented with a loop over y, and a call to copy at each y.
  // TODO: It could be implemented as a copy for each two lines, with a broadcast in y!
  ASSERT_EQ(eval_ctx.copy_calls, H);
  ASSERT_EQ(eval_ctx.copy_elements, W * H);
}

class downsample_y : public testing::TestWithParam<int> {};

INSTANTIATE_TEST_SUITE_P(split, downsample_y, testing::Range(0, 5));

TEST_P(downsample_y, copy) {
  int split = GetParam();

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");

  func downsample = func::make_copy({in, {point(x), point(y * 2)}}, {out, {x, y}});

  if (split > 0) {
    downsample.loops({{y, split}});
  }

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int H = 20;
  const int W = 10;
  buffer<int, 2> in_buf({W, H * 2});
  init_random(in_buf);

  buffer<int, 2> out_buf({W, H});
  out_buf.allocate();
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y), in_buf(x, y * 2));
    }
  }

  ASSERT_EQ(eval_ctx.copy_calls, split == 0 ? 1 : ceil_div(H, split));
  ASSERT_EQ(eval_ctx.copy_elements, W * H);
}

class transpose_test : public testing::TestWithParam<std::vector<int>> {};

INSTANTIATE_TEST_SUITE_P(schedule, transpose_test,
    testing::Values(std::vector<int>{}, std::vector<int>{0}, std::vector<int>{1}, std::vector<int>{2},
        std::vector<int>{0, 1}, std::vector<int>{1, 0}, std::vector<int>{0, 2}, std::vector<int>{2, 0},
        std::vector<int>{1, 2}, std::vector<int>{2, 1}, std::vector<int>{0, 1, 2}, std::vector<int>{2, 1, 0},
        std::vector<int>{1, 0, 2}, std::vector<int>{0, 0, 0}, std::vector<int>{1, 1, 1}, std::vector<int>{2, 2, 2},
        std::vector<int>{1, 0, 2}, std::vector<int>{0, 0, 0}, std::vector<int>{0, 0, 1}));

TEST_P(transpose_test, copy) {
  std::vector<int> permutation = GetParam();

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", permutation.size(), sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 3, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");

  func t = func::make_copy({in, permute<interval_expr>(permutation, {point(x), point(y), point(z)})}, {out, {x, y, z}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int H = 20;
  const int W = 10;
  const int D = 10;
  buffer<int, 3> in_buf(permute<index_t>(permutation, {W, H, D}));
  init_random(in_buf);

  buffer<int, 3> out_buf({W, H, D});
  out_buf.allocate();
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int z = 0; z < D; ++z) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        ASSERT_EQ(out_buf(x, y, z), in_buf(permute<index_t>(permutation, {x, y, z})));
      }
    }
  }

  if (is_permutation(permutation)) {
    ASSERT_EQ(eval_ctx.copy_calls, 1);
  }
}

class broadcast : public testing::TestWithParam<int> {};

INSTANTIATE_TEST_SUITE_P(dim, broadcast, testing::Range(0, 3));

TEST_P(broadcast, copy) {
  const int broadcast_dim = GetParam();

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 3, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 3, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");

  box_expr bounds = {point(x), point(y), point(z)};
  bounds[broadcast_dim] = point(0);
  func crop = func::make_copy({in, bounds}, {out, {x, y, z}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  const int W = 8;
  const int H = 5;
  const int D = 3;

  // Run the pipeline.
  buffer<int, 3> in_buf({W, H, D});
  in_buf.dim(broadcast_dim).set_extent(1);
  init_random(in_buf);

  buffer<int, 3> out_buf({W, H, D});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int z = 0; z < D; ++z) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        index_t i[] = {x, y, z};
        i[broadcast_dim] = 0;
        ASSERT_EQ(out_buf(x, y, z), in_buf(i));
      }
    }
  }

  ASSERT_EQ(eval_ctx.copy_calls, 1);
}

TEST_P(broadcast, copy_sliced) {
  const int broadcast_dim = GetParam();

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 3, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");

  box_expr bounds = {point(x), point(y), point(z)};
  bounds.erase(bounds.begin() + broadcast_dim);
  func crop = func::make_copy({in, bounds}, {out, {x, y, z}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  const int W = 8;
  const int H = 5;
  const int D = 3;

  // Run the pipeline.
  std::vector<index_t> in_extents = {W, H, D};
  in_extents.erase(in_extents.begin() + broadcast_dim);
  buffer<int, 2> in_buf({in_extents[0], in_extents[1]});
  init_random(in_buf);

  buffer<int, 3> out_buf({W, H, D});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int z = 0; z < D; ++z) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        std::vector<index_t> i = {x, y, z};
        i.erase(i.begin() + broadcast_dim);
        ASSERT_EQ(out_buf(x, y, z), in_buf(i));
      }
    }
  }

  ASSERT_EQ(eval_ctx.copy_calls, 1);
}

TEST_P(broadcast, optional) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 3, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 3, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");

  box_expr bounds = {
      select(in->dim(0).extent() == 1, point(in->dim(0).min()), point(x)),
      select(in->dim(1).extent() == 1, point(in->dim(1).min()), point(y)),
      select(in->dim(2).extent() == 1, point(in->dim(2).min()), point(z)),
  };
  func crop = func::make_copy({in, bounds}, {out, {x, y, z}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  const int W = 8;
  const int H = 5;
  const int D = 3;

  // Run the pipeline.
  const int broadcast_dim = GetParam();
  buffer<int, 3> in_buf({W, H, D});
  in_buf.dim(broadcast_dim).set_extent(1);
  init_random(in_buf);

  buffer<int, 3> out_buf({W, H, D});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int z = 0; z < D; ++z) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        index_t i[] = {x, y, z};
        i[broadcast_dim] = 0;
        ASSERT_EQ(out_buf(x, y, z), in_buf(i));
      }
    }
  }

  ASSERT_EQ(eval_ctx.copy_calls, 1);
}

class concatenate : public testing::TestWithParam<std::tuple<int, int>> {};

INSTANTIATE_TEST_SUITE_P(sizes, concatenate, testing::Combine(testing::Values(1, 3), testing::Values(1, 4)),
    test_params_to_string<concatenate::ParamType>);

TEST_P(concatenate, copy) {
  // Make the pipeline
  node_context ctx;

  auto in1 = buffer_expr::make(ctx, "in1", 3, sizeof(int));
  auto in2 = buffer_expr::make(ctx, "in2", 3, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 3, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");

  func concat = func::make_concat({in1, in2}, {out, {x, y, z}}, 1, {0, in1->dim(1).max() + 1, out->dim(1).extent()});

  pipeline p = build_pipeline(ctx, {in1, in2}, {out});

  const int W = 8;
  const int H1 = std::get<0>(GetParam());
  const int H2 = std::get<1>(GetParam());
  const int D = 3;

  // Run the pipeline.
  buffer<int, 3> in1_buf({W, H1, D});
  buffer<int, 3> in2_buf({W, H2, D});
  init_random(in1_buf);
  init_random(in2_buf);
  if (H1 == 1) in1_buf.dim(2).set_stride(0);
  if (H2 == 1) in2_buf.dim(2).set_stride(0);

  buffer<int, 3> out_buf({W, H1 + H2, D});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in1_buf, &in2_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int z = 0; z < D; ++z) {
    for (int y = 0; y < H1 + H2; ++y) {
      for (int x = 0; x < W; ++x) {
        ASSERT_EQ(out_buf(x, y, z), y < H1 ? in1_buf(x, y, z) : in2_buf(x, y - H1, z));
      }
    }
  }

  ASSERT_EQ(eval_ctx.copy_calls, 2);
  ASSERT_EQ(eval_ctx.copy_elements, W * (H1 + H2) * D);
}

TEST(split, copy) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out1 = buffer_expr::make(ctx, "out1", 2, sizeof(int));
  auto out2 = buffer_expr::make(ctx, "out2", 2, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");

  func copy1 = func::make_copy({in, {slinky::point(x), slinky::point(y)}}, {out1, {x, y}});
  func copy2 = func::make_copy({in, {slinky::point(x), slinky::point(y) + out1->dim(1).extent()}}, {out2, {x, y}});

  pipeline p = build_pipeline(ctx, {in}, {out1, out2});

  const int W = 8;
  const int H1 = 5;
  const int H2 = 4;

  // Run the pipeline.
  buffer<int, 3> in_buf({W, H1 + H2});
  init_random(in_buf);

  buffer<int, 3> out1_buf({W, H1});
  buffer<int, 3> out2_buf({W, H2});
  out1_buf.allocate();
  out2_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out1_buf, &out2_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H1; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out1_buf(x, y), in_buf(x, y));
    }
  }
  for (int y = 0; y < H2; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out2_buf(x, y), in_buf(x, y + H1));
    }
  }

  ASSERT_EQ(eval_ctx.copy_calls, 2);
  ASSERT_EQ(eval_ctx.copy_elements, W * (H1 + H2));
}

TEST(stack, copy) {
  // Make the pipeline
  node_context ctx;

  auto in1 = buffer_expr::make(ctx, "in1", 2, sizeof(int));
  auto in2 = buffer_expr::make(ctx, "in2", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 3, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");

  func concat = func::make_stack({in1, in2}, {out, {x, y, z}});

  pipeline p = build_pipeline(ctx, {in1, in2}, {out});

  const int W = 8;
  const int H = 5;

  // Run the pipeline.
  buffer<int, 2> in1_buf({W, H});
  buffer<int, 2> in2_buf({W, H});
  init_random(in1_buf);
  init_random(in2_buf);

  buffer<int, 3> out_buf({W, H, 2});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in1_buf, &in2_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y, 0), in1_buf(x, y));
      ASSERT_EQ(out_buf(x, y, 1), in2_buf(x, y));
    }
  }

  ASSERT_EQ(eval_ctx.copy_calls, 2);
  ASSERT_EQ(eval_ctx.copy_elements, W * H * 2);
}

class reshape : public testing::TestWithParam<bool> {};

INSTANTIATE_TEST_SUITE_P(opaque, reshape, testing::Values(false, true));

TEST_P(reshape, copy) {
  const bool opaque = GetParam();

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 3, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 3, sizeof(int));

  // To be a reshape that we can optimize, we need the buffers to be dense (strides equal to the product of extents of
  // prior dimensions).
  for (auto i : {in, out}) {
    i->dim(0).stride = sizeof(int);
    i->dim(1).stride = i->dim(0).stride * i->dim(0).extent();
    i->dim(2).stride = i->dim(1).stride * i->dim(1).extent();
  }

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");

  // Compute the "flat" index of the coordinates in the output.
  expr flat_out = x + y * out->dim(0).extent() + z * out->dim(0).extent() * out->dim(1).extent();

  // Unpack the coordinates in the input from the flat index of the output.
  box_expr bounds = {
      point(flat_out % in->dim(0).extent()),
      point((flat_out / in->dim(0).extent()) % in->dim(1).extent()),
      point(flat_out / (in->dim(0).extent() * in->dim(1).extent()) % in->dim(2).extent()),
  };

  func copy;
  if (opaque) {
    // Use a copy callback that does a flat memcpy
    copy = func::make(
        [](const buffer<const void>& in, const buffer<void>& out) -> index_t {
          assert(in.size_bytes() == out.size_bytes());
          memcpy(out.base(), in.base(), out.size_bytes());
          return 0;
        },
        {{in, bounds}}, {{out, {x, y, z}}});
  } else {
    // Use a slinky copy
    copy = func::make_copy({in, bounds}, {out, {x, y, z}});
  }

  pipeline p = build_pipeline(ctx, {in}, {out});

  const int W = 8;
  const int H = 5;
  const int D = 3;

  // Run the pipeline.
  buffer<int, 3> in_buf({W, H, D});
  init_random(in_buf);

  // The output should be the same size as the input, but with permuted dimensions.
  buffer<int, 3> out_buf({H, D, W});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  // This should have been a "flat" copy.
  for (int i = 0; i < W * H * D; ++i) {
    ASSERT_EQ(in_buf.base()[i], out_buf.base()[i]);
  }

  if (!opaque) {
    ASSERT_EQ(eval_ctx.copy_calls, W * H * D);
    ASSERT_EQ(eval_ctx.copy_elements, W * H * D);
  }
}

class batch_reshape : public testing::TestWithParam<bool> {};

INSTANTIATE_TEST_SUITE_P(opaque, batch_reshape, testing::Values(false, true));

TEST_P(batch_reshape, copy) {
  const bool opaque = GetParam();

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 4, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 4, sizeof(int));

  // To be a reshape that we can optimize, we need the buffers to be dense (strides equal to the product of extents of
  // prior dimensions).
  for (auto i : {in, out}) {
    i->dim(0).stride = sizeof(int);
    i->dim(1).stride = i->dim(0).stride * i->dim(0).extent();
    i->dim(2).stride = i->dim(1).stride * i->dim(1).extent();
  }

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");
  var w(ctx, "w");

  // Compute the "flat" index of the coordinates in the output.
  expr flat_out = x + y * out->dim(0).extent() + z * out->dim(0).extent() * out->dim(1).extent();

  // Unpack the coordinates in the input from the flat index of the output.
  box_expr bounds = {
      point(flat_out % in->dim(0).extent()),
      point((flat_out / in->dim(0).extent()) % in->dim(1).extent()),
      point(flat_out / (in->dim(0).extent() * in->dim(1).extent()) % in->dim(2).extent()),
      point(w),
  };
  func copy;
  if (opaque) {
    // Use a callback that does a flat memcpy
    copy = func::make(
        [](const buffer<const void>& in, const buffer<void>& out) -> index_t {
          assert(in.size_bytes() == out.size_bytes());
          memcpy(out.base(), in.base(), out.size_bytes());
          return 0;
        },
        {{in, bounds}}, {{out, {x, y, z, w}}});
  } else {
    // Use a slinky copy
    copy = func::make_copy({in, bounds}, {out, {x, y, z, w}});
  }

  pipeline p = build_pipeline(ctx, {in}, {out});

  const int W = 8;
  const int H = 5;
  const int D = 3;
  const int N = 3;

  // Run the pipeline.
  buffer<int, 4> in_buf({W, H, D, N});
  init_random(in_buf);

  // The output should be the same size as the input, but with permuted dimensions.
  buffer<int, 4> out_buf({H, D, W, N});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  // This should have been a "flat" copy.
  for (int n = 0; n < N; ++n) {
    const int* in_base = &in_buf(0, 0, 0, n);
    const int* out_base = &out_buf(0, 0, 0, n);
    for (int i = 0; i < W * H * D; ++i) {
      ASSERT_EQ(in_base[i], out_base[i]);
    }
  }

  if (!opaque) {
    ASSERT_EQ(eval_ctx.copy_calls, W * H * D);
    ASSERT_EQ(eval_ctx.copy_elements, W * H * D * N);
  }
}

}  // namespace slinky
