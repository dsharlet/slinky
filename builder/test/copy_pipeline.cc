#include <gtest/gtest.h>

#include <numeric>

#include "builder/pipeline.h"
#include "builder/replica_pipeline.h"
#include "builder/test/bazel_util.h"
#include "builder/test/context.h"
#include "builder/test/funcs.h"
#include "builder/test/util.h"
#include "runtime/expr.h"
#include "runtime/pipeline.h"

namespace slinky {

std::string get_replica_golden() {
  static std::string golden = read_entire_file(get_bazel_file_path("builder/test/replica_pipeline.cc"));
  return golden;
}

void check_replica_pipeline(const std::string& replica_text) {
  size_t pos = get_replica_golden().find(replica_text);
  ASSERT_NE(pos, std::string::npos) << "Matching replica text not found, expected:\n" << replica_text;
}

TEST(flip_y, pipeline) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(char));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(char));
  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(char));

  var x(ctx, "x");
  var y(ctx, "y");

  func copy = func::make(copy_2d<char>, {{in, {point(x), point(y)}}}, {{intm, {x, y}}});
  func flip = func::make(flip_y<char>, {{intm, {point(x), point(-y)}}}, {{out, {x, y}}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int H = 20;
  const int W = 10;
  buffer<char, 2> in_buf({W, H});
  init_random(in_buf);

  buffer<char, 2> out_buf({W, H});
  out_buf.dim(1).translate(-H + 1);
  out_buf.allocate();
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);
  ASSERT_EQ(eval_ctx.heap.total_size, W * H * sizeof(char));
  ASSERT_EQ(eval_ctx.heap.total_count, 1);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, -y), in_buf(x, y));
    }
  }
}

TEST(padded_copy, pipeline) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(char));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(char));
  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(char));

  var x(ctx, "x");
  var y(ctx, "y");

  // We could just clamp using the bounds directly below, but that would hardcode the bounds we clamp
  // in the pipeline. This way, the bounds can vary at eval-time.
  var w(ctx, "w");
  var h(ctx, "h");

  // Copy the input so we can measure the size of the buffer we think we need internally.
  func copy = func::make(copy_2d<char>, {{in, {point(x), point(y)}}}, {{intm, {x, y}}});
  // This is elementwise, but with a clamp to limit the bounds required of the input.
  func crop = func::make(
      zero_padded_copy<char>, {{intm, {point(clamp(x, 0, w - 1)), point(clamp(y, 0, h - 1))}}}, {{out, {x, y}}});

  crop.loops({y});

  pipeline p = build_pipeline(ctx, {w, h}, {in}, {out});

  const int W = 8;
  const int H = 5;

  // Run the pipeline.
  buffer<char, 2> in_buf({W, H});
  init_random(in_buf);

  // Ask for an output padded in every direction.
  buffer<char, 2> out_buf({W * 3, H * 3});
  out_buf.translate(-W, -H);
  out_buf.allocate();

  index_t args[] = {W, H};
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(args, inputs, outputs, eval_ctx);
  ASSERT_EQ(eval_ctx.heap.total_size, W * H * sizeof(char));
  ASSERT_EQ(eval_ctx.heap.total_count, 1);

  for (int y = -H; y < 2 * H; ++y) {
    for (int x = -W; x < 2 * W; ++x) {
      if (0 <= x && x < W && 0 <= y && y < H) {
        ASSERT_EQ(out_buf(x, y), in_buf(x, y));
      } else {
        ASSERT_EQ(out_buf(x, y), 0);
      }
    }
  }
}

class copied_result : public testing::TestWithParam<int> {};

INSTANTIATE_TEST_SUITE_P(schedule, copied_result, testing::Range(0, 3));

TEST_P(copied_result, pipeline) {
  int schedule = GetParam();

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));

  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(short));

  var x(ctx, "x");
  var y(ctx, "y");

  // In this pipeline, the result is copied to the output. We should just compute the result directly in the output.
  func stencil = func::make(sum3x3<short>, {{in, {bounds(-1, 1) + x, bounds(-1, 1) + y}}}, {{intm, {x, y}}});
  func padded = func::make_copy({intm, {point(x), point(y)}}, {out, {x, y}});

  switch (schedule) {
  case 0: break;
  case 1:
    padded.loops({y});
    stencil.compute_root();
    break;
  case 2: padded.loops({y}); break;
  }

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 20;
  const int H = 10;
  buffer<short, 2> in_buf({W + 2, H + 2});
  in_buf.translate(-1, -1);
  buffer<short, 2> out_buf({W, H});

  init_random(in_buf);
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);
  ASSERT_EQ(eval_ctx.heap.total_count, 0);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      int correct = 0;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          correct += in_buf(x + dx, y + dy);
        }
      }
      ASSERT_EQ(correct, out_buf(x, y)) << x << " " << y;
    }
  }
}

class concatenated_result : public testing::TestWithParam<bool> {};

INSTANTIATE_TEST_SUITE_P(schedule, concatenated_result, testing::Values(false, true));

TEST_P(concatenated_result, pipeline) {
  bool no_alias_buffers = GetParam();
  // Make the pipeline
  node_context ctx;

  auto in1 = buffer_expr::make(ctx, "in1", 2, sizeof(short));
  auto in2 = buffer_expr::make(ctx, "in2", 2, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));

  auto intm1 = buffer_expr::make(ctx, "intm1", 2, sizeof(short));
  auto intm2 = buffer_expr::make(ctx, "intm2", 2, sizeof(short));

  var x(ctx, "x");
  var y(ctx, "y");

  // In this pipeline, the result is copied to the output. We should just compute the result directly in the output.
  func add1 = func::make(add_1<short>, {{{in1, {point(x), point(y)}}}}, {{{intm1, {x, y}}}});
  func add2 = func::make(add_1<short>, {{{in2, {point(x), point(y)}}}}, {{{intm2, {x, y}}}});
  func concatenated =
      func::make_concat({intm1, intm2}, {out, {x, y}}, 1, {0, in1->dim(1).extent(), out->dim(1).extent()});

  pipeline p = build_pipeline(ctx, {in1, in2}, {out}, build_options{.no_alias_buffers = no_alias_buffers});

  // Run the pipeline.
  const int W = 20;
  const int H1 = 4;
  const int H2 = 7;
  buffer<short, 2> in1_buf({W, H1});
  buffer<short, 2> in2_buf({W, H2});
  init_random(in1_buf);
  init_random(in2_buf);

  buffer<short, 2> out_buf({W, H1 + H2});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in1_buf, &in2_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);
  if (!no_alias_buffers) {
    ASSERT_EQ(eval_ctx.heap.total_count, 0);
  }

  for (int y = 0; y < H1 + H2; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y), (y < H1 ? in1_buf(x, y) : in2_buf(x, y - H1)) + 1);
    }
  }

  if (no_alias_buffers == true) {
    check_replica_pipeline(
        define_replica_pipeline(ctx, {in1, in2}, {out}, build_options{.no_alias_buffers = no_alias_buffers}));
  }
}

class transposed_result : public testing::TestWithParam<std::tuple<bool, std::vector<int>>> {};

INSTANTIATE_TEST_SUITE_P(schedule, transposed_result,
    testing::Combine(testing::Values(false, true), testing::Values(std::vector<int>{0, 1, 2}, std::vector<int>{0, 2, 1},
                                                       std::vector<int>{1, 2, 0}, std::vector<int>{0, 0, 0})));

template <typename T>
std::vector<T> permute(span<const int> p, const std::vector<T>& x) {
  std::vector<T> result(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    result[i] = x[p[i]];
  }
  return result;
}

bool is_permutation(span<const int> p) {
  std::vector<int> unpermuted(p.size());
  std::iota(unpermuted.begin(), unpermuted.end(), 0);
  return std::is_permutation(p.begin(), p.end(), unpermuted.begin());
}

TEST_P(transposed_result, pipeline) {
  bool no_alias_buffers = std::get<0>(GetParam());
  const std::vector<int>& permutation = std::get<1>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 3, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 3, sizeof(short));

  auto intm = buffer_expr::make(ctx, "intm", 3, sizeof(short));

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");

  // In this pipeline, the result is copied to the output. We should just compute the result directly in the output.
  func add = func::make(add_1<short>, {{{in, {point(x), point(y), point(z)}}}}, {{{intm, {x, y, z}}}});
  func transposed =
      func::make_copy({intm, permute<interval_expr>(permutation, {point(x), point(y), point(z)})}, {out, {x, y, z}});

  pipeline p = build_pipeline(ctx, {in}, {out}, build_options{.no_alias_buffers = no_alias_buffers});

  // Run the pipeline.
  const int W = 20;
  const int H = 4;
  const int D = 7;
  buffer<short, 3> in_buf(permute<index_t>(permutation, {W, H, D}));
  init_random(in_buf);

  buffer<short, 3> out_buf({W, H, D});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);
  if (is_permutation(permutation) && !no_alias_buffers) {
    ASSERT_EQ(eval_ctx.heap.total_count, 0);
  } else {
    ASSERT_EQ(eval_ctx.heap.total_count, 1);
  }

  for (int z = 0; z < D; ++z) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        ASSERT_EQ(out_buf(x, y, z), in_buf(permute<index_t>(permutation, {x, y, z})) + 1);
      }
    }
  }
}

TEST(stacked_result, pipeline) {
  // Make the pipeline
  node_context ctx;

  auto in1 = buffer_expr::make(ctx, "in1", 2, sizeof(short));
  auto in2 = buffer_expr::make(ctx, "in2", 2, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 3, sizeof(short));

  auto intm1 = buffer_expr::make(ctx, "intm1", 2, sizeof(short));
  auto intm2 = buffer_expr::make(ctx, "intm2", 2, sizeof(short));

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");

  // In this pipeline, the result is copied to the output. We should just compute the result directly in the output.
  func add1 = func::make(add_1<short>, {{{in1, {point(x), point(y)}}}}, {{{intm1, {x, y}}}});
  func add2 = func::make(add_1<short>, {{{in2, {point(x), point(y)}}}}, {{{intm2, {x, y}}}});
  func stacked = func::make_stack({intm1, intm2}, {out, {x, y, z}}, 2);

  pipeline p = build_pipeline(ctx, {in1, in2}, {out});

  // Run the pipeline.
  const int W = 20;
  const int H = 8;
  buffer<short, 2> in1_buf({W, H});
  buffer<short, 2> in2_buf({W, H});
  init_random(in1_buf);
  init_random(in2_buf);

  buffer<short, 3> out_buf({W, H, 2});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in1_buf, &in2_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);
  ASSERT_EQ(eval_ctx.heap.total_count, 0);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y, 0), in1_buf(x, y) + 1);
      ASSERT_EQ(out_buf(x, y, 1), in2_buf(x, y) + 1);
    }
  }

  check_replica_pipeline(define_replica_pipeline(ctx, {in1, in2}, {out}));
}

}  // namespace slinky