#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "base/test/bazel_util.h"
#include "builder/pipeline.h"
#include "builder/replica_pipeline.h"
#include "builder/test/context.h"
#include "builder/test/funcs.h"
#include "builder/test/util.h"
#include "runtime/expr.h"
#include "runtime/pipeline.h"

namespace slinky {

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

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, -y), in_buf(x, y));
    }
  }

  ASSERT_THAT(eval_ctx.heap.allocs, testing::UnorderedElementsAre(W * H * sizeof(char)));
}

class padded_copy : public testing::TestWithParam<std::tuple<int, int, bool, int>> {};

auto offsets = testing::Values(0, 1, -1, 10, -10);

INSTANTIATE_TEST_SUITE_P(offsets, padded_copy,
    testing::Combine(offsets, offsets, testing::Bool(), testing::Values(0, 1, 2)),
    test_params_to_string<padded_copy::ParamType>);

TEST_P(padded_copy, pipeline) {
  int offset_x = std::get<0>(GetParam());
  int offset_y = std::get<1>(GetParam());
  bool in_bounds = std::get<2>(GetParam());
  int split_y = std::get<3>(GetParam());
  std::vector<int> permutation = {0, 1};
  if (std::get<3>(GetParam())) {
    std::swap(permutation[0], permutation[1]);
  }

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(char));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(char));
  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(char));
  auto padded_intm = buffer_expr::make(ctx, "padded_intm", 2, sizeof(char));

  var x(ctx, "x");
  var y(ctx, "y");

  func copy_in = func::make(copy_2d<char>, {{in, {point(x), point(y)}}}, {{intm, {x, y}}});
  func crop = func::make_copy(
      {intm, permute<interval_expr>(permutation, {point(x + offset_x), point(y + offset_y)}), in->bounds()},
      {padded_intm, {x, y}}, {buffer_expr::make<char>(ctx, "padding", 3)});
  func copy_out = func::make(copy_2d<char>, {{padded_intm, {point(x), point(y)}}}, {{out, {x, y}}});

  if (split_y > 0) {
    copy_in.compute_root();
    copy_out.loops({{y, split_y}});
    padded_intm->store_at({&copy_out, y});
  }

  pipeline p = build_pipeline(ctx, {in}, {out});

  const int W = 8;
  const int H = 5;

  // Run the pipeline.
  buffer<char, 2> in_buf(permute<index_t>(permutation, {W, H}));
  if (in_bounds) {
    in_buf.translate(permute<index_t>(permutation, {offset_x, offset_y}));
  }
  init_random(in_buf);

  buffer<char, 2> out_buf({W, H});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      if (in_buf.contains(permute<index_t>(permutation, {x + offset_x, y + offset_y}))) {
        ASSERT_EQ(out_buf(x, y), in_buf(permute<index_t>(permutation, {x + offset_x, y + offset_y})));
      } else {
        ASSERT_EQ(out_buf(x, y), 3);
      }
    }
  }

  ASSERT_THAT(eval_ctx.heap.allocs, testing::UnorderedElementsAre(W * H * sizeof(char)));
  ASSERT_EQ(eval_ctx.copy_calls, split_y == 0 ? 1 : ceil_div(H, split_y));
}

class copy_sequence : public testing::TestWithParam<std::tuple<int, int>> {};

INSTANTIATE_TEST_SUITE_P(one_intermediate, copy_sequence,
    testing::Combine(testing::Values(1), testing::Range(0, 1 << 2)), test_params_to_string<copy_sequence::ParamType>);
INSTANTIATE_TEST_SUITE_P(two_intermediate, copy_sequence,
    testing::Combine(testing::Values(2), testing::Range(0, 1 << 3)), test_params_to_string<copy_sequence::ParamType>);
INSTANTIATE_TEST_SUITE_P(three_intermediate, copy_sequence,
    testing::Combine(testing::Values(3), testing::Range(0, 1 << 4)), test_params_to_string<copy_sequence::ParamType>);
INSTANTIATE_TEST_SUITE_P(four_intermediate, copy_sequence,
    testing::Combine(testing::Values(4), testing::Range(0, 1 << 5)), test_params_to_string<copy_sequence::ParamType>);

TEST_P(copy_sequence, pipeline) {
  int intermediate_count = std::get<0>(GetParam());
  int pad_mask = std::get<1>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 1, sizeof(char));
  auto out = buffer_expr::make(ctx, "out", 1, sizeof(char));
  std::vector<buffer_expr_ptr> intms;
  for (int i = 0; i < intermediate_count; ++i) {
    intms.push_back(buffer_expr::make(ctx, "intm" + std::to_string(i), 1, sizeof(char)));
  }

  in->dim(0).fold_factor = dim::unfolded;
  out->dim(0).fold_factor = dim::unfolded;

  var x(ctx, "x");

  const int N = 32;

  // The padding bounds depend on the stage, so we can look for the different paddings in the output.
  auto pad_min = [=](int stage) { return N / 2 + 4 - (stage * 2 + N / 4); };
  auto pad_max = [=](int stage) { return N / 2 + 4 + N / 4; };

  // Make a sequence of copies, where each copy copies from the next value in the previous buffer in the chain.
  // If the pad mask is one for that stage, we add padding outside the region [1, 4].
  auto make_copy = [&](int stage, buffer_expr_ptr src, buffer_expr_ptr dst) {
    if (((1 << stage) & pad_mask) != 0) {
      return func::make_copy(
          {src, {point(x + 1)}, {bounds(pad_min(stage), pad_max(stage))}}, {dst, {x}}, {
        buffer_expr::make<char>(ctx, "padding", stage)});
    } else {
      return func::make_copy({src, {point(x + 1)}}, {dst, {x}});
    }
  };

  std::vector<func> copies;
  copies.push_back(make_copy(0, in, intms.front()));
  for (int i = 0; i + 1 < intermediate_count; ++i) {
    copies.push_back(make_copy(1 + i, intms[i], intms[i + 1]));
  }
  copies.push_back(make_copy(intermediate_count, intms.back(), out));

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int offset = intermediate_count + 1;
  buffer<char, 1> in_buf({N});
  in_buf.translate(offset);
  init_random(in_buf);

  buffer<char, 1> out_buf({N});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int n = 0; n < N; ++n) {
    int correct = in_buf(n + offset);
    for (int i = 0; (1 << i) <= pad_mask; ++i) {
      if ((pad_mask & (1 << i)) == 0) continue;

      int index = n + offset - i;
      if (index < pad_min(i) || index > pad_max(i)) correct = i;
    }
    ASSERT_EQ(out_buf(n), correct);
  }

  if (pad_mask == 0) {
    ASSERT_EQ(eval_ctx.copy_calls, 1);
    ASSERT_EQ(eval_ctx.heap.allocs.size(), 0);
  } else {
    // NOTE: for this specific padding mask pattern, we can alias fewer buffers.
    if (intermediate_count == 4 && pad_mask % 8 == 5) {
      ASSERT_LE(eval_ctx.heap.allocs.size(), 2);
    } else {
      ASSERT_LE(eval_ctx.heap.allocs.size(), 1);
    }
    // TODO: Try to eliminate more copies when the padding appears between other copies.
  }
}

class copied_output : public testing::TestWithParam<std::tuple<int, int, int>> {};

INSTANTIATE_TEST_SUITE_P(schedule, copied_output, testing::Combine(testing::Range(0, 3), offsets, offsets),
    test_params_to_string<copied_output::ParamType>);

TEST_P(copied_output, pipeline) {
  int schedule = std::get<0>(GetParam());
  int offset_x = std::get<1>(GetParam());
  int offset_y = std::get<2>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));

  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(short));

  // Tell slinky the output is unfolded to allow aliasing it.
  out->dim(0).fold_factor = dim::unfolded;
  out->dim(1).fold_factor = dim::unfolded;

  var x(ctx, "x");
  var y(ctx, "y");

  // In this pipeline, the result is copied to the output. We should just compute the result directly in the output.
  func stencil = func::make(sum3x3<short>, {{in, {bounds(-1, 1) + x, bounds(-1, 1) + y}}}, {{intm, {x, y}}});
  func copied = func::make_copy({intm, {point(x + offset_x), point(y + offset_y)}}, {out, {x, y}});

  switch (schedule) {
  case 0: break;
  case 1:
    copied.loops({y});
    stencil.compute_root();
    break;
  case 2: copied.loops({y}); break;
  }

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 20;
  const int H = 10;
  buffer<short, 2> in_buf({W + 2, H + 2});
  in_buf.translate(-1 + offset_x, -1 + offset_y);
  buffer<short, 2> out_buf({W, H});

  init_random(in_buf);
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      int correct = 0;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          correct += in_buf(x + dx + offset_x, y + dy + offset_y);
        }
      }
      ASSERT_EQ(correct, out_buf(x, y)) << x << " " << y;
    }
  }

  ASSERT_EQ(eval_ctx.heap.allocs.size(), 0);
  ASSERT_EQ(eval_ctx.copy_calls, 0);
}

class copied_input : public testing::TestWithParam<std::tuple<int, int, int>> {};

INSTANTIATE_TEST_SUITE_P(schedule, copied_input, testing::Combine(testing::Range(0, 3), offsets, offsets),
    test_params_to_string<copied_input::ParamType>);

TEST_P(copied_input, pipeline) {
  int schedule = std::get<0>(GetParam());
  int offset_x = std::get<1>(GetParam());
  int offset_y = std::get<2>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));

  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(short));

  // If we want to alias intermediate buffer to the input buffer,
  // we need to tell aliaser that input is unfolded and it's safe to alias.
  in->dim(0).fold_factor = dim::unfolded;
  in->dim(1).fold_factor = dim::unfolded;

  var x(ctx, "x");
  var y(ctx, "y");

  // In this pipeline, the result is copied to the output. We should just compute the result directly in the output.
  func copied = func::make_copy({in, {point(x + offset_x), point(y + offset_y)}}, {intm, {x, y}});
  func stencil = func::make(sum3x3<short>, {{intm, {bounds(-1, 1) + x, bounds(-1, 1) + y}}}, {{out, {x, y}}});

  switch (schedule) {
  case 0: break;
  case 1:
    copied.loops({y});
    stencil.compute_root();
    break;
  case 2: copied.loops({y}); break;
  }

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 20;
  const int H = 10;
  buffer<short, 2> in_buf({W + 2, H + 2});
  in_buf.translate(-1 + offset_x, -1 + offset_y);
  buffer<short, 2> out_buf({W, H});

  init_random(in_buf);
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      int correct = 0;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          correct += in_buf(x + dx + offset_x, y + dy + offset_y);
        }
      }
      ASSERT_EQ(correct, out_buf(x, y)) << x << " " << y;
    }
  }

  ASSERT_EQ(eval_ctx.heap.allocs.size(), 0);
  ASSERT_EQ(eval_ctx.copy_calls, 0);
}

class concatenated_output : public testing::TestWithParam<bool> {};

INSTANTIATE_TEST_SUITE_P(schedule, concatenated_output, testing::Bool());

TEST_P(concatenated_output, pipeline) {
  bool no_alias_buffers = GetParam();
  // Make the pipeline
  node_context ctx;

  auto in1 = buffer_expr::make(ctx, "in1", 2, sizeof(short));
  auto in2 = buffer_expr::make(ctx, "in2", 2, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));

  auto intm1 = buffer_expr::make(ctx, "intm1", 2, sizeof(short));
  auto intm2 = buffer_expr::make(ctx, "intm2", 2, sizeof(short));

  // If we want to alias intermediate buffer to the output buffer,
  // we need to tell aliaser that output is unfolded and it's safe to alias.
  out->dim(0).fold_factor = dim::unfolded;
  out->dim(1).fold_factor = dim::unfolded;

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

  for (int y = 0; y < H1 + H2; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y), (y < H1 ? in1_buf(x, y) : in2_buf(x, y - H1)) + 1);
    }
  }

  if (!no_alias_buffers) {
    ASSERT_EQ(eval_ctx.heap.allocs.size(), 0);
    ASSERT_EQ(eval_ctx.copy_calls, 0);
  }

  if (no_alias_buffers == true) {
    check_replica_pipeline(
        define_replica_pipeline(ctx, {in1, in2}, {out}, build_options{.no_alias_buffers = no_alias_buffers}));
  }
}

class transposed_output : public testing::TestWithParam<std::tuple<bool, int, int, int>> {};

auto iota3 = testing::Values(0, 1, 2);

INSTANTIATE_TEST_SUITE_P(schedule, transposed_output, testing::Combine(testing::Bool(), iota3, iota3, iota3),
    test_params_to_string<transposed_output::ParamType>);

TEST_P(transposed_output, pipeline) {
  bool no_alias_buffers = std::get<0>(GetParam());
  std::vector<int> permutation = {std::get<1>(GetParam()), std::get<2>(GetParam()), std::get<3>(GetParam())};

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 3, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 3, sizeof(short));

  auto intm = buffer_expr::make(ctx, "intm", 3, sizeof(short));

  // If we want to alias intermediate buffer to the output buffer,
  // we need to tell aliaser that output is unfolded and it's safe to alias.
  out->dim(0).fold_factor = dim::unfolded;
  out->dim(1).fold_factor = dim::unfolded;
  out->dim(2).fold_factor = dim::unfolded;

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

  for (int z = 0; z < D; ++z) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        ASSERT_EQ(out_buf(x, y, z), in_buf(permute<index_t>(permutation, {x, y, z})) + 1);
      }
    }
  }

  if (is_permutation(permutation) && !no_alias_buffers) {
    ASSERT_EQ(eval_ctx.heap.allocs.size(), 0);
    ASSERT_EQ(eval_ctx.copy_calls, 0);
  } else {
    ASSERT_EQ(eval_ctx.heap.allocs.size(), 1);
  }
}

TEST(stacked_output, pipeline) {
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

  // If we want to alias intermediate buffer to the output buffer,
  // we need to tell aliaser that output is unfolded and it's safe to alias.
  out->dim(0).fold_factor = dim::unfolded;
  out->dim(1).fold_factor = dim::unfolded;
  out->dim(2).fold_factor = dim::unfolded;

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

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y, 0), in1_buf(x, y) + 1);
      ASSERT_EQ(out_buf(x, y, 1), in2_buf(x, y) + 1);
    }
  }

  ASSERT_EQ(eval_ctx.heap.allocs.size(), 0);
  ASSERT_EQ(eval_ctx.copy_calls, 0);

  check_replica_pipeline(define_replica_pipeline(ctx, {in1, in2}, {out}));
}

class broadcasted_elementwise : public testing::TestWithParam<std::tuple<bool, int, int>> {};

INSTANTIATE_TEST_SUITE_P(dim, broadcasted_elementwise,
    testing::Combine(testing::Bool(), testing::Range(0, 2), testing::Values(0, 1)),
    test_params_to_string<broadcasted_elementwise::ParamType>);

TEST_P(broadcasted_elementwise, input) {
  bool no_alias_buffers = std::get<0>(GetParam());
  const int broadcast_dim = std::get<1>(GetParam());
  const int split_y = std::get<2>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto in1 = buffer_expr::make(ctx, "in1", 2, sizeof(int));
  auto in2 = buffer_expr::make(ctx, "in2", 2, sizeof(int));
  auto in2_broadcasted = buffer_expr::make(ctx, "in2_broadcasted", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");

  box_expr bounds = {
      select(in2->dim(0).extent() == 1, point(in2->dim(0).min()), point(x)),
      select(in2->dim(1).extent() == 1, point(in2->dim(1).min()), point(y)),
  };
  func broadcast = func::make_copy({in2, bounds}, {in2_broadcasted, {x, y}});
  func f = func::make(
      subtract<int>, {{in1, {point(x), point(y)}}, {in2_broadcasted, {point(x), point(y)}}}, {{out, {x, y}}});

  if (split_y > 0) {
    f.loops({{y, split_y}});
  }

  pipeline p = build_pipeline(ctx, {in1, in2}, {out}, build_options{.no_alias_buffers = no_alias_buffers});

  // Run the pipeline.
  const int W = 20;
  const int H = 4;
  buffer<int, 2> in1_buf({W, H});
  buffer<int, 2> in2_buf({W, H});
  in2_buf.dim(broadcast_dim).set_extent(1);
  init_random(in1_buf);
  init_random(in2_buf);

  buffer<int, 2> out_buf({W, H});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in1_buf, &in2_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      index_t i[] = {x, y};
      i[broadcast_dim] = 0;
      ASSERT_EQ(out_buf(x, y), in1_buf(x, y) - in2_buf(i));
    }
  }
  // TODO(vksnk): doesn't fold because mins of the folded buffers are not aligned.
  // if (!no_alias_buffers ) {
  //   ASSERT_EQ(eval_ctx.heap.allocs.size(), 0);
  // }
}

TEST_P(broadcasted_elementwise, internal) {
  bool no_alias_buffers = std::get<0>(GetParam());
  const int broadcast_dim = std::get<1>(GetParam());
  const int split_y = std::get<2>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto in1 = buffer_expr::make(ctx, "in1", 2, sizeof(int));
  auto in2 = buffer_expr::make(ctx, "in2", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(int));
  auto intm_broadcasted = buffer_expr::make(ctx, "intm_broadcasted", 2, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");

  func f = func::make(add_1<int>, {{in2, {point(x), point(y)}}}, {{intm, {x, y}}}, call_stmt::attributes{.name = "f"});

  // Use the bounds of in2 to decide how to broadcast. We can't use the bounds of an internally allocated buffer.
  box_expr bounds = {
      select(in2->dim(0).extent() == 1, point(in2->dim(0).min()), point(x)),
      select(in2->dim(1).extent() == 1, point(in2->dim(1).min()), point(y)),
  };
  func broadcast = func::make_copy({intm, bounds}, {intm_broadcasted, {x, y}});
  func g = func::make(subtract<int>, {{in1, {point(x), point(y)}}, {intm_broadcasted, {point(x), point(y)}}},
      {{out, {x, y}}}, call_stmt::attributes{.name = "g"});

  if (split_y > 0) {
    g.loops({{y, split_y}});
  }
  pipeline p = build_pipeline(ctx, {in1, in2}, {out}, build_options{.no_alias_buffers = no_alias_buffers});

  // Run the pipeline.
  const int W = 20;
  const int H = 4;
  buffer<int, 2> in1_buf({W, H});
  buffer<int, 2> in2_buf({W, H});
  in2_buf.dim(broadcast_dim).set_extent(1);
  init_random(in1_buf);
  init_random(in2_buf);

  buffer<int, 2> out_buf({W, H});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in1_buf, &in2_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      index_t i[] = {x, y};
      i[broadcast_dim] = 0;
      ASSERT_EQ(out_buf(x, y), in1_buf(x, y) - (in2_buf(i) + 1));
    }
  }

  if (!no_alias_buffers) {
    ASSERT_EQ(eval_ctx.heap.allocs.size(), 1);
    ASSERT_EQ(eval_ctx.copy_calls, 0);
  }
}

TEST_P(broadcasted_elementwise, constant) {
  bool no_alias_buffers = std::get<0>(GetParam());
  const int broadcast_dim = std::get<1>(GetParam());
  const int split_y = std::get<2>(GetParam());

  // Make the pipeline
  node_context ctx;

  const int W = 20;
  const int H = 4;

  buffer<int, 2> in2_buf({W, H});
  in2_buf.dim(broadcast_dim).set_extent(1);
  init_random(in2_buf);

  auto in1 = buffer_expr::make(ctx, "in1", 2, sizeof(int));
  auto in2 = buffer_expr::make(ctx, "in2", raw_buffer::make_copy(in2_buf));
  auto in2_broadcasted = buffer_expr::make(ctx, "in2_broadcasted", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");

  box_expr bounds = {
      select(in2->dim(0).extent() == 1, point(in2->dim(0).min()), point(x)),
      select(in2->dim(1).extent() == 1, point(in2->dim(1).min()), point(y)),
  };
  func broadcast = func::make_copy({in2, bounds}, {in2_broadcasted, {x, y}});
  func f = func::make(
      subtract<int>, {{in1, {point(x), point(y)}}, {in2_broadcasted, {point(x), point(y)}}}, {{out, {x, y}}});

  if (split_y > 0) {
    f.loops({{y, split_y}});
  }

  pipeline p = build_pipeline(ctx, {in1}, {out}, build_options{.no_alias_buffers = no_alias_buffers});

  // Run the pipeline.
  buffer<int, 2> in1_buf({W, H});
  init_random(in1_buf);

  buffer<int, 2> out_buf({W, H});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in1_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      index_t i[] = {x, y};
      i[broadcast_dim] = 0;
      ASSERT_EQ(out_buf(x, y), in1_buf(x, y) - in2_buf(i));
    }
  }
  // TODO(vksnk): doesn't fold because mins of the folded buffers are not aligned.
  // if (!no_alias_buffers ) {
  //   ASSERT_EQ(eval_ctx.heap.allocs.size(), 0);
  // }
}

class constrained_transpose : public testing::TestWithParam<std::tuple<bool, bool, bool>> {};

INSTANTIATE_TEST_SUITE_P(dim, constrained_transpose,
    testing::Combine(testing::Bool(), testing::Bool(), testing::Bool()),
    test_params_to_string<constrained_transpose::ParamType>);

TEST_P(constrained_transpose, pipeline) {
  const bool no_alias_buffers = std::get<0>(GetParam());
  const bool intm1_fixed = std::get<1>(GetParam());
  const bool intm2_fixed = std::get<2>(GetParam());
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 3, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 3, sizeof(short));

  auto intm1 = buffer_expr::make(ctx, "intm1", 3, sizeof(short));
  auto intm2 = buffer_expr::make(ctx, "intm2", 3, sizeof(short));

  if (intm1_fixed) {
    intm1->dim(0).stride = sizeof(short);
    intm1->dim(1).stride = sizeof(short) * in->dim(0).extent();
    intm1->dim(2).stride = sizeof(short) * in->dim(0).extent() * in->dim(0).extent();
  }
  if (intm2_fixed) {
    intm2->dim(0).stride = sizeof(short);
    intm2->dim(1).stride = sizeof(short) * out->dim(0).extent();
    intm2->dim(2).stride = sizeof(short) * out->dim(0).extent() * out->dim(0).extent();
  }

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");

  // In this pipeline, the result is copied to the output. We should just compute the result directly in the output.
  func add_in = func::make(add_1<short>, {{{in, {point(x), point(y), point(z)}}}}, {{{intm1, {x, y, z}}}});
  func transposed = func::make_copy({intm1, {point(x), point(z), point(y)}}, {intm2, {x, y, z}});
  func add_out = func::make(add_1<short>, {{intm2, {point(x), point(y), point(z)}}}, {{{out, {x, y, z}}}});

  pipeline p = build_pipeline(ctx, {in}, {out}, build_options{.no_alias_buffers = no_alias_buffers});

  // Run the pipeline.
  const int W = 20;
  const int H = 4;
  const int D = 7;
  buffer<short, 3> in_buf({W, D, H});
  init_random(in_buf);

  buffer<short, 3> out_buf({W, H, D});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int z = 0; z < D; ++z) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        ASSERT_EQ(out_buf(x, y, z), in_buf(x, z, y) + 2);
      }
    }
  }

  if (!no_alias_buffers) {
    if (intm1_fixed && intm2_fixed) {
      // Both the intermediates have stride constraints, we can't alias anything.
      ASSERT_EQ(eval_ctx.heap.allocs.size(), 2);
    } else {
      ASSERT_EQ(eval_ctx.heap.allocs.size(), 1);
      ASSERT_EQ(eval_ctx.copy_calls, 0);
    }
  } else {
    ASSERT_EQ(eval_ctx.heap.allocs.size(), 2);
  }
}

}  // namespace slinky
