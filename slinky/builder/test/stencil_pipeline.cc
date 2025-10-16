#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <functional>
#include <vector>

#include "slinky/builder/pipeline.h"
#include "slinky/builder/test/context.h"
#include "slinky/builder/test/funcs.h"
#include "slinky/builder/test/util.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/pipeline.h"

namespace slinky {

class stencil : public testing::TestWithParam<std::tuple<bool, int, int, int, int>> {};

INSTANTIATE_TEST_SUITE_P(stride_dilation_size, stencil,
    testing::Combine(testing::Bool(), testing::Values(1, 2, 3), testing::Values(1, 2, 3), testing::Values(1, 2, 3),
        testing::Values(0, 3)),
    test_params_to_string<stencil::ParamType>);

TEST_P(stencil, 1d) {
  const bool no_alias_buffers = std::get<0>(GetParam());
  const int S = std::get<1>(GetParam());
  const int D = std::get<2>(GetParam());
  const int K = std::get<3>(GetParam());
  const int split = std::get<4>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 1, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 1, sizeof(short));

  in->dim(0).fold_factor = dim::unfolded;

  auto stencil = buffer_expr::make(ctx, "stencil", 2, sizeof(short));

  var x(ctx, "x");
  var dx(ctx, "dx");

  // This test computes the following stencil operation, in this case a convolution with a kernel of 1s:
  //
  //  for i in [0, N):
  //    for k in [0, K):
  //      out[i] += in[i * S + k * D]
  //
  // Using the following approach:
  // 1. Make a copy of the input such that stencil(x, dx) = in(x * S + dx * D)
  // 2. Compute a reduction of the dx dimension
  //
  // We expect slinky to alias the copy.
  func stencil_copy = func::make_copy({in, {point(x * S + dx * D)}}, {stencil, {x, dx}});
  auto sum_1 = [K](const buffer<const short>& in, const buffer<short>& out) {
    return sum(in, out, /*dims=*/{{1, 0, K - 1}});
  };
  func reduce = func::make(std::move(sum_1), {{stencil, {point(x), min_extent(0, K)}}}, {{{out, {x}}}});

  if (split > 0) {
    reduce.loops({{x, split}});
  }

  pipeline p = build_pipeline(ctx, {in}, {out}, build_options{.no_alias_buffers = no_alias_buffers});

  // Run the pipeline.

  const int N = 10;

  buffer<short, 1> out_buf({N});
  out_buf.allocate();

  buffer<short, 1> in_buf({(N - 1) * S + (K - 1) * D + 1});
  init_random(in_buf);

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int n = 0; n < N; ++n) {
    int expected = 0;
    for (int k = 0; k < K; ++k) {
      expected += in_buf(n * S + k * D);
    }
    ASSERT_EQ(expected, out_buf(n));
  }

  if (!no_alias_buffers) {
    ASSERT_EQ(eval_ctx.heap.allocs.size(), 0);
    ASSERT_EQ(eval_ctx.copy_calls, 0);
  } else {
    ASSERT_EQ(eval_ctx.heap.allocs.size(), 1);
  }
}

TEST_P(stencil, 2d) {
  const bool no_alias_buffers = std::get<0>(GetParam());
  const int S = std::get<1>(GetParam());
  const int D = std::get<2>(GetParam());
  const int K = std::get<3>(GetParam());
  const int split = std::get<4>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));

  in->dim(0).fold_factor = dim::unfolded;
  in->dim(1).fold_factor = dim::unfolded;

  auto stencil = buffer_expr::make(ctx, "stencil", 4, sizeof(short));

  var x(ctx, "x");
  var y(ctx, "y");
  var dx(ctx, "dx");
  var dy(ctx, "dy");

  // See the 1d version for a 1D description of what this is doing in 2D.
  func stencil_copy = func::make_copy({in, {point(x * S + dx * D), point(y * S + dy * D)}}, {stencil, {x, y, dx, dy}});
  auto sum_23 = [K](const buffer<const short>& in, const buffer<short>& out) {
    return sum(in, out, /*dims=*/{{2, 0, K - 1}, {3, 0, K - 1}});
  };
  func reduce = func::make(
      std::move(sum_23), {{stencil, {point(x), point(y), min_extent(0, K), min_extent(0, K)}}}, {{{out, {x, y}}}});

  if (split > 0) {
    reduce.loops({{x, split}, {y, split}});
  }

  pipeline p = build_pipeline(ctx, {in}, {out}, build_options{.no_alias_buffers = no_alias_buffers});

  // Run the pipeline.
  const int W = 10;
  const int H = 5;

  buffer<short, 2> out_buf({W, H});
  out_buf.allocate();

  buffer<short, 2> in_buf({(W - 1) * S + (K - 1) * D + 1, (H - 1) * S + (K - 1) * D + 1});
  init_random(in_buf);

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      int expected = 0;
      for (int dy = 0; dy < K; ++dy) {
        for (int dx = 0; dx < K; ++dx) {
          expected += in_buf(x * S + dx * D, y * S + dy * D);
        }
      }
      ASSERT_EQ(expected, out_buf(x, y));
    }
  }

  if (!no_alias_buffers) {
    ASSERT_EQ(eval_ctx.heap.allocs.size(), 0);
    ASSERT_EQ(eval_ctx.copy_calls, 0);
  } else {
    ASSERT_EQ(eval_ctx.heap.allocs.size(), 1);
  }
}

class stencil_variable : public testing::TestWithParam<std::tuple<bool, int>> {};

INSTANTIATE_TEST_SUITE_P(alias_size_split, stencil_variable,
    testing::Combine(testing::Bool(), testing::Values(0, 3)),
    test_params_to_string<stencil_variable::ParamType>);

TEST_P(stencil_variable, 1d) {
  const bool no_alias_buffers = std::get<0>(GetParam());
  const int split = std::get<1>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 1, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 1, sizeof(short));

  in->dim(0).fold_factor = dim::unfolded;

  auto stencil = buffer_expr::make(ctx, "stencil", 2, sizeof(short));

  var x(ctx, "x");
  var dx(ctx, "dx");
  var s(ctx, "s");
  var d(ctx, "d");
  var k(ctx, "k");

  // This test computes the following stencil operation, in this case a convolution with a kernel of 1s:
  //
  //  for i in [0, N):
  //    for k in [0, K):
  //      out[i] += in[i * s + k * d]
  //
  // Using the following approach:
  // 1. Make a copy of the input such that stencil(x, dx) = in(x * s + dx * d)
  // 2. Compute a reduction of the dx dimension
  //
  // We expect slinky to alias the copy.
  func stencil_copy = func::make_copy({in, {point(x * max(1, s) + dx * max(1, d))}}, {stencil, {x, dx}});
  auto sum_1 = [](const call_stmt* op, eval_context& ctx) {
    const buffer<const short>& in = *ctx.lookup_buffer<const short>(op->inputs[0]);
    const buffer<short>& out = *ctx.lookup_buffer<short>(op->outputs[0]);
    int k = evaluate(op->scalars[0], ctx);
    return sum(in, out, /*dims=*/{{1, 0, k - 1}});
  };
  func reduce = func(std::move(sum_1), {{stencil, {point(x), min_extent(0, k)}}}, {{{out, {x}}}}, {k});

  if (split > 0) {
    reduce.loops({{x, split}});
  }

  pipeline p = build_pipeline(ctx, {s, d, k}, {in}, {out}, {}, build_options{.no_alias_buffers = no_alias_buffers});

  // Run the pipeline with varying stride/dilation parameters.
  for (int S : {1, 2, 3}) {
    for (int D : {1, 2, 3}) {
      for (int K : {1, 2, 3}) {
        const int N = 10;

        buffer<short, 1> out_buf({N});
        out_buf.allocate();

        buffer<short, 1> in_buf({(N - 1) * S + (K - 1) * D + 1});
        init_random(in_buf);

        // Not having span(std::initializer_list<T>) is unfortunate.
        index_t args[] = {S, D, K};
        const raw_buffer* inputs[] = {&in_buf};
        const raw_buffer* outputs[] = {&out_buf};
        test_context eval_ctx;
        p.evaluate(args, inputs, outputs, eval_ctx);

        for (int n = 0; n < N; ++n) {
          int expected = 0;
          for (int k = 0; k < K; ++k) {
            expected += in_buf(n * S + k * D);
          }
          ASSERT_EQ(expected, out_buf(n));
        }

        if (!no_alias_buffers) {
          ASSERT_EQ(eval_ctx.heap.allocs.size(), 0);
          ASSERT_EQ(eval_ctx.copy_calls, 0);
        } else {
          ASSERT_EQ(eval_ctx.heap.allocs.size(), 1);
        }
      }
    }
  }
}

}  // namespace slinky
