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

class stencil : public testing::TestWithParam<std::tuple<int, int, int>> {};

INSTANTIATE_TEST_SUITE_P(stride_dilation_size, stencil,
    testing::Combine(testing::Values(1, 2, 3), testing::Values(1, 2, 3), testing::Values(1, 2, 3)),
    test_params_to_string<stencil::ParamType>);

TEST_P(stencil, x_dx) {
  const int S = std::get<0>(GetParam());
  const int D = std::get<1>(GetParam());
  const int K = std::get<2>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 1, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));

  var x(ctx, "x");
  var dx(ctx, "dx");

  func copy = func::make_copy({in, {point(x * S + dx * D)}}, {out, {x, dx}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int N = 10;

  buffer<short, 2> out_buf({N, K});
  out_buf.allocate();

  buffer<short, 1> in_buf({(N - 1) * S + (K - 1) * D + 1});
  init_random(in_buf);

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      ASSERT_EQ(out_buf(n, k), in_buf(n * S + k * D));
    }
  }
}

TEST_P(stencil, dx_x) {
  const int S = std::get<0>(GetParam());
  const int D = std::get<1>(GetParam());
  const int K = std::get<2>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 1, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));

  var x(ctx, "x");
  var dx(ctx, "dx");

  func copy = func::make_copy({in, {point(x * S + dx * D)}}, {out, {dx, x}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int N = 10;

  buffer<short, 2> out_buf({K, N});
  out_buf.allocate();

  buffer<short, 1> in_buf({(N - 1) * S + (K - 1) * D + 1});
  init_random(in_buf);

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      ASSERT_EQ(out_buf(k, n), in_buf(n * S + k * D));
    }
  }
}

TEST_P(stencil, x_y_dx_dy) {
  const int S = std::get<0>(GetParam());
  const int D = std::get<1>(GetParam());
  const int K = std::get<2>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 4, sizeof(short));

  var x(ctx, "x");
  var y(ctx, "y");
  var dx(ctx, "dx");
  var dy(ctx, "dy");

  func copy = func::make_copy({in, {point(x * S + dx * D), point(y * S + dy * D)}}, {out, {x, y, dx, dy}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 10;
  const int H = 5;

  buffer<short, 4> out_buf({W, H, K, K});
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
      for (int dy = 0; dy < K; ++dy) {
        for (int dx = 0; dx < K; ++dx) {
          ASSERT_EQ(out_buf(x, y, dx, dy), in_buf(x * S + dx * D, y * S + dy * D));
        }
      }
    }
  }
}

TEST_P(stencil, x_dx_y_dy) {
  const int S = std::get<0>(GetParam());
  const int D = std::get<1>(GetParam());
  const int K = std::get<2>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 4, sizeof(short));

  var x(ctx, "x");
  var y(ctx, "y");
  var dx(ctx, "dx");
  var dy(ctx, "dy");

  func copy = func::make_copy({in, {point(x * S + dx * D), point(y * S + dy * D)}}, {out, {x, dx, y, dy}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 10;
  const int H = 5;

  buffer<short, 4> out_buf({W, K, H, K});
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
      for (int dy = 0; dy < K; ++dy) {
        for (int dx = 0; dx < K; ++dx) {
          ASSERT_EQ(out_buf(x, dx, y, dy), in_buf(x * S + dx * D, y * S + dy * D));
        }
      }
    }
  }
}

}  // namespace slinky
