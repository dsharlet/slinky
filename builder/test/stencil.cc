#include <gtest/gtest.h>

#include <cassert>
#include <vector>

#include "builder/pipeline.h"
#include "builder/test/context.h"
#include "builder/test/util.h"
#include "runtime/expr.h"
#include "runtime/pipeline.h"

namespace slinky {

constexpr int offsets[] = {0, 2, -3};

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
  test_context eval_ctx;

  func copy = func::make_copy({in, {point(x * S + dx * D)}}, {out, {x, dx}}, eval_ctx.copy);

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int N = 10;

  for (int min_n : offsets) {
    for (int min_k : offsets) {
      buffer<short, 2> out_buf({N, K});
      out_buf.allocate();
      out_buf.translate(min_n, min_k);

      buffer<short, 1> in_buf({(N - 1) * S + (K - 1) * D + 1});
      init_random(in_buf);
      in_buf.translate(min_n * S + min_k * D);

      // Not having span(std::initializer_list<T>) is unfortunate.
      const raw_buffer* inputs[] = {&in_buf};
      const raw_buffer* outputs[] = {&out_buf};
      eval_ctx.copy_calls = 0;
      p.evaluate(inputs, outputs, eval_ctx);

      for (int n = min_n; n < min_n + N; ++n) {
        for (int k = min_k; k < min_k + K; ++k) {
          ASSERT_EQ(out_buf(n, k), in_buf(n * S + k * D));
        }
      }

      ASSERT_EQ(eval_ctx.copy_calls, 1);
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
  test_context eval_ctx;

  func copy = func::make_copy({in, {point(x * S + dx * D)}}, {out, {dx, x}}, eval_ctx.copy);

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int N = 10;

  for (int min_n : offsets) {
    for (int min_k : offsets) {
      buffer<short, 2> out_buf({K, N});
      out_buf.allocate();
      out_buf.translate(min_k, min_n);

      buffer<short, 1> in_buf({(N - 1) * S + (K - 1) * D + 1});
      init_random(in_buf);
      in_buf.translate(min_n * S + min_k * D);

      // Not having span(std::initializer_list<T>) is unfortunate.
      const raw_buffer* inputs[] = {&in_buf};
      const raw_buffer* outputs[] = {&out_buf};
      eval_ctx.copy_calls = 0;
      p.evaluate(inputs, outputs, eval_ctx);

      for (int n = min_n; n < min_n + N; ++n) {
        for (int k = min_k; k < min_k + K; ++k) {
          ASSERT_EQ(out_buf(k, n), in_buf(n * S + k * D));
        }
      }

      ASSERT_EQ(eval_ctx.copy_calls, 1);
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
  test_context eval_ctx;

  func copy = func::make_copy({in, {point(x * S + dx * D), point(y * S + dy * D)}}, {out, {x, y, dx, dy}}, eval_ctx.copy);

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 10;
  const int H = 5;
  for (int min_x : offsets) {
    for (int min_y : offsets) {
      for (int min_dx : offsets) {
        for (int min_dy : offsets) {
          buffer<short, 4> out_buf({W, H, K, K});
          out_buf.allocate();
          out_buf.translate(min_x, min_y, min_dx, min_dy);

          buffer<short, 2> in_buf({(W - 1) * S + (K - 1) * D + 1, (H - 1) * S + (K - 1) * D + 1});
          init_random(in_buf);
          in_buf.translate(min_x * S + min_dx * D, min_y * S + min_dy * D);

          // Not having span(std::initializer_list<T>) is unfortunate.
          const raw_buffer* inputs[] = {&in_buf};
          const raw_buffer* outputs[] = {&out_buf};
          eval_ctx.copy_calls = 0;
          p.evaluate(inputs, outputs, eval_ctx);

          for (int y = min_y; y < min_y + H; ++y) {
            for (int x = min_x; x < min_x + W; ++x) {
              for (int dy = min_dy; dy < min_dy + K; ++dy) {
                for (int dx = min_dx; dx < min_dx + K; ++dx) {
                  ASSERT_EQ(out_buf(x, y, dx, dy), in_buf(x * S + dx * D, y * S + dy * D));
                }
              }
            }
          }

          ASSERT_EQ(eval_ctx.copy_calls, 1);
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
  test_context eval_ctx;

  func copy = func::make_copy({in, {point(x * S + dx * D), point(y * S + dy * D)}}, {out, {x, dx, y, dy}}, eval_ctx.copy);

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 10;
  const int H = 5;

  for (int min_x : offsets) {
    for (int min_y : offsets) {
      for (int min_dx : offsets) {
        for (int min_dy : offsets) {
          buffer<short, 4> out_buf({W, K, H, K});
          out_buf.allocate();
          out_buf.translate(min_x, min_dx, min_y, min_dy);

          buffer<short, 2> in_buf({(W - 1) * S + (K - 1) * D + 1, (H - 1) * S + (K - 1) * D + 1});
          init_random(in_buf);
          in_buf.translate(min_x * S + min_dx * D, min_y * S + min_dy * D);

          // Not having span(std::initializer_list<T>) is unfortunate.
          const raw_buffer* inputs[] = {&in_buf};
          const raw_buffer* outputs[] = {&out_buf};
          eval_ctx.copy_calls = 0;
          p.evaluate(inputs, outputs, eval_ctx);

          for (int y = min_y; y < min_y + H; ++y) {
            for (int x = min_x; x < min_x + W; ++x) {
              for (int dy = min_dy; dy < min_dy + K; ++dy) {
                for (int dx = min_dx; dx < min_dx + K; ++dx) {
                  ASSERT_EQ(out_buf(x, dx, y, dy), in_buf(x * S + dx * D, y * S + dy * D));
                }
              }
            }
          }

          ASSERT_EQ(eval_ctx.copy_calls, 1);
        }
      }
    }
  }
}

}  // namespace slinky
