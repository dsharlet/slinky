#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "builder/pipeline.h"
#include "builder/test/funcs.h"
#include "builder/test/util.h"
#include "runtime/expr.h"
#include "runtime/pipeline.h"

namespace slinky {

// This implementation of softmax is not intended to be fast, it is only intended to model the data dependencies.
index_t max_dim0(const buffer<const float>& in, const buffer<float>& max_in) {
  for (index_t b = max_in.dim(0).begin(); b < max_in.dim(0).end(); ++b) {
    max_in(b) = -std::numeric_limits<float>::infinity();
    for (index_t c = in.dim(0).begin(); c < in.dim(0).end(); ++c) {
      max_in(b) = std::max(max_in(b), in(c, b));
    }
  }
  return 0;
}

index_t sum_exp(const buffer<const float>& in, const buffer<const float>& max_in, const buffer<float>& exp_in,
    const buffer<float>& sum_exp_in) {
  assert(exp_in.dim(1).min() == sum_exp_in.dim(0).min());
  assert(exp_in.dim(1).max() == sum_exp_in.dim(0).max());
  for (index_t b = exp_in.dim(1).begin(); b < exp_in.dim(1).end(); ++b) {
    sum_exp_in(b) = 0.0f;
    for (index_t c = exp_in.dim(0).begin(); c < exp_in.dim(0).end(); ++c) {
      exp_in(c, b) = std::exp(in(c, b) - max_in(b));
      sum_exp_in(b) += exp_in(c, b);
    }
  }
  return 0;
}

index_t normalize(const buffer<const float>& in, const buffer<const float>& sum_exp_in, const buffer<float>& out) {
  for (index_t b = out.dim(1).begin(); b < out.dim(1).end(); ++b) {
    for (index_t c = out.dim(0).begin(); c < out.dim(0).end(); ++c) {
      out(c, b) = in(c, b) / sum_exp_in(b);
    }
  }
  return 0;
}

index_t fused_softmax(const buffer<const float>& in, const buffer<float>& out) {
  buffer<float, 1> exp_in({out.dim(0).extent()});
  exp_in.allocate();
  for (index_t b = out.dim(1).begin(); b < out.dim(1).end(); ++b) {
    float max_in = -std::numeric_limits<float>::infinity();
    for (index_t c = out.dim(0).begin(); c < out.dim(0).end(); ++c) {
      max_in = std::max(max_in, in(c, b));
    }

    float sum_exp_in = 0.0f;
    for (index_t c = out.dim(0).begin(); c < out.dim(0).end(); ++c) {
      exp_in(c) = std::exp(in(c, b) - max_in);
      sum_exp_in += exp_in(c);
    }

    for (index_t c = out.dim(0).begin(); c < out.dim(0).end(); ++c) {
      out(c, b) = exp_in(c) / sum_exp_in;
    }
  }
  return 0;
}

std::vector<float> run_fused_softmax(std::initializer_list<const float> x) {
  const index_t n = x.size();

  buffer<float, 2> in({n, 1});
  buffer<float, 2> out({n, 1});
  in.allocate();
  out.allocate();

  float* in_x = &in(0, 0);
  for (const float& i : x) {
    *in_x++ = i;
  }

  fused_softmax(in.cast<const float>(), out.cast<float>());

  std::vector<float> result(n);
  for (index_t i = 0; i < n; ++i) {
    result[i] = out(i, 0);
  }
  return result;
}

TEST(fused_softmax, correctness) {
  ASSERT_THAT(run_fused_softmax({0.0f, 0.0f, 0.0f}),
      testing::Pointwise(testing::FloatNear(1e-6f), {1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f}));
  ASSERT_THAT(
      run_fused_softmax({100.0f, 0.0f, 0.0f}), testing::Pointwise(testing::FloatNear(1e-6f), {1.0f, 0.0f, 0.0f}));
  ASSERT_THAT(
      run_fused_softmax({100.0f, 0.0f, 100.0f}), testing::Pointwise(testing::FloatNear(1e-6f), {0.5f, 0.0f, 0.5f}));
}

class softmax : public testing::TestWithParam<std::tuple<int, int>> {};

auto split_factors = testing::Values(0, 1, 4);
INSTANTIATE_TEST_SUITE_P(
    mode, softmax, testing::Combine(split_factors, split_factors), test_params_to_string<softmax::ParamType>);

TEST_P(softmax, pipeline) {
  const int split_c = std::get<0>(GetParam());
  const int split_b = std::get<1>(GetParam());

  // Make the pipeline
  node_context ctx;

  constexpr int rank = 2;

  auto in = buffer_expr::make(ctx, "in", rank, sizeof(float));
  auto out = buffer_expr::make(ctx, "out", rank, sizeof(float));

  auto max_in = buffer_expr::make(ctx, "max_in", rank - 1, sizeof(float));
  auto exp_in = buffer_expr::make(ctx, "exp_in", rank, sizeof(float));
  auto sum_exp_in = buffer_expr::make(ctx, "sum_exp_in", rank - 1, sizeof(float));
  auto softmax_out = buffer_expr::make(ctx, "softmax_out", rank, sizeof(float));

  var c(ctx, "c");
  var b(ctx, "b");

  interval_expr all_c = out->dim(0).bounds;

  func pass1 = func::make(max_dim0, {{in, {all_c, point(b)}}}, {{max_in, {b}}});
  func pass2 =
      func::make(sum_exp, {{in, {all_c, point(b)}}, {max_in, {point(b)}}}, {{exp_in, {c, b}}, {sum_exp_in, {b}}});
  func pass3 = func::make(normalize, {{exp_in, {all_c, point(b)}}, {sum_exp_in, {point(b)}}}, {{softmax_out, {c, b}}});
  // Add a trivial consumer so we can keep the inner loop.
  func pass4 = func::make(add_1<float>, {{softmax_out, {point(c), point(b)}}}, {{out, {c, b}}});

  std::vector<func::loop_info> loops;
  if (split_c > 0) loops.push_back({c, split_c});
  if (split_b > 0) loops.push_back({b, split_b});
  pass4.loops(std::move(loops));

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int D = 10;
  const int B = 10;
  buffer<float, rank> in_buf({D, B});
  buffer<float, rank> out_buf({D, B});

  in_buf.allocate();
  for_each_element([](float* x) { *x = rand(); }, in_buf);
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  // Compare against the fused pipeline.
  buffer<float, rank> ref_buf({D, B});
  ref_buf.allocate();
  fused_softmax(in_buf.cast<const float>(), ref_buf.cast<float>());
  add_1(ref_buf.cast<const float>(), ref_buf.cast<float>());

  for (index_t b = 0; b < B; ++b) {
    auto out_b = span<const float>(&out_buf(0, b), D);
    auto ref_b = span<const float>(&ref_buf(0, b), D);
    ASSERT_THAT(out_b, testing::Pointwise(testing::FloatNear(1e-6f), ref_b));
  }
}

}  // namespace slinky
