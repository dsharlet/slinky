#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "builder/pipeline.h"
#include "builder/test/context.h"
#include "builder/test/funcs.h"
#include "builder/test/util.h"
#include "runtime/expr.h"
#include "runtime/pipeline.h"

namespace slinky {

// This implementation of l2_norm is not intended to be fast, it is only intended to model the data dependencies.
index_t reciprocal_sqrt(const buffer<const float>& in, const buffer<float>& out) {
  assert(in.rank == out.rank);
  for_each_element([&](float* out, const float* in) { *out = 1.0f / std::sqrt(*in); }, out, in);
  return 0;
}

index_t fused_l2_norm(const buffer<const float>& in, const buffer<float>& out) {
  for (index_t b = out.dim(1).begin(); b < out.dim(1).end(); ++b) {
    float sum_sq = 0.0f;
    for (index_t c = out.dim(0).begin(); c < out.dim(0).end(); ++c) {
      sum_sq += in(c, b) * in(c, b);
    }
    const float inv_sqrt_sum_sq = 1.0f / std::sqrt(sum_sq);
    for (index_t c = out.dim(0).begin(); c < out.dim(0).end(); ++c) {
      out(c, b) = in(c, b) * inv_sqrt_sum_sq;
    }
  }
  return 0;
}

std::vector<float> run_fused_l2_norm(std::initializer_list<const float> x) {
  const index_t n = x.size();

  buffer<float, 2> in({n, 1});
  buffer<float, 2> out({n, 1});
  in.allocate();
  out.allocate();

  std::copy(x.begin(), x.end(), &in(0, 0));

  fused_l2_norm(in.cast<const float>(), out.cast<float>());

  std::vector<float> result(n);
  std::copy_n(&out(0, 0), n, result.begin());
  return result;
}

TEST(fused_l2_norm, correctness) {
  ASSERT_THAT(run_fused_l2_norm({0.0f, 0.0f, 1.0f}), testing::Pointwise(testing::FloatNear(1e-6f), {0.0f, 0.0f, 1.0f}));
  ASSERT_THAT(run_fused_l2_norm({1.0f, 1.0f, 0.0f}),
      testing::Pointwise(testing::FloatNear(1e-6f), {1.0f / std::sqrt(2.0f), 1.0f / std::sqrt(2.0f), 0.0f}));
}

class l2_norm : public testing::TestWithParam<int> {};

auto split_factors = testing::Values(0, 1, 7);

INSTANTIATE_TEST_SUITE_P(mode, l2_norm, split_factors);

TEST_P(l2_norm, pipeline) {
  const int split = GetParam();

  // Make the pipeline
  node_context ctx;

  constexpr int rank = 2;

  auto in = buffer_expr::make(ctx, "in", rank, sizeof(float));
  auto out = buffer_expr::make(ctx, "out", rank, sizeof(float));

  auto in_sq = buffer_expr::make(ctx, "in_sq", rank, sizeof(float));
  auto sum_in_sq = buffer_expr::make(ctx, "sum_in_sq", rank - 1, sizeof(float));
  auto inv_sqrt_sum = buffer_expr::make(ctx, "inv_sqrt_sum", rank - 1, sizeof(float));
  auto inv_sqrt_broadcast = buffer_expr::make(ctx, "inv_sqrt_broadcast", rank, sizeof(float));

  var c(ctx, "c");
  var b(ctx, "b");

  interval_expr all_c = out->dim(0).bounds;

  // Add a trivial producer so we can have an inner loop here.
  func pass1 = func::make(
      square<float>, {{in, {point(c), point(b)}}}, {{in_sq, {c, b}}}, call_stmt::attributes{.name = "square"});
  func pass2 = func::make(
      [](const buffer<const float>& in, const buffer<float>& out) -> index_t {
        sum(in, out, {{0, in.dim(0).min(), in.dim(0).max()}});
        return 0;
      },
      {{in_sq, {all_c, point(b)}}}, {{sum_in_sq, {b}}}, call_stmt::attributes{.name = "sum_in_sq"});
  func pass3 = func::make(reciprocal_sqrt, {{sum_in_sq, {point(b)}}}, {{inv_sqrt_sum, {b}}},
      call_stmt::attributes{.name = "reciprocal_sqrt"});
  func broadcast = func::make_copy({inv_sqrt_sum, {point(b)}}, {inv_sqrt_broadcast, {c, b}});
  func pass4 =
      func::make(multiply<float>, {{in, {point(c), point(b)}}, {inv_sqrt_broadcast, {point(c), point(b)}}},
          {{out, {c, b}}}, call_stmt::attributes{.name = "multiply"});

  if (split > 0) {
    pass4.loops({{b, split}});
    pass3.compute_at({&pass4, b});
    pass2.compute_at({&pass4, b});
    pass1.compute_at({&pass4, b});
    in_sq->store_at({&pass4, b});
    sum_in_sq->store_at({&pass4, b});
    inv_sqrt_sum->store_at({&pass4, b});
    inv_sqrt_broadcast->store_at({&pass4, b});
  }

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int D = 30;
  const int B = 20;
  buffer<float, rank> in_buf({D, B});
  buffer<float, rank> out_buf({D, B});

  in_buf.allocate();
  for_each_element([](float* x) { *x = rand(); }, in_buf);
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  // Compare against the fused pipeline.
  buffer<float, rank> ref_buf({D, B});
  ref_buf.allocate();
  fused_l2_norm(in_buf.cast<const float>(), ref_buf.cast<float>());

  for (index_t b = 0; b < B; ++b) {
    auto out_b = span<const float>(&out_buf(0, b), D);
    auto ref_b = span<const float>(&ref_buf(0, b), D);
    ASSERT_THAT(out_b, testing::Pointwise(testing::FloatNear(1e-6f), ref_b));
  }

  if (split > 0) {
    std::vector<int> expected_allocs;
    for (int i = 0; i < B; i += split) {
      const int split_size = std::min(split, B - i);
      expected_allocs.push_back(D * split_size * sizeof(float));
      expected_allocs.push_back(split_size * sizeof(float));
      expected_allocs.push_back(split_size * sizeof(float));
    }
    ASSERT_THAT(eval_ctx.heap.allocs, testing::UnorderedElementsAreArray(expected_allocs));
  } else {
    ASSERT_THAT(eval_ctx.heap.allocs,
        testing::UnorderedElementsAre(D * B * sizeof(float), B * sizeof(float), B * sizeof(float)));
  }
}

}  // namespace slinky
