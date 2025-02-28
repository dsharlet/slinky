#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "builder/pipeline.h"
#include "builder/test/context.h"
#include "builder/test/funcs.h"
#include "builder/test/util.h"
#include "runtime/expr.h"
#include "runtime/pipeline.h"
#include "runtime/visualize.h"

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

class softmax : public testing::TestWithParam<std::tuple<int, int, int, int>> {};

auto split_factors = testing::Values(0, 1, 4);

INSTANTIATE_TEST_SUITE_P(mode, softmax,
    testing::Combine(split_factors, split_factors, testing::Values(0, 1, 2), testing::Values(0)),
    test_params_to_string<softmax::ParamType>);

INSTANTIATE_TEST_SUITE_P(with_copy, softmax,
    testing::Combine(testing::Values(1), testing::Values(1), testing::Values(0), testing::Values(0, 1, 2)),
    test_params_to_string<softmax::ParamType>);

TEST_P(softmax, pipeline) {
  const int split_c = std::get<0>(GetParam());
  const int split_b = std::get<1>(GetParam());
  const int use_compute_at = std::get<2>(GetParam());
  const int copy_at_the_end = std::get<3>(GetParam());

  // Make the pipeline
  node_context ctx;

  constexpr int rank = 2;

  auto in = buffer_expr::make(ctx, "in", rank, sizeof(float));
  auto out = buffer_expr::make(ctx, "out", rank, sizeof(float));

  auto softmax_in = buffer_expr::make(ctx, "softmax_in", rank, sizeof(float));
  auto max_in = buffer_expr::make(ctx, "max_in", rank - 1, sizeof(float));
  auto exp_in = buffer_expr::make(ctx, "exp_in", rank, sizeof(float));
  auto sum_exp_in = buffer_expr::make(ctx, "sum_exp_in", rank - 1, sizeof(float));
  auto softmax_out = buffer_expr::make(ctx, "softmax_out", rank, sizeof(float));
  auto add_out = buffer_expr::make(ctx, "add_out", rank, sizeof(float));

  var c(ctx, "c");
  var b(ctx, "b");

  interval_expr all_c = out->dim(0).bounds;

  // Add a trivial producer so we can have an inner loop here.
  func pass0 = func::make(
      add_1<float>, {{in, {point(c), point(b)}}}, {{softmax_in, {c, b}}}, call_stmt::attributes{.name = "producer"});
  func pass1 = func::make(
      max_dim0, {{softmax_in, {all_c, point(b)}}}, {{max_in, {b}}}, call_stmt::attributes{.name = "max_dim0"});
  func pass2 = func::make(sum_exp, {{in, {all_c, point(b)}}, {max_in, {point(b)}}},
      {{exp_in, {c, b}}, {sum_exp_in, {b}}}, call_stmt::attributes{.name = "exp_in"});
  func pass3 = func::make(normalize, {{exp_in, {all_c, point(b)}}, {sum_exp_in, {point(b)}}}, {{softmax_out, {c, b}}},
      call_stmt::attributes{.name = "normalize"});

  // Add a trivial consumer so we can have an inner loop here too.
  func pass4 = func::make(add_1<float>, {{softmax_out, {point(c), point(b)}}},
      {{copy_at_the_end ? add_out : out, {c, b}}}, call_stmt::attributes{.name = "consumer"});

  func copy;
  if (copy_at_the_end > 0) {
    box_expr bounds;
    if (copy_at_the_end == 1) {
      bounds = {point(c), point(b)};
      // If we want to alias intermediate buffer to the output buffer,
      // we need to tell aliaser that output is unfolded and it's safe to alias.
      out->dim(0).fold_factor = dim::unfolded;
      out->dim(1).fold_factor = dim::unfolded;
    } else {
      bounds = {
          select(in->dim(0).extent() == 1, point(in->dim(0).min()), point(c)),
          select(in->dim(1).extent() == 1, point(in->dim(1).min()), point(b)),
      };
    }
    copy = func::make_copy({add_out, bounds}, {out, {c, b}});
  }

  std::vector<func::loop_info> loops;
  if (split_c > 0) loops.push_back({c, split_c});
  if (split_b > 0) loops.push_back({b, split_b});
  pass0.loops(loops);
  pass4.loops(loops);

  if (use_compute_at && split_b > 0) {
    pass1.compute_at({&pass4, b});
    if (use_compute_at == 2) {
      max_in->store_at({&pass4, b});
      max_in->store_in(memory_type::stack);
    }
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
  buffer<float, rank> softmax_in_buf({D, B});
  ref_buf.allocate();
  softmax_in_buf.allocate();
  add_1(in_buf.cast<const float>(), softmax_in_buf.cast<float>());
  fused_softmax(softmax_in_buf.cast<const float>(), ref_buf.cast<float>());
  add_1(ref_buf.cast<const float>(), ref_buf.cast<float>());

  for (index_t b = 0; b < B; ++b) {
    auto out_b = span<const float>(&out_buf(0, b), D);
    auto ref_b = span<const float>(&ref_buf(0, b), D);
    ASSERT_THAT(out_b, testing::Pointwise(testing::FloatNear(1e-6f), ref_b));
  }

  if (split_b > 0) {
    const int sum_exp_in_size = split_b * sizeof(float);
    const int exp_in_size = split_b * D * sizeof(float);
    const int max_in_size = split_b * sizeof(float);
    const int softmax_in_size = split_b * D * sizeof(float);
    const int softmax_out_size = split_b * (split_c == 0 ? D : split_c) * sizeof(float);
    if (copy_at_the_end == 2) {
      const int add_out_size = D * B * sizeof(float);
      ASSERT_THAT(eval_ctx.heap.allocs, testing::UnorderedElementsAre(sum_exp_in_size, exp_in_size, max_in_size,
                                            softmax_in_size, softmax_out_size, add_out_size));
    } else {
      if (use_compute_at == 2) {
        // TODO(vksnk): investigate why after using store_at on max_in, the storages folding stops working
        // for softmax_in.
        ASSERT_THAT(eval_ctx.heap.allocs,
            testing::UnorderedElementsAre(sum_exp_in_size, exp_in_size, B * softmax_in_size / split_b, softmax_out_size));
      } else {
        ASSERT_THAT(eval_ctx.heap.allocs,
            testing::UnorderedElementsAre(sum_exp_in_size, exp_in_size, max_in_size, softmax_in_size, softmax_out_size));
      }
    }
  } else {
    if (copy_at_the_end == 2) {
      ASSERT_EQ(eval_ctx.heap.allocs.size(), 6);
    } else {
      ASSERT_EQ(eval_ctx.heap.allocs.size(), split_c ? 5 : 4);
    }
  }

  if (split_c == 0 && !use_compute_at) {
    check_visualize("softmax_split_" + std::to_string(split_b) + ".html", p, inputs, outputs, &ctx);
  }
}

}  // namespace slinky
