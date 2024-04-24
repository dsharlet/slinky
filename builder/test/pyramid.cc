#include <gtest/gtest.h>

#include "builder/pipeline.h"
#include "builder/replica_pipeline.h"
#include "builder/substitute.h"
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

// Matrix multiplication (not fast!)
template <typename T>
index_t matmul(const buffer<const T>& a, const buffer<const T>& b, const buffer<T>& c) {
  assert(a.rank == 2);
  assert(b.rank == 2);
  assert(c.rank == 2);
  assert(a.dim(1).begin() == b.dim(0).begin());
  assert(a.dim(1).end() == b.dim(0).end());
  assert(a.dim(1).stride() == sizeof(T));
  assert(b.dim(1).stride() == sizeof(T));
  assert(c.dim(1).stride() == sizeof(T));
  for (index_t i = c.dim(0).begin(); i < c.dim(0).end(); ++i) {
    for (index_t j = c.dim(1).begin(); j < c.dim(1).end(); ++j) {
      c(i, j) = 0;
      for (index_t k = a.dim(1).begin(); k < a.dim(1).end(); ++k) {
        c(i, j) += a(i, k) * b(k, j);
      }
    }
  }
  return 0;
}

const auto loop_modes = testing::Values(loop::serial, loop::parallel);

index_t pyramid_upsample2x(const buffer<const int>& skip, const buffer<const int>& in, const buffer<int>& out) {
  for (index_t y = out.dim(1).begin(); y < out.dim(1).end(); ++y) {
    for (index_t x = out.dim(0).begin(); x < out.dim(0).end(); ++x) {
      out(x, y) = in((x + 0) >> 1, (y + 0) >> 1) + in((x + 1) >> 1, (y + 0) >> 1) + in((x + 0) >> 1, (y + 1) >> 1) +
                  in((x + 1) >> 1, (y + 1) >> 1) + skip(x, y);
    }
  }
  return 0;
}

index_t downsample2x(const buffer<const int>& in, const buffer<int>& out) {
  for (index_t y = out.dim(1).begin(); y < out.dim(1).end(); ++y) {
    for (index_t x = out.dim(0).begin(); x < out.dim(0).end(); ++x) {
      out(x, y) = (in(2 * x + 0, 2 * y + 0) + in(2 * x + 1, 2 * y + 0) + in(2 * x + 0, 2 * y + 1) +
                      in(2 * x + 1, 2 * y + 1) + 2) /
                  4;
    }
  }
  return 0;
}

class pyramid : public testing::TestWithParam<std::tuple<int>> {};

INSTANTIATE_TEST_SUITE_P(mode, pyramid, testing::Combine(loop_modes), test_params_to_string<pyramid::ParamType>);

TEST_P(pyramid, pipeline) {
  int max_workers = std::get<0>(GetParam());
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");

  func downsample = func::make(downsample2x, {{in, {2 * x + bounds(0, 1), 2 * y + bounds(0, 1)}}}, {{intm, {x, y}}});
  func upsample = func::make(pyramid_upsample2x,
      {{in, {point(x), point(y)}}, {intm, {bounds(x, x + 1) / 2, bounds(y, y + 1) / 2}}}, {{out, {x, y}}});

  upsample.loops({{y, 1, max_workers}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 10;
  const int H = 10;
  buffer<int, 2> in_buf({W + 2, H + 2});
  buffer<int, 2> out_buf({W, H});

  init_random(in_buf);
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  const int parallel_extra = max_workers != loop::serial ? 1 : 0;
  ASSERT_EQ(eval_ctx.heap.total_size, (W + 2) / 2 * (2 + parallel_extra) * sizeof(int));
  ASSERT_EQ(eval_ctx.heap.total_count, 1);

  if (max_workers == loop::serial) {
    check_replica_pipeline(define_replica_pipeline(ctx, {in}, {out}));
  }
}

TEST(pyramid_multi, pipeline) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  auto down0 = buffer_expr::make(ctx, "down0", 2, sizeof(int));
  auto down1 = buffer_expr::make(ctx, "down1", 2, sizeof(int));
  auto down2 = buffer_expr::make(ctx, "down2", 2, sizeof(int));
  auto down3 = buffer_expr::make(ctx, "down3", 2, sizeof(int));
  auto up3 = buffer_expr::make(ctx, "up3", 2, sizeof(int));
  auto up2 = buffer_expr::make(ctx, "up2", 2, sizeof(int));
  auto up1 = buffer_expr::make(ctx, "up1", 2, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");

  func downsample0 = func::make(downsample2x, {{in, {2 * x + bounds(0, 1), 2 * y + bounds(0, 1)}}}, {{down0, {x, y}}});
  func downsample1 =
      func::make(downsample2x, {{down0, {2 * x + bounds(0, 1), 2 * y + bounds(0, 1)}}}, {{down1, {x, y}}});
  func downsample2 =
      func::make(downsample2x, {{down1, {2 * x + bounds(0, 1), 2 * y + bounds(0, 1)}}}, {{down2, {x, y}}});
  func downsample3 =
      func::make(downsample2x, {{down2, {2 * x + bounds(0, 1), 2 * y + bounds(0, 1)}}}, {{down3, {x, y}}});
  func upsample3 = func::make(pyramid_upsample2x,
      {{down2, {point(x), point(y)}}, {down3, {bounds(x, x + 1) / 2, bounds(y, y + 1) / 2}}}, {{up3, {x, y}}});
  func upsample2 = func::make(pyramid_upsample2x,
      {{down1, {point(x), point(y)}}, {up3, {bounds(x, x + 1) / 2, bounds(y, y + 1) / 2}}}, {{up2, {x, y}}});
  func upsample1 = func::make(pyramid_upsample2x,
      {{down0, {point(x), point(y)}}, {up2, {bounds(x, x + 1) / 2, bounds(y, y + 1) / 2}}}, {{up1, {x, y}}});
  func upsample = func::make(pyramid_upsample2x,
      {{in, {point(x), point(y)}}, {up1, {bounds(x, x + 1) / 2, bounds(y, y + 1) / 2}}}, {{out, {x, y}}});

  upsample.loops({{y, 1}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 21;
  const int H = 15;
  buffer<int, 2> in_buf({W + 32, H + 32});
  buffer<int, 2> out_buf({W, H});

  init_random(in_buf);
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  // Run the pipeline stages manually to get the reference result.
  buffer<int, 2> ref_down0({(W + 32) / 2, (H + 32) / 2});
  buffer<int, 2> ref_down1({(W + 32) / 4, (H + 32) / 4});
  buffer<int, 2> ref_down2({(W + 32) / 8, (H + 32) / 8});
  buffer<int, 2> ref_down3({(W + 32) / 16, (H + 32) / 16});
  buffer<int, 2> ref_up3({(W + 7) / 8 + 1, (H + 7) / 8 + 1});
  buffer<int, 2> ref_up2({(W + 3) / 4 + 1, (H + 3) / 4 + 1});
  buffer<int, 2> ref_up1({(W + 1) / 2 + 1, (H + 1) / 2 + 1});
  buffer<int, 2> ref_out({W, H});

  ref_down0.allocate();
  ref_down1.allocate();
  ref_down2.allocate();
  ref_down3.allocate();
  ref_up3.allocate();
  ref_up2.allocate();
  ref_up1.allocate();
  ref_out.allocate();

  downsample2x(in_buf.cast<const int>(), ref_down0.cast<int>());
  downsample2x(ref_down0.cast<const int>(), ref_down1.cast<int>());
  downsample2x(ref_down1.cast<const int>(), ref_down2.cast<int>());
  downsample2x(ref_down2.cast<const int>(), ref_down3.cast<int>());
  pyramid_upsample2x(ref_down2.cast<const int>(), ref_down3.cast<const int>(), ref_up3.cast<int>());
  pyramid_upsample2x(ref_down1.cast<const int>(), ref_up3.cast<const int>(), ref_up2.cast<int>());
  pyramid_upsample2x(ref_down0.cast<const int>(), ref_up2.cast<const int>(), ref_up1.cast<int>());
  pyramid_upsample2x(in_buf.cast<const int>(), ref_up1.cast<const int>(), ref_out.cast<int>());

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(ref_out(x, y), out_buf(x, y));
    }
  }
}

}  // namespace slinky
