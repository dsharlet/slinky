#include <gtest/gtest.h>

#include <tuple>

#include "builder/pipeline.h"
#include "builder/test/funcs.h"
#include "builder/test/util.h"

namespace slinky {

class aligned_producer : public testing::TestWithParam<std::tuple<int, int, bool>> {};

INSTANTIATE_TEST_SUITE_P(split_mode, aligned_producer,
    testing::Combine(testing::Range(2, 4), testing::Range(0, 5), testing::Values(true, false)),
    test_params_to_string<aligned_producer::ParamType>);

// An example of two 1D elementwise operations in sequence.
TEST_P(aligned_producer, pipeline) {
  int alignment = std::get<0>(GetParam());
  int split = std::get<1>(GetParam());
  bool schedule_storage = std::get<2>(GetParam());

  if (!schedule_storage && split % alignment != 0) {
    // TODO: This hits the cropping of folded buffers assert.
    return;
  }

  // Make the pipeline
  node_context ctx;

  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));
  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(int));

  intm->dim(0).bounds = align(intm->dim(0).bounds, alignment);

  var x(ctx, "x");
  var y(ctx, "y");

  index_t produced = 0;
  auto assert_aligned = [&](const buffer<int>& out) -> index_t {
    EXPECT_EQ(out.dim(0).min() % alignment, 0);
    EXPECT_EQ(out.dim(0).extent() % alignment, 0);
    for (index_t y = out.dim(1).begin(); y < out.dim(1).end(); ++y) {
      for (index_t x = out.dim(0).begin(); x < out.dim(0).end(); ++x) {
        out(x, y) = y * 1000 + x;
      }
    }
    produced += out.dim(0).extent() * out.dim(1).extent();
    return 0;
  };
  func producer = func::make(std::move(assert_aligned), {}, {{intm, {x, y}}});
  func consumer = func::make(add_1<int>, {{intm, {point(x), point(y)}}}, {{out, {x, y}}});

  if (split > 0) {
    consumer.loops({{x, split}});
    if (schedule_storage) {
      intm->store_at({&consumer, x});
      intm->store_in(memory_type::stack);
    }
  }

  pipeline p = build_pipeline(ctx, {}, {out});

  // Run the pipeline
  const int W = 100;
  const int H = 5;

  buffer<int, 2> out_buf({W, H});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate({}, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y), y * 1000 + x + 1);
    }
  }

  if (split % alignment != 0 && schedule_storage) {
    // If we got sliding window, there won't be redundant compute even with the alignment of the producer.
    ASSERT_GT(produced, W * H);
  } else {
    ASSERT_GE(produced + (alignment - 1) * H, W * H);
  }
}

}  // namespace slinky
