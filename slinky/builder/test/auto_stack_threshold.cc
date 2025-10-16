#include <gtest/gtest.h>

#include "slinky/builder/pipeline.h"
#include "slinky/builder/test/context.h"
#include "slinky/builder/test/funcs.h"
#include "slinky/builder/test/util.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/pipeline.h"

namespace slinky {

// An example of two 2D elementwise operations in sequence.

TEST(auto_stack_threshold, pipeline) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 1, sizeof(uint8_t));
  auto out = buffer_expr::make(ctx, "out", 1, sizeof(uint8_t));
  auto intm = buffer_expr::make(ctx, "intm", 1, sizeof(uint8_t));

  var x(ctx, "x");
  func a = func::make(add_1<uint8_t>, {{in, {point(x)}}}, {{intm, {x}}});
  func b = func::make(add_1<uint8_t>, {{intm, {point(x)}}}, {{out, {x}}});

  // Split the loops such that we limit the number of elements produced to a total number across both dimensions.
  var split_factor(ctx, "split_factor");
  b.loops({{x, split_factor}});

  intm->dim(0).fold_factor = split_factor;

  pipeline p = build_pipeline(ctx, {split_factor}, {in}, {out});

  // Run the pipeline
  const int N = 16 * 1024;

  buffer<uint8_t, 1> in_buf({N});
  in_buf.allocate();
  for (int x = 0; x < N; ++x) {
    in_buf(x) = static_cast<uint8_t>(x);
  }

  buffer<uint8_t, 1> out_buf({N});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  for (int split : {1, 100, 1000, 10000}) {
    const index_t args[] = {split};
    const raw_buffer* inputs[] = {&in_buf};
    const raw_buffer* outputs[] = {&out_buf};
    test_context eval_ctx;
    eval_ctx.config.auto_stack_threshold = 512;
    p.evaluate(args, inputs, outputs, eval_ctx);

    for (int x = 0; x < N; ++x) {
      ASSERT_EQ(out_buf(x), static_cast<uint8_t>(x + 2));
    }

    if (split > static_cast<int>(eval_ctx.config.auto_stack_threshold)) {
      ASSERT_EQ(eval_ctx.heap.allocs.size(), 1);
    } else {
      ASSERT_EQ(eval_ctx.heap.allocs.size(), 0);
    }
  }
}

}  // namespace slinky
