#include <gtest/gtest.h>

#include <cassert>

#include "builder/pipeline.h"
#include "builder/test/funcs.h"
#include "runtime/buffer.h"
#include "runtime/expr.h"
#include "runtime/pipeline.h"

namespace slinky {

// A trivial pipeline with one stage.
TEST(pipeline, checks) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 1, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 1, sizeof(int));

  var x(ctx, "x");

  func mul = func::make(multiply_2<int>, {{in, {point(x)}}}, {{out, {x}}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline
  const int N = 10;

  int checks_failed = 0;

  eval_context eval_ctx;
  eval_config eval_cfg;
  eval_cfg.check_failed = [&](const expr& c) { checks_failed++; };
  eval_ctx.config = &eval_cfg;

  buffer<int, 1> in_buf({N});
  buffer<int, 1> out_buf({N});
  in_buf.allocate();
  const int zero = 0;
  fill(in_buf, &zero);
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  index_t result = p.evaluate(inputs, outputs, eval_ctx);
  ASSERT_EQ(result, 0) << " should succeed";

  // Shouldn't have failed.
  ASSERT_EQ(checks_failed, 0);

  buffer<int, 1> too_small_buf({N - 1});
  const raw_buffer* too_small[] = {&too_small_buf};
  result = p.evaluate(too_small, outputs, eval_ctx);
  ASSERT_NE(result, 0) << " too small should have failed";

  // Input is too small.
  ASSERT_EQ(checks_failed, 1);
}

TEST(pipeline, unused_input) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 1, sizeof(int));
  auto unused = buffer_expr::make(ctx, "unused", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 1, sizeof(int));

  var x(ctx, "x");

  func mul = func::make(multiply_2<int>, {{in, {point(x)}}}, {{out, {x}}});

  pipeline p = build_pipeline(ctx, {in, unused}, {out});

  // Run the pipeline
  const int N = 10;

  eval_context eval_ctx;

  buffer<int, 1> in_buf({N});
  buffer<int, 2> unused_buf;
  buffer<int, 1> out_buf({N});
  in_buf.allocate();
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf, &unused_buf};
  const raw_buffer* outputs[] = {&out_buf};
  index_t result = p.evaluate(inputs, outputs, eval_ctx);
  ASSERT_EQ(result, 0);
}

}  // namespace slinky
