#include <gtest/gtest.h>

#include <cassert>
#include <cstdint>
#include <limits>

#include "slinky/builder/pipeline.h"
#include "slinky/builder/test/funcs.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/evaluate.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/pipeline.h"

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
  copy(scalar<int>(0), in_buf);
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
  copy(scalar<int>(2), in_buf);
  // Leave unused uninitialized so we know if something accesses it (via msan).

  const raw_buffer* inputs[] = {&in_buf, &unused_buf};
  const raw_buffer* outputs[] = {&out_buf};
  index_t result = p.evaluate(inputs, outputs, eval_ctx);
  ASSERT_EQ(result, 0);
}

TEST(pipeline, overflow_check) {
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 1, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 1, sizeof(int));

  var x(ctx, "x");

  func mul = func::make(multiply_2<int>, {{in, {point(x)}}}, {{out, {x}}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  int checks_failed = 0;

  eval_context eval_ctx;
  eval_config eval_cfg;
  eval_cfg.check_failed = [&](const expr& c) { checks_failed++; };
  eval_ctx.config = &eval_cfg;

  buffer<int, 1> out_buf({10});
  out_buf.allocate();

  index_t max_val = std::numeric_limits<index_t>::max();
  index_t huge_stride = max_val / 2 + 1;

  buffer<int, 1> overflow_buf;
  overflow_buf.elem_size = sizeof(int);
  overflow_buf.mutable_dim(0).set_min_extent(0, 3);
  overflow_buf.mutable_dim(0).set_stride(huge_stride);

  const raw_buffer* inputs[] = {&overflow_buf};
  const raw_buffer* outputs[] = {&out_buf};

  index_t result = p.evaluate(inputs, outputs, eval_ctx);
  ASSERT_NE(result, 0);
  ASSERT_EQ(checks_failed, 1);
}

TEST(pipeline, pointer_overflow_check) {
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 1, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 1, sizeof(int));

  var x(ctx, "x");

  func mul = func::make(multiply_2<int>, {{in, {point(x)}}}, {{out, {x}}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  int checks_failed = 0;

  eval_context eval_ctx;
  eval_config eval_cfg;
  eval_cfg.check_failed = [&](const expr& c) { checks_failed++; };
  eval_ctx.config = &eval_cfg;

  buffer<int, 1> out_buf({10});
  out_buf.allocate();

  buffer<int, 1> overflow_buf;
  overflow_buf.elem_size = sizeof(int);
  overflow_buf.mutable_dim(0).set_min_extent(0, 10);
  overflow_buf.mutable_dim(0).set_stride(sizeof(int));

  uintptr_t max_ptr = std::numeric_limits<uintptr_t>::max();
  overflow_buf.raw_buffer::base = reinterpret_cast<void*>(max_ptr - 20);

  const raw_buffer* inputs[] = {&overflow_buf};
  const raw_buffer* outputs[] = {&out_buf};

  index_t result = p.evaluate(inputs, outputs, eval_ctx);
  ASSERT_NE(result, 0);
  ASSERT_EQ(checks_failed, 1);
}

}  // namespace slinky
