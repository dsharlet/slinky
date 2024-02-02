#include <gtest/gtest.h>

#include <cassert>

#include "builder/pipeline.h"
#include "runtime/buffer.h"
#include "runtime/expr.h"
#include "runtime/pipeline.h"

namespace slinky {

template <typename T>
index_t multiply_2(const buffer<const T>& in, const buffer<T>& out) {
  assert(in.rank == out.rank);
  for_each_index(out, [&](auto i) { out(i) = in(i) * 2; });
  return 0;
}

// A trivial pipeline with one stage.
TEST(pipeline, checks) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 1);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 1);

  var x(ctx, "x");

  func mul = func::make(multiply_2<int>, {{in, {point(x)}}}, {{out, {x}}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline
  const int N = 10;

  int checks_failed = 0;

  eval_context eval_ctx;
  eval_ctx.check_failed = [&](const expr& c) { checks_failed++; };

  buffer<int, 1> in_buf({N});
  buffer<int, 1> out_buf({N});

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  index_t result = p.evaluate(inputs, outputs, eval_ctx);
  ASSERT_NE(result, 0) << " null inputs should have failed";

  // The input and output pointers are null.
  ASSERT_EQ(checks_failed, 1);

  in_buf.allocate();
  for_each_index(in_buf, [&](const auto i) { in_buf(i) = 0; });
  out_buf.allocate();
  result = p.evaluate(inputs, outputs, eval_ctx);
  ASSERT_EQ(result, 0) << " should succeed";

  // Shouldn't have failed.
  ASSERT_EQ(checks_failed, 1);

  buffer<int, 1> too_small_buf({N - 1});
  const raw_buffer* too_small[] = {&too_small_buf};
  result = p.evaluate(too_small, outputs, eval_ctx);
  ASSERT_NE(result, 0) << " too small should have failed";

  // Input is too small.
  ASSERT_EQ(checks_failed, 2);
}

}  // namespace slinky
