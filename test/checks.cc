#include "test.h"

#include <cassert>

#include "expr.h"
#include "funcs.h"
#include "pipeline.h"
#include "print.h"

using namespace slinky;

// A trivial pipeline with one stage.
TEST(pipeline_checks) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 1);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 1);

  expr x = make_variable(ctx, "x");

  func mul = func::make<const int, int>(multiply_2<int>, {in, {point(x)}}, {out, {x}});

  pipeline p(ctx, {in}, {out});

  // Run the pipeline
  const int N = 10;

  int checks_failed = 0;

  eval_context eval_ctx;
  eval_ctx.check_failed = [&](const expr& c) {
    checks_failed++;
  };

  buffer<int, 1> in_buf({N});
  buffer<int, 1> out_buf({N});

  const buffer_base* inputs[] = {&in_buf};
  const buffer_base* outputs[] = {&out_buf};
  index_t result = p.evaluate(inputs, outputs, eval_ctx);
  ASSERT(result != 0) << " null inputs should have failed";

  // The input and output pointers are null.
  ASSERT_EQ(checks_failed, 1);

  in_buf.allocate();
  out_buf.allocate();
  result = p.evaluate(inputs, outputs, eval_ctx);
  ASSERT(result == 0) << " should succeed";

  // Shouldn't have failed.
  ASSERT_EQ(checks_failed, 1);

  buffer<int, 1> too_small_buf({N - 1});
  const buffer_base* too_small[] = {&too_small_buf};
  result = p.evaluate(too_small, outputs, eval_ctx);
  ASSERT(result != 0) << " too small should have failed";

  // Input is too small.
  ASSERT_EQ(checks_failed, 2);
}
