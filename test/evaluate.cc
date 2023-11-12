#include "test.h"
#include "pipeline_helpers.h"
#include "expr.h"
#include "evaluate.h"
#include "print.h"

#include <cassert>

using namespace slinky;

TEST(evaluate_arithmetic) {
  node_context ctx;
  expr x = make_variable(ctx, "x");

  eval_context context;
  context.set(ctx.lookup("x"), 4);

  ASSERT_EQ(evaluate(x + 5, context), 9);
  ASSERT_EQ(evaluate(x - 3, context), 1);
  ASSERT_EQ(evaluate(x * 2, context), 8);
  ASSERT_EQ(evaluate(x / 2, context), 2);
  ASSERT_EQ(evaluate(x % 3, context), 1);
}
