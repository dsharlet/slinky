#include "test.h"
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
  ASSERT_EQ(evaluate(x < 4, context), 0);
  ASSERT_EQ(evaluate(x < 5, context), 1);
  ASSERT_EQ(evaluate(x <= 3, context), 0);
  ASSERT_EQ(evaluate(x <= 4, context), 1);
  ASSERT_EQ(evaluate(x > 3, context), 1);
  ASSERT_EQ(evaluate(x > 4, context), 0);
  ASSERT_EQ(evaluate(x >= 4, context), 1);
  ASSERT_EQ(evaluate(x >= 5, context), 0);
  ASSERT_EQ(evaluate(x == 4, context), 1);
  ASSERT_EQ(evaluate(x == 5, context), 0);
  ASSERT_EQ(evaluate(x != 4, context), 0);
  ASSERT_EQ(evaluate(x != 5, context), 1);

  ASSERT_EQ(evaluate((x + 2) / 3, context), 2);
}

TEST(evaluate_call) {
  node_context ctx;
  expr x = make_variable(ctx, "x");
  std::vector<index_t> calls;
  stmt c = call::make([&](std::span<const index_t> scalars, std::span<buffer_base*> buffers) -> index_t {
    calls.push_back(scalars[0]);
    return 0;
  }, { x }, {});

  eval_context context;
  context.set(ctx.lookup("x"), 2);

  int result = evaluate(c, context);
  ASSERT_EQ(result, 0);
  ASSERT_EQ(calls.size(), 1);
  ASSERT_EQ(calls[0], 2);
}

TEST(evaluate_loop) {
  node_context ctx;
  expr x = make_variable(ctx, "x");
  std::vector<index_t> calls;
  stmt c = call::make([&](std::span<const index_t> scalars, std::span<buffer_base*> buffers) -> index_t {
    calls.push_back(scalars[0]);
    return 0;
  }, { x }, {});

  stmt l = loop::make(ctx.lookup("x"), 10, c);

  int result = evaluate(l);
  ASSERT_EQ(result, 0);
  ASSERT_EQ(calls.size(), 10);
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(calls[i], i);
  }
}
