#include "evaluate.h"
#include "expr.h"
#include "print.h"
#include "test.h"

#include <cassert>

using namespace slinky;

TEST(evaluate_arithmetic) {
  node_context ctx;
  var x(ctx, "x");

  eval_context context;
  context[x] = 4;

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
  var x(ctx, "x");
  std::vector<index_t> calls;
  stmt c = call_stmt::make(
      [&](eval_context& ctx) -> index_t {
        calls.push_back(*ctx[x]);
        return 0;
      },
      {}, {});

  eval_context context;
  context[x] = 2;

  int result = evaluate(c, context);
  ASSERT_EQ(result, 0);
  ASSERT_EQ(calls.size(), 1);
  ASSERT_EQ(calls[0], 2);
}

TEST(evaluate_loop) {
  for (loop_mode type : {loop_mode::serial}) {
    node_context ctx;
    var x(ctx, "x");
    std::vector<index_t> calls;
    stmt c = call_stmt::make(
        [&](eval_context& ctx) -> index_t {
          calls.push_back(*ctx[x]);
          return 0;
        },
        {}, {});

    stmt l = loop::make(x.sym(), type, range(2, 12), 3, c);

    int result = evaluate(l);
    ASSERT_EQ(result, 0);
    ASSERT_EQ(calls.size(), 4);
    for (int i = 0; i < 4; ++i) {
      ASSERT_EQ(calls[i], i * 3 + 2);
    }
  }
}
