#include <gtest/gtest.h>

#include <cassert>

#include "runtime/evaluate.h"
#include "runtime/expr.h"
#include "runtime/thread_pool.h"

namespace slinky {

TEST(evaluate, arithmetic) {
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

TEST(evaluate, call) {
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

TEST(evaluate, loop) {
  node_context ctx;
  var x(ctx, "x");

  thread_pool t;

  eval_context eval_ctx;
  eval_ctx.enqueue_many = [&](const thread_pool::task& f) { t.enqueue(t.thread_count(), f); };
  eval_ctx.enqueue_one = [&](thread_pool::task f) { t.enqueue(std::move(f)); };
  eval_ctx.wait_for = [&](std::function<bool()> f) { t.wait_for(std::move(f)); };

  for (loop_mode type : {loop_mode::serial, loop_mode::parallel}) {
    std::atomic<index_t> sum_x = 0;
    stmt c = call_stmt::make(
        [&](eval_context& ctx) -> index_t {
          sum_x += *ctx[x];
          return 0;
        },
        {}, {});

    stmt l = loop::make(x.sym(), type, range(2, 12), 3, c);

    int result = evaluate(l, eval_ctx);
    ASSERT_EQ(result, 0);
    ASSERT_EQ(sum_x, 2 + 5 + 8 + 11);
  }
}

}
