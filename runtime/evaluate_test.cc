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
      [&](const call_stmt*, eval_context& ctx) -> index_t {
        calls.push_back(*ctx[x]);
        return 0;
      },
      {}, {}, {});

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
  eval_ctx.enqueue = [&](int n, const thread_pool::task& f) { t.enqueue(n, f); };
  eval_ctx.wait_for = [&](std::function<bool()> f) { t.wait_for(std::move(f)); };

  for (int max_workers : {loop::serial, 2, 3, loop::parallel}) {
    std::atomic<index_t> sum_x = 0;
    stmt c = call_stmt::make(
        [&](const call_stmt*, eval_context& ctx) -> index_t {
          sum_x += *ctx[x];
          return 0;
        },
        {}, {}, {});

    stmt l = loop::make(x.sym(), max_workers, range(2, 12), 3, c);

    int result = evaluate(l, eval_ctx);
    ASSERT_EQ(result, 0);
    ASSERT_EQ(sum_x, 2 + 5 + 8 + 11);
  }
}

TEST(evaluate_constant, arithmetic) {
  node_context ctx;
  var x(ctx, "x");

  ASSERT_EQ(evaluate_constant(x + 5), std::nullopt);
  ASSERT_EQ(evaluate_constant(x - 3), std::nullopt);
  ASSERT_EQ(evaluate_constant(2 * x), std::nullopt);
  ASSERT_EQ(evaluate_constant(x / 2), std::nullopt);
  ASSERT_EQ(evaluate_constant(4 % x), std::nullopt);
  ASSERT_EQ(evaluate_constant(x < 4), std::nullopt);
  ASSERT_EQ(evaluate_constant(x < 5), std::nullopt);
  ASSERT_EQ(evaluate_constant(3 <= x), std::nullopt);
  ASSERT_EQ(evaluate_constant(x <= 4), std::nullopt);
  ASSERT_EQ(evaluate_constant(x > 3), std::nullopt);
  ASSERT_EQ(evaluate_constant(x > 4), std::nullopt);
  ASSERT_EQ(evaluate_constant(x >= 4), std::nullopt);
  ASSERT_EQ(evaluate_constant(x >= 5), std::nullopt);
  ASSERT_EQ(evaluate_constant(x == 4), std::nullopt);
  ASSERT_EQ(evaluate_constant(x == 5), std::nullopt);
  ASSERT_EQ(evaluate_constant(x != 4), std::nullopt);
  ASSERT_EQ(evaluate_constant(x != 5), std::nullopt);
}

}  // namespace slinky
