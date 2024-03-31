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
  eval_ctx.atomic_call = [&](thread_pool::task f) { t.atomic_call(std::move(f)); };

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

TEST(evaluate, semaphore) {
  node_context ctx;
  var x(ctx, "x");
  var buf(ctx, "buf");

  thread_pool t;

  const int min = 0;
  const int max = 10;
  constexpr index_t fold_factor = 3;
  index_t produce_sem = fold_factor;
  index_t produced_sem = 0;

  std::atomic<int> data[fold_factor];
  std::fill_n(data, fold_factor, min - 1);
  constexpr index_t elem_size = sizeof(int);

  auto make_eval_ctx = [&]() {
    eval_context eval_ctx;
    eval_ctx.enqueue_many = [&](const thread_pool::task& f) { t.enqueue(t.thread_count(), f); };
    eval_ctx.enqueue = [&](int n, const thread_pool::task& f) { t.enqueue(n, f); };
    eval_ctx.wait_for = [&](std::function<bool()> f) { t.wait_for(std::move(f)); };
    eval_ctx.atomic_call = [&](thread_pool::task f) { t.atomic_call(std::move(f)); };
    return eval_ctx;
  };

  stmt produce = call_stmt::make(
      [&](const call_stmt*, eval_context& ctx) -> index_t {
        buffer<std::atomic<int>>& b = *reinterpret_cast<buffer<std::atomic<int>>*>(*ctx.lookup(buf.sym()));
        for (index_t x = b.dim(0).begin(); x < b.dim(0).end(); ++x) {
          b(x) = x;
        }
        return 0;
      },
      {}, {buf.sym()}, {});
  produce = block::make({
      check::make(semaphore_wait(reinterpret_cast<index_t>(&produce_sem), buffer_extent(buf, 0))),
      produce,
      check::make(semaphore_signal(reinterpret_cast<index_t>(&produced_sem), buffer_extent(buf, 0))),
  });
  produce = crop_dim::make(buf.sym(), 0, {x, x}, produce);
  produce = loop::make(x.sym(), loop::serial, {min, max}, 1, produce);
  produce = make_buffer::make(
      buf.sym(), reinterpret_cast<index_t>(&data[0]), elem_size, {{{min, max}, elem_size, fold_factor}}, produce);

  index_t sum = 0;
  stmt consume = call_stmt::make(
      [&](const call_stmt*, eval_context& ctx) -> index_t {
        buffer<std::atomic<int>>& b = *reinterpret_cast<buffer<std::atomic<int>>*>(*ctx.lookup(buf.sym()));
        for (index_t x = b.dim(0).begin(); x < b.dim(0).end(); ++x) {
          sum += b(x);
        }
        return 0;
      },
      {buf.sym()}, {}, {});
  consume = block::make({
      check::make(semaphore_wait(reinterpret_cast<index_t>(&produced_sem), buffer_extent(buf, 0))),
      consume,
      check::make(semaphore_signal(reinterpret_cast<index_t>(&produce_sem), buffer_extent(buf, 0))),
  });
  consume = crop_dim::make(buf.sym(), 0, {x, x}, consume);
  consume = loop::make(x.sym(), loop::serial, {min, max}, 1, consume);
  consume = make_buffer::make(
      buf.sym(), reinterpret_cast<index_t>(&data[0]), elem_size, {{{min, max}, elem_size, fold_factor}}, consume);

  std::thread consume_t([&]() {
    eval_context eval_ctx = make_eval_ctx();
    int result = evaluate(consume, eval_ctx);
    ASSERT_EQ(result, 0);
  });
  std::thread produce_t([&]() {
    eval_context eval_ctx = make_eval_ctx();
    int result = evaluate(produce, eval_ctx);
    ASSERT_EQ(result, 0);
  });

  consume_t.join();
  produce_t.join();

  ASSERT_EQ(sum, max * (max + 1) / 2 - min * (min + 1) / 2);
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
