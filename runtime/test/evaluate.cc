#include <gtest/gtest.h>

#include <cassert>

#include "base/thread_pool.h"
#include "runtime/evaluate.h"
#include "runtime/expr.h"

namespace slinky {

namespace {

node_context ctx;
var x(ctx, "x");
var y(ctx, "y");

expr define_undef(const expr& a, const expr& def) { return call::make(intrinsic::define_undef, {a, def}); }

}  // namespace

TEST(evaluate, arithmetic) {
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

  ASSERT_EQ(evaluate(and_then({expr(true), expr(true)})), true);
  ASSERT_EQ(evaluate(and_then({expr(true), expr(false)})), false);
  ASSERT_EQ(evaluate(and_then({expr(false), indeterminate()})), false);
  ASSERT_EQ(evaluate(or_else({expr(true), expr(true)})), true);
  ASSERT_EQ(evaluate(or_else({expr(false), expr(true)})), true);
  ASSERT_EQ(evaluate(or_else({expr(false), expr(false)})), false);
  ASSERT_EQ(evaluate(or_else({expr(true), indeterminate()})), true);
}

TEST(evaluate, undef) {
  eval_context context;
  context[x] = 4;

  ASSERT_EQ(evaluate(define_undef(select(true, expr(), x), 0), context), 0);
  ASSERT_EQ(evaluate(define_undef(select(false, expr(), x), 0), context), 4);
  ASSERT_EQ(evaluate(define_undef(select(true, expr(), x) + 2, 0), context), 0);
  ASSERT_EQ(evaluate(define_undef(select(false, expr(), x) + 2, 0), context), 6);
  ASSERT_EQ(evaluate(define_undef(select(true, expr(), x) + 2, 0) + 1, context), 1);
  ASSERT_EQ(evaluate(define_undef(select(false, expr(), x) + 2, 0) + 1, context), 7);
}

TEST(evaluate, call) {
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
  thread_pool t;

  eval_context ctx;
  ctx.enqueue_many = [&](thread_pool::task f) { t.enqueue(t.thread_count(), std::move(f)); };
  ctx.enqueue = [&](int n, thread_pool::task f) { t.enqueue(n, std::move(f)); };
  ctx.wait_for = [&](const std::function<bool()>& f) { t.wait_for(f); };
  ctx.atomic_call = [&](const thread_pool::task& f) { t.atomic_call(f); };

  for (int max_workers : {loop::serial, 2, 3, loop::parallel}) {
    std::atomic<index_t> sum_x = 0;
    stmt c = call_stmt::make(
        [&](const call_stmt*, eval_context& ctx) -> index_t {
          sum_x += *ctx[x];
          return 0;
        },
        {}, {}, {});

    stmt l = loop::make(x, max_workers, range(2, 12), 3, c);

    int result = evaluate(l, ctx);
    ASSERT_EQ(result, 0);
    ASSERT_EQ(sum_x, 2 + 5 + 8 + 11);
  }
}

void assert_buffer_extents_are(const raw_buffer& buf, const std::vector<int>& extents) {
  ASSERT_EQ(buf.rank, extents.size());
  for (std::size_t d = 0; d < extents.size(); ++d) {
    ASSERT_EQ(extents[d], buf.dim(d).extent());
  }
}

stmt make_check(var buffer, std::vector<int> extents) {
  return call_stmt::make(
      [=](const call_stmt*, eval_context& ctx) -> index_t {
        assert_buffer_extents_are(*ctx.lookup_buffer(buffer), extents);
        return 0;
      },
      {}, {buffer}, {});
}

TEST(evaluate, crop_dim) {
  eval_context ctx;
  buffer<void, 2> buf({10, 20});
  ctx[x] = reinterpret_cast<index_t>(&buf);

  evaluate(crop_dim::make(x, x, 0, {1, 3}, make_check(x, {3, 20})), ctx);
  evaluate(crop_dim::make(y, x, 0, {1, 3}, block::make({make_check(x, {10, 20}), make_check(y, {3, 20})})), ctx);
  assert_buffer_extents_are(buf, {10, 20});
}

TEST(evaluate, crop_buffer) {
  eval_context ctx;
  buffer<void, 4> buf({10, 20, 30, 40});
  ctx[x] = reinterpret_cast<index_t>(&buf);

  evaluate(crop_buffer::make(x, x, {{1, 3}, {}, {2, 5}}, make_check(x, {3, 20, 4, 40})), ctx);
  evaluate(crop_buffer::make(y, x, {{1, 3}, {}, {2, 5}},
               block::make({make_check(x, {10, 20, 30, 40}), make_check(y, {3, 20, 4, 40})})),
      ctx);
  assert_buffer_extents_are(buf, {10, 20, 30, 40});
}

TEST(evaluate, slice_dim) {
  eval_context ctx;
  buffer<void, 3> buf({10, 20, 30});
  ctx[x] = reinterpret_cast<index_t>(&buf);

  evaluate(slice_dim::make(x, x, 1, 2, make_check(x, {10, 30})), ctx);
  evaluate(slice_dim::make(y, x, 1, 2, block::make({make_check(x, {10, 20, 30}), make_check(y, {10, 30})})), ctx);
  assert_buffer_extents_are(buf, {10, 20, 30});
}

TEST(evaluate, slice_buffer) {
  eval_context ctx;
  buffer<void, 4> buf({10, 20, 30, 40});
  ctx[x] = reinterpret_cast<index_t>(&buf);

  evaluate(slice_buffer::make(x, x, {{}, 4, {}, 2}, make_check(x, {10, 30})), ctx);
  evaluate(
      slice_buffer::make(y, x, {{}, 4, {}, 2}, block::make({make_check(x, {10, 20, 30, 40}), make_check(y, {10, 30})})),
      ctx);
  assert_buffer_extents_are(buf, {10, 20, 30, 40});
}

TEST(evaluate, transpose) {
  eval_context ctx;
  buffer<void, 4> buf({10, 20, 30, 40});
  ctx[x] = reinterpret_cast<index_t>(&buf);

  evaluate(transpose::make(x, x, {0, 1}, make_check(x, {10, 20})), ctx);
  evaluate(transpose::make(x, x, {3, 1}, make_check(x, {40, 20})), ctx);
  evaluate(transpose::make(y, x, {0, 1}, block::make({make_check(x, {10, 20, 30, 40}), make_check(y, {10, 20})})), ctx);
  evaluate(transpose::make(y, x, {2, 1}, block::make({make_check(x, {10, 20, 30, 40}), make_check(y, {30, 20})})), ctx);
  assert_buffer_extents_are(buf, {10, 20, 30, 40});
}

TEST(evaluate, clone_buffer) {
  eval_context ctx;
  buffer<void, 2> buf({10, 20});
  ctx[y] = reinterpret_cast<index_t>(&buf);

  evaluate(clone_buffer::make(x, y, block::make({make_check(x, {10, 20}), make_check(y, {10, 20})})), ctx);
  assert_buffer_extents_are(buf, {10, 20});
}

TEST(evaluate, semaphore) {
  thread_pool t;

  eval_context ctx;
  ctx.wait_for = [&](const std::function<bool()>& f) { t.wait_for(f); };
  ctx.atomic_call = [&](const thread_pool::task& f) { t.atomic_call(f); };

  index_t sem1 = 0;
  index_t sem2 = 0;
  index_t sem3 = 0;
  auto make_wait = [&](index_t& sem) { return check::make(semaphore_wait(reinterpret_cast<index_t>(&sem))); };
  auto make_signal = [&](index_t& sem) { return check::make(semaphore_signal(reinterpret_cast<index_t>(&sem))); };

  std::atomic<int> state = 0;

  std::thread th([&]() {
    evaluate(make_wait(sem1), ctx);
    state++;
    evaluate(make_signal(sem2), ctx);
    evaluate(make_wait(sem3), ctx);
    state++;
  });
  ASSERT_EQ(state, 0);
  evaluate(make_signal(sem1), ctx);
  evaluate(make_wait(sem2), ctx);
  ASSERT_EQ(state, 1);
  evaluate(make_signal(sem3), ctx);
  th.join();
  ASSERT_EQ(state, 2);
}

TEST(evaluate_constant, arithmetic) {
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
