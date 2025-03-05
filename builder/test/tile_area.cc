#include <gtest/gtest.h>

#include "base/thread_pool.h"
#include "builder/pipeline.h"
#include "builder/test/context.h"
#include "builder/test/funcs.h"
#include "builder/test/util.h"
#include "runtime/expr.h"
#include "runtime/pipeline.h"

namespace slinky {

// An example of two 2D elementwise operations in sequence.
TEST(tile_area, pipeline) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));
  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");

  // Track the max number of elements produced by any one call.
  int max_elem_count_seen = 0;

  auto m2 = [&](const buffer<const int>& a, const buffer<int>& b) -> index_t {
    max_elem_count_seen = std::max<int>(max_elem_count_seen, b.elem_count());
    return multiply_2<int>(a, b);
  };
  auto a1 = [&](const buffer<const int>& a, const buffer<int>& b) -> index_t {
    max_elem_count_seen = std::max<int>(max_elem_count_seen, b.elem_count());
    return add_1<int>(a, b);
  };

  func mul = func::make(std::move(m2), {{in, {point(x), point(y)}}}, {{intm, {x, y}}});
  func add = func::make(std::move(a1), {{intm, {point(x), point(y)}}}, {{out, {x, y}}});

  // Split the loops such that we limit the number of elements produced to a total number across both dimensions.
  var split_x(ctx, "split_x");
  var split_y(ctx, "split_y");
  add.loops({{x, split_x}, {y, split_y}});

  // Make the split area a parameter to the pipeline, and use the `lets` feature to define these splits as global
  // variables.
  var split_area(ctx, "split_area");
  std::vector<std::pair<var, expr>> lets = {
      {split_x, min(out->dim(0).extent(), split_area)},
      {split_y, max(1, split_area / split_x)},
  };
  pipeline p = build_pipeline(ctx, {split_area}, {in}, {out}, std::move(lets));

  // Run the pipeline
  const int W = 10;
  const int H = 10;

  buffer<int, 2> in_buf({W, H});
  in_buf.allocate();
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      in_buf(x, y) = y * W + x;
    }
  }

  buffer<int, 2> out_buf({W, H});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  for (int split_area : {1, 5, 10, 20, W * H, W * H * 2}) {
    const index_t args[] = {split_area};
    const raw_buffer* inputs[] = {&in_buf};
    const raw_buffer* outputs[] = {&out_buf};
    test_context eval_ctx;
    p.evaluate(args, inputs, outputs, eval_ctx);

    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        ASSERT_EQ(out_buf(x, y), 2 * (y * W + x) + 1);
      }
    }

    ASSERT_LE(max_elem_count_seen, split_area);
  }
}

class test_thread_pool_impl : public thread_pool_impl {
public:
  void run(task_ref t, task_id id = unique_task_id) override {
    run_called = true;
    thread_pool_impl::run(t, id);
  }

  bool run_called = false;
};

// An example of two 2D elementwise operations in sequence.
TEST(conditional_parallel, pipeline) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));
  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");

  func mul = func::make(multiply_2<int>, {{in, {point(x), point(y)}}}, {{intm, {x, y}}});
  func add = func::make(add_1<int>, {{intm, {point(x), point(y)}}}, {{out, {x, y}}});

  // This makes loops configurable by parameter which can be passed from outside.
  var max_workers_x(ctx, "max_workers_x");
  var max_workers_y(ctx, "max_workers_y");

  add.loops({{x, 2, max_workers_x}, {y, 1, max_workers_y}});

  pipeline p = build_pipeline(ctx, {max_workers_x, max_workers_y}, {in}, {out});

  // Run the pipeline
  const int W = 10;
  const int H = 10;

  buffer<int, 2> in_buf({W, H});
  in_buf.allocate();
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      in_buf(x, y) = y * W + x;
    }
  }

  buffer<int, 2> out_buf({W, H});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  for (int max_x : {loop::serial, loop::parallel}) {
    for (int max_y : {loop::serial, loop::parallel}) {
      const index_t args[] = {max_x, max_y};
      const raw_buffer* inputs[] = {&in_buf};
      const raw_buffer* outputs[] = {&out_buf};
      test_context eval_ctx;
      test_thread_pool_impl tp;
      eval_ctx.config.thread_pool = &tp;
      p.evaluate(args, inputs, outputs, eval_ctx);

      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          ASSERT_EQ(out_buf(x, y), 2 * (y * W + x) + 1);
        }
      }

      ASSERT_EQ(tp.run_called, max_x != loop::serial || max_y != loop::serial);
    }
  }
}

}  // namespace slinky
