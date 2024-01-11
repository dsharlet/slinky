#include "pipeline.h"
#include "expr.h"
#include "funcs.h"
#include "print.h"
#include "test.h"
#include "thread_pool.h"

#include <cassert>

using namespace slinky;

thread_pool threads;

struct memory_info {
  std::atomic<std::ptrdiff_t> live_count = 0;
  std::atomic<std::ptrdiff_t> live_size = 0;
  std::atomic<std::ptrdiff_t> total_count = 0;
  std::atomic<std::ptrdiff_t> total_size = 0;

  void track_allocate(std::size_t size) {
    live_count += 1;
    live_size += size;
    total_count += 1;
    total_size += size;
  }

  void track_free(std::size_t size) {
    live_count -= 1;
    live_size -= size;
  }
};

class test_context : public eval_context {
public:
  memory_info heap;

  test_context() {
    allocate = [this](symbol_id, raw_buffer* b) {
      b->allocate();
      heap.track_allocate(b->size_bytes());
    };
    free = [this](symbol_id, raw_buffer* b) {
      b->free();
      heap.track_free(b->size_bytes());
    };

    enqueue_many = [&](const thread_pool::task& t) { threads.enqueue(threads.thread_count(), t); };
    enqueue_one = [&](thread_pool::task t) { threads.enqueue(std::move(t)); };
    work_on_tasks = [&](std::function<bool()> while_true) { return threads.work_on_tasks(std::move(while_true)); };
  }
};

// A trivial pipeline with one stage
TEST(pipeline_trivial) {
  for (int split : {0, 1, 2, 3}) {
    for (loop_mode lm : {loop_mode::serial, loop_mode::parallel}) {
      // Make the pipeline
      node_context ctx;

      auto in = buffer_expr::make(ctx, "in", sizeof(int), 1);
      auto out = buffer_expr::make(ctx, "out", sizeof(int), 1);

      var x(ctx, "x");

      func mul = func::make<const int, int>(multiply_2<int>, {in, {point(x)}}, {out, {x}});
      if (split > 0) {
        mul.loops({{x, split, lm}});
      }

      pipeline p(ctx, {in}, {out});

      // Run the pipeline
      const int N = 10;

      buffer<int, 1> in_buf({N});
      in_buf.allocate();
      for (int i = 0; i < N; ++i) {
        in_buf(i) = i;
      }

      buffer<int, 1> out_buf({N});
      out_buf.allocate();

      // Not having std::span(std::initializer_list<T>) is unfortunate.
      const raw_buffer* inputs[] = {&in_buf};
      const raw_buffer* outputs[] = {&out_buf};
      test_context eval_ctx;
      p.evaluate(inputs, outputs, eval_ctx);
      ASSERT_EQ(eval_ctx.heap.total_size, 0);

      for (int i = 0; i < N; ++i) {
        ASSERT_EQ(out_buf(i), 2 * i);
      }
    }
  }
}

// An example of two 1D elementwise operations in sequence.
TEST(pipeline_elementwise_1d) {
  for (int split : {0, 1, 2, 3}) {
    for (bool schedule_storage : {false, true}) {
      for (loop_mode lm : {loop_mode::serial, loop_mode::parallel}) {
        // Make the pipeline
        node_context ctx;

        auto in = buffer_expr::make(ctx, "in", sizeof(int), 1);
        auto out = buffer_expr::make(ctx, "out", sizeof(int), 1);
        auto intm = buffer_expr::make(ctx, "intm", sizeof(int), 1);

        var x(ctx, "x");

        func mul = func::make<const int, int>(multiply_2<int>, {in, {point(x)}}, {intm, {x}});
        func add = func::make<const int, int>(add_1<int>, {intm, {point(x)}}, {out, {x}});

        if (split > 0) {
          add.loops({{x, split, lm}});
          mul.compute_at({&add, x});
          if (schedule_storage) {
            intm->store_at({&add, x});
            intm->store_in(memory_type::stack);
          }
        }

        pipeline p(ctx, {in}, {out});

        // Run the pipeline
        const int N = 10;

        buffer<int, 1> in_buf({N});
        in_buf.allocate();
        for (int i = 0; i < N; ++i) {
          in_buf(i) = i;
        }

        buffer<int, 1> out_buf({N});
        out_buf.allocate();

        // Not having std::span(std::initializer_list<T>) is unfortunate.
        const raw_buffer* inputs[] = {&in_buf};
        const raw_buffer* outputs[] = {&out_buf};
        test_context eval_ctx;
        p.evaluate(inputs, outputs, eval_ctx);
        if (schedule_storage) {
          ASSERT_EQ(eval_ctx.heap.total_count, 0);  // The intermediate only needs stack.
        }

        for (int i = 0; i < N; ++i) {
          ASSERT_EQ(out_buf(i), 2 * i + 1);
        }
      }
    }
  }
}

// Two matrix multiplies: D = (A x B) x C.
TEST(pipeline_matmuls) {
  for (int split : {0, 1, 2, 3}) {
    for (loop_mode lm : {loop_mode::serial, loop_mode::parallel}) {
      // Make the pipeline
      node_context ctx;

      auto a = buffer_expr::make(ctx, "a", sizeof(int), 2);
      auto b = buffer_expr::make(ctx, "b", sizeof(int), 2);
      auto c = buffer_expr::make(ctx, "c", sizeof(int), 2);
      auto abc = buffer_expr::make(ctx, "abc", sizeof(int), 2);

      auto ab = buffer_expr::make(ctx, "ab", sizeof(int), 2);

      var i(ctx, "i");
      var j(ctx, "j");

      // The bounds required of the dimensions consumed by the reduction depend on the size of the
      // buffers passed in. Note that we haven't used any constants yet.
      auto K_ab = a->dim(1).bounds;
      auto K_abc = c->dim(0).bounds;

      // We use int for this pipeline so we can test for correctness exactly.
      func matmul_ab = func::make<const int, const int, int>(
          matmul<int>, {a, {point(i), K_ab}}, {b, {K_ab, point(j)}}, {ab, {i, j}});
      func matmul_abc = func::make<const int, const int, int>(
          matmul<int>, {ab, {point(i), K_abc}}, {c, {K_abc, point(j)}}, {abc, {i, j}});

      a->dim(1).stride = a->elem_size();
      b->dim(1).stride = b->elem_size();
      c->dim(1).stride = c->elem_size();
      abc->dim(1).stride = abc->elem_size();

      // TODO: There should be a more user friendly way to control the strides.
      ab->dim(1).stride = static_cast<index_t>(sizeof(int));
      ab->dim(0).stride = ab->dim(1).extent() * ab->dim(1).stride;

      if (split > 0) {
        matmul_abc.loops({{i, split, lm}});
        matmul_ab.compute_at({&matmul_abc, i});

        if (lm == loop_mode::parallel) {
          ab->store_at({&matmul_abc, i});
        }
      }

      pipeline p(ctx, {a, b, c}, {abc});

      // Run the pipeline.
      const int M = 10;
      const int N = 10;
      buffer<int, 2> a_buf({N, M});
      buffer<int, 2> b_buf({N, M});
      buffer<int, 2> c_buf({N, M});
      buffer<int, 2> abc_buf({N, M});
      // TODO: There should be a more user friendly way to initialize a buffer with strides other than the default
      // order.
      std::swap(a_buf.dim(1), a_buf.dim(0));
      std::swap(b_buf.dim(1), b_buf.dim(0));
      std::swap(c_buf.dim(1), c_buf.dim(0));
      std::swap(abc_buf.dim(1), abc_buf.dim(0));

      init_random(a_buf);
      init_random(b_buf);
      init_random(c_buf);
      abc_buf.allocate();

      // Not having std::span(std::initializer_list<T>) is unfortunate.
      const raw_buffer* inputs[] = {&a_buf, &b_buf, &c_buf};
      const raw_buffer* outputs[] = {&abc_buf};
      test_context eval_ctx;
      p.evaluate(inputs, outputs, eval_ctx);
      if (split > 0 && lm == loop_mode::serial) {
        ASSERT_EQ(eval_ctx.heap.total_size, N * sizeof(int) * split);
      }

      buffer<int, 2> ref_ab({N, M});
      buffer<int, 2> ref_abc({N, M});
      std::swap(ref_ab.dim(1), ref_ab.dim(0));
      std::swap(ref_abc.dim(1), ref_abc.dim(0));
      ref_ab.allocate();
      ref_abc.allocate();
      matmul<int>(a_buf.cast<const int>(), b_buf.cast<const int>(), ref_ab.cast<int>());
      matmul<int>(ref_ab.cast<const int>(), c_buf.cast<const int>(), ref_abc.cast<int>());
      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
          ASSERT_EQ(ref_abc(j, i), abc_buf(j, i));
        }
      }
    }
  }
}

index_t upsample2x(const buffer<const int>& in, const buffer<int>& out) {
  for (index_t y = out.dim(1).begin(); y < out.dim(1).end(); ++y) {
    for (index_t x = out.dim(0).begin(); x < out.dim(0).end(); ++x) {
      out(x, y) = in(x >> 1, y >> 1);
    }
  }
  return 0;
}

index_t downsample2x(const buffer<const int>& in, const buffer<int>& out) {
  for (index_t y = out.dim(1).begin(); y < out.dim(1).end(); ++y) {
    for (index_t x = out.dim(0).begin(); x < out.dim(0).end(); ++x) {
      out(x, y) = (in(2 * x + 0, 2 * y + 0) + in(2 * x + 1, 2 * y + 0) + in(2 * x + 0, 2 * y + 1) +
                      in(2 * x + 1, 2 * y + 1) + 2) /
                  4;
    }
  }
  return 0;
}

TEST(pipeline_pyramid) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 2);

  auto intm = buffer_expr::make(ctx, "intm", sizeof(int), 2);

  var x(ctx, "x");
  var y(ctx, "y");

  func downsample =
      func::make<const int, int>(downsample2x, {in, {2 * x + bounds(0, 1), 2 * y + bounds(0, 1)}}, {intm, {x, y}});
  func upsample = func::make<const int, int>(upsample2x, {intm, {point(x) / 2, point(y) / 2}}, {out, {x, y}});

  pipeline p(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 10;
  const int H = 10;
  buffer<int, 2> in_buf({W, H});
  buffer<int, 2> out_buf({W, H});

  init_random(in_buf);
  out_buf.allocate();

  // Not having std::span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);
  ASSERT_EQ(eval_ctx.heap.total_size, W * H * sizeof(int) / 4);
  ASSERT_EQ(eval_ctx.heap.total_count, 1);
}

TEST(pipeline_stencil) {
  for (int split : {0, 1, 2}) {
    for (loop_mode lm : {loop_mode::serial, loop_mode::parallel}) {
      // Make the pipeline
      node_context ctx;

      auto in = buffer_expr::make(ctx, "in", sizeof(short), 2);
      auto out = buffer_expr::make(ctx, "out", sizeof(short), 2);

      auto intm = buffer_expr::make(ctx, "intm", sizeof(short), 2);

      var x(ctx, "x");
      var y(ctx, "y");

      func add = func::make<const short, short>(add_1<short>, {in, {point(x), point(y)}}, {intm, {x, y}});
      func stencil =
          func::make<const short, short>(sum3x3<short>, {intm, {bounds(-1, 1) + x, bounds(-1, 1) + y}}, {out, {x, y}});

      if (split > 0) {
        stencil.loops({{y, split, lm}});
        add.compute_at({&stencil, y});
      }

      pipeline p(ctx, {in}, {out});

      // Run the pipeline.
      const int W = 20;
      const int H = 10;
      buffer<short, 2> in_buf({W + 2, H + 2});
      in_buf.translate(-1, -1);
      buffer<short, 2> out_buf({W, H});

      init_random(in_buf);
      out_buf.allocate();

      // Not having std::span(std::initializer_list<T>) is unfortunate.
      const raw_buffer* inputs[] = {&in_buf};
      const raw_buffer* outputs[] = {&out_buf};
      test_context eval_ctx;
      p.evaluate(inputs, outputs, eval_ctx);
      if (lm == loop_mode::serial && split > 0) {
        ASSERT_EQ(eval_ctx.heap.total_size, (W + 2) * (2 + split) * sizeof(short));
      }
      ASSERT_EQ(eval_ctx.heap.total_count, 1);

      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          int correct = 0;
          for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
              correct += in_buf(x + dx, y + dy) + 1;
            }
          }
          ASSERT_EQ(correct, out_buf(x, y)) << x << " " << y;
        }
      }
    }
  }
}

TEST(pipeline_stencil_chain) {
  for (int split : {0, 1, 2}) {
    for (loop_mode lm : {loop_mode::serial, loop_mode::parallel}) {
      // Make the pipeline
      node_context ctx;

      auto in = buffer_expr::make(ctx, "in", sizeof(short), 2);
      auto out = buffer_expr::make(ctx, "out", sizeof(short), 2);

      auto intm = buffer_expr::make(ctx, "add_result", sizeof(short), 2);
      auto intm2 = buffer_expr::make(ctx, "stencil1_result", sizeof(short), 2);

      var x(ctx, "x");
      var y(ctx, "y");

      func add = func::make<const short, short>(add_1<short>, {in, {point(x), point(y)}}, {intm, {x, y}});
      func stencil1 = func::make<const short, short>(
          sum3x3<short>, {intm, {bounds(-1, 1) + x, bounds(-1, 1) + y}}, {intm2, {x, y}});
      func stencil2 =
          func::make<const short, short>(sum3x3<short>, {intm2, {bounds(-1, 1) + x, bounds(-1, 1) + y}}, {out, {x, y}});

      if (split > 0) {
        stencil2.loops({{y, split, lm}});
        add.compute_at({&stencil2, y});
        stencil1.compute_at({&stencil2, y});
      }

      pipeline p(ctx, {in}, {out});

      // Run the pipeline.
      const int W = 20;
      const int H = 10;
      buffer<short, 2> in_buf({W + 4, H + 4});
      in_buf.translate(-2, -2);
      buffer<short, 2> out_buf({W, H});

      init_random(in_buf);
      out_buf.allocate();

      // Not having std::span(std::initializer_list<T>) is unfortunate.
      const raw_buffer* inputs[] = {&in_buf};
      const raw_buffer* outputs[] = {&out_buf};
      test_context eval_ctx;
      p.evaluate(inputs, outputs, eval_ctx);
      if (split > 0 && lm == loop_mode::serial) {
        ASSERT_EQ(
            eval_ctx.heap.total_size, (W + 2) * (split + 2) * sizeof(short) + (W + 4) * (split + 2) * sizeof(short));
      }
      ASSERT_EQ(eval_ctx.heap.total_count, 2);

      // Run the pipeline stages manually to get the reference result.
      buffer<short, 2> ref_intm({W + 4, H + 4});
      buffer<short, 2> ref_intm2({W + 2, H + 2});
      buffer<short, 2> ref_out({W, H});
      ref_intm.translate(-2, -2);
      ref_intm2.translate(-1, -1);
      ref_intm.allocate();
      ref_intm2.allocate();
      ref_out.allocate();

      add_1<short>(in_buf.cast<const short>(), ref_intm.cast<short>());
      sum3x3<short>(ref_intm.cast<const short>(), ref_intm2.cast<short>());
      sum3x3<short>(ref_intm2.cast<const short>(), ref_out.cast<short>());

      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          ASSERT_EQ(ref_out(x, y), out_buf(x, y));
        }
      }
    }
  }
}

TEST(pipeline_flip_y) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(char), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(char), 2);
  auto intm = buffer_expr::make(ctx, "intm", sizeof(char), 2);

  var x(ctx, "x");
  var y(ctx, "y");

  func copy = func::make<const char, char>(::copy<char>, {in, {point(x), point(y)}}, {intm, {x, y}});
  func flip = func::make<const char, char>(flip_y<char>, {intm, {point(x), point(-y)}}, {out, {x, y}});

  pipeline p(ctx, {in}, {out});

  // Run the pipeline.
  const int H = 20;
  const int W = 10;
  buffer<char, 2> in_buf({W, H});
  init_random(in_buf);

  buffer<char, 2> out_buf({W, H});
  out_buf.dim(1).translate(-H + 1);
  out_buf.allocate();
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);
  ASSERT_EQ(eval_ctx.heap.total_size, W * H * sizeof(char));
  ASSERT_EQ(eval_ctx.heap.total_count, 1);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, -y), in_buf(x, y));
    }
  }
}

TEST(pipeline_padded_copy) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(char), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(char), 2);
  auto intm = buffer_expr::make(ctx, "intm", sizeof(char), 2);

  var x(ctx, "x");
  var y(ctx, "y");

  // We could just clamp using the bounds directly below, but that would hardcode the bounds we clamp
  // in the pipeline. This way, the bounds can vary at eval-time.
  var w(ctx, "w");
  var h(ctx, "h");

  // Copy the input so we can measure the size of the buffer we think we need internally.
  func copy = func::make<const char, char>(::copy<char>, {in, {point(x), point(y)}}, {intm, {x, y}});
  // This is elementwise, but with a clamp to limit the bounds required of the input.
  func crop = func::make<const char, char>(
      ::zero_padded_copy<char>, {intm, {point(clamp(x, 0, w - 1)), point(clamp(y, 0, h - 1))}}, {out, {x, y}});

  crop.loops({y});

  pipeline p(ctx, {w, h}, {in}, {out});

  const int W = 8;
  const int H = 5;

  // Run the pipeline.
  buffer<char, 2> in_buf({W, H});
  init_random(in_buf);

  // Ask for an output padded in every direction.
  buffer<char, 2> out_buf({W * 3, H * 3});
  out_buf.translate(-W, -H);
  out_buf.allocate();

  index_t args[] = {W, H};
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(args, inputs, outputs, eval_ctx);
  ASSERT_EQ(eval_ctx.heap.total_size, W * H * sizeof(char));
  ASSERT_EQ(eval_ctx.heap.total_count, 1);

  for (int y = -H; y < 2 * H; ++y) {
    for (int x = -W; x < 2 * W; ++x) {
      if (0 <= x && x < W && 0 <= y && y < H) {
        ASSERT_EQ(out_buf(x, y), in_buf(x, y));
      } else {
        ASSERT_EQ(out_buf(x, y), 0);
      }
    }
  }
}

TEST(pipeline_multiple_outputs) {
  for (int split : {0, 1, 2, 3}) {
    for (loop_mode lm : {loop_mode::serial, loop_mode::parallel}) {
      // Make the pipeline
      node_context ctx;

      auto in = buffer_expr::make(ctx, "in", sizeof(int), 3);
      auto sum_x = buffer_expr::make(ctx, "sum_x", sizeof(int), 2);
      auto sum_xy = buffer_expr::make(ctx, "sum_xy", sizeof(int), 1);

      var x(ctx, "x");
      var y(ctx, "y");
      var z(ctx, "z");

      auto X = in->dim(0).bounds;
      auto Y = in->dim(1).bounds;

      // For a 3D input in(x, y, z), compute sum_x = sum(input(:, y, z)) and sum_xy = sum(input(:, :, z)) in one stage.
      auto sum_x_xy = [](const buffer<const int>& in, const buffer<int>& sum_x, const buffer<int>& sum_xy) -> index_t {
        assert(sum_x.dim(1).min() == sum_xy.dim(0).min());
        for (index_t z = sum_xy.dim(0).min(); z <= sum_xy.dim(0).max(); ++z) {
          sum_xy(z) = 0;
          for (index_t y = sum_x.dim(0).min(); y <= sum_x.dim(0).max(); ++y) {
            sum_x(y, z) = 0;
            for (index_t x = in.dim(0).min(); x <= in.dim(0).max(); ++x) {
              sum_x(y, z) += in(x, y, z);
              sum_xy(z) += in(x, y, z);
            }
          }
        }
        return 0;
      };
      func sums = func::make<const int, int, int>(sum_x_xy, {in, {X, Y, point(z)}}, {sum_x, {y, z}}, {sum_xy, {z}});

      if (split > 0) {
        sums.loops({{z, split, lm}});
      }

      pipeline p(ctx, {in}, {sum_x, sum_xy});

      // Run the pipeline.
      const int H = 20;
      const int W = 10;
      const int D = 5;
      buffer<int, 3> in_buf({W, H, D});
      init_random(in_buf);

      buffer<int, 2> sum_x_buf({H, D});
      buffer<int, 1> sum_xy_buf({D});
      sum_x_buf.allocate();
      sum_xy_buf.allocate();
      const raw_buffer* inputs[] = {&in_buf};
      const raw_buffer* outputs[] = {&sum_x_buf, &sum_xy_buf};
      test_context eval_ctx;
      p.evaluate(inputs, outputs, eval_ctx);

      for (int z = 0; z < D; ++z) {
        int expected_xy = 0;
        for (int y = 0; y < H; ++y) {
          int expected_x = 0;
          for (int x = 0; x < W; ++x) {
            expected_x += in_buf(x, y, z);
            expected_xy += in_buf(x, y, z);
          }
          ASSERT_EQ(sum_x_buf(y, z), expected_x);
        }
        ASSERT_EQ(sum_xy_buf(z), expected_xy);
      }
    }
  }
}

TEST(pipeline_outer_product) {
  for (int split_i : {0, 1, 2, 3}) {
    for (int split_j : {0, 1, 2, 3}) {
      for (loop_mode lm : {loop_mode::serial, loop_mode::parallel}) {
        // Make the pipeline
        node_context ctx;

        auto a = buffer_expr::make(ctx, "a", sizeof(int), 1);
        auto b = buffer_expr::make(ctx, "b", sizeof(int), 1);
        auto out = buffer_expr::make(ctx, "out", sizeof(int), 2);

        var i(ctx, "i");
        var j(ctx, "j");

        func outer =
            func::make<const int, const int, int>(outer_product<int>, {a, {point(i)}}, {b, {point(j)}}, {out, {i, j}});

        std::vector<func::loop_info> loops;
        if (split_i > 0) loops.emplace_back(i, split_i, lm);
        if (split_j > 0) loops.emplace_back(j, split_j, lm);
        outer.loops(loops);

        pipeline p(ctx, {a, b}, {out});

        // Run the pipeline.
        const int M = 20;
        const int N = 10;
        buffer<int, 1> a_buf({M});
        buffer<int, 1> b_buf({N});
        init_random(a_buf);
        init_random(b_buf);

        buffer<int, 2> out_buf({M, N});
        out_buf.allocate();
        const raw_buffer* inputs[] = {&a_buf, &b_buf};
        const raw_buffer* outputs[] = {&out_buf};
        test_context eval_ctx;
        p.evaluate(inputs, outputs, eval_ctx);

        for (int j = 0; j < N; ++j) {
          for (int i = 0; i < M; ++i) {
            ASSERT_EQ(out_buf(i, j), a_buf(i) * b_buf(j));
          }
        }
      }
    }
  }
}

TEST(pipeline_unrelated) {
  // Make the pipeline
  node_context ctx;

  auto in1 = buffer_expr::make(ctx, "in1", sizeof(short), 2);
  auto out1 = buffer_expr::make(ctx, "out1", sizeof(short), 2);
  auto intm1 = buffer_expr::make(ctx, "intm1", sizeof(short), 2);

  auto in2 = buffer_expr::make(ctx, "in2", sizeof(int), 1);
  auto out2 = buffer_expr::make(ctx, "out2", sizeof(int), 1);
  auto intm2 = buffer_expr::make(ctx, "intm2", sizeof(int), 1);

  var x(ctx, "x");
  var y(ctx, "y");

  func add1 = func::make<const short, short>(add_1<short>, {in1, {point(x), point(y)}}, {intm1, {x, y}});
  func stencil1 =
      func::make<const short, short>(sum3x3<short>, {intm1, {bounds(-1, 1) + x, bounds(-1, 1) + y}}, {out1, {x, y}});

  func mul2 = func::make<const int, int>(multiply_2<int>, {in2, {point(x)}}, {intm2, {x}});
  func add2 = func::make<const int, int>(add_1<int>, {intm2, {point(x)}}, {out2, {x}});

  stencil1.loops({{y, 2}});
  add1.compute_at({&stencil1, y});

  pipeline p(ctx, {in1, in2}, {out1, out2});

  // Run the pipeline.
  const int W1 = 20;
  const int H1 = 10;
  buffer<short, 2> in1_buf({W1 + 2, H1 + 2});
  in1_buf.translate(-1, -1);
  buffer<short, 2> out1_buf({W1, H1});

  init_random(in1_buf);
  out1_buf.allocate();

  const int N2 = 30;
  buffer<int, 1> in2_buf({N2});
  in2_buf.allocate();
  for (int i = 0; i < N2; ++i) {
    in2_buf(i) = i;
  }

  buffer<int, 1> out2_buf({N2});
  out2_buf.allocate();

  // Not having std::span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in1_buf, &in2_buf};
  const raw_buffer* outputs[] = {&out1_buf, &out2_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  ASSERT_EQ(eval_ctx.heap.total_size, (W1 + 2) * 4 * sizeof(short));
  ASSERT_EQ(eval_ctx.heap.total_count, 1);  // intm2 aliased to out2.

  for (int y = 0; y < H1; ++y) {
    for (int x = 0; x < W1; ++x) {
      int correct = 0;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          correct += in1_buf(x + dx, y + dy) + 1;
        }
      }
      ASSERT_EQ(correct, out1_buf(x, y)) << x << " " << y;
    }
  }

  for (int i = 0; i < N2; ++i) {
    ASSERT_EQ(out2_buf(i), 2 * i + 1);
  }
}

TEST(pipeline_copied_result) {
  for (int schedule : {0, 1, 2}) {
    // Make the pipeline
    node_context ctx;

    auto in = buffer_expr::make(ctx, "in", sizeof(short), 2);
    auto out = buffer_expr::make(ctx, "out", sizeof(short), 2);

    auto intm = buffer_expr::make(ctx, "intm", sizeof(short), 2);

    var x(ctx, "x");
    var y(ctx, "y");

    // In this pipeline, the result is copied to the output. We should just compute the result directly in the output.
    func stencil =
        func::make<const short, short>(sum3x3<short>, {in, {bounds(-1, 1) + x, bounds(-1, 1) + y}}, {intm, {x, y}});
    func padded = func::make_copy({intm, {point(x), point(y)}}, {out, {x, y}});

    switch (schedule) {
    case 0: break;
    case 1: padded.loops({y}); break;
    case 2:
      padded.loops({y});
      stencil.compute_at({&padded, y});
      break;
    }

    pipeline p(ctx, {in}, {out});

    // Run the pipeline.
    const int W = 20;
    const int H = 10;
    buffer<short, 2> in_buf({W + 2, H + 2});
    in_buf.translate(-1, -1);
    buffer<short, 2> out_buf({W, H});

    init_random(in_buf);
    out_buf.allocate();

    // Not having std::span(std::initializer_list<T>) is unfortunate.
    const raw_buffer* inputs[] = {&in_buf};
    const raw_buffer* outputs[] = {&out_buf};
    test_context eval_ctx;
    p.evaluate(inputs, outputs, eval_ctx);
    ASSERT_EQ(eval_ctx.heap.total_count, 0);

    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        int correct = 0;
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            correct += in_buf(x + dx, y + dy);
          }
        }
        ASSERT_EQ(correct, out_buf(x, y)) << x << " " << y;
      }
    }
  }
}

TEST(pipeline_padded_stencil) {
  for (int schedule : {0, 1, 2, 3}) {
    // Make the pipeline
    node_context ctx;

    auto in = buffer_expr::make(ctx, "in", sizeof(short), 2);
    auto out = buffer_expr::make(ctx, "out", sizeof(short), 2);

    auto intm = buffer_expr::make(ctx, "intm", sizeof(short), 2);
    auto padded_intm = buffer_expr::make(ctx, "padded_intm", sizeof(short), 2);

    var x(ctx, "x");
    var y(ctx, "y");

    var w(ctx, "w");
    var h(ctx, "h");

    func add = func::make<const short, short>(add_1<short>, {in, {point(x), point(y)}}, {intm, {x, y}});
    func padded =
        func::make_copy({intm, {point(clamp(x, 0, w - 1)), point(clamp(y, 0, h - 1))}}, {padded_intm, {x, y}}, {6, 0});
    func stencil = func::make<const short, short>(
        sum3x3<short>, {padded_intm, {bounds(-1, 1) + x, bounds(-1, 1) + y}}, {out, {x, y}});

    switch (schedule) {
    case 0: break;
    case 1: stencil.loops({y}); break;
    case 2:
      stencil.loops({y});
      padded.compute_at({&stencil, y});
      break;
    case 3:
      stencil.loops({y});
      padded.compute_at({&stencil, y});
      add.compute_at({&stencil, y});
      break;
    }

    pipeline p(ctx, {w, h}, {in}, {out});

    // Run the pipeline.
    const int W = 20;
    const int H = 10;
    buffer<short, 2> in_buf({W, H});
    buffer<short, 2> out_buf({W, H});

    init_random(in_buf);
    out_buf.allocate();

    // Not having std::span(std::initializer_list<T>) is unfortunate.
    index_t args[] = {W, H};
    const raw_buffer* inputs[] = {&in_buf};
    const raw_buffer* outputs[] = {&out_buf};
    test_context eval_ctx;
    p.evaluate(args, inputs, outputs, eval_ctx);
    ASSERT_EQ(eval_ctx.heap.total_size, (W + 2) * (H + 2) * sizeof(short));
    ASSERT_EQ(eval_ctx.heap.total_count, 1);

    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        int correct = 0;
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            if (0 <= x + dx && x + dx < W && 0 <= y + dy && y + dy < H) {
              correct += in_buf(x + dx, y + dy) + 1;
            } else {
              correct += 6;
            }
          }
        }
        ASSERT_EQ(correct, out_buf(x, y)) << x << " " << y;
      }
    }
  }
}