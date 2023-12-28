#include "pipeline.h"
#include "expr.h"
#include "funcs.h"
#include "print.h"
#include "test.h"

#include <cassert>

using namespace slinky;

// A trivial pipeline with one stage.
TEST(pipeline_trivial) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 1);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 1);

  expr x = make_variable(ctx, "x");

  func mul = func::make<const int, int>(multiply_2<int>, {in, {interval(x)}}, {out, {x}});

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
  const buffer_base* inputs[] = {&in_buf};
  const buffer_base* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);
  ASSERT_EQ(eval_ctx.heap.peak_size, 0);
  ASSERT_EQ(eval_ctx.stack.peak_size, 0);

  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(out_buf(i), 2 * i);
  }
}

index_t multiply_2_assert_1_element(const buffer<const int>& in, const buffer<int>& out) {
  assert(in.rank == out.rank);
  assert(out.rank == 1);
  std::size_t count = 0;
  for (index_t i = out.dim(0).begin(); i < out.dim(0).end(); ++i) {
    out(i) = in(i)*2;
    ++count;
  }
  ASSERT_EQ(count, 1);
  return 0;
}

// A trivial pipeline with one stage, where the loop over the one dimesion is explicit.
TEST(pipeline_trivial_explicit) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 1);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 1);

  expr x = make_variable(ctx, "x");

  func mul = func::make<const int, int>(multiply_2_assert_1_element, {in, {interval(x)}}, {out, {x}});
  mul.loops({x});

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
  const buffer_base* inputs[] = {&in_buf};
  const buffer_base* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);
  ASSERT_EQ(eval_ctx.heap.total_size, 0);
  ASSERT_EQ(eval_ctx.stack.total_size, 0);

  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(out_buf(i), 2 * i);
  }
}

// An example of two 1D elementwise operations in sequence.
TEST(pipeline_elementwise_1d) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 1);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 1);
  auto intm = buffer_expr::make(ctx, "intm", sizeof(int), 1);

  expr x = make_variable(ctx, "x");

  func mul = func::make<const int, int>(multiply_2<int>, {in, {interval(x)}}, {intm, {x}});
  func add = func::make<const int, int>(add_1<int>, {intm, {interval(x)}}, {out, {x}});

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
  const buffer_base* inputs[] = {&in_buf};
  const buffer_base* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);
  ASSERT_EQ(eval_ctx.heap.total_size, N * sizeof(int));
  ASSERT_EQ(eval_ctx.heap.total_count, 1);

  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(out_buf(i), 2 * i + 1);
  }
}

// An example of two 1D elementwise operations in sequence.
TEST(pipeline_elementwise_1d_explicit) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 1);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 1);
  auto intm = buffer_expr::make(ctx, "intm", sizeof(int), 1);

  expr x = make_variable(ctx, "x");

  func mul = func::make<const int, int>(multiply_2<int>, {in, {interval(x)}}, {intm, {x}});
  func add = func::make<const int, int>(add_1<int>, {intm, {interval(x)}}, {out, {x}});

  add.loops({x});
  mul.compute_at({&add, x});

  intm->store_at({&add, x});
  intm->store_in(memory_type::stack);

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
  const buffer_base* inputs[] = {&in_buf};
  const buffer_base* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);
  ASSERT_EQ(eval_ctx.stack.peak_size, sizeof(int));
  ASSERT_EQ(eval_ctx.stack.peak_count, 1);
  ASSERT_EQ(eval_ctx.stack.total_size, N * sizeof(int));
  ASSERT_EQ(eval_ctx.stack.total_count, N);
  ASSERT_EQ(eval_ctx.heap.total_size, 0);
  ASSERT_EQ(eval_ctx.heap.total_count, 0);

  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(out_buf(i), 2 * i + 1);
  }
}

// Two matrix multiplies: D = (A x B) x C.
TEST(pipeline_matmuls) {
  // Make the pipeline
  node_context ctx;

  auto a = buffer_expr::make(ctx, "a", sizeof(int), 2);
  auto b = buffer_expr::make(ctx, "b", sizeof(int), 2);
  auto c = buffer_expr::make(ctx, "c", sizeof(int), 2);
  auto abc = buffer_expr::make(ctx, "abc", sizeof(int), 2);

  a->dim(1).stride_bytes = a->elem_size();
  b->dim(1).stride_bytes = b->elem_size();
  c->dim(1).stride_bytes = c->elem_size();
  abc->dim(1).stride_bytes = abc->elem_size();

  auto ab = buffer_expr::make(ctx, "ab", sizeof(int), 2);

  expr i = make_variable(ctx, "i");
  expr j = make_variable(ctx, "j");
  expr k = make_variable(ctx, "k");

  // The bounds required of the dimensions consumed by the reduction depend on the size of the
  // buffers passed in. Note that we haven't used any constants yet.
  interval K_ab(a->dim(1).min, a->dim(1).max());
  interval K_abc(c->dim(0).min, c->dim(0).max());

  // We use int for this pipeline so we can test for correctness exactly.
  func matmul_ab = func::make<const int, const int, int>(
      matmul<int>, {a, {interval(i), K_ab}}, {b, {K_ab, interval(j)}}, {ab, {i, j}});
  func matmul_abc = func::make<const int, const int, int>(
      matmul<int>, {ab, {interval(i), K_abc}}, {c, {K_abc, interval(j)}}, {abc, {i, j}});

  // TODO: There should be a more user friendly way to control the strides.
  ab->dim(1).stride_bytes = static_cast<index_t>(sizeof(int));
  ab->dim(0).stride_bytes = ab->dim(1).extent * ab->dim(1).stride_bytes;

  //matmul_abc.loops({i});
  //matmul_ab.compute_at({&matmul_abc, i});

  pipeline p(ctx, {a, b, c}, {abc});

  // Run the pipeline.
  const int M = 10;
  const int N = 10;
  buffer<int, 2> a_buf({N, M});
  buffer<int, 2> b_buf({N, M});
  buffer<int, 2> c_buf({N, M});
  buffer<int, 2> abc_buf({N, M});
  // TODO: There should be a more user friendly way to initialize a buffer with strides other than the default order.
  std::swap(a_buf.dim(1), a_buf.dim(0));
  std::swap(b_buf.dim(1), b_buf.dim(0));
  std::swap(c_buf.dim(1), c_buf.dim(0));
  std::swap(abc_buf.dim(1), abc_buf.dim(0));

  init_random(a_buf);
  init_random(b_buf);
  init_random(c_buf);
  abc_buf.allocate();

  // Not having std::span(std::initializer_list<T>) is unfortunate.
  const buffer_base* inputs[] = {&a_buf, &b_buf, &c_buf};
  const buffer_base* outputs[] = {&abc_buf};
  eval_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);
  ASSERT_EQ(eval_ctx.heap.total_size, M * N * sizeof(int));
  ASSERT_EQ(eval_ctx.heap.total_count, 1);
  ASSERT_EQ(eval_ctx.stack.total_size, 0);
  ASSERT_EQ(eval_ctx.stack.total_count, 0);

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

  expr x = make_variable(ctx, "x");
  expr y = make_variable(ctx, "y");

  func downsample =
      func::make<const int, int>(downsample2x, {in, {2 * x + interval(0, 1), 2 * y + interval(0, 1)}}, {intm, {x, y}});
  func upsample = func::make<const int, int>(upsample2x, {intm, {interval(x) / 2, interval(y) / 2}}, {out, {x, y}});

  pipeline p(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 10;
  const int H = 10;
  buffer<int, 2> in_buf({W, H});
  buffer<int, 2> out_buf({W, H});

  init_random(in_buf);
  out_buf.allocate();

  // Not having std::span(std::initializer_list<T>) is unfortunate.
  const buffer_base* inputs[] = {&in_buf};
  const buffer_base* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);
  ASSERT_EQ(eval_ctx.heap.total_size, W * H * sizeof(int) / 4);
  ASSERT_EQ(eval_ctx.heap.total_count, 1);
  ASSERT_EQ(eval_ctx.stack.total_size, 0);
  ASSERT_EQ(eval_ctx.stack.total_count, 0);
}

TEST(pipeline_stencil) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(short), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(short), 2);

  auto intm = buffer_expr::make(ctx, "intm", sizeof(short), 2);

  expr x = make_variable(ctx, "x");
  expr y = make_variable(ctx, "y");

  func add = func::make<const short, short>(add_1<short>, {in, {interval(x), interval(y)}}, {intm, {x, y}});
  func stencil =
      func::make<const short, short>(sum3x3<short>, {intm, {interval(-1, 1) + x, interval(-1, 1) + y}}, {out, {x, y}});

  stencil.loops({y});
  add.compute_at({&stencil, y});

  pipeline p(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 20;
  const int H = 10;
  buffer<short, 2> in_buf({W + 2, H + 2});
  in_buf.dim(0).translate(-1);
  in_buf.dim(1).translate(-1);
  buffer<short, 2> out_buf({W, H});

  init_random(in_buf);
  out_buf.allocate();

  // Not having std::span(std::initializer_list<T>) is unfortunate.
  const buffer_base* inputs[] = {&in_buf};
  const buffer_base* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);
  ASSERT_EQ(eval_ctx.heap.total_size, (W + 2) * (H + 2) * sizeof(short));
  ASSERT_EQ(eval_ctx.heap.total_count, 1);
  ASSERT_EQ(eval_ctx.stack.total_size, 0);
  ASSERT_EQ(eval_ctx.stack.total_count, 0);

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

TEST(pipeline_flip_y) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(char), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(char), 2);
  auto intm = buffer_expr::make(ctx, "intm", sizeof(char), 2);

  expr x = make_variable(ctx, "x");
  expr y = make_variable(ctx, "y");

  func copy = func::make<const char, char>(::copy<char>, {in, {interval(x), interval(y)}}, {intm, {x, y}});
  func flip = func::make<const char, char>(flip_y<char>, {intm, {interval(x), interval(-y)}}, {out, {x, y}});

  pipeline p(ctx, {in}, {out});

  // Run the pipeline.
  const int H = 20;
  const int W = 10;
  buffer<char, 2> in_buf({W, H});
  init_random(in_buf);

  buffer<char, 2> out_buf({W, H});
  out_buf.dim(1).translate(-H + 1);
  out_buf.allocate();
  const buffer_base* inputs[] = {&in_buf};
  const buffer_base* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);
  ASSERT_EQ(eval_ctx.heap.total_size, W * H * sizeof(char));
  ASSERT_EQ(eval_ctx.heap.total_count, 1);
  ASSERT_EQ(eval_ctx.stack.total_size, 0);
  ASSERT_EQ(eval_ctx.stack.total_count, 0);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, -y), in_buf(x, y));
    }
  }
}