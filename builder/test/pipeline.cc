#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <numeric>

#include "builder/pipeline.h"
#include "builder/replica_pipeline.h"
#include "builder/substitute.h"
#include "builder/test/context.h"
#include "builder/test/funcs.h"
#include "builder/test/util.h"
#include "runtime/expr.h"
#include "runtime/pipeline.h"
#include "runtime/print.h"

namespace slinky {

// Replaces callbacks within the call_stmt with nullptr, so they can be compared.
class call_nullifier : public node_mutator {
public:
  void visit(const call_stmt* op) override { set_result(call_stmt::make(nullptr, op->inputs, op->outputs, op->attrs)); }

  using node_mutator::visit;
};

stmt nullify_calls(const stmt& s) { return call_nullifier().mutate(s); }

// Matrix multiplication (not fast!)
template <typename T>
index_t matmul(const buffer<const T>& a, const buffer<const T>& b, const buffer<T>& c) {
  assert(a.rank == 2);
  assert(b.rank == 2);
  assert(c.rank == 2);
  assert(a.dim(1).begin() == b.dim(0).begin());
  assert(a.dim(1).end() == b.dim(0).end());
  assert(a.dim(1).stride() == sizeof(T));
  assert(b.dim(1).stride() == sizeof(T));
  assert(c.dim(1).stride() == sizeof(T));
  for (index_t i = c.dim(0).begin(); i < c.dim(0).end(); ++i) {
    for (index_t j = c.dim(1).begin(); j < c.dim(1).end(); ++j) {
      c(i, j) = 0;
      for (index_t k = a.dim(1).begin(); k < a.dim(1).end(); ++k) {
        c(i, j) += a(i, k) * b(k, j);
      }
    }
  }
  return 0;
}

const auto loop_modes = testing::Values(loop::serial, loop::parallel);

class trivial : public testing::TestWithParam<std::tuple<int, int>> {};

INSTANTIATE_TEST_SUITE_P(
    split_mode, trivial, testing::Combine(loop_modes, testing::Range(0, 4)), test_params_to_string<trivial::ParamType>);

// A trivial pipeline with one stage
TEST_P(trivial, pipeline) {
  int max_workers = std::get<0>(GetParam());
  int split = std::get<1>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 1, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 1, sizeof(int));

  var x(ctx, "x");

  func mul =
      func::make(multiply_2<int>, {{in, {point(x)}}}, {{out, {x}}}, call_stmt::attributes{.allow_in_place = 0x1});
  if (split > 0) {
    mul.loops({{x, split, max_workers}});
  }

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline
  const int N = 10;

  buffer<int, 1> in_buf({N});
  in_buf.allocate();
  for (int i = 0; i < N; ++i) {
    in_buf(i) = i;
  }

  buffer<int, 1> out_buf({N});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(out_buf(i), 2 * i);
  }

  ASSERT_EQ(eval_ctx.heap.allocs.size(), 0);
}

class elementwise : public testing::TestWithParam<std::tuple<int, int, bool>> {};

INSTANTIATE_TEST_SUITE_P(split_schedule_mode, elementwise,
    testing::Combine(loop_modes, testing::Range(0, 4), testing::Bool()), test_params_to_string<elementwise::ParamType>);

// An example of two 1D elementwise operations in sequence.
TEST_P(elementwise, pipeline_1d) {
  int max_workers = std::get<0>(GetParam());
  int split = std::get<1>(GetParam());
  bool schedule_storage = std::get<2>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 1, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 1, sizeof(int));
  auto intm = buffer_expr::make(ctx, "intm", 1, sizeof(int));

  var x(ctx, "x");

  if (split == 0) {
    // If we want to alias intermediate buffer to the output buffer,
    // we need to tell aliaser that output is unfolded and it's safe to alias.
    out->dim(0).fold_factor = dim::unfolded;
  }

  // Here we explicitly use std::functions (in the form of a
  // func::callable typedef) to wrap the local calls
  // purely to verify that the relevant func::make calls work correctly.
  func::callable<const int, int> m2 = multiply_2<int>;
  func::callable<const int, int> a1 = add_1<int>;

  func mul = func::make(
      std::move(m2), {{in, {point(x)}}}, {{intm, {x}}}, call_stmt::attributes{.allow_in_place = 0x1, .name = "mul"});
  func add = func::make(
      std::move(a1), {{intm, {point(x)}}}, {{out, {x}}}, call_stmt::attributes{.allow_in_place = 0x1, .name = "add"});

  if (split > 0) {
    add.loops({{x, split, max_workers}});
    if (schedule_storage) {
      intm->store_at({&add, x});
      intm->store_in(memory_type::stack);
    }
  }

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline
  const int N = 10;

  buffer<int, 1> in_buf({N});
  in_buf.allocate();
  for (int i = 0; i < N; ++i) {
    in_buf(i) = i;
  }

  buffer<int, 1> out_buf({N});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(out_buf(i), 2 * i + 1);
  }

  if (split > 0 && schedule_storage) {
    ASSERT_EQ(eval_ctx.heap.allocs.size(), 0);  // The intermediate only needs stack.
  }
}

// An example of two 2D elementwise operations in sequence.
TEST_P(elementwise, pipeline_2d) {
  int max_workers = std::get<0>(GetParam());
  int split = std::get<1>(GetParam());
  bool schedule_storage = std::get<2>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));
  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(int));

  if (split == 0) {
    // If we want to alias intermediate buffer to the output buffer,
    // we need to tell aliaser that output is unfolded and it's safe to alias.
    out->dim(0).fold_factor = dim::unfolded;
    out->dim(1).fold_factor = dim::unfolded;
  }

  var x(ctx, "x");
  var y(ctx, "y");

  // Here we explicitly use lambdas to wrap the local calls,
  // purely to verify that the relevant func::make calls work correctly.
  auto m2 = [](const buffer<const int>& a, raw_buffer b) -> index_t { return multiply_2<int>(a, b.cast<int>()); };
  auto a1 = [](const raw_buffer& a, const raw_buffer* b) -> index_t {
    return add_1<int>(a.cast<const int>(), b->cast<int>());
  };

  func mul = func::make(
      std::move(m2), {{in, {point(x), point(y)}}}, {{intm, {x, y}}}, call_stmt::attributes{.allow_in_place = 0x1});
  func add = func::make(
      std::move(a1), {{intm, {point(x), point(y)}}}, {{out, {x, y}}}, call_stmt::attributes{.allow_in_place = 0x1});

  if (split > 0) {
    add.loops({{x, split, max_workers}, {y, split, max_workers}});
    if (schedule_storage) {
      intm->store_at({&add, x});
      intm->store_in(memory_type::stack);
    }
  }

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline
  const int W = 15;
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
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y), 2 * (y * W + x) + 1);
    }
  }

  if (split > 0 && schedule_storage) {
    ASSERT_EQ(eval_ctx.heap.allocs.size(), 0);  // The intermediate only needs stack.
  }
}

class store_at : public testing::TestWithParam<bool> {};

INSTANTIATE_TEST_SUITE_P(alias_in_place, store_at, testing::Bool());

// An example of two 2D elementwise operations in sequence with intermediate buffer stored at
// the inner loop level.
TEST_P(store_at, elementwise_2d) {
  bool alias_in_place = GetParam();
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));
  auto intm1 = buffer_expr::make(ctx, "intm1", 2, sizeof(int));
  auto intm2 = buffer_expr::make(ctx, "intm2", 2, sizeof(int));
  auto intm3 = buffer_expr::make(ctx, "intm3", 2, sizeof(int));
  auto intm4 = buffer_expr::make(ctx, "intm4", 2, sizeof(int));

  // If we want to alias intermediate buffer to the output buffer,
  // we need to tell aliaser that output is unfolded and it's safe to alias.
  out->dim(0).fold_factor = dim::unfolded;
  out->dim(1).fold_factor = dim::unfolded;

  var x(ctx, "x");
  var y(ctx, "y");

  func mul1 = func::make(multiply_2<int>, {{in, {point(x), point(y)}}}, {{intm1, {x, y}}},
      call_stmt::attributes{.allow_in_place = alias_in_place});
  func add1 = func::make(add_1<int>, {{intm1, {point(x), point(y)}}}, {{intm2, {x, y}}},
      call_stmt::attributes{.allow_in_place = alias_in_place});
  func mul2 = func::make(multiply_2<int>, {{intm2, {point(x), point(y)}}}, {{intm3, {x, y}}},
      call_stmt::attributes{.allow_in_place = alias_in_place});
  func add2 = func::make(add_1<int>, {{intm3, {point(x), point(y)}}}, {{intm4, {x, y}}},
      call_stmt::attributes{.allow_in_place = alias_in_place});
  func mul3 = func::make(multiply_2<int>, {{intm4, {point(x), point(y)}}}, {{out, {x, y}}},
      call_stmt::attributes{.allow_in_place = alias_in_place});

  mul3.loops({{y, 1, loop::serial}});
  buffer_expr_ptr intms[] = {intm1, intm2, intm3, intm4};
  for (auto& b : intms) {
    b->store_at({&mul3, y});
    b->store_in(memory_type::stack);
  }

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline
  const int W = 15;
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
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y), 2 * (2 * (2 * (y * W + x) + 1) + 1));
    }
  }

  ASSERT_EQ(eval_ctx.heap.allocs.size(), 0);  // The intermediate only needs stack.
}

// Two separate loops where intermediate buffer can be folded in theory, but shouldn't
// because consumer is in a different loop.
TEST(elementwise, outside_fold) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));
  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");

  // Here we explicitly use lambdas to wrap the local calls,
  // purely to verify that the relevant func::make calls work correctly.
  auto m2 = [](const buffer<const int>& a, const buffer<int>& b) -> index_t { return multiply_2<int>(a, b); };
  auto a1 = [](const buffer<const int>& a, const buffer<int>& b) -> index_t { return add_1<int>(a, b); };

  func mul = func::make(
      std::move(m2), {{in, {point(x), point(y)}}}, {{intm, {x, y}}}, call_stmt::attributes{.allow_in_place = 0x1});
  func add = func::make(
      std::move(a1), {{intm, {point(x), point(y)}}}, {{out, {x, y}}}, call_stmt::attributes{.allow_in_place = 0x1});

  mul.loops({{y, 1}});
  mul.compute_root();
  add.loops({{y, 1}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline
  const int W = 15;
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
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y), 2 * (y * W + x) + 1);
    }
  }
}

class matmuls : public testing::TestWithParam<std::tuple<int, int>> {};

INSTANTIATE_TEST_SUITE_P(
    split_mode, matmuls, testing::Combine(loop_modes, testing::Range(0, 4)), test_params_to_string<matmuls::ParamType>);

// Two matrix multiplies: D = (A x B) x C.
TEST_P(matmuls, pipeline) {
  int max_workers = std::get<0>(GetParam());
  int split = std::get<1>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto a = buffer_expr::make(ctx, "a", 2, sizeof(int));
  auto b = buffer_expr::make(ctx, "b", 2, sizeof(int));
  auto c = buffer_expr::make(ctx, "c", 2, sizeof(int));
  auto abc = buffer_expr::make(ctx, "abc", 2, sizeof(int));

  auto ab = buffer_expr::make(ctx, "ab", 2, sizeof(int));

  var i(ctx, "i");
  var j(ctx, "j");

  // The bounds required of the dimensions consumed by the reduction depend on the size of the
  // buffers passed in. Note that we haven't used any constants yet.
  auto K_ab = a->dim(1).bounds;
  auto K_abc = c->dim(0).bounds;

  // We use int for this pipeline so we can test for correctness exactly.
  func matmul_ab = func::make(matmul<int>, {{a, {point(i), K_ab}}, {b, {K_ab, point(j)}}}, {{ab, {i, j}}});
  func matmul_abc = func::make(matmul<int>, {{ab, {point(i), K_abc}}, {c, {K_abc, point(j)}}}, {{abc, {i, j}}});

  a->dim(1).stride = a->elem_size();
  b->dim(1).stride = b->elem_size();
  c->dim(1).stride = c->elem_size();
  abc->dim(1).stride = abc->elem_size();

  // TODO: There should be a more user friendly way to control the strides.
  ab->dim(1).stride = ab->elem_size();

  if (split > 0) {
    matmul_abc.loops({{i, split, max_workers}});

    if (max_workers != loop::serial) {
      ab->store_at({&matmul_abc, i});
    }
  }

  pipeline p = build_pipeline(ctx, {a, b, c}, {abc});

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

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&a_buf, &b_buf, &c_buf};
  const raw_buffer* outputs[] = {&abc_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

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

  if (split > 0 && max_workers == loop::serial) {
    ASSERT_THAT(eval_ctx.heap.allocs, testing::UnorderedElementsAre(N * sizeof(int) * split));
  }

  if (split == 1 && max_workers == loop::serial) {
    check_replica_pipeline(define_replica_pipeline(ctx, {a, b, c}, {abc}));
  }
}

class stencil : public testing::TestWithParam<std::tuple<int, int, int>> {};

INSTANTIATE_TEST_SUITE_P(split_split_mode, stencil,
    testing::Combine(loop_modes, testing::Range(0, 4), testing::Range(0, 4)),
    test_params_to_string<stencil::ParamType>);

TEST_P(stencil, pipeline) {
  int max_workers = std::get<0>(GetParam());
  int split = std::get<1>(GetParam());
  int split_intermediate = std::get<2>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));

  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(short));

  var x(ctx, "x");
  var y(ctx, "y");

  var s(ctx, "s");
  var t(ctx, "t");

  func add = func::make(add_1<short>, {{in, {point(s), point(t)}}}, {{intm, {s, t}}});
  func stencil = func::make(sum3x3<short>, {{intm, {bounds(-1, 1) + x, bounds(-1, 1) + y}}}, {{out, {x, y}}});

  if (split > 0) {
    stencil.loops({{y, split, max_workers}});
  }

  if (split_intermediate > 0) {
    add.loops({{t, split_intermediate, loop::serial}});
  }

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 20;
  const int H = 30;
  buffer<short, 2> in_buf({W + 2, H + 2});
  in_buf.translate(-1, -1);
  buffer<short, 2> out_buf({W, H});

  init_random(in_buf);
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

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

  if (split > 0) {
    const int parallel_extra = max_workers != loop::serial ? split : 0;
    const int intm_size = (W + 2) * (split + parallel_extra + 2) * sizeof(short);
    ASSERT_THAT(eval_ctx.heap.allocs, testing::UnorderedElementsAre(intm_size));
  } else {
    ASSERT_EQ(eval_ctx.heap.allocs.size(), 1);
  }

  // Also visualize this pipeline.
  if (max_workers == loop::serial && split_intermediate == 0) {
    check_visualize("stencil_split_" + std::to_string(split) + ".html", p, inputs, outputs, &ctx);
  }
}

class slide_2d : public testing::TestWithParam<std::tuple<int, int, bool>> {};

INSTANTIATE_TEST_SUITE_P(split_split_mode, slide_2d, testing::Combine(loop_modes, loop_modes, testing::Bool()),
    test_params_to_string<slide_2d::ParamType>);

TEST_P(slide_2d, pipeline) {
  int max_workers_x = std::get<0>(GetParam());
  int max_workers_y = std::get<1>(GetParam());
  bool constrain_min = std::get<2>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));

  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(short));

  if (constrain_min) {
    out->dim(0).bounds.min = 0;
    out->dim(1).bounds.min = 0;
  }

  var x(ctx, "x");
  var y(ctx, "y");

  std::atomic<int> add_count = 0;
  auto add_counter = [&add_count](const buffer<const short>& in, const buffer<short>& out) -> index_t {
    add_count += out.dim(0).extent() * out.dim(1).extent();
    return add_1<short>(in, out);
  };

  func add = func::make(std::move(add_counter), {{in, {point(x), point(y)}}}, {{intm, {x, y}}});
  func stencil = func::make(sum3x3<short>, {{intm, {bounds(-1, 1) + x, bounds(-1, 1) + y}}}, {{out, {x, y}}});

  stencil.loops({{x, 1, max_workers_x}, {y, 1, max_workers_y}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 20;
  const int H = 10;
  buffer<short, 2> in_buf({W + 2, H + 2});
  in_buf.translate(-1, -1);
  buffer<short, 2> out_buf({W, H});

  init_random(in_buf);
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

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

  ASSERT_THAT(eval_ctx.heap.allocs, testing::UnorderedElementsAre((W + 2) * 3 * sizeof(short)));
  ASSERT_EQ(add_count, (W + 2) * (H + 2));
}

class stencil_chain : public testing::TestWithParam<std::tuple<int, int>> {};

INSTANTIATE_TEST_SUITE_P(split_split_mode, stencil_chain, testing::Combine(loop_modes, testing::Range(0, 5)),
    test_params_to_string<stencil_chain::ParamType>);

TEST_P(stencil_chain, pipeline) {
  int max_workers = std::get<0>(GetParam());
  int split = std::get<1>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));

  auto intm = buffer_expr::make(ctx, "add_result", 2, sizeof(short));
  auto intm2 = buffer_expr::make(ctx, "stencil1_result", 2, sizeof(short));

  var x(ctx, "x");
  var y(ctx, "y");

  func add =
      func::make(add_1<short>, {{in, {point(x), point(y)}}}, {{intm, {x, y}}}, call_stmt::attributes{.name = "add_1"});
  func stencil1 = func::make(sum3x3<short>, {{intm, {bounds(-1, 1) + x, bounds(-1, 1) + y}}}, {{intm2, {x, y}}},
      call_stmt::attributes{.name = "sum3x3"});
  func stencil2 = func::make(sum3x3<short>, {{intm2, {bounds(-1, 1) + x, bounds(-1, 1) + y}}}, {{out, {x, y}}},
      call_stmt::attributes{.name = "sum3x3"});

  if (split > 0) {
    stencil2.loops({{y, split, max_workers}});
  }

  pipeline p = build_pipeline(ctx, {in}, {out}, build_options{.trace = true});

  // Run the pipeline.
  const int W = 20;
  const int H = 30;
  buffer<short, 2> in_buf({W + 4, H + 4});
  in_buf.translate(-2, -2);
  buffer<short, 2> out_buf({W, H});

  init_random(in_buf);
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;

  std::string test_name = "stencil_chain_split_" + std::string(max_workers == loop::serial ? "serial" : "parallel") +
                          "_split_" + std::to_string(split);
  setup_tracing(eval_ctx.config, test_name + ".json");

  p.evaluate(inputs, outputs, eval_ctx);

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

  if (split > 0) {
    const int parallel_extra = max_workers != loop::serial ? split * 2 : 0;
    const int intm_size = (W + 2) * (split + parallel_extra + 2) * sizeof(short);
    const int intm2_size = (W + 4) * (split + parallel_extra + 2) * sizeof(short);
    ASSERT_THAT(eval_ctx.heap.allocs, testing::UnorderedElementsAre(intm_size, intm2_size));
  } else {
    ASSERT_EQ(eval_ctx.heap.allocs.size(), 2);
  }

  // Also visualize this pipeline.
  if (max_workers == loop::serial) {
    check_visualize(test_name + ".html", p, inputs, outputs, &ctx);
  }
}

class multiple_outputs : public testing::TestWithParam<std::tuple<int, int, bool>> {};

INSTANTIATE_TEST_SUITE_P(split_mode, multiple_outputs,
    testing::Combine(loop_modes, testing::Range(0, 4), testing::Bool()),
    test_params_to_string<multiple_outputs::ParamType>);

TEST_P(multiple_outputs, pipeline) {
  int max_workers = std::get<0>(GetParam());
  int split = std::get<1>(GetParam());
  bool ignore_one_output = std::get<2>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 3, sizeof(int));
  auto sum_x = buffer_expr::make(ctx, "sum_x", 2, sizeof(int));
  auto sum_xy = buffer_expr::make(ctx, "sum_xy", 1, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");

  auto X = in->dim(0).bounds;
  auto Y = in->dim(1).bounds;

  // For a 3D input in(x, y, z), compute sum_x = sum(input(:, y, z)) and sum_xy = sum(input(:, :, z)) in one stage.
  func::callable<const int, int, int> sum_x_xy = [](const buffer<const int>& in, const buffer<int>& sum_x,
                                                     const buffer<int>& sum_xy) -> index_t {
    for (index_t z = std::min(sum_xy.dim(0).min(), sum_x.dim(1).min());
         z <= std::max(sum_xy.dim(0).max(), sum_x.dim(1).max()); ++z) {
      if (sum_xy.contains(z)) sum_xy(z) = 0;
      for (index_t y = sum_x.dim(0).min(); y <= sum_x.dim(0).max(); ++y) {
        if (sum_x.contains(y, z)) sum_x(y, z) = 0;
        for (index_t x = in.dim(0).min(); x <= in.dim(0).max(); ++x) {
          if (sum_x.contains(y, z)) sum_x(y, z) += in(x, y, z);
          if (sum_xy.contains(z)) sum_xy(z) += in(x, y, z);
        }
      }
    }
    return 0;
  };
  func sums = func::make(std::move(sum_x_xy), {{in, {X, Y, point(z)}}}, {{sum_x, {y, z}}, {sum_xy, {z}}});

  if (split > 0) {
    sums.loops({{z, split, max_workers}});
  }

  if (ignore_one_output) {
    pipeline p = build_pipeline(ctx, {in}, {sum_x});

    // Run the pipeline.
    const int H = 20;
    const int W = 10;
    const int D = 5;
    buffer<int, 3> in_buf({W, H, D});
    init_random(in_buf);

    buffer<int, 2> sum_x_buf({H, D});
    sum_x_buf.allocate();
    const raw_buffer* inputs[] = {&in_buf};
    const raw_buffer* outputs[] = {&sum_x_buf};
    test_context eval_ctx;
    p.evaluate(inputs, outputs, eval_ctx);

    for (int z = 0; z < D; ++z) {
      for (int y = 0; y < H; ++y) {
        int expected_x = 0;
        for (int x = 0; x < W; ++x) {
          expected_x += in_buf(x, y, z);
        }
        ASSERT_EQ(sum_x_buf(y, z), expected_x);
      }
    }
  } else {
    pipeline p = build_pipeline(ctx, {in}, {sum_x, sum_xy});

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

  if (split == 1 && max_workers == loop::serial && !ignore_one_output) {
    check_replica_pipeline(define_replica_pipeline(ctx, {in}, {sum_x, sum_xy}));
  }
}

class outer_product : public testing::TestWithParam<std::tuple<int, int, int>> {};

INSTANTIATE_TEST_SUITE_P(split_split_mode, outer_product,
    testing::Combine(loop_modes, testing::Range(0, 3), testing::Range(0, 3)),
    test_params_to_string<outer_product::ParamType>);

TEST_P(outer_product, pipeline) {
  int max_workers = std::get<0>(GetParam());
  int split_i = std::get<1>(GetParam());
  int split_j = std::get<2>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto a = buffer_expr::make(ctx, "a", 1, sizeof(int));
  auto b = buffer_expr::make(ctx, "b", 1, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  auto intm_a = buffer_expr::make(ctx, "intm_a", 1, sizeof(int));
  auto intm_b = buffer_expr::make(ctx, "intm_b", 1, sizeof(int));

  var i(ctx, "i");
  var j(ctx, "j");

  func add_a = func::make(add_1<int>, {{a, {point(i)}}}, {{intm_a, {i}}});
  func add_b = func::make(add_1<int>, {{b, {point(i)}}}, {{intm_b, {i}}});

  func outer = func::make(
      [](const buffer<const int>& a, const buffer<const int>& b, const buffer<int>& c) -> index_t {
        for (index_t j = c.dim(1).begin(); j < c.dim(1).end(); ++j) {
          for (index_t i = c.dim(0).begin(); i < c.dim(0).end(); ++i) {
            c(i, j) = a(i) * b(j);
          }
        }
        return 0;
      },
      {{intm_a, {point(i)}}, {intm_b, {point(j)}}}, {{out, {i, j}}});

  std::vector<func::loop_info> loops;
  if (split_i > 0) loops.emplace_back(i, split_i, max_workers);
  if (split_j > 0) loops.emplace_back(j, split_j, max_workers);
  outer.loops(loops);

  pipeline p = build_pipeline(ctx, {a, b}, {out});

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
      ASSERT_EQ(out_buf(i, j), (a_buf(i) + 1) * (b_buf(j) + 1));
    }
  }
}

TEST(unrelated, pipeline) {
  // Make the pipeline
  auto make_pipeline = []() {
    node_context ctx;

    auto in1 = buffer_expr::make(ctx, "in1", 2, sizeof(short));
    auto out1 = buffer_expr::make(ctx, "out1", 2, sizeof(short));
    auto intm1 = buffer_expr::make(ctx, "intm1", 2, sizeof(short));

    auto in2 = buffer_expr::make(ctx, "in2", 1, sizeof(int));
    auto out2 = buffer_expr::make(ctx, "out2", 1, sizeof(int));
    auto intm2 = buffer_expr::make(ctx, "intm2", 1, sizeof(int));

    // If we want to alias intermediate buffer to the output buffer,
    // we need to tell aliaser that output is unfolded and it's safe to alias.
    out2->dim(0).fold_factor = dim::unfolded;

    var x(ctx, "x");
    var y(ctx, "y");

    func add1 = func::make(add_1<short>, {{in1, {point(x), point(y)}}}, {{intm1, {x, y}}},
        call_stmt::attributes{.allow_in_place = 0x1, .name = "add1"});
    func stencil1 = func::make(sum3x3<short>, {{intm1, {bounds(-1, 1) + x, bounds(-1, 1) + y}}}, {{out1, {x, y}}});

    func mul2 = func::make(multiply_2<int>, {{in2, {point(x)}}}, {{intm2, {x}}},
        call_stmt::attributes{.allow_in_place = 0x1, .name = "mul2"});
    func add2 = func::make(
        add_1<int>, {{intm2, {point(x)}}}, {{out2, {x}}}, call_stmt::attributes{.allow_in_place = 0x1, .name = "add2"});

    stencil1.loops({{y, 2}});

    check_replica_pipeline(define_replica_pipeline(ctx, {in1, in2}, {out1, out2}));

    return build_pipeline(ctx, {in1, in2}, {out1, out2});
  };
  pipeline p = make_pipeline();
  pipeline p2 = make_pipeline();
  ASSERT_TRUE(match(nullify_calls(p.body), nullify_calls(p2.body)));

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

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in1_buf, &in2_buf};
  const raw_buffer* outputs[] = {&out1_buf, &out2_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

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

  // intm2 aliased to out2.
  // TODO: Bring back aliasing in-place calls.
  // ASSERT_THAT(eval_ctx.heap.allocs, testing::UnorderedElementsAre((W1 + 2) * 4 * sizeof(short)));
}

class padded_stencil : public testing::TestWithParam<int> {};

INSTANTIATE_TEST_SUITE_P(schedule, padded_stencil, testing::Range(0, 3));

TEST_P(padded_stencil, pipeline) {
  int schedule = GetParam();

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));

  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(short));
  auto padded_intm = buffer_expr::make(ctx, "padded_intm", 2, sizeof(short));

  var x(ctx, "x");
  var y(ctx, "y");

  func add = func::make(add_1<short>, {{in, {point(x), point(y)}}}, {{intm, {x, y}}});
  func padded = func::make_copy(
      {intm, {point(x), point(y)}, in->bounds()}, {padded_intm, {x, y}}, {buffer_expr::make<short>(ctx, "padding", 6)});
  func stencil = func::make(sum3x3<short>, {{padded_intm, {bounds(-1, 1) + x, bounds(-1, 1) + y}}}, {{out, {x, y}}});

  switch (schedule) {
  case 0: break;
  case 1:
    stencil.loops({y});
    padded.compute_root();
    break;
  case 2: stencil.loops({y}); break;
  }

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 20;
  const int H = 30;
  buffer<short, 2> in_buf({W, H});
  buffer<short, 2> out_buf({W, H});

  init_random(in_buf);
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

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

  if (schedule == 2) {
    const int intm_size = W * sizeof(short);
    const int padded_intm_size = (W + 2) * 3 * sizeof(short);
    ASSERT_THAT(eval_ctx.heap.allocs, testing::UnorderedElementsAre(intm_size, padded_intm_size));
  } else {
    ASSERT_EQ(eval_ctx.heap.allocs.size(), 1);
  }

  // Also visualize this pipeline.
  check_visualize("padded_stencil_" + std::to_string(schedule) + ".html", p, inputs, outputs, &ctx);

  if (schedule == 1) {
    check_replica_pipeline(define_replica_pipeline(ctx, {in}, {out}));
  }
}

interval_expr dilate(interval_expr x, int dx) { return {x.min - dx, x.max + dx}; }

class padded_stencil_separable : public testing::TestWithParam<std::tuple<bool, int>> {};

INSTANTIATE_TEST_SUITE_P(alias_schedule, padded_stencil_separable,
    testing::Combine(testing::Bool(), testing::Range(0, 3)),
    test_params_to_string<padded_stencil_separable::ParamType>);

TEST_P(padded_stencil_separable, pipeline) {
  bool require_dense_x = std::get<0>(GetParam());
  int split_y = std::get<1>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));

  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(short));
  auto padded_intm_t = buffer_expr::make(ctx, "padded_intm_t", 2, sizeof(short));
  auto padded_intm = buffer_expr::make(ctx, "padded_intm", 2, sizeof(short));
  auto stencil_intm = buffer_expr::make(ctx, "stencil_intm", 2, sizeof(short));

  if (require_dense_x) {
    intm->dim(0).stride = sizeof(short);
    padded_intm_t->dim(0).stride = sizeof(short);
    padded_intm->dim(0).stride = sizeof(short);
    stencil_intm->dim(0).stride = sizeof(short);
  }

  var x(ctx, "x");
  var y(ctx, "y");

  int adds = 0;
  int stencil_xs = 0;
  int stencil_ys = 0;

  func add = func::make(
      [&](const buffer<const short>& a, const buffer<short>& b) -> index_t {
        if (require_dense_x) {
          // Make sure we've respected the stride constraints, which prevent the transposes from aliasing.
          if (a.dim(0).stride() != sizeof(short) || b.dim(0).stride() != sizeof(short)) return 1;
        }
        auto result = add_1<short>(a, b);
        adds += b.elem_count();
        return result;
      },
      {{in, {point(x), point(y)}}}, {{intm, {x, y}}}, call_stmt::attributes{.name = "add"});
  // transpose so we compute the stencil in x.
  func padded_t = func::make_copy({intm, {point(x), point(y)}, in->bounds()}, {padded_intm_t, {y, x}},
      {buffer_expr::make<short>(ctx, "padding", 6)});
  func stencil_x = func::make(
      [&](const buffer<const short>& a, const buffer<short>& b) -> index_t {
        if (require_dense_x) {
          // Make sure we've respected the stride constraints, which prevent the transposes from aliasing.
          if (a.dim(0).stride() != sizeof(short) || b.dim(0).stride() != sizeof(short)) return 1;
        }
        index_t result = sum1x3<short>(a, b);
        stencil_xs += b.elem_count();
        return result;
      },
      {{padded_intm_t, {point(x), bounds(-1, 1) + y}}}, {{stencil_intm, {x, y}}},
      call_stmt::attributes{.name = "stencil_x"});
  // transpose back, compute the stencil in y.
  func padded = func::make_copy({stencil_intm, {point(y), point(x)}}, {padded_intm, {x, y}});
  func stencil_y = func::make(
      [&](const buffer<const short>& a, const buffer<short>& b) -> index_t {
        if (require_dense_x) {
          // Make sure we've respected the stride constraints, which prevent the transposes from aliasing.
          if (a.dim(0).stride() != sizeof(short) || b.dim(0).stride() != sizeof(short)) return 1;
        }
        index_t result = sum1x3<short>(a, b);
        stencil_ys += b.elem_count();
        return result;
      },
      {{padded_intm, {point(x), bounds(-1, 1) + y}}}, {{out, {x, y}}}, call_stmt::attributes{.name = "stencil_y"});

  if (split_y > 0) {
    stencil_y.loops({{y, split_y}});
  }

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 20;
  const int H = 30;
  buffer<short, 2> in_buf({W, H});
  buffer<short, 2> out_buf({W, H});

  init_random(in_buf);
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  ASSERT_EQ(0, p.evaluate(inputs, outputs, eval_ctx));

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

  ASSERT_EQ(adds, W * H);
  ASSERT_EQ(stencil_xs, W * (H + 2));
  ASSERT_EQ(stencil_ys, W * H);

  if (split_y == 1) {
    const index_t intm_size = W * split_y * sizeof(short);
    const index_t padded_intm_t_size = (W + 2) * split_y * sizeof(short);
    const index_t stencil_intm_size = W * split_y * sizeof(short);
    const index_t padded_intm_size = W * (split_y + 2) * sizeof(short);

    if (!require_dense_x) {
      // We can't alias stencil_intm and padded_intm like we can without splitting because of fold factor constraints.
      ASSERT_THAT(eval_ctx.heap.allocs,
          testing::UnorderedElementsAre(std::max(intm_size, padded_intm_t_size), stencil_intm_size, padded_intm_size));
    } else {
      // We can't alias anything when we require the strides to be dense.
      ASSERT_THAT(eval_ctx.heap.allocs,
          testing::UnorderedElementsAre(intm_size, padded_intm_t_size, stencil_intm_size, padded_intm_size));
    }
  } else if (split_y == 0) {
    const index_t intm_size = W * H * sizeof(short);
    const index_t padded_intm_t_size = (W + 2) * (H + 2) * sizeof(short);
    const index_t stencil_intm_size = W * (H + 2) * sizeof(short);
    const index_t padded_intm_size = W * (H + 2) * sizeof(short);

    if (!require_dense_x) {
      ASSERT_THAT(eval_ctx.heap.allocs, testing::UnorderedElementsAre(std::max(intm_size, padded_intm_t_size),
                                            std::max(stencil_intm_size, padded_intm_size)));
    } else {
      // We can't alias anything when we require the strides to be dense.
      ASSERT_THAT(eval_ctx.heap.allocs,
          testing::UnorderedElementsAre(intm_size, padded_intm_t_size, stencil_intm_size, padded_intm_size));
    }
  } else if (split_y == 2) {
    // TODO(vksnk): aliasing is not happening with split_y == 2, because of the misagligned mins of the folded buffers.
  }
}

TEST(constant, pipeline) {
  // Make the pipeline
  node_context ctx;

  const int W = 20;
  const int H = 10;

  slinky::dim dims[2];
  dims[0].set_bounds(0, W);
  dims[0].set_stride(1 * sizeof(short));
  dims[1].set_bounds(0, H);
  dims[1].set_stride(W * sizeof(short));

  auto constant_buf = raw_buffer::make(2, sizeof(short), dims);
  fill_random(constant_buf->cast<short>());

  auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));

  auto constant = buffer_expr::make(ctx, "constant", std::move(constant_buf));

  var x(ctx, "x");
  var y(ctx, "y");

  func add = func::make(add_1<short>, {{constant, {point(x), point(y)}}}, {{out, {x, y}}});

  pipeline p = build_pipeline(ctx, {}, {out});

  // Run the pipeline.
  buffer<short, 2> out_buf({W, H});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate({}, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y), *reinterpret_cast<short*>(constant->constant()->address_at(x, y)) + 1);
    }
  }
}

class parallel_stencils : public testing::TestWithParam<int> {};

INSTANTIATE_TEST_SUITE_P(schedule, parallel_stencils, testing::Range(0, 5));

TEST_P(parallel_stencils, pipeline) {
  int schedule = GetParam();

  // Make the pipeline
  node_context ctx;

  auto in1 = buffer_expr::make(ctx, "in1", 2, sizeof(short));
  auto in2 = buffer_expr::make(ctx, "in2", 2, sizeof(short));
  auto intm1 = buffer_expr::make(ctx, "intm1", 2, sizeof(short));
  auto intm2 = buffer_expr::make(ctx, "intm2", 2, sizeof(short));
  auto intm3 = buffer_expr::make(ctx, "intm3", 2, sizeof(short));
  auto intm4 = buffer_expr::make(ctx, "intm4", 2, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));

  var x(ctx, "x");
  var y(ctx, "y");

  func add1 =
      func::make(add_1<short>, {{in1, {point(x), point(y)}}}, {{intm1, {x, y}}}, call_stmt::attributes{.name = "add1"});
  func mul2 = func::make(
      multiply_2<short>, {{in2, {point(x), point(y)}}}, {{intm2, {x, y}}}, call_stmt::attributes{.name = "mul2"});
  func stencil1 = func::make(sum3x3<short>, {{intm1, {bounds(-1, 1) + x, bounds(-1, 1) + y}}}, {{intm3, {x, y}}},
      call_stmt::attributes{.name = "sum3x3"});
  func stencil2 = func::make(sum5x5<short>, {{intm2, {bounds(-2, 2) + x, bounds(-2, 2) + y}}}, {{intm4, {x, y}}},
      call_stmt::attributes{.name = "sum5x5"});
  func diff = func::make(subtract<short>, {{intm3, {point(x), point(y)}}, {intm4, {point(x), point(y)}}},
      {{out, {x, y}}}, call_stmt::attributes{.name = "subtract"});

  if (schedule == 0) {
    diff.loops({{y, 1}});
  } else if (schedule == 1) {
    diff.loops({{y, 2}});
    stencil1.loops({{y, 1}});
    stencil2.loops({{y, 2}});
    add1.compute_root();
    mul2.compute_at({&diff, y});
  } else if (schedule == 2) {
    diff.loops({{y, 2}});
    stencil1.loops({{y, 2}});
    stencil2.loops({{y, 2}});
  } else if (schedule == 3) {
    diff.loops({{y, 1, loop::parallel}});
  } else if (schedule == 4) {
    diff.loops({{y, 1234567, loop::parallel}});
  }

  pipeline p = build_pipeline(ctx, {in1, in2}, {out});

  // Run the pipeline.
  const int W = 20;
  const int H = 30;
  buffer<short, 2> in1_buf({W + 2, H + 2});
  buffer<short, 2> in2_buf({W + 4, H + 4});
  in1_buf.translate(-1, -1);
  in2_buf.translate(-2, -2);
  buffer<short, 2> out_buf({W, H});

  init_random(in1_buf);
  init_random(in2_buf);
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in1_buf, &in2_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  // Run the pipeline stages manually to get the reference result.
  buffer<short, 2> ref_intm1({W + 2, H + 2});
  buffer<short, 2> ref_intm2({W + 4, H + 4});
  buffer<short, 2> ref_intm3({W, H});
  buffer<short, 2> ref_intm4({W, H});
  buffer<short, 2> ref_out({W, H});
  ref_intm1.translate(-1, -1);
  ref_intm2.translate(-2, -2);
  ref_intm1.allocate();
  ref_intm2.allocate();
  ref_intm3.allocate();
  ref_intm4.allocate();
  ref_out.allocate();

  add_1<short>(in1_buf.cast<const short>(), ref_intm1.cast<short>());
  multiply_2<short>(in2_buf.cast<const short>(), ref_intm2.cast<short>());
  sum3x3<short>(ref_intm1.cast<const short>(), ref_intm3.cast<short>());
  sum5x5<short>(ref_intm2.cast<const short>(), ref_intm4.cast<short>());
  subtract<short>(ref_intm3.cast<const short>(), ref_intm4.cast<const short>(), ref_out.cast<short>());

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(ref_out(x, y), out_buf(x, y));
    }
  }

  // Also visualize this pipeline
  if (schedule < 3) {
    check_visualize("parallel_stencils_" + std::to_string(schedule) + ".html", p, inputs, outputs, &ctx);
  }
}

class diamond_stencils : public testing::TestWithParam<int> {};

INSTANTIATE_TEST_SUITE_P(schedule, diamond_stencils, testing::Range(0, 5));

TEST_P(diamond_stencils, pipeline) {
  int schedule = GetParam();

  auto make_pipeline = [schedule]() {
    node_context ctx;

    auto in = buffer_expr::make(ctx, "in1", 2, sizeof(short));
    auto intm2 = buffer_expr::make(ctx, "intm2", 2, sizeof(short));
    auto intm3 = buffer_expr::make(ctx, "intm3", 2, sizeof(short));
    auto intm4 = buffer_expr::make(ctx, "intm4", 2, sizeof(short));
    auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));

    var x(ctx, "x");
    var y(ctx, "y");

    func mul2 = func::make(multiply_2<short>, {{in, {point(x), point(y)}}}, {{intm2, {x, y}}});
    func stencil1 = func::make(sum3x3<short>, {{intm2, {bounds(-1, 1) + x, bounds(-1, 1) + y}}}, {{intm3, {x, y}}});
    func stencil2 = func::make(sum5x5<short>, {{intm2, {bounds(-2, 2) + x, bounds(-2, 2) + y}}}, {{intm4, {x, y}}});
    func diff =
        func::make(subtract<short>, {{intm3, {point(x), point(y)}}, {intm4, {point(x), point(y)}}}, {{out, {x, y}}});

    if (schedule == 1) {
      diff.loops({{y, 1}});
    } else if (schedule == 2) {
      diff.loops({{y, 1}});
      stencil1.loops({{y, 2}});
      stencil2.loops({{y, 2}});
    } else if (schedule == 3) {
      diff.loops({{y, 1}});
      stencil1.loops({{y, 2}});
      stencil2.loops({{y, 2}});
      mul2.compute_root();
    } else if (schedule == 4) {
      diff.loops({{y, 1, loop::parallel}});
    }

    return build_pipeline(ctx, {in}, {out});
  };
  pipeline p = make_pipeline();
  pipeline p2 = make_pipeline();
  ASSERT_TRUE(match(nullify_calls(p.body), nullify_calls(p2.body)));

  // Run the pipeline.
  const int W = 20;
  const int H = 10;
  buffer<short, 2> in_buf({W + 4, H + 4});
  in_buf.translate(-2, -2);
  buffer<short, 2> out_buf({W, H});

  init_random(in_buf);
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  // Run the pipeline stages manually to get the reference result.
  buffer<short, 2> ref_intm2({W + 4, H + 4});
  buffer<short, 2> ref_intm3({W, H});
  buffer<short, 2> ref_intm4({W, H});
  buffer<short, 2> ref_out({W, H});
  ref_intm2.translate(-2, -2);
  ref_intm2.allocate();
  ref_intm3.allocate();
  ref_intm4.allocate();
  ref_out.allocate();

  multiply_2<short>(in_buf.cast<const short>(), ref_intm2.cast<short>());
  sum3x3<short>(ref_intm2.cast<const short>(), ref_intm3.cast<short>());
  sum5x5<short>(ref_intm2.cast<const short>(), ref_intm4.cast<short>());
  subtract<short>(ref_intm3.cast<const short>(), ref_intm4.cast<const short>(), ref_out.cast<short>());

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(ref_out(x, y), out_buf(x, y));
    }
  }
}

TEST(fork, pipeline) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in1", 2, sizeof(short));
  auto intm2 = buffer_expr::make(ctx, "intm2", 2, sizeof(short));
  auto out1 = buffer_expr::make(ctx, "out1", 2, sizeof(short));
  auto out2 = buffer_expr::make(ctx, "out2", 2, sizeof(short));

  var x(ctx, "x");
  var y(ctx, "y");

  func mul2 = func::make(multiply_2<short>, {{in, {point(x), point(y)}}}, {{intm2, {x, y}}});
  func add1 = func::make(add_1<short>, {{intm2, {point(x), point(y)}}}, {{out1, {x, y}}});
  func add2 = func::make(add_1<short>, {{intm2, {point(x), point(y)}}}, {{out2, {x, y}}});

  add2.loops({{y, 1}});

  pipeline p = build_pipeline(ctx, {in}, {out1, out2});

  // Run the pipeline.
  const int W = 32;
  const int H = 32;
  buffer<short, 2> in_buf({W, H});
  buffer<short, 2> out1_buf({W, H});
  buffer<short, 2> out2_buf({W, H});

  init_random(in_buf);
  out1_buf.allocate();
  out2_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out1_buf, &out2_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out1_buf(x, y), in_buf(x, y) * 2 + 1);
      ASSERT_EQ(out2_buf(x, y), in_buf(x, y) * 2 + 1);
    }
  }

  check_replica_pipeline(define_replica_pipeline(ctx, {in}, {out1, out2}));
}

TEST(split, pipeline) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(short));
  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(short));
  auto intm1 = buffer_expr::make(ctx, "intm1", 2, sizeof(short));
  auto intm2 = buffer_expr::make(ctx, "intm2", 2, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));

  const int W = 16;
  const int H = 32;

  var x(ctx, "x");
  var y(ctx, "y");

  // This test is designed to check that if we alias an elementwise output to an input, that we use the memory layout of
  // the input, but the bounds of the output.
  func add_in = func::make(add_1<short>, {{in, {point(x), point(y)}}}, {{intm, {x, y}}});
  func add1 = func::make(
      add_1<short>, {{intm, {point(x), point(y)}}}, {{intm1, {x, y}}}, call_stmt::attributes{.allow_in_place = 0x1});
  func add2 = func::make(
      [](const buffer<const short>& in, const buffer<short>& out) -> index_t {
        for (index_t y = out.dim(1).min(); y <= out.dim(1).max(); ++y) {
          for (index_t x = out.dim(0).min(); x <= out.dim(0).max(); ++x) {
            out(x, y) = in(x + W, y) + 1;
          }
        }
        return 0;
      },
      {{intm, {point(x + W), point(y)}}}, {{intm2, {x, y}}}, call_stmt::attributes{.allow_in_place = 0x1});

  func sum_out = func::make(subtract<short>, {{intm1, {point(x), point(y)}}, {intm2, {point(x), point(y)}}},
      {{out, {x, y}}}, call_stmt::attributes{.allow_in_place = 0x2});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  buffer<short, 2> in_buf({W * 2, H});
  buffer<short, 2> out_buf({W, H});

  init_random(in_buf);
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y), in_buf(x, y) - in_buf(x + W, y));
    }
  }
}

class upsample : public testing::TestWithParam<std::tuple<int, int>> {};

INSTANTIATE_TEST_SUITE_P(split_mode, upsample, testing::Combine(loop_modes, testing::Range(0, 2)),
    test_params_to_string<upsample::ParamType>);

TEST_P(upsample, pipeline) {
  int max_workers = std::get<0>(GetParam());
  int split = std::get<1>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));

  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(short));

  var x(ctx, "x");
  var y(ctx, "y");

  func add = func::make(add_1<short>, {{in, {point(x), point(x)}}}, {{intm, {x, y}}});
  func upsample = func::make(upsample_nn_2x<short>, {{intm, {point(x) / 2, point(y) / 2}}}, {{out, {x, y}}});

  if (split > 0) {
    upsample.loops({{y, split, max_workers}});
  }

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 20;
  const int H = 30;
  buffer<short, 2> in_buf({W / 2, H / 2});
  buffer<short, 2> out_buf({W, H});

  init_random(in_buf);
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      int correct = in_buf(x / 2, y / 2) + 1;
      ASSERT_EQ(correct, out_buf(x, y)) << x << " " << y;
    }
  }

  if (split > 0 && max_workers == loop::serial) {
    const int intm_size = W / 2 * sizeof(short);
    ASSERT_THAT(eval_ctx.heap.allocs, testing::UnorderedElementsAre(intm_size));
  } else {
    ASSERT_EQ(eval_ctx.heap.allocs.size(), 1);
  }
}

}  // namespace slinky
