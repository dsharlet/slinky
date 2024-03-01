#include <gtest/gtest.h>

#include <cassert>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>

#include "builder/pipeline.h"
#include "builder/replica_pipeline.h"
#include "runtime/expr.h"
#include "runtime/pipeline.h"
#include "runtime/thread_pool.h"
#include "tools/cpp/runfiles/runfiles.h"

#if 1
#define LOG_REPLICA_TEXT(STR)                                                                                          \
  do {                                                                                                                 \
  } while (0)
#else
#define LOG_REPLICA_TEXT(STR)                                                                                          \
  do {                                                                                                                 \
    std::cerr << "REPLICA_TEXT:\n" << (STR) << "\n";                                                                   \
  } while (0)
#endif

namespace slinky {

using bazel::tools::cpp::runfiles::Runfiles;

std::string read_entire_file(const std::string& pathname) {
  std::ifstream f(pathname, std::ios::in | std::ios::binary);
  std::string result;

  f.seekg(0, std::ifstream::end);
  size_t size = f.tellg();
  result.resize(size);
  f.seekg(0, std::ifstream::beg);
  f.read(result.data(), result.size());
  if (!f.good()) {
    std::cerr << "Unable to read file: " << pathname;
    std::abort();
  }
  f.close();
  return result;
}

class ReplicaPipelineTest : public testing::Test {
protected:
  void SetUp() override {
    // TODO: for testing purposes, remove
    // replica_pipeline_test_src = read_entire_file("/Users/srj/GitHub/slinky/builder/replica_pipeline_test.cc");
    // return;

    std::string error;
    runfiles.reset(Runfiles::CreateForTest(BAZEL_CURRENT_REPOSITORY, &error));
    if (runfiles == nullptr) {
      std::cerr << "Can't find runfile directory: " << error;
      std::abort();
    }

    // As of Bazel 6.x, apparently `_main` is the toplevel for runfiles
    // (https://github.com/bazelbuild/bazel/issues/18128)
    auto full_path = runfiles->Rlocation("_main/builder/replica_pipeline_test.cc");
    replica_pipeline_test_src = read_entire_file(full_path);
  }

  std::unique_ptr<Runfiles> runfiles;
  std::string replica_pipeline_test_src;
};

thread_pool threads;

class test_context : public eval_context {
public:
  test_context() {
    enqueue_many = [&](const thread_pool::task& t) { threads.enqueue(threads.thread_count(), t); };
    enqueue_one = [&](thread_pool::task t) { threads.enqueue(std::move(t)); };
    wait_for = [&](std::function<bool()> condition) { return threads.wait_for(std::move(condition)); };
  }
};

std::mt19937& rng() {
  static std::mt19937 r{static_cast<uint32_t>(time(nullptr))};
  return r;
}

template <typename T>
T random() {
  union {
    T value;
    char bytes[sizeof(T)];
  } u;
  for (int i = 0; i < sizeof(T); i++) {
    u.bytes[i] = static_cast<char>(rng()());
  }
  return u.value;
}

template <typename T, std::size_t N>
void init_random(buffer<T, N>& buf) {
  buf.allocate();
  std::size_t flat_size = buf.size_bytes() / sizeof(T);
  for (std::size_t i = 0; i < flat_size; ++i) {
    buf.base()[i] = random<T>();
  }
}

// clang-format off
static std::function<pipeline()> kMultipleOutputsReplica =
// BEGIN define_replica_pipeline() output
[]() -> ::slinky::pipeline {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", sizeof(uint32_t), 3);
  auto sum_x = buffer_expr::make(ctx, "sum_x", sizeof(uint32_t), 2);
  auto y = var(ctx, "y");
  auto z = var(ctx, "z");
  auto _1 = variable::make(in->sym());
  auto _2 = buffer_min(_1, 0);
  auto _3 = buffer_max(_1, 0);
  auto _4 = buffer_min(_1, 1);
  auto _5 = buffer_max(_1, 1);
  auto _replica_fn_6 = [=](const buffer<const void>& i0, const buffer<void>& o0, const buffer<void>& o1) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0, &o1};
    const func::input fins[] = {{in, {{_2, _3}, {_4, _5}, point(z)}}};
    const std::vector<var> fout_dims[] = {{y, z}, {z}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _7 = buffer_min(_1, 0);
  auto _8 = buffer_max(_1, 0);
  auto _9 = buffer_min(_1, 1);
  auto _10 = buffer_max(_1, 1);
  auto sum_xy = buffer_expr::make(ctx, "sum_xy", sizeof(uint32_t), 1);
  auto _fn_0 = func::make(std::move(_replica_fn_6), {{in, {{_7, _8}, {_9, _10}, point(z)}}}, {{sum_x, {y, z}}, {sum_xy, {z}}});
  _fn_0.loops({{z, 1, loop_mode::serial}});
  auto p = build_pipeline(ctx, {}, {in}, {sum_x, sum_xy}, {});
  return p;
}
// END define_replica_pipeline() output
;
// clang-format on

TEST_F(ReplicaPipelineTest, multiple_outputs) {
  constexpr int split = 1;
  constexpr loop_mode lm = loop_mode::serial;
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
  func::callable<const int, int, int> sum_x_xy = [](const buffer<const int>& in, const buffer<int>& sum_x,
                                                     const buffer<int>& sum_xy) -> index_t {
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
  func sums = func::make(std::move(sum_x_xy), {{in, {X, Y, point(z)}}}, {{sum_x, {y, z}}, {sum_xy, {z}}});

  if (split > 0) {
    sums.loops({{z, split, lm}});
  }

  pipeline p = build_pipeline(ctx, {in}, {sum_x, sum_xy});
  pipeline p_replica = kMultipleOutputsReplica();

  // Look at the source code to this test to verify that we
  // we have something that matches exactly
  std::string replica_text = define_replica_pipeline(ctx, {in}, {sum_x, sum_xy});
  LOG_REPLICA_TEXT(replica_text);
  size_t pos = replica_pipeline_test_src.find(replica_text);
  ASSERT_NE(pos, std::string::npos) << "Matching replica text not found, expected:\n" << replica_text;

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
  {
    test_context eval_ctx;
    p.evaluate(inputs, outputs, eval_ctx);
  }
  {
    test_context eval_ctx;
    p_replica.evaluate(inputs, outputs, eval_ctx);
  }
}

// clang-format off
static std::function<pipeline()> kMatmulReplica =
// BEGIN define_replica_pipeline() output
[]() -> ::slinky::pipeline {
  node_context ctx;
  auto a = buffer_expr::make(ctx, "a", sizeof(uint32_t), 2);
  auto b = buffer_expr::make(ctx, "b", sizeof(uint32_t), 2);
  auto c = buffer_expr::make(ctx, "c", sizeof(uint32_t), 2);
  auto abc = buffer_expr::make(ctx, "abc", sizeof(uint32_t), 2);
  auto i = var(ctx, "i");
  auto j = var(ctx, "j");
  auto ab = buffer_expr::make(ctx, "ab", sizeof(uint32_t), 2);
  auto _2 = variable::make(a->sym());
  auto _3 = buffer_min(_2, 1);
  auto _4 = buffer_max(_2, 1);
  auto _5 = buffer_min(_2, 1);
  auto _6 = buffer_max(_2, 1);
  auto _replica_fn_7 = [=](const buffer<const void>& i0, const buffer<const void>& i1, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0, &i1};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{a, {point(i), {_3, _4}}}, {b, {{_5, _6}, point(j)}}};
    const std::vector<var> fout_dims[] = {{i, j}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _8 = buffer_min(_2, 1);
  auto _9 = buffer_max(_2, 1);
  auto _10 = buffer_min(_2, 1);
  auto _11 = buffer_max(_2, 1);
  auto _fn_1 = func::make(std::move(_replica_fn_7), {{a, {point(i), {_8, _9}}}, {b, {{_10, _11}, point(j)}}}, {{ab, {i, j}}});
  auto _12 = variable::make(c->sym());
  auto _13 = buffer_min(_12, 0);
  auto _14 = buffer_max(_12, 0);
  auto _15 = buffer_min(_12, 0);
  auto _16 = buffer_max(_12, 0);
  auto _replica_fn_17 = [=](const buffer<const void>& i0, const buffer<const void>& i1, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0, &i1};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{ab, {point(i), {_13, _14}}}, {c, {{_15, _16}, point(j)}}};
    const std::vector<var> fout_dims[] = {{i, j}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _18 = buffer_min(_12, 0);
  auto _19 = buffer_max(_12, 0);
  auto _20 = buffer_min(_12, 0);
  auto _21 = buffer_max(_12, 0);
  auto _fn_0 = func::make(std::move(_replica_fn_17), {{ab, {point(i), {_18, _19}}}, {c, {{_20, _21}, point(j)}}}, {{abc, {i, j}}});
  _fn_0.loops({{i, 1, loop_mode::serial}});
  auto p = build_pipeline(ctx, {}, {a, b, c}, {abc}, {});
  return p;
}
// END define_replica_pipeline() output
;
// clang-format on

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

// Two matrix multiplies: D = (A x B) x C.
TEST_F(ReplicaPipelineTest, matmul) {
  constexpr int split = 1;
  constexpr loop_mode lm = loop_mode::serial;

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
  func matmul_ab = func::make(matmul<int>, {{a, {point(i), K_ab}}, {b, {K_ab, point(j)}}}, {{ab, {i, j}}});
  func matmul_abc = func::make(matmul<int>, {{ab, {point(i), K_abc}}, {c, {K_abc, point(j)}}}, {{abc, {i, j}}});

  a->dim(1).stride = a->elem_size();
  b->dim(1).stride = b->elem_size();
  c->dim(1).stride = c->elem_size();
  abc->dim(1).stride = abc->elem_size();

  // TODO: There should be a more user friendly way to control the strides.
  ab->dim(1).stride = static_cast<index_t>(sizeof(int));
  ab->dim(0).stride = ab->dim(1).extent() * ab->dim(1).stride;

  if (split > 0) {
    matmul_abc.loops({{i, split, lm}});

    if (lm == loop_mode::parallel) {
      ab->store_at({&matmul_abc, i});
    }
  }

  pipeline p = build_pipeline(ctx, {a, b, c}, {abc});
  pipeline p_replica = kMatmulReplica();

  // Look at the source code to this test to verify that we
  // we have something that matches exactly
  std::string replica_text = define_replica_pipeline(ctx, {a, b, c}, {abc});
  LOG_REPLICA_TEXT(replica_text);
  size_t pos = replica_pipeline_test_src.find(replica_text);
  ASSERT_NE(pos, std::string::npos) << "Matching replica text not found, expected:\n" << replica_text;

  // Run the pipeline
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

  const raw_buffer* inputs[] = {&a_buf, &b_buf, &c_buf};
  const raw_buffer* outputs[] = {&abc_buf};
  {
    test_context eval_ctx;
    p.evaluate(inputs, outputs, eval_ctx);
  }
  {
    test_context eval_ctx;
    p_replica.evaluate(inputs, outputs, eval_ctx);
  }
}

// clang-format off
static std::function<pipeline()> kPyramidReplica =
// BEGIN define_replica_pipeline() output
[]() -> ::slinky::pipeline {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", sizeof(uint32_t), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(uint32_t), 2);
  auto x = var(ctx, "x");
  auto y = var(ctx, "y");
  auto intm = buffer_expr::make(ctx, "intm", sizeof(uint32_t), 2);
  auto _2 = x * 2;
  auto _3 = _2 + 0;
  auto _4 = x * 2;
  auto _5 = _4 + 1;
  auto _6 = y * 2;
  auto _7 = _6 + 0;
  auto _8 = y * 2;
  auto _9 = _8 + 1;
  auto _replica_fn_10 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{in, {{_3, _5}, {_7, _9}}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _11 = x * 2;
  auto _12 = _11 + 0;
  auto _13 = x * 2;
  auto _14 = _13 + 1;
  auto _15 = y * 2;
  auto _16 = _15 + 0;
  auto _17 = y * 2;
  auto _18 = _17 + 1;
  auto _fn_1 = func::make(std::move(_replica_fn_10), {{in, {{_12, _14}, {_16, _18}}}}, {{intm, {x, y}}});
  auto _19 = x / 2;
  auto _20 = x + 1;
  auto _21 = _20 / 2;
  auto _22 = y / 2;
  auto _23 = y + 1;
  auto _24 = _23 / 2;
  auto _replica_fn_25 = [=](const buffer<const void>& i0, const buffer<const void>& i1, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0, &i1};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{in, {point(x), point(y)}}, {intm, {{_19, _21}, {_22, _24}}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _26 = x / 2;
  auto _27 = x + 1;
  auto _28 = _27 / 2;
  auto _29 = y / 2;
  auto _30 = y + 1;
  auto _31 = _30 / 2;
  auto _fn_0 = func::make(std::move(_replica_fn_25), {{in, {point(x), point(y)}}, {intm, {{_26, _28}, {_29, _31}}}}, {{out, {x, y}}});
  _fn_0.loops({{y, 1, loop_mode::serial}});
  auto p = build_pipeline(ctx, {}, {in}, {out}, {});
  return p;
}
// END define_replica_pipeline() output
;
// clang-format on

index_t pyramid_upsample2x(const buffer<const int>& skip, const buffer<const int>& in, const buffer<int>& out) {
  for (index_t y = out.dim(1).begin(); y < out.dim(1).end(); ++y) {
    for (index_t x = out.dim(0).begin(); x < out.dim(0).end(); ++x) {
      out(x, y) = in((x + 0) >> 1, (y + 0) >> 1) + in((x + 1) >> 1, (y + 0) >> 1) + in((x + 0) >> 1, (y + 1) >> 1) +
                  in((x + 1) >> 1, (y + 1) >> 1) + skip(x, y);
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

TEST_F(ReplicaPipelineTest, pyramid) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 2);

  auto intm = buffer_expr::make(ctx, "intm", sizeof(int), 2);

  var x(ctx, "x");
  var y(ctx, "y");

  func downsample = func::make(downsample2x, {{in, {2 * x + bounds(0, 1), 2 * y + bounds(0, 1)}}}, {{intm, {x, y}}});
  func upsample = func::make(pyramid_upsample2x,
      {{in, {point(x), point(y)}}, {intm, {bounds(x, x + 1) / 2, bounds(y, y + 1) / 2}}}, {{out, {x, y}}});

  upsample.loops({{y, 1}});

  pipeline p = build_pipeline(ctx, {in}, {out});
  pipeline p_replica = kPyramidReplica();

  // Look at the source code to this test to verify that we
  // we have something that matches exactly
  std::string replica_text = define_replica_pipeline(ctx, {in}, {out});
  LOG_REPLICA_TEXT(replica_text);
  size_t pos = replica_pipeline_test_src.find(replica_text);
  ASSERT_NE(pos, std::string::npos) << "Matching replica text not found, expected:\n" << replica_text;

  // Run the pipeline.
  const int W = 8;
  const int H = 8;
  buffer<int, 2> in_buf({W + 4, H + 4});
  in_buf.translate(-2, -2);
  init_random(in_buf);

  buffer<int, 2> out_buf({W, H});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  {
    test_context eval_ctx;
    p.evaluate(inputs, outputs, eval_ctx);
  }
  {
    test_context eval_ctx;
    p_replica.evaluate(inputs, outputs, eval_ctx);
  }
}

template <typename T>
index_t add_1(const buffer<const T>& in, const buffer<T>& out) {
  assert(in.rank == out.rank);
  for_each_index(out, [&](auto i) { out(i) = in(i) + 1; });
  return 0;
}

template <typename T>
index_t subtract(const buffer<const T>& a, const buffer<const T>& b, const buffer<T>& out) {
  assert(a.rank == out.rank);
  assert(b.rank == out.rank);
  for_each_index(out, [&](auto i) { out(i) = a(i) - b(i); });
  return 0;
}

// A 2D stencil, sums [x + dx0, x + dx1] x [y + dy0, y + dy]
template <typename T, int dx0, int dy0, int dx1, int dy1>
index_t sum_stencil(const buffer<const T>& in, const buffer<T>& out) {
  assert(in.rank == 2);
  assert(out.rank == 2);
  for (index_t y = out.dim(1).begin(); y < out.dim(1).end(); ++y) {
    for (index_t x = out.dim(0).begin(); x < out.dim(0).end(); ++x) {
      T sum = 0;
      for (index_t dy = dy0; dy <= dy1; ++dy) {
        for (index_t dx = dx0; dx <= dx1; ++dx) {
          sum += in(x + dx, y + dy);
        }
      }
      out(x, y) = sum;
    }
  }
  return 0;
}

// A centered 2D 3x3 stencil operation.
template <typename T>
index_t sum3x3(const buffer<const T>& in, const buffer<T>& out) {
  return sum_stencil<T, -1, -1, 1, 1>(in, out);
}

// A centered 2D 5x5 stencil operation.
template <typename T>
index_t sum5x5(const buffer<const T>& in, const buffer<T>& out) {
  return sum_stencil<T, -2, -2, 2, 2>(in, out);
}

template <typename T>
index_t multiply_2(const buffer<const T>& in, const buffer<T>& out) {
  assert(in.rank == out.rank);
  for_each_index(out, [&](auto i) { out(i) = in(i) * 2; });
  return 0;
}

// clang-format off
static std::function<pipeline()> kUnrelatedReplica =
// BEGIN define_replica_pipeline() output
[]() -> ::slinky::pipeline {
  node_context ctx;
  auto in1 = buffer_expr::make(ctx, "in1", sizeof(uint16_t), 2);
  auto in2 = buffer_expr::make(ctx, "in2", sizeof(uint32_t), 1);
  auto out1 = buffer_expr::make(ctx, "out1", sizeof(uint16_t), 2);
  auto x = var(ctx, "x");
  auto y = var(ctx, "y");
  auto intm1 = buffer_expr::make(ctx, "intm1", sizeof(uint16_t), 2);
  auto _replica_fn_2 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{in1, {point(x), point(y)}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_1 = func::make(std::move(_replica_fn_2), {{in1, {point(x), point(y)}}}, {{intm1, {x, y}}});
  auto _3 = x + -1;
  auto _4 = x + 1;
  auto _5 = y + -1;
  auto _6 = y + 1;
  auto _replica_fn_7 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{intm1, {{_3, _4}, {_5, _6}}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _8 = x + -1;
  auto _9 = x + 1;
  auto _10 = y + -1;
  auto _11 = y + 1;
  auto _fn_0 = func::make(std::move(_replica_fn_7), {{intm1, {{_8, _9}, {_10, _11}}}}, {{out1, {x, y}}});
  _fn_0.loops({{y, 2, loop_mode::serial}});
  auto out2 = buffer_expr::make(ctx, "out2", sizeof(uint32_t), 1);
  auto intm2 = buffer_expr::make(ctx, "intm2", sizeof(uint32_t), 1);
  auto _replica_fn_14 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{in2, {point(x)}}};
    const std::vector<var> fout_dims[] = {{x}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_13 = func::make(std::move(_replica_fn_14), {{in2, {point(x)}}}, {{intm2, {x}}});
  auto _replica_fn_15 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{intm2, {point(x)}}};
    const std::vector<var> fout_dims[] = {{x}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_12 = func::make(std::move(_replica_fn_15), {{intm2, {point(x)}}}, {{out2, {x}}});
  auto p = build_pipeline(ctx, {}, {in1, in2}, {out1, out2}, {});
  return p;
}
// END define_replica_pipeline() output
;
// clang-format on

TEST_F(ReplicaPipelineTest, unrelated) {
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

  func add1 = func::make(add_1<short>, {{in1, {point(x), point(y)}}}, {{intm1, {x, y}}},
      call_stmt::callable_attrs{.allow_in_place = true});
  func stencil1 = func::make(sum3x3<short>, {{intm1, {bounds(-1, 1) + x, bounds(-1, 1) + y}}}, {{out1, {x, y}}});

  func mul2 = func::make(
      multiply_2<int>, {{in2, {point(x)}}}, {{intm2, {x}}}, call_stmt::callable_attrs{.allow_in_place = true});
  func add2 =
      func::make(add_1<int>, {{intm2, {point(x)}}}, {{out2, {x}}}, call_stmt::callable_attrs{.allow_in_place = true});

  stencil1.loops({{y, 2}});

  pipeline p = build_pipeline(ctx, {in1, in2}, {out1, out2});
  pipeline p_replica = kUnrelatedReplica();

  // Look at the source code to this test to verify that we
  // we have something that matches exactly
  std::string replica_text = define_replica_pipeline(ctx, {in1, in2}, {out1, out2});
  LOG_REPLICA_TEXT(replica_text);
  size_t pos = replica_pipeline_test_src.find(replica_text);
  ASSERT_NE(pos, std::string::npos) << "Matching replica text not found, expected:\n" << replica_text;

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
  {
    test_context eval_ctx;
    p.evaluate(inputs, outputs, eval_ctx);
  }
  {
    test_context eval_ctx;
    p_replica.evaluate(inputs, outputs, eval_ctx);
  }
}

// clang-format off
static std::function<pipeline()> kConcatenatedReplica =
// BEGIN define_replica_pipeline() output
[]() -> ::slinky::pipeline {
  node_context ctx;
  auto in1 = buffer_expr::make(ctx, "in1", sizeof(uint16_t), 2);
  auto in2 = buffer_expr::make(ctx, "in2", sizeof(uint16_t), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(uint16_t), 2);
  auto intm1 = buffer_expr::make(ctx, "intm1", sizeof(uint16_t), 2);
  auto x = var(ctx, "x");
  auto y = var(ctx, "y");
  auto _replica_fn_2 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{in1, {point(x), point(y)}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_1 = func::make(std::move(_replica_fn_2), {{in1, {point(x), point(y)}}}, {{intm1, {x, y}}});
  auto _3 = y - 0;
  auto _4 = y - 0;
  auto _5 = variable::make(in1->sym());
  auto _6 = buffer_max(_5, 1);
  auto _7 = buffer_min(_5, 1);
  auto _8 = _6 - _7;
  auto _9 = _8 + 1;
  auto _10 = _9 - 1;
  auto intm2 = buffer_expr::make(ctx, "intm2", sizeof(uint16_t), 2);
  auto _replica_fn_12 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{in2, {point(x), point(y)}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_11 = func::make(std::move(_replica_fn_12), {{in2, {point(x), point(y)}}}, {{intm2, {x, y}}});
  auto _13 = buffer_max(_5, 1);
  auto _14 = buffer_min(_5, 1);
  auto _15 = _13 - _14;
  auto _16 = _15 + 1;
  auto _17 = y - _16;
  auto _18 = buffer_max(_5, 1);
  auto _19 = buffer_min(_5, 1);
  auto _20 = _18 - _19;
  auto _21 = _20 + 1;
  auto _22 = y - _21;
  auto _23 = buffer_max(_5, 1);
  auto _24 = buffer_min(_5, 1);
  auto _25 = _23 - _24;
  auto _26 = _25 + 1;
  auto _27 = variable::make(out->sym());
  auto _28 = buffer_max(_27, 1);
  auto _29 = buffer_min(_27, 1);
  auto _30 = _28 - _29;
  auto _31 = _30 + 1;
  auto _32 = _31 - 1;
  auto _fn_0 = func::make_copy({{intm1, {point(x), {_3, _4}}, {point(expr()), {0, _10}}, {}}, {intm2, {point(x), {_17, _22}}, {point(expr()), {_26, _32}}, {}}}, {out, {x, y}});
  auto p = build_pipeline(ctx, {}, {in1, in2}, {out}, {.no_alias_buffers = true});
  return p;
}
// END define_replica_pipeline() output
;
// clang-format on

TEST_F(ReplicaPipelineTest, concatenated) {
  constexpr bool no_alias_buffers = true;

  // Make the pipeline
  node_context ctx;

  auto in1 = buffer_expr::make(ctx, "in1", sizeof(short), 2);
  auto in2 = buffer_expr::make(ctx, "in2", sizeof(short), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(short), 2);

  auto intm1 = buffer_expr::make(ctx, "intm1", sizeof(short), 2);
  auto intm2 = buffer_expr::make(ctx, "intm2", sizeof(short), 2);

  var x(ctx, "x");
  var y(ctx, "y");

  // In this pipeline, the result is copied to the output. We should just compute the result directly in the output.
  func add1 = func::make(add_1<short>, {{{in1, {point(x), point(y)}}}}, {{{intm1, {x, y}}}});
  func add2 = func::make(add_1<short>, {{{in2, {point(x), point(y)}}}}, {{{intm2, {x, y}}}});
  func concatenated =
      func::make_concat({intm1, intm2}, {out, {x, y}}, 1, {0, in1->dim(1).extent(), out->dim(1).extent()});

  pipeline p = build_pipeline(ctx, {in1, in2}, {out}, build_options{.no_alias_buffers = no_alias_buffers});
  pipeline p_replica = kConcatenatedReplica();

  // Look at the source code to this test to verify that we
  // we have something that matches exactly
  std::string replica_text =
      define_replica_pipeline(ctx, {in1, in2}, {out}, build_options{.no_alias_buffers = no_alias_buffers});
  LOG_REPLICA_TEXT(replica_text);
  size_t pos = replica_pipeline_test_src.find(replica_text);
  ASSERT_NE(pos, std::string::npos) << "Matching replica text not found, expected:\n" << replica_text;

  // Run the pipeline.
  const int W = 20;
  const int H1 = 4;
  const int H2 = 7;
  buffer<short, 2> in1_buf({W, H1});
  buffer<short, 2> in2_buf({W, H2});
  init_random(in1_buf);
  init_random(in2_buf);

  buffer<short, 2> out_buf({W, H1 + H2});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in1_buf, &in2_buf};
  const raw_buffer* outputs[] = {&out_buf};
  {
    test_context eval_ctx;
    p.evaluate(inputs, outputs, eval_ctx);
  }
  {
    test_context eval_ctx;
    p_replica.evaluate(inputs, outputs, eval_ctx);
  }
}

// clang-format off
static std::function<pipeline()> kStackedReplica =
// BEGIN define_replica_pipeline() output
[]() -> ::slinky::pipeline {
  node_context ctx;
  auto in1 = buffer_expr::make(ctx, "in1", sizeof(uint16_t), 2);
  auto in2 = buffer_expr::make(ctx, "in2", sizeof(uint16_t), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(uint16_t), 3);
  auto intm1 = buffer_expr::make(ctx, "intm1", sizeof(uint16_t), 2);
  auto x = var(ctx, "x");
  auto y = var(ctx, "y");
  auto _replica_fn_2 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{in1, {point(x), point(y)}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_1 = func::make(std::move(_replica_fn_2), {{in1, {point(x), point(y)}}}, {{intm1, {x, y}}});
  auto intm2 = buffer_expr::make(ctx, "intm2", sizeof(uint16_t), 2);
  auto _replica_fn_4 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{in2, {point(x), point(y)}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_3 = func::make(std::move(_replica_fn_4), {{in2, {point(x), point(y)}}}, {{intm2, {x, y}}});
  auto _fn_0 = func::make_copy({{intm1, {point(x), point(y)}, {}, {expr(), expr(), 0}}, {intm2, {point(x), point(y)}, {}, {expr(), expr(), 1}}}, {out, {x, y}});
  auto p = build_pipeline(ctx, {}, {in1, in2}, {out}, {});
  return p;
}
// END define_replica_pipeline() output
;
// clang-format on

TEST_F(ReplicaPipelineTest, stacked) {
  // Make the pipeline
  node_context ctx;

  auto in1 = buffer_expr::make(ctx, "in1", sizeof(short), 2);
  auto in2 = buffer_expr::make(ctx, "in2", sizeof(short), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(short), 3);

  auto intm1 = buffer_expr::make(ctx, "intm1", sizeof(short), 2);
  auto intm2 = buffer_expr::make(ctx, "intm2", sizeof(short), 2);

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");

  // In this pipeline, the result is copied to the output. We should just compute the result directly in the output.
  func add1 = func::make(add_1<short>, {{{in1, {point(x), point(y)}}}}, {{{intm1, {x, y}}}});
  func add2 = func::make(add_1<short>, {{{in2, {point(x), point(y)}}}}, {{{intm2, {x, y}}}});
  func stacked = func::make_stack({intm1, intm2}, {out, {x, y, z}}, 2);

  pipeline p = build_pipeline(ctx, {in1, in2}, {out});
  pipeline p_replica = kStackedReplica();

  // Look at the source code to this test to verify that we
  // we have something that matches exactly
  std::string replica_text =
      define_replica_pipeline(ctx, {in1, in2}, {out});
  LOG_REPLICA_TEXT(replica_text);
  size_t pos = replica_pipeline_test_src.find(replica_text);
  ASSERT_NE(pos, std::string::npos) << "Matching replica text not found, expected:\n" << replica_text;

  // Run the pipeline.
  const int W = 20;
  const int H = 8;
  buffer<short, 2> in1_buf({W, H});
  buffer<short, 2> in2_buf({W, H});
  init_random(in1_buf);
  init_random(in2_buf);

  buffer<short, 3> out_buf({W, H, 2});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in1_buf, &in2_buf};
  const raw_buffer* outputs[] = {&out_buf};
  {
    test_context eval_ctx;
    p.evaluate(inputs, outputs, eval_ctx);
  }
  {
    test_context eval_ctx;
    p_replica.evaluate(inputs, outputs, eval_ctx);
  }
}

// clang-format off
static std::function<pipeline()> kDiamondStencilsReplica =
// BEGIN define_replica_pipeline() output
[]() -> ::slinky::pipeline {
  node_context ctx;
  auto in1 = buffer_expr::make(ctx, "in1", sizeof(uint16_t), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(uint16_t), 2);
  auto x = var(ctx, "x");
  auto y = var(ctx, "y");
  auto intm3 = buffer_expr::make(ctx, "intm3", sizeof(uint16_t), 2);
  auto intm2 = buffer_expr::make(ctx, "intm2", sizeof(uint16_t), 2);
  auto _replica_fn_3 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{in1, {point(x), point(y)}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_2 = func::make(std::move(_replica_fn_3), {{in1, {point(x), point(y)}}}, {{intm2, {x, y}}});
  auto _4 = x + -1;
  auto _5 = x + 1;
  auto _6 = y + -1;
  auto _7 = y + 1;
  auto _replica_fn_8 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{intm2, {{_4, _5}, {_6, _7}}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _9 = x + -1;
  auto _10 = x + 1;
  auto _11 = y + -1;
  auto _12 = y + 1;
  auto _fn_1 = func::make(std::move(_replica_fn_8), {{intm2, {{_9, _10}, {_11, _12}}}}, {{intm3, {x, y}}});
  auto intm4 = buffer_expr::make(ctx, "intm4", sizeof(uint16_t), 2);
  auto _14 = x + -2;
  auto _15 = x + 2;
  auto _16 = y + -2;
  auto _17 = y + 2;
  auto _replica_fn_18 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{intm2, {{_14, _15}, {_16, _17}}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _19 = x + -2;
  auto _20 = x + 2;
  auto _21 = y + -2;
  auto _22 = y + 2;
  auto _fn_13 = func::make(std::move(_replica_fn_18), {{intm2, {{_19, _20}, {_21, _22}}}}, {{intm4, {x, y}}});
  auto _replica_fn_23 = [=](const buffer<const void>& i0, const buffer<const void>& i1, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0, &i1};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{intm3, {point(x), point(y)}}, {intm4, {point(x), point(y)}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_0 = func::make(std::move(_replica_fn_23), {{intm3, {point(x), point(y)}}, {intm4, {point(x), point(y)}}}, {{out, {x, y}}});
  _fn_0.loops({{y, 1, loop_mode::serial}});
  auto p = build_pipeline(ctx, {}, {in1}, {out}, {});
  return p;
}
// END define_replica_pipeline() output
;
// clang-format on

TEST_F(ReplicaPipelineTest, diamond_stencils) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in1", sizeof(short), 2);
  auto intm2 = buffer_expr::make(ctx, "intm2", sizeof(short), 2);
  auto intm3 = buffer_expr::make(ctx, "intm3", sizeof(short), 2);
  auto intm4 = buffer_expr::make(ctx, "intm4", sizeof(short), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(short), 2);

  var x(ctx, "x");
  var y(ctx, "y");

  func mul2 = func::make(multiply_2<short>, {{in, {point(x), point(y)}}}, {{intm2, {x, y}}});
  func stencil1 = func::make(sum3x3<short>, {{intm2, {bounds(-1, 1) + x, bounds(-1, 1) + y}}}, {{intm3, {x, y}}});
  func stencil2 = func::make(sum5x5<short>, {{intm2, {bounds(-2, 2) + x, bounds(-2, 2) + y}}}, {{intm4, {x, y}}});
  func diff =
      func::make(subtract<short>, {{intm3, {point(x), point(y)}}, {intm4, {point(x), point(y)}}}, {{out, {x, y}}});

  diff.loops({{y, 1}});

  pipeline p = build_pipeline(ctx, {in}, {out});
  pipeline p_replica = kDiamondStencilsReplica();

  // Look at the source code to this test to verify that we
  // we have something that matches exactly
  std::string replica_text = define_replica_pipeline(ctx, {in}, {out});
  LOG_REPLICA_TEXT(replica_text);
  size_t pos = replica_pipeline_test_src.find(replica_text);
  ASSERT_NE(pos, std::string::npos) << "Matching replica text not found, expected:\n" << replica_text;

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
  {
    test_context eval_ctx;
    p.evaluate(inputs, outputs, eval_ctx);
  }
  {
    test_context eval_ctx;
    p_replica.evaluate(inputs, outputs, eval_ctx);
  }
}

// clang-format off
static std::function<pipeline()> kPaddedStencilReplica =
// BEGIN define_replica_pipeline() output
[]() -> ::slinky::pipeline {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", sizeof(uint16_t), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(uint16_t), 2);
  auto x = var(ctx, "x");
  auto y = var(ctx, "y");
  auto padded_intm = buffer_expr::make(ctx, "padded_intm", sizeof(uint16_t), 2);
  auto intm = buffer_expr::make(ctx, "intm", sizeof(uint16_t), 2);
  auto _replica_fn_3 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{in, {point(x), point(y)}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_2 = func::make(std::move(_replica_fn_3), {{in, {point(x), point(y)}}}, {{intm, {x, y}}});
  auto _4 = variable::make(in->sym());
  auto _5 = buffer_min(_4, 0);
  auto _6 = buffer_max(_4, 0);
  auto _7 = buffer_min(_4, 1);
  auto _8 = buffer_max(_4, 1);
  auto _fn_1 = func::make_copy({{intm, {point(x), point(y)}, {{_5, _6}, {_7, _8}}, {}}}, {padded_intm, {x, y}});
  _fn_1.compute_root();
  auto _9 = x + -1;
  auto _10 = x + 1;
  auto _11 = y + -1;
  auto _12 = y + 1;
  auto _replica_fn_13 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{padded_intm, {{_9, _10}, {_11, _12}}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _14 = x + -1;
  auto _15 = x + 1;
  auto _16 = y + -1;
  auto _17 = y + 1;
  auto _fn_0 = func::make(std::move(_replica_fn_13), {{padded_intm, {{_14, _15}, {_16, _17}}}}, {{out, {x, y}}});
  _fn_0.loops({{y, 1, loop_mode::serial}});
  auto p = build_pipeline(ctx, {}, {in}, {out}, {});
  return p;
}
// END define_replica_pipeline() output
;
// clang-format on

TEST_F(ReplicaPipelineTest, padded_stencil) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(short), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(short), 2);

  auto intm = buffer_expr::make(ctx, "intm", sizeof(short), 2);
  auto padded_intm = buffer_expr::make(ctx, "padded_intm", sizeof(short), 2);

  var x(ctx, "x");
  var y(ctx, "y");

  func add = func::make(add_1<short>, {{in, {point(x), point(y)}}}, {{intm, {x, y}}});
  func padded = func::make_copy({intm, {point(x), point(y)}, in->bounds()}, {padded_intm, {x, y}}, {{6, 0}});
  func stencil = func::make(sum3x3<short>, {{padded_intm, {bounds(-1, 1) + x, bounds(-1, 1) + y}}}, {{out, {x, y}}});

  stencil.loops({y});
  padded.compute_root();

  pipeline p = build_pipeline(ctx, {in}, {out});
  pipeline p_replica = kPaddedStencilReplica();

  // Look at the source code to this test to verify that we
  // we have something that matches exactly
  std::string replica_text = define_replica_pipeline(ctx, {in}, {out});
  LOG_REPLICA_TEXT(replica_text);
  size_t pos = replica_pipeline_test_src.find(replica_text);
  ASSERT_NE(pos, std::string::npos) << "Matching replica text not found, expected:\n" << replica_text;

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
  {
    test_context eval_ctx;
    p.evaluate(inputs, outputs, eval_ctx);
  }
  {
    test_context eval_ctx;
    p_replica.evaluate(inputs, outputs, eval_ctx);
  }
}

}  // namespace slinky
