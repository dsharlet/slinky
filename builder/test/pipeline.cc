#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <fstream>
#include <numeric>

#include "builder/pipeline.h"
#include "builder/replica_pipeline.h"
#include "builder/substitute.h"
#include "builder/test/bazel_util.h"
#include "builder/test/util.h"
#include "runtime/chrome_trace.h"
#include "runtime/expr.h"
#include "runtime/pipeline.h"
#include "runtime/print.h"
#include "runtime/thread_pool.h"
#include "runtime/visualize.h"

namespace slinky {

std::string get_replica_golden() {
  static std::string golden = read_entire_file(get_bazel_file_path("builder/test/replica_pipeline.cc"));
  return golden;
}

void check_replica_pipeline(const std::string& replica_text) {
  size_t pos = get_replica_golden().find(replica_text);
  ASSERT_NE(pos, std::string::npos) << "Matching replica text not found, expected:\n" << replica_text;
}

void check_visualize(const std::string& filename, const pipeline& p, pipeline::buffers inputs,
    pipeline::buffers outputs, const node_context* ctx) {
  std::stringstream viz_stream;
  visualize(viz_stream, p, inputs, outputs, ctx);
  std::string viz = viz_stream.str();

  std::string golden_path = get_bazel_file_path("builder/test/visualize/" + filename);
  if (is_bazel_test()) {
    std::string golden = read_entire_file(golden_path);
    ASSERT_FALSE(golden.empty());
    // If this check fails, and you believe the changes to the visualization is correct, run this
    // test outside of bazel from the root of the repo to update the golden files.
    ASSERT_TRUE(golden == viz);
  } else {
    std::ofstream file(golden_path);
    file << viz;
  }
}

void setup_tracing(eval_context& ctx, const std::string& filename) {
  struct tracer {
    std::string trace_file;
    // Store the trace in a stringstream and write it at the end, to avoid overhead influencing the trace.
    std::stringstream buffer;
    chrome_trace trace;

    tracer(const std::string& filename) : trace_file(filename), trace(buffer) {}
    ~tracer() {
      std::ofstream file(trace_file);
      file << buffer.str();
    }
  };

  auto t = std::make_shared<tracer>(filename);

  ctx.trace_begin = [t](const char* op) -> index_t {
    t->trace.begin(op);
    // chrome_trace expects trace_begin and trace_end to pass the string, while slinky's API expects to pass a token to
    // trace_end returned by trace_begin. Because `index_t` must be able to hold a pointer, we'll just use the token to
    // store the pointer.
    return reinterpret_cast<index_t>(op);
  };
  ctx.trace_end = [t](index_t token) { t->trace.end(reinterpret_cast<const char*>(token)); };
}

thread_pool threads;

struct memory_info {
  std::atomic<index_t> live_count = 0;
  std::atomic<index_t> live_size = 0;
  std::atomic<index_t> total_count = 0;
  std::atomic<index_t> total_size = 0;

  void track_allocate(index_t size) {
    live_count += 1;
    live_size += size;
    total_count += 1;
    total_size += size;
  }

  void track_free(index_t size) {
    live_count -= 1;
    live_size -= size;
  }
};

class test_context : public eval_context {
public:
  memory_info heap;

  test_context() {
    allocate = [this](var, raw_buffer* b) {
      void* allocation = b->allocate();
      heap.track_allocate(b->size_bytes());
      return allocation;
    };
    free = [this](var, raw_buffer* b, void* allocation) {
      ::free(allocation);
      heap.track_free(b->size_bytes());
    };

    enqueue_many = [&](thread_pool::task t) { threads.enqueue(threads.thread_count(), std::move(t)); };
    enqueue = [&](int n, thread_pool::task t) { threads.enqueue(n, std::move(t)); };
    wait_for = [&](const std::function<bool()>& condition) { return threads.wait_for(condition); };
    atomic_call = [&](const thread_pool::task& t) { return threads.atomic_call(t); };
  }
};

// This file provides a number of toy funcs for test pipelines.

// Copy from input to output.
// TODO: We should be able to just do this with raw_buffer and not make it a template.
template <typename T>
index_t copy_2d(const buffer<const T>& in, const buffer<T>& out) {
  copy(in, out, nullptr);
  return 0;
}

template <typename T>
index_t zero_padded_copy(const buffer<const T>& in, const buffer<T>& out) {
  assert(in.rank == out.rank);
  T zero = 0;
  slinky::copy(in, out, &zero);
  return 0;
}

// Copy rows, where the output y is -y in the input.
template <typename T>
index_t flip_y(const buffer<const T>& in, const buffer<T>& out) {
  assert(in.rank == 2);
  assert(out.rank == 2);
  std::size_t size = out.dim(0).extent() * out.elem_size;
  for (index_t y = out.dim(1).begin(); y < out.dim(1).end(); ++y) {
    const T* src = &in(out.dim(0).min(), -y);
    T* dst = &out(out.dim(0).min(), y);
    std::copy(src, src + size, dst);
  }
  return 0;
}

template <typename T>
index_t multiply_2(const buffer<const T>& in, const buffer<T>& out) {
  assert(in.rank == out.rank);
  for_each_index(out, [&](auto i) { out(i) = in(i) * 2; });
  return 0;
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

// init_random() for raw_buffer requires allocation be done by caller
template <typename T>
void fill_random(raw_buffer& x) {
  assert(x.base != nullptr);
  for_each_index(x, [&](auto i) { *reinterpret_cast<T*>(x.address_at(i)) = (rand() % 20) - 10; });
}

template <typename T, std::size_t N>
void init_random(buffer<T, N>& x) {
  x.allocate();
  fill_random<T>(x);
}

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
      func::make(multiply_2<int>, {{in, {point(x)}}}, {{out, {x}}}, call_stmt::attributes{.allow_in_place = true});
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
  ASSERT_EQ(eval_ctx.heap.total_size, 0);

  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(out_buf(i), 2 * i);
  }
}

class elementwise : public testing::TestWithParam<std::tuple<int, int, bool>> {};

INSTANTIATE_TEST_SUITE_P(split_schedule_mode, elementwise,
    testing::Combine(loop_modes, testing::Range(0, 4), testing::Values(false, true)),
    test_params_to_string<elementwise::ParamType>);

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

  // Here we explicitly use std::functions (in the form of a
  // func::callable typedef) to wrap the local calls
  // purely to verify that the relevant func::make calls work correctly.
  func::callable<const int, int> m2 = multiply_2<int>;
  func::callable<const int, int> a1 = add_1<int>;

  func mul =
      func::make(std::move(m2), {{in, {point(x)}}}, {{intm, {x}}}, call_stmt::attributes{.allow_in_place = true});
  func add =
      func::make(std::move(a1), {{intm, {point(x)}}}, {{out, {x}}}, call_stmt::attributes{.allow_in_place = true});

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
  if (schedule_storage) {
    ASSERT_EQ(eval_ctx.heap.total_count, 0);  // The intermediate only needs stack.
  }

  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(out_buf(i), 2 * i + 1);
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

  var x(ctx, "x");
  var y(ctx, "y");

  // Here we explicitly use lambdas to wrap the local calls,
  // purely to verify that the relevant func::make calls work correctly.
  auto m2 = [](const buffer<const int>& a, const buffer<int>& b) -> index_t { return multiply_2<int>(a, b); };
  auto a1 = [](const buffer<const int>& a, const buffer<int>& b) -> index_t { return add_1<int>(a, b); };

  func mul = func::make(
      std::move(m2), {{in, {point(x), point(y)}}}, {{intm, {x, y}}}, call_stmt::attributes{.allow_in_place = true});
  func add = func::make(
      std::move(a1), {{intm, {point(x), point(y)}}}, {{out, {x, y}}}, call_stmt::attributes{.allow_in_place = true});

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
  if (schedule_storage) {
    ASSERT_EQ(eval_ctx.heap.total_count, 0);  // The intermediate only needs stack.
  } else {
    ASSERT_EQ(eval_ctx.heap.total_count, 0);  // The buffers should alias.
  }

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
  ab->dim(1).stride = static_cast<index_t>(sizeof(int));
  ab->dim(0).stride = ab->dim(1).extent() * ab->dim(1).stride;

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
  if (split > 0 && max_workers == loop::serial) {
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

  if (split == 1 && max_workers == loop::serial) {
    check_replica_pipeline(define_replica_pipeline(ctx, {a, b, c}, {abc}));
  }
}

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

class pyramid : public testing::TestWithParam<std::tuple<int>> {};

INSTANTIATE_TEST_SUITE_P(mode, pyramid, testing::Combine(loop_modes), test_params_to_string<pyramid::ParamType>);

TEST_P(pyramid, pipeline) {
  int max_workers = std::get<0>(GetParam());
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");

  func downsample = func::make(downsample2x, {{in, {2 * x + bounds(0, 1), 2 * y + bounds(0, 1)}}}, {{intm, {x, y}}});
  func upsample = func::make(pyramid_upsample2x,
      {{in, {point(x), point(y)}}, {intm, {bounds(x, x + 1) / 2, bounds(y, y + 1) / 2}}}, {{out, {x, y}}});

  upsample.loops({{y, 1, max_workers}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 10;
  const int H = 10;
  buffer<int, 2> in_buf({W + 2, H + 2});
  buffer<int, 2> out_buf({W, H});

  init_random(in_buf);
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  const int parallel_extra = max_workers != loop::serial ? 1 : 0;
  ASSERT_EQ(eval_ctx.heap.total_size, (W + 2) / 2 * (2 + parallel_extra) * sizeof(int));
  ASSERT_EQ(eval_ctx.heap.total_count, 1);

  if (max_workers == loop::serial) {
    check_replica_pipeline(define_replica_pipeline(ctx, {in}, {out}));
  }
}

TEST(pyramid_multi, pipeline) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  auto down0 = buffer_expr::make(ctx, "down0", 2, sizeof(int));
  auto down1 = buffer_expr::make(ctx, "down1", 2, sizeof(int));
  auto down2 = buffer_expr::make(ctx, "down2", 2, sizeof(int));
  auto down3 = buffer_expr::make(ctx, "down3", 2, sizeof(int));
  auto up3 = buffer_expr::make(ctx, "up3", 2, sizeof(int));
  auto up2 = buffer_expr::make(ctx, "up2", 2, sizeof(int));
  auto up1 = buffer_expr::make(ctx, "up1", 2, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");

  func downsample0 = func::make(downsample2x, {{in, {2 * x + bounds(0, 1), 2 * y + bounds(0, 1)}}}, {{down0, {x, y}}});
  func downsample1 =
      func::make(downsample2x, {{down0, {2 * x + bounds(0, 1), 2 * y + bounds(0, 1)}}}, {{down1, {x, y}}});
  func downsample2 =
      func::make(downsample2x, {{down1, {2 * x + bounds(0, 1), 2 * y + bounds(0, 1)}}}, {{down2, {x, y}}});
  func downsample3 =
      func::make(downsample2x, {{down2, {2 * x + bounds(0, 1), 2 * y + bounds(0, 1)}}}, {{down3, {x, y}}});
  func upsample3 = func::make(pyramid_upsample2x,
      {{down2, {point(x), point(y)}}, {down3, {bounds(x, x + 1) / 2, bounds(y, y + 1) / 2}}}, {{up3, {x, y}}});
  func upsample2 = func::make(pyramid_upsample2x,
      {{down1, {point(x), point(y)}}, {up3, {bounds(x, x + 1) / 2, bounds(y, y + 1) / 2}}}, {{up2, {x, y}}});
  func upsample1 = func::make(pyramid_upsample2x,
      {{down0, {point(x), point(y)}}, {up2, {bounds(x, x + 1) / 2, bounds(y, y + 1) / 2}}}, {{up1, {x, y}}});
  func upsample = func::make(pyramid_upsample2x,
      {{in, {point(x), point(y)}}, {up1, {bounds(x, x + 1) / 2, bounds(y, y + 1) / 2}}}, {{out, {x, y}}});

  upsample.loops({{y, 1}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 21;
  const int H = 15;
  buffer<int, 2> in_buf({W + 32, H + 32});
  buffer<int, 2> out_buf({W, H});

  init_random(in_buf);
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  // Run the pipeline stages manually to get the reference result.
  buffer<int, 2> ref_down0({(W + 32) / 2, (H + 32) / 2});
  buffer<int, 2> ref_down1({(W + 32) / 4, (H + 32) / 4});
  buffer<int, 2> ref_down2({(W + 32) / 8, (H + 32) / 8});
  buffer<int, 2> ref_down3({(W + 32) / 16, (H + 32) / 16});
  buffer<int, 2> ref_up3({(W + 7) / 8 + 1, (H + 7) / 8 + 1});
  buffer<int, 2> ref_up2({(W + 3) / 4 + 1, (H + 3) / 4 + 1});
  buffer<int, 2> ref_up1({(W + 1) / 2 + 1, (H + 1) / 2 + 1});
  buffer<int, 2> ref_out({W, H});

  ref_down0.allocate();
  ref_down1.allocate();
  ref_down2.allocate();
  ref_down3.allocate();
  ref_up3.allocate();
  ref_up2.allocate();
  ref_up1.allocate();
  ref_out.allocate();

  downsample2x(in_buf.cast<const int>(), ref_down0.cast<int>());
  downsample2x(ref_down0.cast<const int>(), ref_down1.cast<int>());
  downsample2x(ref_down1.cast<const int>(), ref_down2.cast<int>());
  downsample2x(ref_down2.cast<const int>(), ref_down3.cast<int>());
  pyramid_upsample2x(ref_down2.cast<const int>(), ref_down3.cast<const int>(), ref_up3.cast<int>());
  pyramid_upsample2x(ref_down1.cast<const int>(), ref_up3.cast<const int>(), ref_up2.cast<int>());
  pyramid_upsample2x(ref_down0.cast<const int>(), ref_up2.cast<const int>(), ref_up1.cast<int>());
  pyramid_upsample2x(in_buf.cast<const int>(), ref_up1.cast<const int>(), ref_out.cast<int>());

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(ref_out(x, y), out_buf(x, y));
    }
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
  if (split > 0) {
    const int parallel_extra = max_workers != loop::serial ? split : 0;
    ASSERT_EQ(eval_ctx.heap.total_size, (W + 2) * align_up(split + parallel_extra + 2, split) * sizeof(short));
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

  // Also visualize this pipeline.
  if (max_workers == loop::serial && split_intermediate == 0) {
    check_visualize("stencil_split_" + std::to_string(split) + ".html", p, inputs, outputs, &ctx);
  }
}

class slide_2d : public testing::TestWithParam<std::tuple<int, int>> {};

INSTANTIATE_TEST_SUITE_P(
    split_split_mode, slide_2d, testing::Combine(loop_modes, loop_modes), test_params_to_string<slide_2d::ParamType>);

TEST_P(slide_2d, pipeline) {
  int max_workers_x = std::get<0>(GetParam());
  int max_workers_y = std::get<1>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));

  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(short));

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
  ASSERT_EQ(eval_ctx.heap.total_size, (W + 2) * 3 * sizeof(short));
  ASSERT_EQ(eval_ctx.heap.total_count, 1);
  ASSERT_EQ(add_count, (W + 2) * (H + 2));

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

class stencil_chain : public testing::TestWithParam<std::tuple<int, int>> {};

INSTANTIATE_TEST_SUITE_P(split_split_mode, stencil_chain, testing::Combine(loop_modes, testing::Range(0, 3)),
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
  setup_tracing(eval_ctx, test_name + ".json");

  p.evaluate(inputs, outputs, eval_ctx);

  if (split > 0) {
    const int parallel_extra = max_workers != loop::serial ? split * 2 : 0;
    ASSERT_EQ(eval_ctx.heap.total_size, (W + 2) * align_up(split + parallel_extra + 2, split) * sizeof(short) +
                                            (W + 4) * align_up(split + parallel_extra + 2, split) * sizeof(short));
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

  // Also visualize this pipeline.
  if (max_workers == loop::serial) {
    check_visualize(test_name + ".html", p, inputs, outputs, &ctx);
  }
}

TEST(flip_y, pipeline) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(char));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(char));
  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(char));

  var x(ctx, "x");
  var y(ctx, "y");

  func copy = func::make(copy_2d<char>, {{in, {point(x), point(y)}}}, {{intm, {x, y}}});
  func flip = func::make(flip_y<char>, {{intm, {point(x), point(-y)}}}, {{out, {x, y}}});

  pipeline p = build_pipeline(ctx, {in}, {out});

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

TEST(padded_copy, pipeline) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(char));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(char));
  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(char));

  var x(ctx, "x");
  var y(ctx, "y");

  // We could just clamp using the bounds directly below, but that would hardcode the bounds we clamp
  // in the pipeline. This way, the bounds can vary at eval-time.
  var w(ctx, "w");
  var h(ctx, "h");

  // Copy the input so we can measure the size of the buffer we think we need internally.
  func copy = func::make(copy_2d<char>, {{in, {point(x), point(y)}}}, {{intm, {x, y}}});
  // This is elementwise, but with a clamp to limit the bounds required of the input.
  func crop = func::make(
      zero_padded_copy<char>, {{intm, {point(clamp(x, 0, w - 1)), point(clamp(y, 0, h - 1))}}}, {{out, {x, y}}});

  crop.loops({y});

  pipeline p = build_pipeline(ctx, {w, h}, {in}, {out});

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

class multiple_outputs : public testing::TestWithParam<std::tuple<int, int>> {};

INSTANTIATE_TEST_SUITE_P(split_mode, multiple_outputs, testing::Combine(loop_modes, testing::Range(0, 4)),
    test_params_to_string<multiple_outputs::ParamType>);

TEST_P(multiple_outputs, pipeline) {
  int max_workers = std::get<0>(GetParam());
  int split = std::get<1>(GetParam());

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
    sums.loops({{z, split, max_workers}});
  }

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

  if (split == 1 && max_workers == loop::serial) {
    check_replica_pipeline(define_replica_pipeline(ctx, {in}, {sum_x, sum_xy}));
  }
}

class outer_product : public testing::TestWithParam<std::tuple<int, int, int>> {};

INSTANTIATE_TEST_SUITE_P(split_split_mode, outer_product,
    testing::Combine(loop_modes, testing::Range(0, 4), testing::Range(0, 4)),
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

  var i(ctx, "i");
  var j(ctx, "j");

  func outer = func::make(
      [](const buffer<const int>& a, const buffer<const int>& b, const buffer<int>& c) -> index_t {
        for (index_t j = c.dim(1).begin(); j < c.dim(1).end(); ++j) {
          for (index_t i = c.dim(0).begin(); i < c.dim(0).end(); ++i) {
            c(i, j) = a(i) * b(j);
          }
        }
        return 0;
      },
      {{a, {point(i)}}, {b, {point(j)}}}, {{out, {i, j}}});

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
      ASSERT_EQ(out_buf(i, j), a_buf(i) * b_buf(j));
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

    var x(ctx, "x");
    var y(ctx, "y");

    func add1 = func::make(
        add_1<short>, {{in1, {point(x), point(y)}}}, {{intm1, {x, y}}}, call_stmt::attributes{.allow_in_place = true});
    func stencil1 = func::make(sum3x3<short>, {{intm1, {bounds(-1, 1) + x, bounds(-1, 1) + y}}}, {{out1, {x, y}}});

    func mul2 =
        func::make(multiply_2<int>, {{in2, {point(x)}}}, {{intm2, {x}}}, call_stmt::attributes{.allow_in_place = true});
    func add2 =
        func::make(add_1<int>, {{intm2, {point(x)}}}, {{out2, {x}}}, call_stmt::attributes{.allow_in_place = true});

    stencil1.loops({{y, 2}});

    check_replica_pipeline(define_replica_pipeline(ctx, {in1, in2}, {out1, out2}));

    return build_pipeline(ctx, {in1, in2}, {out1, out2});
  };
  pipeline p = make_pipeline();
  pipeline p2 = make_pipeline();
  ASSERT_TRUE(match(p.body, p2.body));

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

class copied_result : public testing::TestWithParam<int> {};

INSTANTIATE_TEST_SUITE_P(schedule, copied_result, testing::Range(0, 3));

TEST_P(copied_result, pipeline) {
  int schedule = GetParam();

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));

  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(short));

  var x(ctx, "x");
  var y(ctx, "y");

  // In this pipeline, the result is copied to the output. We should just compute the result directly in the output.
  func stencil = func::make(sum3x3<short>, {{in, {bounds(-1, 1) + x, bounds(-1, 1) + y}}}, {{intm, {x, y}}});
  func padded = func::make_copy({intm, {point(x), point(y)}}, {out, {x, y}});

  switch (schedule) {
  case 0: break;
  case 1:
    padded.loops({y});
    stencil.compute_root();
    break;
  case 2: padded.loops({y}); break;
  }

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

class concatenated_result : public testing::TestWithParam<bool> {};

INSTANTIATE_TEST_SUITE_P(schedule, concatenated_result, testing::Values(false, true));

TEST_P(concatenated_result, pipeline) {
  bool no_alias_buffers = GetParam();
  // Make the pipeline
  node_context ctx;

  auto in1 = buffer_expr::make(ctx, "in1", 2, sizeof(short));
  auto in2 = buffer_expr::make(ctx, "in2", 2, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));

  auto intm1 = buffer_expr::make(ctx, "intm1", 2, sizeof(short));
  auto intm2 = buffer_expr::make(ctx, "intm2", 2, sizeof(short));

  var x(ctx, "x");
  var y(ctx, "y");

  // In this pipeline, the result is copied to the output. We should just compute the result directly in the output.
  func add1 = func::make(add_1<short>, {{{in1, {point(x), point(y)}}}}, {{{intm1, {x, y}}}});
  func add2 = func::make(add_1<short>, {{{in2, {point(x), point(y)}}}}, {{{intm2, {x, y}}}});
  func concatenated =
      func::make_concat({intm1, intm2}, {out, {x, y}}, 1, {0, in1->dim(1).extent(), out->dim(1).extent()});

  pipeline p = build_pipeline(ctx, {in1, in2}, {out}, build_options{.no_alias_buffers = no_alias_buffers});

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
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);
  if (!no_alias_buffers) {
    ASSERT_EQ(eval_ctx.heap.total_count, 0);
  }

  for (int y = 0; y < H1 + H2; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y), (y < H1 ? in1_buf(x, y) : in2_buf(x, y - H1)) + 1);
    }
  }

  // Also visualize this pipeline.
  check_visualize("concatenate_" + std::to_string(no_alias_buffers) + ".html", p, inputs, outputs, &ctx);

  if (no_alias_buffers == true) {
    check_replica_pipeline(
        define_replica_pipeline(ctx, {in1, in2}, {out}, build_options{.no_alias_buffers = no_alias_buffers}));
  }
}

class transposed_result : public testing::TestWithParam<std::tuple<bool, std::vector<int>>> {};

INSTANTIATE_TEST_SUITE_P(schedule, transposed_result,
    testing::Combine(testing::Values(false, true), testing::Values(std::vector<int>{0, 1, 2}, std::vector<int>{0, 2, 1},
                                                       std::vector<int>{1, 2, 0}, std::vector<int>{0, 0, 0})));

template <typename T>
std::vector<T> permute(span<const int> p, const std::vector<T>& x) {
  std::vector<T> result(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    result[i] = x[p[i]];
  }
  return result;
}

bool is_permutation(span<const int> p) {
  std::vector<int> unpermuted(p.size());
  std::iota(unpermuted.begin(), unpermuted.end(), 0);
  return std::is_permutation(p.begin(), p.end(), unpermuted.begin());
}

TEST_P(transposed_result, pipeline) {
  bool no_alias_buffers = std::get<0>(GetParam());
  const std::vector<int>& permutation = std::get<1>(GetParam());

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 3, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 3, sizeof(short));

  auto intm = buffer_expr::make(ctx, "intm", 3, sizeof(short));

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");

  // In this pipeline, the result is copied to the output. We should just compute the result directly in the output.
  func add = func::make(add_1<short>, {{{in, {point(x), point(y), point(z)}}}}, {{{intm, {x, y, z}}}});
  func transposed =
      func::make_copy({intm, permute<interval_expr>(permutation, {point(x), point(y), point(z)})}, {out, {x, y, z}});

  pipeline p = build_pipeline(ctx, {in}, {out}, build_options{.no_alias_buffers = no_alias_buffers});

  // Run the pipeline.
  const int W = 20;
  const int H = 4;
  const int D = 7;
  buffer<short, 3> in_buf(permute<index_t>(permutation, {W, H, D}));
  init_random(in_buf);

  buffer<short, 3> out_buf({W, H, D});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);
  if (is_permutation(permutation) && !no_alias_buffers) {
    ASSERT_EQ(eval_ctx.heap.total_count, 0);
  } else {
    ASSERT_EQ(eval_ctx.heap.total_count, 1);
  }

  for (int z = 0; z < D; ++z) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        ASSERT_EQ(out_buf(x, y, z), in_buf(permute<index_t>(permutation, {x, y, z})) + 1);
      }
    }
  }
}

TEST(stacked_result, pipeline) {
  // Make the pipeline
  node_context ctx;

  auto in1 = buffer_expr::make(ctx, "in1", 2, sizeof(short));
  auto in2 = buffer_expr::make(ctx, "in2", 2, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 3, sizeof(short));

  auto intm1 = buffer_expr::make(ctx, "intm1", 2, sizeof(short));
  auto intm2 = buffer_expr::make(ctx, "intm2", 2, sizeof(short));

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");

  // In this pipeline, the result is copied to the output. We should just compute the result directly in the output.
  func add1 = func::make(add_1<short>, {{{in1, {point(x), point(y)}}}}, {{{intm1, {x, y}}}});
  func add2 = func::make(add_1<short>, {{{in2, {point(x), point(y)}}}}, {{{intm2, {x, y}}}});
  func stacked = func::make_stack({intm1, intm2}, {out, {x, y, z}}, 2);

  pipeline p = build_pipeline(ctx, {in1, in2}, {out});

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
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);
  ASSERT_EQ(eval_ctx.heap.total_count, 0);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y, 0), in1_buf(x, y) + 1);
      ASSERT_EQ(out_buf(x, y, 1), in2_buf(x, y) + 1);
    }
  }

  check_replica_pipeline(define_replica_pipeline(ctx, {in1, in2}, {out}));
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
  func padded = func::make_copy({intm, {point(x), point(y)}, in->bounds()}, {padded_intm, {x, y}}, {{6, 0}});
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
  if (schedule == 2) {
    // TODO: We need to be able to find the upper bound of
    // max((x + 1), buffer_min(a, b)) - min((x + 1), buffer_max(a, b)) to fold this.
    // ASSERT_EQ(eval_ctx.heap.total_size, W * 2 * sizeof(short) + (W + 2) * 3 * sizeof(short));
    ASSERT_EQ(eval_ctx.heap.total_count, 2);
  }

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

  // Also visualize this pipeline.
  check_visualize("padded_stencil_" + std::to_string(schedule) + ".html", p, inputs, outputs, &ctx);

  if (schedule == 1) {
    check_replica_pipeline(define_replica_pipeline(ctx, {in}, {out}));
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
  fill_random<short>(*constant_buf);

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

INSTANTIATE_TEST_SUITE_P(schedule, parallel_stencils, testing::Range(0, 4));

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

  func add1 = func::make(add_1<short>, {{in1, {point(x), point(y)}}}, {{intm1, {x, y}}});
  func mul2 = func::make(multiply_2<short>, {{in2, {point(x), point(y)}}}, {{intm2, {x, y}}});
  func stencil1 = func::make(sum3x3<short>, {{intm1, {bounds(-1, 1) + x, bounds(-1, 1) + y}}}, {{intm3, {x, y}}});
  func stencil2 = func::make(sum5x5<short>, {{intm2, {bounds(-2, 2) + x, bounds(-2, 2) + y}}}, {{intm4, {x, y}}});
  func diff =
      func::make(subtract<short>, {{intm3, {point(x), point(y)}}, {intm4, {point(x), point(y)}}}, {{out, {x, y}}});

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

INSTANTIATE_TEST_SUITE_P(schedule, diamond_stencils, testing::Range(0, 4));

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

    if (schedule == 0) {
      diff.loops({{y, 1}});
    } else if (schedule == 1) {
      diff.loops({{y, 1}});
      stencil1.loops({{y, 2}});
      stencil2.loops({{y, 2}});
    } else if (schedule == 2) {
      diff.loops({{y, 1}});
      stencil1.loops({{y, 2}});
      stencil2.loops({{y, 2}});
      mul2.compute_root();
    } else if (schedule == 3) {
      diff.loops({{y, 1, loop::parallel}});
    }

    return build_pipeline(ctx, {in}, {out});
  };
  pipeline p = make_pipeline();
  pipeline p2 = make_pipeline();
  ASSERT_TRUE(match(p.body, p2.body));

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
  auto intm3 = buffer_expr::make(ctx, "intm3", 2, sizeof(short));
  auto intm4 = buffer_expr::make(ctx, "intm4", 2, sizeof(short));

  var x(ctx, "x");
  var y(ctx, "y");

  func mul2 = func::make(multiply_2<short>, {{in, {point(x), point(y)}}}, {{intm2, {x, y}}});
  func add1 = func::make(add_1<short>, {{intm2, {point(x), point(y)}}}, {{intm3, {x, y}}});
  func add2 = func::make(add_1<short>, {{intm2, {point(x), point(y)}}}, {{intm4, {x, y}}});

  add2.loops({{y, 1}});

  pipeline p = build_pipeline(ctx, {in}, {intm3, intm4});

  // Run the pipeline.
  const int W = 32;
  const int H = 32;
  buffer<short, 2> in_buf({W, H});
  buffer<short, 2> intm3_buf({W, H});
  buffer<short, 2> intm4_buf({W, H});

  init_random(in_buf);
  intm3_buf.allocate();
  intm4_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&intm3_buf, &intm4_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  // Run the pipeline stages manually to get the reference result.
  buffer<short, 2> ref_intm2({W, H});
  buffer<short, 2> ref_intm3({W, H});
  buffer<short, 2> ref_intm4({W, H});

  ref_intm2.allocate();
  ref_intm3.allocate();
  ref_intm4.allocate();

  multiply_2<short>(in_buf.cast<const short>(), ref_intm2.cast<short>());
  add_1<short>(ref_intm2.cast<const short>(), ref_intm3.cast<short>());
  add_1<short>(ref_intm2.cast<const short>(), ref_intm4.cast<short>());

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(ref_intm3(x, y), intm3_buf(x, y));
    }
  }

  for (int y = 0; y < W; ++y) {
    for (int x = 0; x < H; ++x) {
      ASSERT_EQ(ref_intm4(x, y), intm4_buf(x, y));
    }
  }

  check_replica_pipeline(define_replica_pipeline(ctx, {in}, {intm3, intm4}));
}
}  // namespace slinky
