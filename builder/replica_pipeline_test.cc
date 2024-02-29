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
static std::function<pipeline()> kTrivialReplicas[2][2] = {
  {
    {
// split = 0, lm = serial
// BEGIN define_replica_pipeline() output
[]() -> ::slinky::pipeline {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", sizeof(uint32_t), 1);
  auto out = buffer_expr::make(ctx, "out", sizeof(uint32_t), 1);
  auto x = var(ctx, "x");
  auto _replica_fn_1 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{in, {point(x)}}};
    const std::vector<var> fout_dims[] = {{x}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_0 = func::make(std::move(_replica_fn_1), {{in, {point(x)}}}, {{out, {x}}});
  auto p = build_pipeline(ctx, {}, {in}, {out}, {});
  return p;
}
// END define_replica_pipeline() output
    },
    {
// split = 1, lm = serial
// BEGIN define_replica_pipeline() output
[]() -> ::slinky::pipeline {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", sizeof(uint32_t), 1);
  auto out = buffer_expr::make(ctx, "out", sizeof(uint32_t), 1);
  auto x = var(ctx, "x");
  auto _replica_fn_1 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{in, {point(x)}}};
    const std::vector<var> fout_dims[] = {{x}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_0 = func::make(std::move(_replica_fn_1), {{in, {point(x)}}}, {{out, {x}}});
  auto p = build_pipeline(ctx, {}, {in}, {out}, {});
  return p;
}
// END define_replica_pipeline() output
    },
  },
  {
    {
// split = 0, lm = parallel
// BEGIN define_replica_pipeline() output
[]() -> ::slinky::pipeline {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", sizeof(uint32_t), 1);
  auto out = buffer_expr::make(ctx, "out", sizeof(uint32_t), 1);
  auto x = var(ctx, "x");
  auto _replica_fn_1 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{in, {point(x)}}};
    const std::vector<var> fout_dims[] = {{x}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_0 = func::make(std::move(_replica_fn_1), {{in, {point(x)}}}, {{out, {x}}});
  _fn_0.loops({{x, 1, loop_mode::serial}});
  auto p = build_pipeline(ctx, {}, {in}, {out}, {});
  return p;
}
// END define_replica_pipeline() output
    },
    {
// split = 1, lm = parallel
// BEGIN define_replica_pipeline() output
[]() -> ::slinky::pipeline {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", sizeof(uint32_t), 1);
  auto out = buffer_expr::make(ctx, "out", sizeof(uint32_t), 1);
  auto x = var(ctx, "x");
  auto _replica_fn_1 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{in, {point(x)}}};
    const std::vector<var> fout_dims[] = {{x}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_0 = func::make(std::move(_replica_fn_1), {{in, {point(x)}}}, {{out, {x}}});
  _fn_0.loops({{x, 1, loop_mode::parallel}});
  auto p = build_pipeline(ctx, {}, {in}, {out}, {});
  return p;
}
// END define_replica_pipeline() output
    },
  },
};
// clang-format on

template <typename T>
index_t xor_hash_trivial(const buffer<const T>& in, const buffer<T>& out) {
  const T value = 0;
  fill(out, &value);
  for_each_index(out, [&](auto i) { out(i) ^= in(i); });
  return 0;
}

// A trivial pipeline with one stage
TEST_F(ReplicaPipelineTest, trivial) {
  for (int split : {0, 1}) {
    for (loop_mode lm : {loop_mode::serial, loop_mode::parallel}) {

      // Make the pipeline
      node_context ctx;

      auto in = buffer_expr::make(ctx, "in", sizeof(int), 1);
      auto out = buffer_expr::make(ctx, "out", sizeof(int), 1);

      var x(ctx, "x");

      func test = func::make(xor_hash_trivial<int>, {{in, {point(x)}}}, {{out, {x}}}, {.allow_in_place = true});
      if (split > 0) {
        test.loops({{x, split, lm}});
      }

      pipeline p = build_pipeline(ctx, {in}, {out});
      pipeline p_replica = kTrivialReplicas[split][static_cast<int>(lm)]();

      // Look at the source code to this test to verify that we
      // we have something that matches exactly
      std::string replica_text = define_replica_pipeline(ctx, {in}, {out});
      LOG_REPLICA_TEXT(replica_text);
      size_t pos = replica_pipeline_test_src.find(replica_text);
      ASSERT_NE(pos, std::string::npos) << "Matching replica text not found, expected:\n" << replica_text;

      // Run the pipeline
      const int N = 10;

      buffer<int, 1> in_buf({N});
      in_buf.allocate();
      for (int i = 0; i < N; ++i) {
        in_buf(i) = i;
      }

      buffer<int, 1> out_buf({N});
      out_buf.allocate();

      {
        const raw_buffer* inputs[] = {&in_buf};
        const raw_buffer* outputs[] = {&out_buf};
        test_context eval_ctx;
        p.evaluate(inputs, outputs, eval_ctx);
      }

      buffer<int, 1> out_buf_replica({N});
      out_buf_replica.allocate();

      {
        const raw_buffer* inputs[] = {&in_buf};
        const raw_buffer* outputs[] = {&out_buf_replica};
        test_context eval_ctx;
        p_replica.evaluate(inputs, outputs, eval_ctx);
      }

      for (int i = 0; i < N; ++i) {
        ASSERT_EQ(out_buf(i), out_buf_replica(i));
      }
    }
  }
}

// clang-format off
static std::function<pipeline()> kMatmulReplicas[2][2] = {
  {
    {
// split = 0, lm = serial
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
  auto p = build_pipeline(ctx, {}, {a, b, c}, {abc}, {});
  return p;
}
// END define_replica_pipeline() output
    },
    {
// split = 1, lm = serial
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
  auto p = build_pipeline(ctx, {}, {a, b, c}, {abc}, {});
  return p;
}
// END define_replica_pipeline() output
    },
  },
  {
    {
// split = 0, lm = parallel
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
    },
    {
// split = 1, lm = parallel
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
  _fn_0.loops({{i, 1, loop_mode::parallel}});
  auto p = build_pipeline(ctx, {}, {a, b, c}, {abc}, {});
  return p;
}
// END define_replica_pipeline() output
    },
  },
};
// clang-format on

template <typename T>
index_t xor_hash_matmul(const buffer<const T>& a, const buffer<const T>& b, const buffer<T>& c) {
  const T value = 0;
  for (index_t i = c.dim(0).begin(); i < c.dim(0).end(); ++i) {
    for (index_t j = c.dim(1).begin(); j < c.dim(1).end(); ++j) {
      c(i, j) = value;
      for (index_t k = a.dim(1).begin(); k < a.dim(1).end(); ++k) {
        c(i, j) = c(i, j) ^ a(i, k) ^ b(k, j);
      }
    }
  }
  return 0;
}

// Two matrix multiplies: D = (A x B) x C.
TEST_F(ReplicaPipelineTest, matmul) {
  for (int split : {0, 1}) {
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
      func matmul_ab = func::make(xor_hash_matmul<int>, {{a, {point(i), K_ab}}, {b, {K_ab, point(j)}}}, {{ab, {i, j}}});
      func matmul_abc =
          func::make(xor_hash_matmul<int>, {{ab, {point(i), K_abc}}, {c, {K_abc, point(j)}}}, {{abc, {i, j}}});

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
      pipeline p_replica = kMatmulReplicas[split][static_cast<int>(lm)]();

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

      {
        const raw_buffer* inputs[] = {&a_buf, &b_buf, &c_buf};
        const raw_buffer* outputs[] = {&abc_buf};
        test_context eval_ctx;
        p.evaluate(inputs, outputs, eval_ctx);
      }

      buffer<int, 2> abc_buf_replica({N, M});
      std::swap(abc_buf_replica.dim(1), abc_buf_replica.dim(0));
      abc_buf_replica.allocate();

      {
        const raw_buffer* inputs[] = {&a_buf, &b_buf, &c_buf};
        const raw_buffer* outputs[] = {&abc_buf_replica};
        test_context eval_ctx;
        p_replica.evaluate(inputs, outputs, eval_ctx);
      }

      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
          ASSERT_EQ(abc_buf(j, i), abc_buf_replica(j, i));
        }
      }
    }
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

  func downsample =
      func::make(downsample2x, {{in, {2 * x + bounds(0, 1), 2 * y + bounds(0, 1)}}}, {{intm, {x, y}}});
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

  {
    const raw_buffer* inputs[] = {&in_buf};
    const raw_buffer* outputs[] = {&out_buf};
    test_context eval_ctx;
    p.evaluate(inputs, outputs, eval_ctx);
  }

  buffer<int, 2> out_buf_replica({W, H});
  out_buf_replica.allocate();

  {
    const raw_buffer* inputs[] = {&in_buf};
    const raw_buffer* outputs[] = {&out_buf_replica};
    test_context eval_ctx;
    p_replica.evaluate(inputs, outputs, eval_ctx);
  }
}

}  // namespace slinky
