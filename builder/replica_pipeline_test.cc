#include <gtest/gtest.h>

#include <cassert>
#include <fstream>
#include <functional>
#include <iostream>

#include "builder/pipeline.h"
#include "builder/replica_pipeline.h"
#include "runtime/expr.h"
#include "runtime/pipeline.h"
#include "runtime/thread_pool.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace slinky {

using bazel::tools::cpp::runfiles::Runfiles;

std::string read_entire_file(const std::string &pathname) {
  try {
    std::ifstream f(pathname, std::ios::in | std::ios::binary);
    std::string result;

    f.seekg(0, std::ifstream::end);
    size_t size = f.tellg();
    result.resize(size);
    f.seekg(0, std::ifstream::beg);
    f.read(result.data(), result.size());
    if (!f.good()) {
      std::cerr << "Unable to read file: " << pathname;
      abort();
    }
    f.close();
    return result;
  } catch (...) {
    std::cerr<<"HEY WAIT NOW\n";
    return "";
  }
}

class ReplicaPipelineTest : public testing::Test {
 protected:
  void SetUp() override {
    std::string error;
    runfiles.reset(Runfiles::CreateForTest(BAZEL_CURRENT_REPOSITORY, &error));
    if (runfiles == nullptr) {
      std::cerr << "Can't find runfile directory: " << error;
      abort();
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

template<typename T>
T output_fill_value() {
  T value;
  memset(&value, internal::kReplicaBufferFillValue, sizeof(value));
  return value;
}

template <typename T>
index_t xor_hash_trivial(const buffer<const T>& in, const buffer<T>& out) {
  const T value = output_fill_value<T>();
  fill(out, &value);
  for_each_index(out, [&](auto i) { out(i) ^= in(i); });
  return 0;
}

// clang-format off
static std::function<pipeline()> kTrivialReplicas[2][2] = {
  {
    {
// split = 0, lm = serial
[]() -> ::slinky::pipeline {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", sizeof(uint32_t), 1);
  auto out = buffer_expr::make(ctx, "out", sizeof(uint32_t), 1);
  auto x = var(ctx, "x");
  auto _replica_fn_1 =   [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const box_expr fin_bounds[] = {{point(x)}};
    const std::vector<var> fout_dims[] = {{x}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fin_bounds, fout_dims);
  };
  auto _fn_0 = func::make(std::move(_replica_fn_1), {{in, {point(x)}}}, {{out, {x}}});
  auto p = build_pipeline(ctx, {}, {in}, {out}, {});
  return p;
}
    },
    {
// split = 1, lm = serial
[]() -> ::slinky::pipeline {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", sizeof(uint32_t), 1);
  auto out = buffer_expr::make(ctx, "out", sizeof(uint32_t), 1);
  auto x = var(ctx, "x");
  auto _replica_fn_1 =   [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const box_expr fin_bounds[] = {{point(x)}};
    const std::vector<var> fout_dims[] = {{x}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fin_bounds, fout_dims);
  };
  auto _fn_0 = func::make(std::move(_replica_fn_1), {{in, {point(x)}}}, {{out, {x}}});
  _fn_0.loops({{x, 1, loop_mode::serial}});
  auto p = build_pipeline(ctx, {}, {in}, {out}, {});
  return p;
}
    },
  },
  {
    {
// split = 0, lm = parallel
[]() -> ::slinky::pipeline {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", sizeof(uint32_t), 1);
  auto out = buffer_expr::make(ctx, "out", sizeof(uint32_t), 1);
  auto x = var(ctx, "x");
  auto _replica_fn_1 =   [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const box_expr fin_bounds[] = {{point(x)}};
    const std::vector<var> fout_dims[] = {{x}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fin_bounds, fout_dims);
  };
  auto _fn_0 = func::make(std::move(_replica_fn_1), {{in, {point(x)}}}, {{out, {x}}});
  auto p = build_pipeline(ctx, {}, {in}, {out}, {});
  return p;
}
    },
    {
// split = 1, lm = parallel
[]() -> ::slinky::pipeline {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", sizeof(uint32_t), 1);
  auto out = buffer_expr::make(ctx, "out", sizeof(uint32_t), 1);
  auto x = var(ctx, "x");
  auto _replica_fn_1 =   [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const box_expr fin_bounds[] = {{point(x)}};
    const std::vector<var> fout_dims[] = {{x}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fin_bounds, fout_dims);
  };
  auto _fn_0 = func::make(std::move(_replica_fn_1), {{in, {point(x)}}}, {{out, {x}}});
  _fn_0.loops({{x, 1, loop_mode::parallel}});
  auto p = build_pipeline(ctx, {}, {in}, {out}, {});
  return p;
}
    },
  },
};
// clang-format on

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

        for (int i = 0; i < N; ++i) {
          ASSERT_EQ(out_buf(i), output_fill_value<int>() ^ in_buf(i));
        }
      }

      buffer<int, 1> out_buf_replica({N});
      out_buf_replica.allocate();

      {
        const raw_buffer* inputs[] = {&in_buf};
        const raw_buffer* outputs[] = {&out_buf_replica};
        test_context eval_ctx;
        p.evaluate(inputs, outputs, eval_ctx);

        for (int i = 0; i < N; ++i) {
          ASSERT_EQ(out_buf(i), out_buf_replica(i));
        }
      }
    }
  }
}

}  // namespace slinky
