#include "expr.h"
#include "pipeline.h"
#include "print.h"
#include "test.h"
#include "funcs.h"

#include <cassert>

using namespace slinky;

double benchmark_pipeline(const pipeline& p, index_t total_size, index_t copy_size) {
  buffer<char, 2> in_buf({copy_size, total_size / copy_size});
  in_buf.allocate();
  for (index_t i = 0; i < total_size; ++i) {
    in_buf.base()[i] = rand() % 64;
  }

  buffer<char, 2> out_buf({copy_size, total_size / copy_size});
  out_buf.allocate();
  memset(out_buf.base(), 0, total_size);
  const buffer_base* inputs[] = {&in_buf};
  const buffer_base* outputs[] = {&out_buf};
  double t = benchmark([&]() { p.evaluate(inputs, outputs); });

  for (size_t i = 0; i < total_size; ++i) {
    ASSERT_EQ(out_buf.base()[i], in_buf.base()[i]);
  }

  return t;
}

void benchmark_pipelines(bool explicit_y) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(char), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(char), 2);

  expr x = make_variable(ctx, "x");
  expr y = make_variable(ctx, "y");

  func copy = func::make<const char, char>(::copy<char>, {in, {point(x), point(y)}}, {out, {x, y}});

  if (explicit_y) {
    copy.loops({y});
  }

  pipeline p(ctx, {in}, {out}, build_options{.no_checks = true});

  for (int total_size : {128 * 1024, 8 * 1024 * 1024}) {
    for (int copy_size : {4096}) {
      double t = benchmark_pipeline(p, total_size, copy_size);
      std::cout << total_size << " " << copy_size << ": " << total_size / (t * 1e9) << "GB/s" << std::endl;
    }
  }
}

// Benchmark copying a 2D buffer, where the loop over y is explicit.
TEST(performance_copy_explicit_y) { benchmark_pipelines(true); }
TEST(performance_copy_implicit_y) { benchmark_pipelines(false); }

