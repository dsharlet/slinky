#include "test.h"
#include "expr.h"
#include "pipeline.h"
#include "print.h"

#include <cassert>

using namespace slinky;

// These functions use buffer<>::operator(), which is not designed to be fast.
// TODO: Maybe eliminate this helper entirely and move it to be only for tests.
index_t multiply_2(const buffer<const int>& in, const buffer<int>& out) {
  for (index_t i = out.dims[0].begin(); i < out.dims[0].end(); ++i) {
    out(i) = in(i) * 2;
  }
  return 0;
}

index_t add_1(const buffer<const int>& in, const buffer<int>& out) {
  for (index_t i = out.dims[0].begin(); i < out.dims[0].end(); ++i) {
    out(i) = in(i) + 1;
  }
  return 0;
}

// An example of two 1D elementwise operations in sequence.
TEST(pipeline_elementwise_1d) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 1);
  auto out = buffer_expr::make(ctx, "out", 1);
  auto intm = buffer_expr::make(ctx, "intm", 1);

  expr x = make_variable(ctx, "x");

  func mul = func::make<const int, int>(multiply_2, { in, {interval(x)} }, { intm, {x} });
  func add = func::make<const int, int>(add_1, { intm, {interval(x)} }, { out, {x} });

  pipeline p({ in }, { out });

  // Run the pipeline
  const int N = 10;

  buffer<int, 1> in_buf({ N });
  in_buf.allocate();
  for (int i = 0; i < N; ++i) {
    in_buf(i) = i;
  }

  buffer<int, 1> out_buf({ N });
  out_buf.allocate();

  //eval_context ctx;

  //p.evaluate(ctx);

  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(out_buf(i), 2 * i + 1);
  }
}

// This matrix multiply operates on integers, so we can test for correctness exactly.
index_t matmul(const buffer<const int>& a, const buffer<const int>& b, const buffer<int>& c) {
  for (index_t i = c.dims[0].begin(); i < c.dims[0].end(); ++i) {
    for (index_t j = c.dims[1].begin(); j < c.dims[1].end(); ++j) {
      c(i, j) = 0;
      for (index_t k = a.dims[1].begin(); k < a.dims[1].end(); ++k) {
        c(i, j) += a(i, k) * b(k, j);
      }
    }
  }
  return 0;
}

// Two matrix multiplies: D = (A x B) x C.
TEST(pipeline_matmuls) {
  // Make the pipeline
  node_context ctx;

  auto a = buffer_expr::make(ctx, "a", 2);
  auto b = buffer_expr::make(ctx, "b", 2);
  auto c = buffer_expr::make(ctx, "c", 2);
  auto d = buffer_expr::make(ctx, "d", 2);

  auto ab = buffer_expr::make(ctx, "ab", 2);

  expr i = make_variable(ctx, "i");
  expr j = make_variable(ctx, "j");
  expr k = make_variable(ctx, "k");

  // The bounds required of the dimensions consumed by the reduction depend on the size of the buffers passed in.
  // Note that we haven't used any constants yet.
  expr K_ab = a->dim(1).extent;
  expr K_d = c->dim(0).extent;

  func matmul_ab = func::make<const int, const int, int>(matmul, { a, { interval(i), interval(0, K_ab) } }, { b, {interval(0, K_ab), interval(j)} }, { ab, {i, j} });
  func matmul_abc = func::make<const int, const int, int>(matmul, { ab, { interval(i), interval(0, K_d) } }, { c, {interval(0, K_d), interval(j)} }, { d, {i, j} });

  // Run the pipeline.

}