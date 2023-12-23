#include "expr.h"
#include "pipeline.h"
#include "print.h"
#include "test.h"

#include <cassert>

using namespace slinky;

index_t copy(const buffer<const char>& in, const buffer<char>& out) {
  const char* src = &in(out.dims[0].min, out.dims[1].min);
  char* dst = &out(out.dims[0].min, out.dims[1].min);
  std::size_t size = out.dims[0].extent * out.elem_size;
  for (int y = out.dims[1].begin(); y < out.dims[1].end(); ++y) {
    std::copy(src, src + size, dst);
    dst += out.dims[1].stride_bytes;
    src += in.dims[1].stride_bytes;
  }
  return 0;
}

// A trivial pipeline with one stage.
TEST(performance_copy) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(char), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(char), 2);

  expr x = make_variable(ctx, "x");
  expr y = make_variable(ctx, "y");

  func copy = func::make<const char, char>(::copy, {in, {interval(x), interval(y)}}, {out, {x, y}});

  copy.loops({y});

  pipeline p(ctx, {in}, {out});

  for (int total_size : {128 * 1024, 8 * 1024 * 1024}) {
    for (int copy_size : {4096, 32 * 1024}) {
      buffer<char, 2> in_buf({copy_size, total_size / copy_size});
      in_buf.allocate();
      for (index_t i = 0; i < total_size; ++i) {
        in_buf.base()[i] = rand() % 64;
      }

      buffer<char, 2> out_buf({copy_size, total_size / copy_size});
      out_buf.allocate();
      memset(out_buf.base(), 0, total_size);

      // For the reference time, just call copy directly.
      double reference = benchmark([&]() { ::copy(in_buf.cast<const char>(), out_buf.cast<char>()); });
      for (size_t i = 0; i < total_size; ++i) {
        ASSERT_EQ(out_buf.base()[i], in_buf.base()[i]);
      }

      memset(out_buf.base(), 0, total_size);
      const buffer_base* inputs[] = {&in_buf};
      const buffer_base* outputs[] = {&out_buf};
      double slinky = benchmark([&]() { p.evaluate(inputs, outputs); });

      for (size_t i = 0; i < total_size; ++i) {
        ASSERT_EQ(out_buf.base()[i], in_buf.base()[i]);
      }

      std::cout << total_size << " " << copy_size << ", slinky: " << total_size / (slinky * 1e9)
                << "GB/s, reference: " << total_size / (reference * 1e9) << "GB/s" << std::endl;
    }
  }
}
