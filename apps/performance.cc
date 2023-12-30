#include "pipeline.h"
#include "benchmark.h"

#include <cstdlib>
#include <cassert>
#include <iostream>

using namespace slinky;

// Copy from input to output.
// TODO: We should be able to just do this with raw_buffer and not make it a template.
template <typename T>
index_t copy(const buffer<const T>& in, const buffer<T>& out) {
  const T* src = &in(out.dim(0).min(), out.dim(1).min());
  T* dst = &out(out.dim(0).min(), out.dim(1).min());
  std::size_t size = out.dim(0).extent() * out.elem_size;
  for (index_t y = out.dim(1).begin(); y < out.dim(1).end(); ++y) {
    std::copy(src, src + size, dst);
    dst += out.dim(1).stride_bytes();
    src += in.dim(1).stride_bytes();
  }
  return 0;
}

pipeline make_pipeline(bool explicit_y) {
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

  return p;
}

int main(int argc, const char** argv) {
  pipeline loop = make_pipeline(true);
  pipeline no_loop = make_pipeline(false);

  const int total_sizes[] = {32, 128, 512, 2048, 4096, 8192};
  const int copy_sizes[] = {1, 2, 4, 8, 16, 32};

  std::cout << std::endl;
  for (int total_size : total_sizes) {
    std::cout << "total size (KB): " << total_size << std::endl;
    total_size *= 1024;

    std::cout << "| copy size (KB) | loop (GB/s) | no loop (GB/s) | ratio |" << std::endl;
    std::cout << "|----------------|-------------|----------------|-------|" << std::endl;
    for (int copy_size : copy_sizes) {
      std::cout << "| " << copy_size << " | ";
      copy_size *= 1024;

      if (total_size < copy_size) continue;

      buffer<char, 2> in_buf({copy_size, total_size / copy_size});
      buffer<char, 2> out_buf({copy_size, total_size / copy_size});
      in_buf.allocate();
      out_buf.allocate();
      for (index_t i = 0; i < total_size; ++i) {
        in_buf.base()[i] = rand() % 64;
      }

      const raw_buffer* inputs[] = {&in_buf};
      const raw_buffer* outputs[] = {&out_buf};

      memset(out_buf.base(), 0, total_size);
      double loop_t = benchmark([&]() { loop.evaluate(inputs, outputs); });
      assert(memcmp(out_buf.base(), in_buf.base(), total_size) == 0);
      std::cout << total_size / (loop_t * 1e9) << " | ";

      memset(out_buf.base(), 0, total_size);
      double no_loop_t = benchmark([&]() { no_loop.evaluate(inputs, outputs); });
      assert(memcmp(out_buf.base(), in_buf.base(), total_size) == 0);
      std::cout << total_size / (no_loop_t * 1e9) << " | ";

      std::cout << no_loop_t / loop_t << " | " << std::endl;
    }
    std::cout << std::endl;
  }

  return 0;
}
