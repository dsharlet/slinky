#include "apps/benchmark.h"
#include "builder/pipeline.h"
#include "runtime/pipeline.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <iomanip>
#include <iostream>

namespace slinky {

// Unfortunately, here in 2024 on modern OSes, the standard `memcpy` is over 10x slower than this on AMD CPUs. Over a
// certain size, `memcpy` uses a `rep movsb` sequence, which is apparently really bad on AMD Zen:
// https://bugs.launchpad.net/ubuntu/+source/glibc/+bug/2030515
// We can work around this by memcpying 2048 bytes or less at a time.
void memcpy_workaround(char* dst, const char* src, std::size_t size) {
  constexpr std::size_t chunk_size = 2048;
  for (std::size_t i = 0; i < size; i += chunk_size) {
    std::size_t size_i = std::min(size - i, chunk_size);
    memcpy(dst, src, size_i);
    dst += chunk_size;
    src += chunk_size;
  }
}

// Copy from input to output.
// TODO: We should be able to just do this with raw_buffer and not make it a template.
template <typename T>
index_t copy_2d(const buffer<const T>& in, const buffer<T>& out) {
  const T* src = &in(out.dim(0).min(), out.dim(1).min());
  T* dst = &out(out.dim(0).min(), out.dim(1).min());
  std::size_t size_bytes = out.dim(0).extent() * out.elem_size;
  for (index_t y = out.dim(1).begin(); y < out.dim(1).end(); ++y) {
    memcpy_workaround((char*)dst, (const char*)src, size_bytes);
    dst += out.dim(1).stride();
    src += in.dim(1).stride();
  }
  return 0;
}

pipeline make_pipeline(bool explicit_y) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(char), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(char), 2);

  var x(ctx, "x");
  var y(ctx, "y");

  func copy = func::make(copy_2d<char>, {{in, {point(x), point(y)}}}, {{out, {x, y}}});

  if (explicit_y) {
    copy.loops({y});
  }

  pipeline p = build_pipeline(ctx, {in}, {out}, build_options{.no_checks = true});

  return p;
}

}  // namespace slinky

int main(int argc, const char** argv) {
  using namespace slinky;

  pipeline loop = make_pipeline(true);
  pipeline no_loop = make_pipeline(false);

  const int total_sizes[] = {32, 128, 512, 2048, 8192};
  const int copy_sizes[] = {1, 2, 4, 8, 16, 32};

  std::cout << std::endl;
  for (int total_size : total_sizes) {
    std::cout << "### " << total_size << " KB" << std::endl;
    total_size *= 1024;

    std::cout << "| copy size (KB) | loop (GB/s) | no loop (GB/s) | ratio |" << std::endl;
    std::cout << "|----------------|-------------|----------------|-------|" << std::endl;
    for (int copy_size : copy_sizes) {
      std::cout << "| " << std::setw(14) << copy_size << " | ";
      copy_size *= 1024;

      if (total_size < copy_size) continue;

      buffer<char, 2> in_buf({copy_size, total_size / copy_size});
      buffer<char, 2> out_buf({copy_size, total_size / copy_size});
      in_buf.allocate();
      out_buf.allocate();
      for (index_t i = 0; i < total_size; ++i) {
        in_buf.base()[i] = rand() % 64;
      }

      eval_context ctx;

      const raw_buffer* inputs[] = {&in_buf};
      const raw_buffer* outputs[] = {&out_buf};

      memset(out_buf.base(), 0, total_size);
      double loop_t = benchmark([&]() { loop.evaluate(inputs, outputs, ctx); });
      assert(memcmp(out_buf.base(), in_buf.base(), total_size) == 0);
      std::cout << std::setw(11) << total_size / (loop_t * 1e9) << " | ";

      memset(out_buf.base(), 0, total_size);
      double no_loop_t = benchmark([&]() { no_loop.evaluate(inputs, outputs, ctx); });
      assert(memcmp(out_buf.base(), in_buf.base(), total_size) == 0);
      std::cout << std::setw(14) << total_size / (no_loop_t * 1e9) << " | ";

      std::cout << std::setw(5) << std::setprecision(3) << std::fixed << no_loop_t / loop_t << " | " << std::endl;
    }
    std::cout << std::endl;
  }

  return 0;
}
