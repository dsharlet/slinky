#include "apps/benchmark.h"

#include <cassert>
#include <cstring>
#include <iostream>

void copy_chunks(char* dst, const char* src, int total_size, int chunk_size) {
  for (int i = 0; i + chunk_size <= total_size; i += chunk_size) {
    memcpy(dst + i, src + i, chunk_size);
  }
}

int main(int argc, const char** argv) {
  const int total_sizes[] = {32, 128, 512, 2048, 8192};
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

      char* src = new char[total_size];
      char* dst = new char[total_size];
      memset(src, 0, total_size);

      memset(dst, 0, total_size);
      double loop_t = benchmark([&]() { copy_chunks(dst, src, total_size, copy_size); });
      assert_used(dst);
      assert(memcmp(src, dst, total_size) == 0);
      std::cout << total_size / (loop_t * 1e9) << " | ";

      memset(dst, 0, total_size);
      double no_loop_t = benchmark([&]() { copy_chunks(dst, src, total_size, total_size); });
      assert_used(dst);
      assert(memcmp(src, dst, total_size) == 0);
      std::cout << total_size / (no_loop_t * 1e9) << " | ";

      std::cout << no_loop_t / loop_t << " | " << std::endl;

      delete[] src;
      delete[] dst;
    }
    std::cout << std::endl;
  }

  return 0;
}
