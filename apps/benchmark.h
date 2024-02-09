#ifndef SLINKY_APPS_BENCHMARK_H
#define SLINKY_APPS_BENCHMARK_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

inline bool is_bazel_test() {
  const char* bazel_test = getenv("BAZEL_TEST");
  return bazel_test && strcmp(bazel_test, "1") == 0;
}

// Benchmark a call.
template <class F>
double benchmark(F op) {
  op();

  const int max_trials = 10;
  const double min_time_s = is_bazel_test() ? 0.0 : 0.5;
  double time_per_iteration_s = 0;
  long iterations = 1;
  for (int trials = 0; trials < max_trials; trials++) {
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < iterations; j++) {
      op();
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    time_per_iteration_s = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / (iterations * 1e9);
    if (time_per_iteration_s * iterations > min_time_s) {
      break;
    }

    long next_iterations = static_cast<long>(std::ceil((min_time_s * 2) / time_per_iteration_s));
    iterations = std::min(std::max(next_iterations, iterations), iterations * 10);
  }
  return time_per_iteration_s;
}

// Tricks the compiler into not stripping away dead objects.
template <class T>
__attribute__((noinline)) void assert_used(const T&) {}

// Tricks the compiler into not constant folding the result of x.
template <class T>
__attribute__((noinline)) T not_constant(T x) {
  return x;
}

#endif  // SLINKY_APPS_BENCHMARK_H
