# Tests use C++20 (matching per_file_copt in .bazelrc)
add_library(slinky_base_test_util
    cmake_util.cc  # CMake stub replacing bazel_util.cc
)
target_link_libraries(slinky_base_test_util PUBLIC slinky_base GTest::gtest)
target_compile_features(slinky_base_test_util PUBLIC cxx_std_20)

# --- Unit tests ---

add_executable(slinky_base_arithmetic_test arithmetic.cc)
target_link_libraries(slinky_base_arithmetic_test PRIVATE
    slinky_base slinky_base_test_util GTest::gtest_main)
target_compile_features(slinky_base_arithmetic_test PRIVATE cxx_std_20)
gtest_discover_tests(slinky_base_arithmetic_test)

add_executable(slinky_base_modulus_remainder_test modulus_remainder.cc)
target_link_libraries(slinky_base_modulus_remainder_test PRIVATE
    slinky_base GTest::gtest_main)
target_compile_features(slinky_base_modulus_remainder_test PRIVATE cxx_std_20)
gtest_discover_tests(slinky_base_modulus_remainder_test)

add_executable(slinky_base_set_test set.cc)
target_link_libraries(slinky_base_set_test PRIVATE
    slinky_base GTest::gtest_main)
target_compile_features(slinky_base_set_test PRIVATE cxx_std_20)
gtest_discover_tests(slinky_base_set_test)

add_executable(slinky_base_thread_pool_test thread_pool.cc)
target_link_libraries(slinky_base_thread_pool_test PRIVATE
    slinky_thread_pool_impl GTest::gtest_main)
target_compile_features(slinky_base_thread_pool_test PRIVATE cxx_std_20)
gtest_discover_tests(slinky_base_thread_pool_test)

# --- Benchmarks ---

add_executable(slinky_base_arithmetic_benchmark arithmetic_benchmark.cc)
target_link_libraries(slinky_base_arithmetic_benchmark PRIVATE
    slinky_base benchmark::benchmark_main)
target_compile_features(slinky_base_arithmetic_benchmark PRIVATE cxx_std_20)

add_executable(slinky_base_thread_pool_benchmark thread_pool_benchmark.cc)
target_link_libraries(slinky_base_thread_pool_benchmark PRIVATE
    slinky_thread_pool_impl benchmark::benchmark_main)
target_compile_features(slinky_base_thread_pool_benchmark PRIVATE cxx_std_20)

add_executable(slinky_base_atomic_wait_benchmark atomic_wait_benchmark.cc)
target_link_libraries(slinky_base_atomic_wait_benchmark PRIVATE
    slinky_base benchmark::benchmark_main)
target_compile_features(slinky_base_atomic_wait_benchmark PRIVATE cxx_std_20)
