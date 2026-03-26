# --- Unit tests ---

add_executable(slinky_runtime_buffer_test buffer.cc)
target_link_libraries(slinky_runtime_buffer_test PRIVATE
    slinky_runtime slinky_base_test_util GTest::gmock GTest::gtest_main)
target_compile_features(slinky_runtime_buffer_test PRIVATE cxx_std_20)
gtest_discover_tests(slinky_runtime_buffer_test)

add_executable(slinky_runtime_depends_on_test depends_on.cc)
target_link_libraries(slinky_runtime_depends_on_test PRIVATE
    slinky_runtime GTest::gmock GTest::gtest_main)
target_compile_features(slinky_runtime_depends_on_test PRIVATE cxx_std_20)
gtest_discover_tests(slinky_runtime_depends_on_test)

add_executable(slinky_runtime_evaluate_test evaluate.cc)
target_link_libraries(slinky_runtime_evaluate_test PRIVATE
    slinky_base slinky_thread_pool_impl slinky_runtime GTest::gtest_main)
target_compile_features(slinky_runtime_evaluate_test PRIVATE cxx_std_20)
gtest_discover_tests(slinky_runtime_evaluate_test)

# --- Benchmarks ---

add_executable(slinky_runtime_buffer_benchmark buffer_benchmark.cc)
target_link_libraries(slinky_runtime_buffer_benchmark PRIVATE
    slinky_runtime benchmark::benchmark_main)
target_compile_features(slinky_runtime_buffer_benchmark PRIVATE cxx_std_20)

add_executable(slinky_runtime_evaluate_benchmark evaluate_benchmark.cc)
target_link_libraries(slinky_runtime_evaluate_benchmark PRIVATE
    slinky_runtime slinky_thread_pool_impl benchmark::benchmark_main)
target_compile_features(slinky_runtime_evaluate_benchmark PRIVATE cxx_std_20)
