slinky_add_library(base_test_util
    SOURCES cmake_util.cc  # CMake stub replacing bazel_util.cc
    HEADERS
        bazel_util.h
        seeded_test.h
    LINK PUBLIC slinky::base GTest::gtest
)
target_compile_definitions(slinky_base_test_util
    PRIVATE SLINKY_SOURCE_DIR="${slinky_SOURCE_DIR}"
)

# --- Unit tests ---

slinky_add_test(arithmetic LINK slinky::base slinky::base_test_util)
slinky_add_test(modulus_remainder LINK slinky::base)
slinky_add_test(set LINK slinky::base)
slinky_add_test(thread_pool LINK slinky::thread_pool_impl)

# --- Benchmarks ---

slinky_add_test(arithmetic BENCHMARK
    SOURCES arithmetic_benchmark.cc
    LINK slinky::base
)
slinky_add_test(thread_pool BENCHMARK
    SOURCES thread_pool_benchmark.cc
    LINK slinky::thread_pool_impl
)
slinky_add_test(atomic_wait BENCHMARK
    SOURCES atomic_wait_benchmark.cc
    LINK slinky::base
)
