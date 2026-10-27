# --- Unit tests ---

slinky_add_test(buffer
    LINK slinky::runtime slinky::base_test_util GTest::gmock
)
slinky_add_test(depends_on LINK slinky::runtime GTest::gmock)
slinky_add_test(evaluate
    LINK slinky::base slinky::thread_pool_impl slinky::runtime
)
slinky_add_test(memory_pool LINK slinky::runtime GTest::gmock)

# --- Benchmarks ---

slinky_add_test(buffer BENCHMARK
    SOURCES buffer_benchmark.cc
    LINK slinky::runtime
)
slinky_add_test(evaluate BENCHMARK
    SOURCES evaluate_benchmark.cc
    LINK slinky::runtime slinky::thread_pool_impl
)
