slinky_add_test(memcpy BENCHMARK
    SOURCES
        memcpy.cc
        benchmark.h
    LINK slinky::base
)

slinky_add_test(performance BENCHMARK
    SOURCES
        performance.cc
        benchmark.h
    LINK slinky::base slinky::builder slinky::runtime
)
