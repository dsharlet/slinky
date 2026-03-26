add_library(slinky_builder_test_util
    context.cc
    util.cc
)
target_link_libraries(slinky_builder_test_util PUBLIC
    slinky_runtime
    slinky_visualize
    slinky_chrome_trace
    slinky_thread_pool_impl
    slinky_base_test_util
    slinky_replica_pipeline
    GTest::gmock
    GTest::gtest
)
target_compile_features(slinky_builder_test_util PUBLIC cxx_std_20)

# Helper macro to reduce repetition.
# Tests that read source or golden files run from the repo root so that
# get_bazel_file_path() relative paths resolve correctly.
macro(add_builder_test name)
    add_executable(slinky_builder_${name}_test ${name}.cc)
    target_link_libraries(slinky_builder_${name}_test PRIVATE ${ARGN} GTest::gtest_main)
    target_compile_features(slinky_builder_${name}_test PRIVATE cxx_std_20)
    gtest_discover_tests(slinky_builder_${name}_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endmacro()

add_builder_test(checks
    slinky_builder_test_util slinky_builder slinky_runtime)

add_builder_test(copy
    slinky_builder_test_util slinky_base slinky_builder slinky_runtime)

add_builder_test(elementwise
    slinky_base_test_util slinky_builder slinky_runtime)

add_builder_test(pipeline
    slinky_builder_test_util slinky_base_test_util slinky_builder
    slinky_replica_pipeline slinky_runtime)

add_builder_test(copy_pipeline
    slinky_builder_test_util slinky_base_test_util slinky_builder
    slinky_replica_pipeline slinky_runtime)

add_builder_test(pyramid
    slinky_builder_test_util slinky_base_test_util slinky_builder
    slinky_replica_pipeline slinky_runtime)

add_builder_test(softmax
    slinky_builder_test_util slinky_base_test_util slinky_builder
    slinky_runtime slinky_visualize)

add_builder_test(l2_norm
    slinky_builder_test_util slinky_base_test_util slinky_builder slinky_runtime)

add_builder_test(aligned_producer
    slinky_builder_test_util slinky_builder slinky_runtime)

add_builder_test(tile_area
    slinky_builder_test_util slinky_thread_pool_impl slinky_builder slinky_runtime)

add_builder_test(auto_stack_threshold
    slinky_builder_test_util slinky_builder slinky_runtime)

add_builder_test(cannot_alias
    slinky_builder_test_util slinky_builder slinky_runtime)

add_builder_test(replica_pipeline
    slinky_builder_test_util slinky_builder slinky_replica_pipeline)

add_builder_test(substitute
    slinky_builder slinky_runtime GTest::gmock)

add_builder_test(optimizations
    slinky_builder slinky_runtime GTest::gmock)

add_builder_test(stencil
    slinky_builder_test_util slinky_builder slinky_runtime)

add_builder_test(stencil_pipeline
    slinky_builder_test_util slinky_base_test_util slinky_builder slinky_runtime)

add_builder_test(rewrite
    slinky_builder GTest::gmock)

add_subdirectory(simplify)
