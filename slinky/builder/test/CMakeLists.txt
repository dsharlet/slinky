slinky_add_library(builder_test_util
    SOURCES
        context.cc
        util.cc
    HEADERS
        context.h
        funcs.h
        util.h
    LINK
        PUBLIC
        slinky::runtime
        slinky::visualize
        slinky::chrome_trace
        slinky::thread_pool_impl
        slinky::base_test_util
        GTest::gmock
        GTest::gtest
)

set(common slinky::builder_test_util slinky::builder slinky::runtime)

slinky_add_test(checks LINK ${common})
slinky_add_test(copy LINK ${common} slinky::base)
slinky_add_test(elementwise LINK ${common} slinky::base_test_util)
slinky_add_test(pipeline LINK ${common} slinky::base_test_util)
slinky_add_test(copy_pipeline LINK ${common} slinky::base_test_util)
slinky_add_test(pyramid LINK ${common} slinky::base_test_util)
slinky_add_test(softmax LINK ${common} slinky::base_test_util slinky::visualize)
slinky_add_test(l2_norm LINK ${common} slinky::base_test_util)
slinky_add_test(aligned_producer LINK ${common})
slinky_add_test(tile_area LINK ${common} slinky::thread_pool_impl)
slinky_add_test(auto_stack_threshold LINK ${common})
slinky_add_test(cannot_alias LINK ${common})
slinky_add_test(substitute LINK slinky::builder slinky::runtime GTest::gmock)
slinky_add_test(optimizations LINK slinky::builder slinky::runtime GTest::gmock)
slinky_add_test(stencil LINK ${common})
slinky_add_test(stencil_pipeline LINK ${common} slinky::base_test_util)
slinky_add_test(rewrite LINK slinky::builder GTest::gmock)

add_subdirectory(simplify)
