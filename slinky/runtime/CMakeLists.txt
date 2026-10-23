slinky_add_library(runtime
    SOURCES
        buffer.cc
        depends_on.cc
        evaluate.cc
        expr_stmt.cc
        memory_pool.cc
        pipeline.cc
        print.cc
    HEADERS
        buffer.h
        depends_on.h
        evaluate.h
        expr.h
        memory_pool.h
        pipeline.h
        print.h
        stmt.h
    LINK
        PUBLIC
            slinky::base
            slinky::thread_pool
        PRIVATE slinky::chrome_trace
)

slinky_add_library(visualize
    SOURCES visualize.cc
    HEADERS visualize.h
    LINK PUBLIC slinky::runtime
)

if (SLINKY_ENABLE_TESTS)
    add_subdirectory(test)
endif ()
