slinky_add_library(base
    SOURCES arithmetic.cc
    HEADERS
    allocator.h
    arithmetic.h
    atomic_wait.h
    function_ref.h
    modulus_remainder.h
    ref_count.h
    set.h
    span.h
    util.h
)

slinky_add_library(thread_pool
    SOURCES thread_pool.cc
    HEADERS thread_pool.h
    LINK PUBLIC slinky::base
)

slinky_add_library(thread_pool_impl
    SOURCES thread_pool_impl.cc
    HEADERS thread_pool_impl.h
    LINK PUBLIC slinky::thread_pool Threads::Threads
)

slinky_add_library(chrome_trace
    SOURCES chrome_trace.cc
    HEADERS chrome_trace.h
)

if (SLINKY_ENABLE_TESTS)
    add_subdirectory(test)
endif ()
