add_library(slinky_base
    arithmetic.cc
)

add_library(slinky_thread_pool
    thread_pool.cc
)
target_link_libraries(slinky_thread_pool PUBLIC slinky_base)

add_library(slinky_thread_pool_impl
    thread_pool_impl.cc
)
target_link_libraries(slinky_thread_pool_impl PUBLIC slinky_thread_pool)

add_library(slinky_chrome_trace
    chrome_trace.cc
)

if(SLINKY_ENABLE_TESTS)
    add_subdirectory(test)
endif()
