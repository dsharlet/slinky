add_library(slinky_runtime
    buffer.cc
    depends_on.cc
    evaluate.cc
    expr_stmt.cc
    pipeline.cc
    print.cc
)
target_link_libraries(slinky_runtime PUBLIC
    slinky_base
    slinky_chrome_trace
    slinky_thread_pool
)

add_library(slinky_visualize
    visualize.cc
)
target_link_libraries(slinky_visualize PUBLIC slinky_runtime slinky_base)

if(SLINKY_ENABLE_TESTS)
    add_subdirectory(test)
endif()
