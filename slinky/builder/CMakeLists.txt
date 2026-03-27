add_library(slinky_builder
    pipeline.cc
    node_mutator.cc
    optimizations.cc
    simplify.cc
    simplify_bounds.cc
    simplify_exprs.cc
    slide_and_fold_storage.cc
    substitute.cc
)
target_link_libraries(slinky_builder PUBLIC
    slinky_base
    slinky_chrome_trace
    slinky_runtime
)

add_library(slinky_replica_pipeline
    replica_pipeline.cc
)
target_link_libraries(slinky_replica_pipeline PUBLIC slinky_runtime slinky_builder)

if(SLINKY_ENABLE_TESTS)
    add_subdirectory(test)
endif()
