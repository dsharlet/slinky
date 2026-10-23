slinky_add_library(builder
    SOURCES
        compact_nodes.cc
        node_mutator.cc
        optimizations.cc
        pipeline.cc
        simplify.cc
        simplify_bounds.cc
        simplify_exprs.cc
        slide_and_fold_storage.cc
        substitute.cc
    HEADERS
        node_mutator.h
        optimizations.h
        pipeline.h
        rewrite.h
        simplify.h
        simplify_rules.h
        slide_and_fold_storage.h
        substitute.h
    LINK
        PUBLIC
            slinky::base
            slinky::runtime
        PRIVATE slinky::chrome_trace
)

if (SLINKY_ENABLE_TESTS)
    add_subdirectory(test)
endif ()
