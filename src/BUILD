package(
    default_visibility = ["//visibility:private"],
)

cc_library(
    name = "runtime",
    srcs = [
        "buffer.cc",
    ],
    hdrs = [
        "arithmetic.h", 
        "buffer.h",
        "span.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "slinky",
    srcs = [
        "evaluate.cc",
        "expr.cc",
        "infer_bounds.cc",
        "node_mutator.cc",
        "optimizations.cc",
        "pipeline.cc",
        "print.cc",
        "simplify.cc",
        "substitute.cc",
    ],
    hdrs = [
        "arithmetic.h",
        "evaluate.h",
        "expr.h",
        "infer_bounds.h",
        "node_mutator.h",
        "optimizations.h",
        "pipeline.h",
        "print.h",
        "ref_count.h",
        "simplify.h",
        "span.h",
        "substitute.h",
        "symbol_map.h",
    ],
    deps = [":runtime"],
    visibility = ["//visibility:public"],
)