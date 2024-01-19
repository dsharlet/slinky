package(
    default_visibility = ["//visibility:private"],
)

cc_library(
    name = "runtime",
    srcs = [
        "buffer.cc",
    ],
    hdrs = [
        "buffer.h",
        "util.h", 
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
        "evaluate.h",
        "expr.h",
        "infer_bounds.h",
        "node_mutator.h",
        "optimizations.h",
        "pipeline.h",
        "print.h",
        "simplify.h",
        "substitute.h",
        "util.h", 
    ],
    deps = [":runtime"],
    visibility = ["//visibility:public"],
)