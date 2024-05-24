package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:private"],
)

cc_library(
    name = "simplify_rules",
    hdrs = ["simplify_rules.h"],
    visibility = ["//builder:__subpackages__"],
)

# This library provides the functionality to lower a graph of funcs into a pipeline suitable for
# evaluation in the runtime.
cc_library(
    name = "builder",
    srcs = [
        "pipeline.cc",
        "node_mutator.cc",
        "optimizations.cc",
        "rewrite.h",
        "simplify.cc",
        "simplify_bounds.cc",
        "simplify_exprs.cc",
        "slide_and_fold_storage.cc",
        "substitute.cc",
    ],
    hdrs = [
        "pipeline.h",
        "node_mutator.h",
        "optimizations.h",
        "simplify.h",
        "slide_and_fold_storage.h",
        "substitute.h",
    ],
    deps = [
        ":simplify_rules",
        "//base",
        "//runtime",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "replica_pipeline",
    srcs = [
        "replica_pipeline.cc",
    ],
    hdrs = [
        "replica_pipeline.h",
    ],
    deps = ["//runtime", ":builder"],
    visibility = ["//visibility:public"],
)
