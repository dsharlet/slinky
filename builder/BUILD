package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:private"],
)

# This library provides the functionality to lower a graph of funcs into a pipeline suitable for
# evaluation in the runtime.
cc_library(
    name = "builder",
    srcs = [
        "pipeline.cc",
        "infer_bounds.cc",
        "node_mutator.cc",
        "optimizations.cc",
        "rewrite.cc",
        "rewrite.h",
        "simplify.cc",
        "simplify_bounds.cc",
        "simplify_exprs.cc",
        "substitute.cc",
    ],
    hdrs = [
        "pipeline.h",
        "infer_bounds.h",
        "node_mutator.h",
        "optimizations.h",
        "simplify.h",
        "substitute.h",
    ],
    deps = ["//runtime"],
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

cc_test(
    name = "checks_test",
    srcs = ["checks_test.cc"],
    deps = [
        ":builder",
        "@googletest//:gtest_main",
        "//runtime",
    ],
    size="small",
)

cc_test(
    name = "copy_test",
    srcs = ["copy_test.cc"],
    deps = [
        ":builder",
        "@googletest//:gtest_main",
        "//runtime",
    ],
    size="small",
)

cc_test(
    name = "elementwise_test",
    srcs = ["elementwise_test.cc"],
    deps = [
        ":builder",
        "@googletest//:gtest_main",
        "//runtime",
    ],
    size="small",
)

cc_test(
    name = "pipeline_test",
    srcs = [
        "pipeline_test.cc", 
        "bazel_util.h",
    ],
    data = ["replica_pipeline_test.cc"],
    deps = [
        ":builder",
        ":replica_pipeline",
        "@bazel_tools//tools/cpp/runfiles",
        "@googletest//:gtest_main",
        "//runtime",
        "//runtime:thread_pool",
        "//runtime:visualize",
    ],
    size="small",
)

cc_test(
    name = "replica_pipeline_test",
    srcs = ["replica_pipeline_test.cc"],
    deps = [
        ":builder",
        ":replica_pipeline",
        "@googletest//:gtest_main",
    ],
    size="small",
)

cc_test(
    name = "simplify_test",
    srcs = ["simplify_test.cc"],
    deps = [
        ":builder",
        "@googletest//:gtest_main",
        "//runtime",
    ],
    size="small",
)

cc_test(
    name = "substitute_test",
    srcs = ["substitute_test.cc"],
    deps = [
        ":builder",
        "@googletest//:gtest_main",
        "//runtime",
    ],
    size="small",
)