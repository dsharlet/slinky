package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:private"],
)

# This library provides the functionality to lower a graph of funcs into a pipeline suitable for
# evaluation in the runtime.
cc_library(
    name = "builder",
    srcs = [
        "func.cc",
        "infer_bounds.cc",
        "node_mutator.cc",
        "optimizations.cc",
        "simplify.cc",
        "substitute.cc",
    ],
    hdrs = [
        "func.h",
        "infer_bounds.h",
        "node_mutator.h",
        "optimizations.h",
        "simplify.h",
        "substitute.h",
    ],
    deps = ["//runtime"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "checks_test",
    srcs = ["checks_test.cc"],
    deps = [
        ":builder",
        "//runtime",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "copy_test",
    srcs = ["copy_test.cc"],
    deps = [
        ":builder",
        "//runtime",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "elementwise_test",
    srcs = ["elementwise_test.cc"],
    deps = [
        ":builder",
        "//runtime",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "pipeline_test",
    srcs = ["pipeline_test.cc"],
    deps = [
        ":builder",
        "//runtime",
        "//runtime:thread_pool",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "simplify_test",
    srcs = ["simplify_test.cc"],
    deps = [
        ":builder",
        "//runtime",
        "@googletest//:gtest_main",
    ],
)