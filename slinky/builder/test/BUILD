load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:private"],
)

cc_library(
    name = "util",
    testonly = True,
    srcs = [
        "context.cc", 
        "util.cc",
    ],
    hdrs = [
        "context.h",
        "funcs.h",
        "util.h",
    ],
    deps = [
        "//slinky/runtime",
        "//slinky/runtime:visualize",
        "//slinky/base:chrome_trace",
        "//slinky/base:thread_pool_impl",
        "//slinky/base/test:util",
        "//slinky/builder:replica_pipeline",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "checks",
    srcs = ["checks.cc"],
    deps = [
        ":util",
        "//slinky/builder",
        "//slinky/runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "copy",
    srcs = ["copy.cc"],
    deps = [
        ":util",
        "//slinky/base",
        "//slinky/builder",
        "//slinky/runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "elementwise",
    srcs = ["elementwise.cc"],
    deps = [
        "//slinky/base/test:util",
        "//slinky/builder",
        "//slinky/runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "pipeline",
    srcs = ["pipeline.cc"],
    data = ["replica_pipeline.cc"] + glob(["visualize/*.html"]),
    deps = [
        ":util",
        "//slinky/base/test:util",
        "//slinky/builder",
        "//slinky/builder:replica_pipeline",
        "//slinky/runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "copy_pipeline",
    srcs = ["copy_pipeline.cc"],
    data = ["replica_pipeline.cc"],
    deps = [
        ":util",
        "//slinky/base/test:util",
        "//slinky/builder",
        "//slinky/builder:replica_pipeline",
        "//slinky/runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "pyramid",
    srcs = ["pyramid.cc"],
    data = ["replica_pipeline.cc"],
    deps = [
        ":util",
        "//slinky/base/test:util",
        "//slinky/builder",
        "//slinky/builder:replica_pipeline",
        "//slinky/runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "softmax",
    srcs = ["softmax.cc"],
    data = glob(["visualize/*.html"]),
    deps = [
        ":util",
        "//slinky/base/test:util",
        "//slinky/builder",
        "//slinky/runtime",
        "//slinky/runtime:visualize",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "l2_norm",
    srcs = ["l2_norm.cc"],
    deps = [
        ":util",
        "//slinky/base/test:util",
        "//slinky/builder",
        "//slinky/runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "aligned_producer",
    srcs = ["aligned_producer.cc"],
    deps = [
        ":util",
        "//slinky/builder",
        "//slinky/runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "tile_area",
    srcs = ["tile_area.cc"],
    deps = [
        ":util",
        "//slinky/base:thread_pool_impl",
        "//slinky/builder",
        "//slinky/runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "auto_stack_threshold",
    srcs = ["auto_stack_threshold.cc"],
    deps = [
        ":util",
        "//slinky/builder",
        "//slinky/runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "cannot_alias",
    srcs = ["cannot_alias.cc"],
    deps = [
        ":util",
        "//slinky/builder",
        "//slinky/runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "replica_pipeline",
    srcs = ["replica_pipeline.cc"],
    deps = [
        ":util",
        "//slinky/builder",
        "//slinky/builder:replica_pipeline",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "substitute",
    srcs = ["substitute.cc"],
    deps = [
        "//slinky/builder",
        "//slinky/runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "optimizations",
    srcs = ["optimizations.cc"],
    deps = [
        "//slinky/builder",
        "//slinky/runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "stencil",
    srcs = ["stencil.cc"],
    deps = [
        ":util",
        "//slinky/builder",
        "//slinky/runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "stencil_pipeline",
    srcs = ["stencil_pipeline.cc"],
    deps = [
        ":util",
        "//slinky/base/test:util",
        "//slinky/builder",
        "//slinky/runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "rewrite",
    srcs = ["rewrite.cc"],
    deps = [
        "//slinky/builder",
        "@googletest//:gtest_main",
    ],
    size = "small",
)
