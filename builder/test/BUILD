package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:private"],
)

cc_library(
    name = "util",
    testonly = True,
    srcs = ["context.cc"],
    hdrs = [
        "context.h",
        "funcs.h",
        "util.h",
    ],
    deps = [
        "//runtime",
        "//base:chrome_trace",
        "//base:thread_pool",
        "//base/test:util",
        "@bazel_tools//tools/cpp/runfiles",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "checks",
    srcs = ["checks.cc"],
    deps = [
        ":util",
        "//builder",
        "//runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "copy",
    srcs = ["copy.cc"],
    deps = [
        ":util",
        "//builder",
        "//runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "elementwise",
    srcs = ["elementwise.cc"],
    deps = [
        "//builder",
        "//runtime",
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
        "//base/test:util",
        "//builder",
        "//builder:replica_pipeline",
        "//runtime",
        "//runtime:visualize",
        "@bazel_tools//tools/cpp/runfiles",  # Needed for -DBAZEL_CURRENT_REPOSITORY even though it's used transitively via util
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
        "//base/test:util",
        "//builder",
        "//builder:replica_pipeline",
        "//runtime",
        "@bazel_tools//tools/cpp/runfiles",  # Needed for -DBAZEL_CURRENT_REPOSITORY even though it's used transitively via util
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
        "//base/test:util",
        "//builder",
        "//builder:replica_pipeline",
        "//runtime",
        "@bazel_tools//tools/cpp/runfiles",  # Needed for -DBAZEL_CURRENT_REPOSITORY even though it's used transitively via util
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "softmax",
    srcs = ["softmax.cc"],
    deps = [
        ":util",
        "//base/test:util",
        "//builder",
        "//runtime",
        "//runtime:visualize",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "aligned_producer",
    srcs = ["aligned_producer.cc"],
    deps = [
        ":util",
        "//builder",
        "//runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "replica_pipeline",
    srcs = ["replica_pipeline.cc"],
    deps = [
        ":util",
        "//builder",
        "//builder:replica_pipeline",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "simplify",
    srcs = ["simplify.cc"],
    deps = [
        "//builder",
        "//runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "substitute",
    srcs = ["substitute.cc"],
    deps = [
        "//builder",
        "//runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)