package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:private"],
)

cc_test(
    name = "checks",
    srcs = ["checks.cc"],
    deps = [
        "//builder",
        "//runtime",
        "@googletest//:gtest_main",
    ],
    size="small",
)

cc_test(
    name = "copy",
    srcs = ["copy.cc"],
    deps = [
        "//builder",
        "//runtime",
        "@googletest//:gtest_main",
    ],
    size="small",
)

cc_test(
    name = "elementwise",
    srcs = ["elementwise.cc"],
    deps = [
        "//builder",
        "//runtime",
        "@googletest//:gtest_main",
    ],
    size="small",
)

cc_test(
    name = "pipeline",
    srcs = [
        "pipeline.cc", 
        "bazel_util.h",
        "util.h",
    ],
    data = ["replica_pipeline.cc"] + glob(["builder/visualize/*.html"]),
    deps = [
        "//builder",
        "//builder:replica_pipeline",
        "//runtime",
        "//runtime:thread_pool",
        "//runtime:visualize",
        "@bazel_tools//tools/cpp/runfiles",
        "@googletest//:gtest_main",
    ],
    size="small",
)

cc_test(
    name = "replica_pipeline",
    srcs = ["replica_pipeline.cc"],
    deps = [
        "//builder",
        "//builder:replica_pipeline",
        "@googletest//:gtest_main",
    ],
    size="small",
)

cc_test(
    name = "simplify",
    srcs = ["simplify.cc"],
    deps = [
        "//builder",
        "//runtime",
        "@googletest//:gtest_main",
    ],
    size="small",
)

cc_test(
    name = "substitute",
    srcs = ["substitute.cc"],
    deps = [
        "//builder",
        "//runtime",
        "@googletest//:gtest_main",
    ],
    size="small",
)