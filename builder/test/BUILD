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
        "funcs.h",
        "util.h",
    ],
    data = ["replica_pipeline.cc"] + glob(["visualize/*.html"]),
    deps = [
        "//builder",
        "//builder:replica_pipeline",
        "//runtime",
        "//runtime:chrome_trace",
        "//runtime:thread_pool",
        "//runtime:visualize",
        "@bazel_tools//tools/cpp/runfiles",
        "@googletest//:gtest_main",
    ],
    size="small",
)

cc_test(
    name = "copy_pipeline",
    srcs = [
        "copy_pipeline.cc", 
        "bazel_util.h",
        "funcs.h",
        "util.h",
    ],
    data = ["replica_pipeline.cc"],
    deps = [
        "//builder",
        "//builder:replica_pipeline",
        "//runtime",
        "@bazel_tools//tools/cpp/runfiles",
        "@googletest//:gtest_main",
    ],
    size="small",
)

cc_test(
    name = "pyramid",
    srcs = [
        "pyramid.cc", 
        "bazel_util.h",
        "funcs.h",
        "util.h",
    ],
    data = ["replica_pipeline.cc"],
    deps = [
        "//builder",
        "//builder:replica_pipeline",
        "//runtime",
        "//runtime:thread_pool",
        "@bazel_tools//tools/cpp/runfiles",
        "@googletest//:gtest_main",
    ],
    size="small",
)

cc_test(
    name = "aligned_producer",
    srcs = [
        "aligned_producer.cc",
        "funcs.h",
        "util.h",
    ],
    deps = [
        "//builder",
        "//runtime",
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