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
        "//runtime",
        "//runtime:visualize",
        "//base:chrome_trace",
        "//base:thread_pool",
        "//base/test:util",
        "//builder:replica_pipeline",
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
        "//base/test:util",
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
    name = "tile_area",
    srcs = ["tile_area.cc"],
    deps = [
        ":util",
        "//builder",
        "//runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "cannot_alias",
    srcs = ["cannot_alias.cc"],
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
    name = "substitute",
    srcs = ["substitute.cc"],
    deps = [
        "//builder",
        "//runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "optimizations",
    srcs = ["optimizations.cc"],
    deps = [
        "//builder",
        "//runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)
