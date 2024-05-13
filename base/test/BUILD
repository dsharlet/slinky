package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:private"],
)

cc_library(
    name = "util",
    srcs = ["bazel_util.cc"],
    hdrs = ["bazel_util.h"],
    deps = [
        "@bazel_tools//tools/cpp/runfiles",
        "@googletest//:gtest",
    ],
    testonly = True,
    visibility = ["//visibility:public"],
)

cc_test(
    name = "arithmetic",
    srcs = ["arithmetic.cc"],
    deps = [
        "//base",
        "@googletest//:gtest_main",
    ],
    size = "small",
)
