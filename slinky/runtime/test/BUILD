load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:private"],
)

cc_test(
    name = "buffer",
    srcs = ["buffer.cc"],
    deps = [
        "//slinky/base/test:util",
        "//slinky/runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "depends_on",
    srcs = ["depends_on.cc"],
    deps = [
        "//slinky/runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "evaluate",
    srcs = ["evaluate.cc"],
    deps = [
        "//slinky/base",
        "//slinky/base:thread_pool_impl",
        "//slinky/runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "buffer_benchmark",
    srcs = ["buffer_benchmark.cc"],
    deps = [
        "//slinky/runtime",
        "@google_benchmark//:benchmark_main",
    ],
    args=["--benchmark_min_time=1x"],
    size = "small",
)

cc_test(
    name = "evaluate_benchmark",
    srcs = ["evaluate_benchmark.cc"],
    deps = [
        "//slinky/runtime",
        "//slinky/base:thread_pool_impl",
        "@google_benchmark//:benchmark_main",
    ],
    args=["--benchmark_min_time=1x"],
    size = "small",
)
