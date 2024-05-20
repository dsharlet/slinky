package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:private"],
)

cc_test(
    name = "buffer",
    srcs = ["buffer.cc"],
    deps = [
        "//base/test:util",
        "//runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "depends_on",
    srcs = ["depends_on.cc"],
    deps = [
        "//runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "evaluate",
    srcs = ["evaluate.cc"],
    deps = [
        "//runtime",
        "//base:thread_pool",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "buffer_benchmark",
    srcs = ["buffer_benchmark.cc"],
    deps = [
        "//runtime",
        "@google_benchmark//:benchmark_main",
    ],
    args=["--benchmark_min_time=0.1s"],
    size = "small",
)

cc_test(
    name = "evaluate_benchmark",
    srcs = ["evaluate_benchmark.cc"],
    deps = [
        "//runtime",
        "//base:thread_pool",
        "@google_benchmark//:benchmark_main",
    ],
    args=["--benchmark_min_time=0.1s"],
    size = "small",
)
