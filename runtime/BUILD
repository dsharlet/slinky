package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:private"],
)

# This library contains the necessary functionality to evaluate pipelines.
cc_library(
    name = "runtime",
    srcs = [
        "buffer.cc",
        "depends_on.cc",
        "evaluate.cc",
        "expr.cc",
        "pipeline.cc",
        "print.cc",
    ],
    hdrs = [
        "buffer.h",
        "depends_on.h",
        "evaluate.h",
        "expr.h",
        "pipeline.h",
        "print.h",
        "util.h", 
    ],
    visibility = ["//visibility:public"],
)

# This thread pool is not directly used by the runtime, it is only used by tests in this package.
# It could be used in client code, but the expectation is that the client code implements the
# thread pool interface with some existing parallelism provider rather than using this implementation.
cc_library(
    name = "thread_pool",
    srcs = ["thread_pool.cc"],
    hdrs = ["thread_pool.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "visualize",
    srcs = ["visualize.cc"],
    hdrs = ["visualize.h"],
    deps = [":runtime"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "arithmetic_test",
    srcs = ["arithmetic_test.cc"],
    deps = [
        ":runtime",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "buffer_test",
    srcs = ["buffer_test.cc"],
    deps = [
        ":runtime",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "evaluate_test",
    srcs = ["evaluate_test.cc"],
    deps = [
        ":runtime",
        ":thread_pool",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "buffer_benchmark",
    srcs = ["buffer_benchmark.cc"],
    deps = [
        ":runtime",
        "@google_benchmark//:benchmark_main",
    ],
    testonly = 1,
)

cc_binary(
    name = "evaluate_benchmark",
    srcs = ["evaluate_benchmark.cc"],
    deps = [
        ":runtime",
        ":thread_pool",
        "@google_benchmark//:benchmark_main",
    ],
    testonly = 1,
)
