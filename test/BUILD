
cc_library(
    name = "thread_pool",
    srcs = ["thread_pool.cc"],
    hdrs = ["thread_pool.h"],
)

cc_test(
    name = "arithmetic",
    srcs = ["arithmetic.cc"],
    deps = [
        "//src:runtime",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "buffer",
    srcs = ["buffer.cc"],
    deps = [
        "//src:runtime",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "checks",
    srcs = ["checks.cc", "funcs.h"],
    deps = [
        "//src:runtime",
        "//src:slinky",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "copy",
    srcs = ["copy.cc", "funcs.h"],
    deps = [
        "//src:runtime",
        "//src:slinky",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "elementwise",
    srcs = ["elementwise.cc", "funcs.h"],
    deps = [
        "//src:runtime",
        "//src:slinky",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "evaluate",
    srcs = ["evaluate.cc"],
    deps = [
        ":thread_pool",
        "//src:runtime",
        "//src:slinky",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "pipeline",
    srcs = ["pipeline.cc", "funcs.h"],
    deps = [
        ":thread_pool",
        "//src:runtime",
        "//src:slinky",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "simplify",
    srcs = ["simplify.cc"],
    deps = [
        "//src:runtime",
        "//src:slinky",
        "@googletest//:gtest_main",
    ],
)