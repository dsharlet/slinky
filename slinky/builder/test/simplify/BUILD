load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:private"],
)

cc_test(
    name = "simplify",
    srcs = [
        "expr_generator.h",
        "rule_tester.h",
        "simplify.cc", 
    ] + glob(["*rules.cc"]),
    deps = [
        "//slinky/base",
        "//slinky/base/test:util",
        "//slinky/builder",
        "//slinky/runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "expr_helpers",
    srcs = [
        "expr_generator.h",
        "expr_helpers.cc",
    ],
    deps = [
        "//slinky/base",
        "//slinky/base/test:util",
        "//slinky/builder",
        "//slinky/runtime",
        "@googletest//:gtest_main",
    ],
    size = "small",
)
