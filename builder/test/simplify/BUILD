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
        "//base/test:util",
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