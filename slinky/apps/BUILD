load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:private"],
)

cc_test(
    name = "memcpy",
    srcs = [
        "benchmark.h",
        "memcpy.cc", 
    ],
    deps = [
        "//slinky/base",
    ],
    size="small",
)

cc_test(
    name = "performance",
    srcs = [
        "benchmark.h",
        "performance.cc",
    ],
    deps = [
        "//slinky/base",
        "//slinky/builder", 
        "//slinky/runtime",
    ],
    size="small",
)

cc_binary(
    name = "simplifier_to_smt",
    srcs = ["simplifier_to_smt.cc"],
    # This is an offline tool whose runtime is negligible, and optimizing it made it by far the slowest compile.
    copts = ["-O0"],
    visibility = ["//visibility:public"],
    deps = [
        "//slinky/base",
        "//slinky/builder",
        "//slinky/runtime",
    ],
)
