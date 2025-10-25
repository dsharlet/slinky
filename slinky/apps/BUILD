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
