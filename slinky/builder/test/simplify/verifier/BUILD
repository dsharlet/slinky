load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:private"],
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
