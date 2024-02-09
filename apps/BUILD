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
)

cc_test(
    name = "performance",
    srcs = [
        "benchmark.h",
        "performance.cc",
    ],
    deps = [
        "//builder", 
        "//runtime",
    ],
)
