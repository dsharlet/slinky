package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:private"],
)

cc_binary(
    name = "memcpy",
    srcs = [
        "benchmark.h",
        "memcpy.cc", 
    ],
)

cc_binary(
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
