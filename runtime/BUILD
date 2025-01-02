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
        "expr_stmt.cc",
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
        "stmt.h",
    ],
    deps = [
        "//base",
        "//base:chrome_trace",
        "//base:thread_pool",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "visualize",
    srcs = ["visualize.cc"],
    hdrs = ["visualize.h"],
    deps = [":runtime"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "validate",
    srcs = ["validate.cc"],
    hdrs = ["validate.h"],
    deps = [
        "//base",
        ":runtime",
    ],
    visibility = ["//visibility:public"],
)
