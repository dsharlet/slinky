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
