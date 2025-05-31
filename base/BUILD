package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:private"],
)

cc_library(
    name = "base",
    hdrs = [
        "allocator.h",
        "arithmetic.h",
        "atomic_wait.h",
        "function_ref.h",
        "modulus_remainder.h",
        "ref_count.h",
        "set.h",
        "span.h",
        "util.h",
    ],
    srcs = [
        "arithmetic.cc",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "thread_pool",
    srcs = ["thread_pool.cc"],
    hdrs = [
        "function_ref.h",
        "ref_count.h",
        "thread_pool.h",
        "util.h"
    ],
    deps = [":base"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "thread_pool_impl",
    srcs = ["thread_pool_impl.cc"],
    hdrs = [
        "function_ref.h",
        "ref_count.h",
        "thread_pool_impl.h",
        "util.h"
    ],
    deps = [":thread_pool"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "chrome_trace",
    srcs = ["chrome_trace.cc"],
    hdrs = ["chrome_trace.h"],
    visibility = ["//visibility:public"],
)
