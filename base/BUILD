load("@rules_cc//cc:cc_library.bzl", "cc_library")

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

config_setting(
    name = "use_absl_synchronization_true",
    define_values = {"slinky_use_absl_synchronization": "true"},
)

config_setting(
    name = "use_absl_synchronization_false",
    define_values = {"slinky_use_absl_synchronization": "false"},
)

alias(
    name = "use_absl_synchronization",
    actual = select({
        ":use_absl_synchronization_true": ":use_absl_synchronization_true",
        ":use_absl_synchronization_false": ":use_absl_synchronization_true",
        "//conditions:default": ":use_absl_synchronization_true",
    })
)

cc_library(
    name = "thread_pool_impl",
    srcs = ["thread_pool_impl.cc"],
    hdrs = [
        "function_ref.h",
        "ref_count.h",
        "thread_pool_impl.h",
        "thread_synchronization.h",
        "util.h"
    ],
    defines = select({
        ":use_absl_synchronization": ["SLINKY_USE_ABSL_SYNCHRONIZATION"],
        "//conditions:default": [],
    }),
    deps = [":thread_pool"] + select({
        ":use_absl_synchronization": ["@abseil-cpp//absl/synchronization"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "chrome_trace",
    srcs = ["chrome_trace.cc"],
    hdrs = ["chrome_trace.h"],
    visibility = ["//visibility:public"],
)
