# From https://github.com/erenon/bazel_clang_tidy
#
# To use: bazel build //... --config clang-tidy
#
# Assumes clang-tidy-17 is in PATH

load(
    "@bazel_tools//tools/build_defs/repo:git.bzl",
    "git_repository",
)

git_repository(
       name = "bazel_clang_tidy",
       commit = "43bef6852a433f3b2a6b001daecc8bc91d791b92",
       remote = "https://github.com/erenon/bazel_clang_tidy.git",
)
