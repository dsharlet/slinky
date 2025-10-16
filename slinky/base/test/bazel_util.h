#ifndef SLINKY_BASE_TEST_BAZEL_UTIL_H
#define SLINKY_BASE_TEST_BAZEL_UTIL_H

#include <string>

namespace slinky {

bool is_bazel_test();

// Get the path to a file relative to the root of the repo in the current bazel invocation.
std::string get_bazel_file_path(const std::string& repo_path);

}  // namespace slinky

#endif  // SLINKY_BASE_TEST_BAZEL_UTIL_H
