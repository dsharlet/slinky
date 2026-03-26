// CMake build stub for bazel_util — Bazel runfiles are not available here.
// is_bazel_test() always returns false, so get_bazel_file_path() falls back
// to returning the path relative to the working directory (repo root).
#include "slinky/base/test/bazel_util.h"

namespace slinky {

bool is_bazel_test() { return false; }

std::string get_bazel_file_path(const std::string& repo_path) { return repo_path; }

}  // namespace slinky
