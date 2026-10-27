// CMake build stub for bazel_util — Bazel runfiles are not available here.
// is_bazel_test() always returns false, and get_bazel_file_path() resolves
// paths against the source tree, so tests can run from the build tree.
#include "slinky/base/test/bazel_util.h"

namespace slinky {

bool is_bazel_test() { return false; }

std::string get_bazel_file_path(const std::string& repo_path) { return SLINKY_SOURCE_DIR "/" + repo_path; }

}  // namespace slinky
