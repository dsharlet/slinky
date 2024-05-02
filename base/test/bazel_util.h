#ifndef SLINKY_BUILDER_BAZEL_UTIL_H
#define SLINKY_BUILDER_BAZEL_UTIL_H

#include <string>

#include "tools/cpp/runfiles/runfiles.h"

namespace slinky {

inline bool is_bazel_test() { return getenv("BAZEL_TEST") != nullptr; }

// Get the path to a file relative to the root of the repo in the current bazel invocation.
inline std::string get_bazel_file_path(const std::string& repo_path) {
  if (is_bazel_test()) {
    using bazel::tools::cpp::runfiles::Runfiles;

    std::string error;
    std::unique_ptr<Runfiles> runfiles(Runfiles::CreateForTest(BAZEL_CURRENT_REPOSITORY, &error));
    if (runfiles == nullptr) {
      std::cerr << "Can't find runfile directory: " << error;
      std::abort();
    }

    return runfiles->Rlocation("_main/" + repo_path);
  } else {
    // When running without bazel, just assume we're in the repo root.
    return repo_path;
  }
}

}  // namespace slinky

#endif