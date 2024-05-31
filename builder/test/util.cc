#include "builder/test/util.h"

#include <fstream>

#include "base/test/bazel_util.h"
#include "runtime/visualize.h"

namespace slinky {

std::string remove_windows_newlines(std::string s) {
  s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
  return s;
}

std::string read_entire_file(const std::string& pathname) {
  std::ifstream f(pathname);
  std::stringstream buffer;
  buffer << f.rdbuf();
  return remove_windows_newlines(buffer.str());
}

void check_visualize(const std::string& filename, const pipeline& p, pipeline::buffers inputs,
    pipeline::buffers outputs, const node_context* ctx) {
  // return ;
  std::stringstream viz_stream;
  visualize(viz_stream, p, inputs, outputs, ctx);
  std::string viz = viz_stream.str();

  std::string golden_path = get_bazel_file_path("builder/test/visualize/" + filename);
  if (is_bazel_test()) {
    std::string golden = read_entire_file(golden_path);
    ASSERT_FALSE(golden.empty());
    // If this check fails, and you believe the changes to the visualization is correct, run this
    // test outside of bazel from the root of the repo to update the golden files.
    ASSERT_TRUE(golden == viz);
  } else {
    std::ofstream file(golden_path);
    file << viz;
  }
}

std::string get_replica_golden() {
  static std::string golden = read_entire_file(get_bazel_file_path("builder/test/replica_pipeline.cc"));
  return golden;
}

void check_replica_pipeline(const std::string& replica_text) {
  size_t pos = get_replica_golden().find(replica_text);
  ASSERT_NE(pos, std::string::npos) << "Matching replica text not found, expected:\n" << replica_text;
}

}  // namespace slinky
