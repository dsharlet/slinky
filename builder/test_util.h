#ifndef SLINKY_BUILDER_TEST_UTIL_H
#define SLINKY_BUILDER_TEST_UTIL_H

#include <string>

#include <gtest/gtest.h>

namespace slinky {

template <typename T>
std::string str_join(const std::string&, const T& t) {
  using std::to_string;
  return to_string(t);
}

template <typename T, typename... Ts>
std::string str_join(const std::string& separator, const T& t, const Ts&... ts) {
  using std::to_string;
  return to_string(t) + separator + str_join(separator, ts...);
}

template <typename T, std::size_t... Is>
std::string test_params_to_string_impl(const T& t, std::index_sequence<Is...>) {
  return str_join("_", std::get<Is>(t)...);
}

template <typename T>
std::string test_params_to_string(const testing::TestParamInfo<T>& info) {
  constexpr std::size_t n = std::tuple_size<T>();
  return test_params_to_string_impl(info.param, std::make_index_sequence<n>());
}

std::string read_entire_file(const std::string& pathname) {
  std::ifstream f(pathname, std::ios::in | std::ios::binary);
  std::string result;

  f.seekg(0, std::ifstream::end);
  size_t size = f.tellg();
  result.resize(size);
  f.seekg(0, std::ifstream::beg);
  f.read(result.data(), result.size());
  if (!f.good()) {
    std::cerr << "Unable to read file: " << pathname;
    std::abort();
  }
  f.close();
  return result;
}

std::string read_entire_runfile(const std::string& rlocation) {
  return read_entire_file(get_bazel_file_path(rlocation));
}

std::string remove_windows_newlines(std::string s) {
  s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
  return s;
}

}  // namespace slinky

#endif  // SLINKY_BUILDER_TEST_UTIL_H