#ifndef SLINKY_BUILDER_TEST_UTIL_H
#define SLINKY_BUILDER_TEST_UTIL_H

#include <algorithm>
#include <numeric>
#include <string>

#include <gtest/gtest.h>

#include "slinky/runtime/pipeline.h"

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
  std::string result = test_params_to_string_impl(info.param, std::make_index_sequence<n>());
  std::replace(result.begin(), result.end(), '-', '_');
  return result;
}

std::string remove_windows_newlines(std::string s);

std::string read_entire_file(const std::string& pathname);

inline bool is_permutation(span<const int> p) {
  std::vector<int> unpermuted(p.size());
  std::iota(unpermuted.begin(), unpermuted.end(), 0);
  return std::is_permutation(p.begin(), p.end(), unpermuted.begin());
}

void check_visualize(const std::string& filename, const pipeline& p, pipeline::buffers inputs,
    pipeline::buffers outputs, const node_context* ctx);

std::string get_replica_golden();

void check_replica_pipeline(const std::string& replica_text);

}  // namespace slinky

#endif  // SLINKY_BUILDER_TEST_UTIL_H