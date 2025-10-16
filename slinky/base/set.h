#ifndef SLINKY_BASE_SET_H
#define SLINKY_BASE_SET_H

#include <algorithm>
#include <set>

namespace slinky {

// The algorithm at https://en.cppreference.com/w/cpp/algorithm/set_intersection, but detects any intersection.
template <typename It>
bool empty_intersection(It a_begin, It a_end, It b_begin, It b_end) {
  It a = a_begin;
  It b = b_begin;
  while (a != a_end && b != b_end) {
    if (*a == *b) {
      return false;
    } else if (*a < *b) {
      ++a;
    } else {
      ++b;
    }
  }
  return true;
}

template <typename T>
bool empty_intersection(const std::set<T>& a, const std::set<T>& b) {
  return empty_intersection(a.begin(), a.end(), b.begin(), b.end());
}

}  // namespace slinky

#endif  // SLINKY_BASE_SET_H
