#ifndef SLINKY_BASE_FLAT_SET_H
#define SLINKY_BASE_FLAT_SET_H

#include <algorithm>
#include <cassert>
#include <iterator>
#include <vector>

namespace slinky {

// Implements a set with std::vector<T> storage. The vector is kept sorted to enable efficient set operations.
template <typename T>
class flat_set {
public:
  using value_type = T;
  using iterator = typename std::vector<T>::iterator;

private:
  std::vector<T> s_;

public:
  flat_set() = default;
  flat_set(const flat_set&) = default;
  flat_set(flat_set&&) = default;
  flat_set(std::initializer_list<T> values) : s_(values) {
    std::sort(s_.begin(), s_.end());
    s_.erase(std::unique(s_.begin(), s_.end()), s_.end());
  }
  flat_set& operator=(const flat_set&) = default;
  flat_set& operator=(flat_set&&) = default;

  const value_type* data() const { return s_.data(); }
  std::size_t size() const { return s_.size(); }
  bool empty() const { return s_.empty(); }
  auto begin() const { return s_.begin(); }
  auto end() const { return s_.end(); }
  void clear() { s_.clear(); }
  const value_type& operator[](std::size_t i) const { return s_[i]; }

  // The rest of the these functions are drop-in replacements for std::set<T> members of the same name.
  auto insert(const T& x) {
    auto i = std::lower_bound(s_.begin(), s_.end(), x);
    if (i == s_.end() || *i != x) {
      return std::make_pair(s_.insert(i, x), true);
    } else {
      return std::make_pair(i, false);
    }
  }
  void erase(iterator i) { s_.erase(i); }

  int count(const T& x) const { return std::binary_search(s_.begin(), s_.end(), x) ? 1 : 0; }

  auto find(const T& x) {
    auto i = std::lower_bound(s_.begin(), s_.end(), x);
    return i != s_.end() && *i == x ? i : s_.end();
  }

  // The set union and intersection operators are consistent with our overloads for intervals (which are also sets).
  flat_set operator|(const flat_set& b) const {
    flat_set result;
    result.s_.reserve(size() + b.size());
    std::set_union(s_.begin(), s_.end(), b.begin(), b.end(), std::back_inserter(result.s_));
    return result;
  }
  flat_set operator&(const flat_set& b) const {
    flat_set result;
    result.s_.reserve(std::min(size(), b.size()));
    std::set_intersection(s_.begin(), s_.end(), b.begin(), b.end(), std::back_inserter(result.s_));
    return result;
  }

  flat_set& operator|=(const flat_set& b) { return *this = *this | b; }
  flat_set& operator&=(const flat_set& b) { return *this = *this & b; }
};

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
bool empty_intersection(const flat_set<T>& a, const flat_set<T>& b) {
  return empty_intersection(a.begin(), a.end(), b.begin(), b.end());
}

}  // namespace slinky

#endif  // SLINKY_BASE_SPAN_H
