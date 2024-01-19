#ifndef SLINKY_SPAN_H
#define SLINKY_SPAN_H

#include <type_traits>
#include <cstddef>
#include <vector>
#include <array>

namespace slinky {

// Don't want to depend on C++20, so just provide our own span-like helper. Differences:
// - const-only
// - No fixed size extents
template <typename T>
class span {
  using value_type = std::remove_const_t<T>;
  const value_type* data_;
  std::size_t size_;

public:
  span() : data_(nullptr), size_(0) {}
  span(const span&) = default;
  span(span&&) = default;
  span(const value_type* data, std::size_t size) : data_(data), size_(size) {}
  span(const value_type* begin, const value_type* end) : data_(begin), size_(end - begin) {}
  template <std::size_t N>
  span(const value_type (&x)[N]) : data_(&x[0]), size_(N) {}
  template <std::size_t N>
  span(const std::array<value_type, N>& x) : data_(std::data(x)), size_(N) {}
  span(const std::vector<value_type>& c) : data_(std::data(c)), size_(std::size(c)) {}

  const value_type* data() const { return data_; }
  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  const value_type* begin() const { return data_; }
  const value_type* end() const { return data_ + size_; }

  const value_type& operator[](std::size_t i) const { return data_[i]; }

  span subspan(std::size_t offset) { return span(data_ + offset, size_ - offset); }
  span subspan(std::size_t offset, std::size_t size) { return span(data_ + offset, size); }
};

}  // namespace slinky

#endif  // SLINKY_REF_COUNT_H