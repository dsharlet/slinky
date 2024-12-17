#ifndef SLINKY_BASE_SPAN_H
#define SLINKY_BASE_SPAN_H

#include <array>
#include <vector>

namespace slinky {

// Don't want to depend on C++20, so just provide our own span-like helper. Differences:
// - const-only
// - No fixed size extents
template <typename T>
class span {
public:
  using value_type = std::remove_const_t<T>;

private:
  const value_type* data_;
  std::size_t size_;

public:
  span() : data_(nullptr), size_(0) {}
  span(const value_type* data, std::size_t size) : data_(data), size_(size) {}
  span(const value_type* begin, const value_type* end) : data_(begin), size_(end - begin) {}
  template <std::size_t N>
  span(const value_type (&x)[N]) : data_(&x[0]), size_(N) {}
  span(const value_type (&x)[0]) : data_(nullptr), size_(0) {}
  template <std::size_t N>
  span(const std::array<value_type, N>& x) : data_(std::data(x)), size_(N) {}
  span(const std::vector<value_type>& c) : data_(std::data(c)), size_(std::size(c)) {}

  // Allow shallow copying/assignment.
  span(const span&) = default;
  span(span&&) = default;
  span& operator=(const span&) = default;
  span& operator=(span&&) = default;

  const value_type* data() const { return data_; }
  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  const value_type* begin() const { return data_; }
  const value_type* end() const { return data_ + size_; }

  const value_type& operator[](std::size_t i) const { return data_[i]; }

  span subspan(std::size_t offset) const { return span(data_ + offset, size_ - offset); }
  span subspan(std::size_t offset, std::size_t size) const { return span(data_ + offset, size); }
};

template <typename T>
std::vector<T> permute(span<const int> p, const std::vector<T>& x) {
  std::vector<T> result(p.size());
  for (std::size_t i = 0; i < p.size(); ++i) {
    result[i] = x[p[i]];
  }
  return result;
}

}  // namespace slinky

#endif  // SLINKY_BASE_SPAN_H
