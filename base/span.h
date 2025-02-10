#ifndef SLINKY_BASE_SPAN_H
#define SLINKY_BASE_SPAN_H

#include <array>
#include <cassert>
#include <vector>

namespace slinky {

constexpr std::size_t dynamic_extent = std::numeric_limits<std::size_t>::max();

// Don't want to depend on C++20, so just provide our own span-like helper. Differences:
// - const-only
// - No fixed size extents
template <typename T, std::size_t Extent = dynamic_extent>
class span {
public:
  using value_type = std::remove_const_t<T>;

private:
  const value_type* data_;

public:
  span(const value_type* data, std::size_t size) : data_(data) { assert(size == Extent); }
  span(const value_type* begin, const value_type* end) : data_(begin) { assert(end - begin == Extent); }
  template <std::size_t N>
  span(const value_type (&x)[N]) : data_(&x[0]) {
    static_assert(N == Extent, "");
  }
  span(const std::array<value_type, Extent>& x) : data_(std::data(x)) {}
  span(const std::vector<value_type>& c) : data_(std::data(c)) { assert(c.size() == Extent); }

  // Allow shallow copying/assignment.
  span(const span&) = default;
  span(span&&) = default;
  span& operator=(const span&) = default;
  span& operator=(span&&) = default;

  const value_type* data() const { return data_; }
  static constexpr std::size_t size() { return Extent; }
  static constexpr bool empty() { return Extent == 0; }
  const value_type* begin() const { return data_; }
  const value_type* end() const { return data_ + size(); }

  const value_type& operator[](std::size_t i) const { return data_[i]; }
  const value_type& front() const { return data_[0]; }
  const value_type& back() const { return data_[size() - 1]; }

  span<T, dynamic_extent> subspan(std::size_t offset) const {
    return span<T, dynamic_extent>(data_ + offset, size() - offset);
  }
  span<T, dynamic_extent> subspan(std::size_t offset, std::size_t size) const {
    return span<T, dynamic_extent>(data_ + offset, size);
  }
};

template <typename T>
class span<T, dynamic_extent> {
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
  bool empty() const { return size() == 0; }
  const value_type* begin() const { return data_; }
  const value_type* end() const { return data_ + size(); }

  const value_type& operator[](std::size_t i) const { return data_[i]; }
  const value_type& front() const { return data_[0]; }
  const value_type& back() const { return data_[size() - 1]; }

  span subspan(std::size_t offset) const { return span(data_ + offset, size() - offset); }
  span subspan(std::size_t offset, std::size_t size) const { return span(data_ + offset, size); }
};

template <typename T, std::size_t Extent = dynamic_extent>
class mutable_span {
public:
  using value_type = std::remove_const_t<T>;

private:
  value_type* data_;

public:
  mutable_span(value_type* data, std::size_t size) : data_(data) { assert(size == Extent); }
  mutable_span(value_type* begin, value_type* end) : data_(begin) { assert(end - begin == Extent); }
  template <std::size_t N>
  mutable_span(value_type (&x)[N]) : data_(&x[0]) {
    static_assert(N == Extent, "");
  }
  mutable_span(std::array<value_type, Extent>& x) : data_(std::data(x)) {}
  mutable_span(std::vector<value_type>& c) : data_(std::data(c)) { assert(c.size() == Extent); }

  // Allow shallow copying/assignment.
  mutable_span(const mutable_span&) = default;
  mutable_span(mutable_span&&) = default;
  mutable_span& operator=(const mutable_span&) = default;
  mutable_span& operator=(mutable_span&&) = default;

  value_type* data() const { return data_; }
  static constexpr std::size_t size() { return Extent; }
  static constexpr bool empty() { return Extent == 0; }
  value_type* begin() const { return data_; }
  value_type* end() const { return data_ + size(); }

  value_type& operator[](std::size_t i) const { return data_[i]; }
  value_type& front() const { return data_[0]; }
  value_type& back() const { return data_[size() - 1]; }

  mutable_span<T, dynamic_extent> subspan(std::size_t offset) const {
    return mutable_span<T, dynamic_extent>(data_ + offset, size() - offset);
  }
  mutable_span<T, dynamic_extent> subspan(std::size_t offset, std::size_t size) const {
    return mutable_span<T, dynamic_extent>(data_ + offset, size);
  }
};

template <typename T>
class mutable_span<T, dynamic_extent> {
public:
  using value_type = std::remove_const_t<T>;

private:
  value_type* data_;
  std::size_t size_;

public:
  mutable_span() : data_(nullptr), size_(0) {}
  mutable_span(value_type* data, std::size_t size) : data_(data), size_(size) {}
  mutable_span(value_type* begin, value_type* end) : data_(begin), size_(end - begin) {}
  template <std::size_t N>
  mutable_span(value_type (&x)[N]) : data_(&x[0]), size_(N) {}
  mutable_span(value_type (&x)[0]) : data_(nullptr), size_(0) {}
  template <std::size_t N>
  mutable_span(std::array<value_type, N>& x) : data_(std::data(x)), size_(N) {}
  mutable_span(std::vector<value_type>& c) : data_(std::data(c)), size_(std::size(c)) {}

  // Allow shallow copying/assignment.
  mutable_span(const mutable_span&) = default;
  mutable_span(mutable_span&&) = default;
  mutable_span& operator=(const mutable_span&) = default;
  mutable_span& operator=(mutable_span&&) = default;

  value_type* data() const { return data_; }
  std::size_t size() const { return size_; }
  bool empty() const { return size() == 0; }
  value_type* begin() const { return data_; }
  value_type* end() const { return data_ + size(); }

  value_type& operator[](std::size_t i) const { return data_[i]; }
  value_type& front() const { return data_[0]; }
  value_type& back() const { return data_[size() - 1]; }

  mutable_span subspan(std::size_t offset) const { return mutable_span(data_ + offset, size() - offset); }
  mutable_span subspan(std::size_t offset, std::size_t size) const { return mutable_span(data_ + offset, size); }
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
