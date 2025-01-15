#ifndef SLINKY_BASE_EMBEDDED_VECTOR_H
#define SLINKY_BASE_EMBEDDED_VECTOR_H

namespace slinky {

// A vector-like type that does not own its own memory, but does constructor and destruct its elements.
// For use in objects that allocate a block of memory for the object itself plus extra data. This object
// can be pointed at that extra data to manage that memory.
template <typename T>
class embedded_vector {
  T* data_;
  std::size_t size_;

public:
  using value_type = T;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using iterator = pointer;
  using const_iterator = const_pointer;

  embedded_vector() = delete;
  embedded_vector(T* data, std::size_t size) : data_(data), size_(size) {
    // Call default constructor.
    for (T& i : *this) {
      new (&i) T();
    }
  }
  ~embedded_vector() {
    for (T& i : *this) {
      i.~T();
    }
  }

  // We can't implement any copy/move functions.
  embedded_vector(const embedded_vector&) = delete;
  embedded_vector(embedded_vector&&) = delete;
  void operator=(const embedded_vector&) = delete;
  void operator=(embedded_vector&&) = delete;

  T* data() { return data_; }
  const T* data() const { return data_; }
  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  T& operator[](std::size_t i) { return data_[i]; }
  const T& operator[](std::size_t i) const { return data_[i]; }
  T& at(std::size_t i) {
    assert(i < size_);
    return data_[i];
  }
  const T& at(std::size_t i) const {
    assert(i < size_);
    return data_[i];
  }
  T& front() { return *data_; }
  T& back() { return *(data_ + size_ - 1); }
  const T& front() const { return *data_; }
  const T& back() const { return *(data_ + size_ - 1); }

  iterator begin() { return data_; }
  iterator end() { return data_ + size_; }
  const_iterator begin() const { return data_; }
  const_iterator end() const { return data_ + size_; }
};

}  // namespace slinky

#endif  // SLINKY_BASE_EMBEDDED_VECTOR_H