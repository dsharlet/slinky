#ifndef SLINKY_RUNTIME_UTIL_H
#define SLINKY_RUNTIME_UTIL_H

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

namespace slinky {

#define SLINKY_ALLOCA(T, N) reinterpret_cast<T*>(alloca((N) * sizeof(T)))
#define SLINKY_ALWAYS_INLINE __attribute__((always_inline))

// Signed integer division in C/C++ is terrible. These implementations
// of Euclidean division and mod are taken from:
// https://github.com/halide/Halide/blob/1a0552bb6101273a0e007782c07e8dafe9bc5366/src/CodeGen_Internal.cpp#L358-L408
template <typename T>
T euclidean_div(T a, T b) {
  if (b == 0) {
    return 0;
  }
  T q = a / b;
  T r = a - q * b;
  T bs = b >> (sizeof(T) * 8 - 1);
  T rs = r >> (sizeof(T) * 8 - 1);
  return q - (rs & bs) + (rs & ~bs);
}

template <typename T>
T euclidean_mod(T a, T b) {
  if (b == 0) {
    return 0;
  }
  T r = a % b;
  T sign_mask = r >> (sizeof(T) * 8 - 1);
  return r + (sign_mask & std::abs(b));
}

// Compute a / b, rounding down.
template <typename T>
inline T floor_div(T a, T b) {
  return euclidean_div(a, b);
}

// Compute a / b, rounding to nearest.
template <typename T>
inline T round_div(T a, T b) {
  return floor_div(a + (b >> 1), b);
}

// Compute a / b, rounding upwards.
template <typename T>
inline T ceil_div(T a, T b) {
  return floor_div(a + b - 1, b);
}

// Align x up to the next multiplie of n.
template <typename T>
inline T align_up(T x, T n) {
  return ceil_div(x, n) * n;
}

// Align x down to the next multiplie of n.
template <typename T>
inline T align_down(T x, T n) {
  return floor_div(x, n) * n;
}

template <typename T>
inline T saturate_add(T a, T b) {
  T result;
  if (!__builtin_add_overflow(a, b, &result)) {
    return result;
  } else {
    return (a >> 1) + (b >> 1) > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
  }
}

template <typename T>
inline T saturate_sub(T a, T b) {
  T result;
  if (!__builtin_sub_overflow(a, b, &result)) {
    return result;
  } else {
    return (a >> 1) - (b >> 1) > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
  }
}
template <typename T>
inline T saturate_negate(T x) {
  if (x == std::numeric_limits<T>::min()) {
    return std::numeric_limits<T>::max();
  } else {
    return -x;
  }
}

template <typename T>
inline int sign(T x) {
  return x >= 0 ? 1 : -1;
}

template <typename T>
inline T saturate_mul(T a, T b) {
  T result;
  if (!__builtin_mul_overflow(a, b, &result)) {
    return result;
  } else {
    return sign(a) * sign(b) > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
  }
}

template <typename T>
inline T saturate_div(T a, T b) {
  // This is safe from overflow unless a is max and b is -1.
  if (a == std::numeric_limits<T>::max() && b == -1) {
    return std::numeric_limits<T>::min();
  } else {
    return euclidean_div(a, b);
  }
}

template <typename T>
inline T saturate_mod(T a, T b) {
  // Can this overflow...?
  return euclidean_mod(a, b);
}

// Don't want to depend on C++20, so just provide our own span-like helper. Differences:
// - const-only
// - No fixed size extents
// - Has std::initializer_list constructor!!!
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
  span(const std::initializer_list<value_type>& l) : data_(&*l.begin()), size_(l.size()) {}

  const value_type* data() const { return data_; }
  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  const value_type* begin() const { return data_; }
  const value_type* end() const { return data_ + size_; }

  const value_type& operator[](std::size_t i) const { return data_[i]; }

  span subspan(std::size_t offset) { return span(data_ + offset, size_ - offset); }
  span subspan(std::size_t offset, std::size_t size) { return span(data_ + offset, size); }
};

// Base class for reference counted objects.
template <typename T>
class ref_counted {
  mutable std::atomic<int> ref_count_{0};

public:
  int ref_count() const { return ref_count_; }
  void add_ref() const { ++ref_count_; }
  void release() const {
    if (--ref_count_ == 0) {
      // This const_cast is ugly, but:
      // https://stackoverflow.com/questions/755196/deleting-a-pointer-to-const-t-const
      T::destroy(const_cast<T*>(static_cast<const T*>(this)));
    }
  }

  virtual ~ref_counted() = default;
};

// A smart pointer to a ref_counted base.
template <typename T>
class ref_count {
  T* value;

public:
  ref_count(T* v = nullptr) : value(v) {
    if (value) value->add_ref();
  }
  ref_count(const ref_count& other) : ref_count(other.value) {}
  ref_count(ref_count&& other) noexcept : value(other.value) { other.value = nullptr; }
  ~ref_count() {
    if (value) value->release();
  }

  ref_count& operator=(T* v) {
    if (value != v) {
      if (value) value->release();
      value = v;
      if (value) value->add_ref();
    }
    return *this;
  }

  ref_count& operator=(const ref_count& other) { return operator=(other.value); }

  ref_count& operator=(ref_count&& other) noexcept {
    std::swap(value, other.value);
    other = nullptr;
    return *this;
  }

  SLINKY_ALWAYS_INLINE T& operator*() { return *value; }
  SLINKY_ALWAYS_INLINE const T& operator*() const { return *value; }
  SLINKY_ALWAYS_INLINE T* operator->() { return value; }
  SLINKY_ALWAYS_INLINE const T* operator->() const { return value; }

  SLINKY_ALWAYS_INLINE operator T*() { return value; }
  SLINKY_ALWAYS_INLINE operator const T*() const { return value; }
};

// A vector-like type that does not own its own memory, but does constructor and destruct its elements.
// For use in objects that allocate a block of memory for the object itself plus extra data. This object
// can be pointed at that extra data to manage that memory.
template <typename T>
class embedded_vector {
public:
  using value_type = T;
  using size_type = std::size_t;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using iterator = pointer;
  using const_iterator = const_pointer;

private:
  T* data_;
  size_type capacity_;
  size_type size_;

public:
  embedded_vector() = delete;
  embedded_vector(T* data, size_type capacity) : data_(data), capacity_(capacity), size_(0) {}
  ~embedded_vector() { resize(0); }

  // We can't implement any constructor that doesn't give us a pointer and capacity, and we can't do anything useful
  // with moves.
  embedded_vector(const embedded_vector&) = delete;
  embedded_vector(embedded_vector&&) = delete;
  void operator=(const embedded_vector&) = delete;
  void operator=(embedded_vector&&) = delete;

  void assign(size_type count, const T& value) {
    for (size_type i = 0; i < std::min(size_, count); ++i) {
      data_[i] = value;
    }
    if (size_ < count) {
      for (size_type i = size_; i < count; ++i) {
        push_back(value);
      }
    } else if (size_ > count) {
      resize(count);
    }
  }
  template <class It>
  void assign(It begin, It end) {
    resize(0);
    for (It i = begin; i != end; ++i) {
      push_back(*i);
    }
  }
  void assign(std::initializer_list<T> ilist) { assign(ilist.begin(), ilist.end()); }

  T* data() { return data_; }
  const T* data() const { return data_; }
  size_type size() const { return size_; }
  size_type capacity() const { return capacity_; }
  bool empty() const { return size_ == 0; }

  void resize(size_type new_size) {
    assert(new_size <= capacity_);
    // Construct new elements.
    for (size_type i = size_; i < new_size; ++i) {
      new (&data_[i]) T();
    }
    // Destruct removed elements.
    for (size_type i = new_size; i < size_; ++i) {
      data_[i].~T();
    }
    size_ = new_size;
  }
  void clear() { resize(0); }
  void push_back(const T& x) { new (&data_[size_++]) T(x); }
  void push_back(T&& x) { new (&data_[size_++]) T(x); }
  template <typename... Args>
  reference emplace_back(Args&&... args) {
    new (&data_[size_++]) T(std::forward<Args>(args)...);
    return back();
  }
  void pop_back() {
    back().~T();
    --size_;
  }

  T& operator[](size_type i) { return data_[i]; }
  const T& operator[](size_type i) const { return data_[i]; }
  T& at(size_type i) {
    assert(i < size_);
    return data_[i];
  }
  const T& at(size_type i) const {
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

#endif  // SLINKY_RUNTIME_UTIL_H
