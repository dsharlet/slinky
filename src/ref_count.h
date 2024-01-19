#ifndef SLINKY_REF_COUNT_H
#define SLINKY_REF_COUNT_H

#include <algorithm>
#include <atomic>

namespace slinky {

// Base class for reference counted objects.
class ref_counted {
  mutable std::atomic<int> ref_count_{0};

public:
  int ref_count() const { return ref_count_; }
  void add_ref() const { ++ref_count_; }
  void release() const {
    if (--ref_count_ == 0) delete this;
  }

  virtual ~ref_counted() {}
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
  ref_count(ref_count&& other) : value(other.value) { other.value = nullptr; }
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

  ref_count& operator=(const ref_count& other) {
    return operator=(other.value);
  }

  ref_count& operator=(ref_count&& other) {
    std::swap(value, other.value);
    other = nullptr;
    return *this;
  }

  T& operator*() { return *value; }
  const T& operator*() const { return *value; }
  T* operator->() { return value; }
  const T* operator->() const { return value; }

  operator T*() { return value; }
  operator const T*() const { return value; }
};

}  // namespace slinky

#endif  // SLINKY_REF_COUNT_H