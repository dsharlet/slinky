#ifndef SLINKY_REF_COUNT_H
#define SLINKY_REF_COUNT_H

#include <atomic>

namespace slinky {

// Base class for reference counted objects.
class ref_counted {
  mutable std::atomic<int> ref_count{0};

public:
  void add_ref() const { ++ref_count; }
  void release() const {
    if (--ref_count == 0) delete this;
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
  ref_count(ref_count&& other) : ref_count(other.value) { other.value = nullptr; }
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
    if (value != other.value) {
      if (value) value->release();
      value = other.value;
      if (value) value->add_ref();
    }
    return *this;
  }

  ref_count& operator=(ref_count&& other) {
    if (value == other.value) {
      other = nullptr;
    } else {
      if (value) value->release();
      value = other.value;
      other.value = nullptr;
    }
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