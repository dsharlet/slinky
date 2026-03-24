#ifndef SLINKY_BASE_REF_COUNT_H
#define SLINKY_BASE_REF_COUNT_H

#include <algorithm>
#include <atomic>

#include "slinky/base/util.h"

namespace slinky {

// Base class for reference counted objects.
template <typename T>
class ref_counted {
  mutable std::atomic<int> ref_count_{0};

public:
  ref_counted() = default;
  virtual ~ref_counted() = default;

  // std::atomic<int> does not have any copy or move constructors. This makes sense, such operations would be racy.
  // However, we want to allow making copies of reference counted objects. In this case, what should happen to the
  // reference count? The reference count is for the allocation, and will determine when the value is freed. In that
  // case, it makes sense that the reference count should stay with the object, and not change or be copied when the
  // object is copied or assigned.
  ref_counted(const ref_counted&) {}
  ref_counted(ref_counted&&) {}
  ref_counted& operator=(const ref_counted&) { return *this; }
  ref_counted& operator=(ref_counted&&) { return *this; }

  int ref_count() const { return ref_count_.load(std::memory_order_relaxed); }
  void add_ref() const { ref_count_.fetch_add(1, std::memory_order_relaxed); }
  void release() const {
    if (ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      // This const_cast is ugly, but:
      // https://stackoverflow.com/questions/755196/deleting-a-pointer-to-const-t-const
      T::destroy(const_cast<T*>(static_cast<const T*>(this)));
    }
  }
};

// A smart pointer to a ref_counted base.
template <typename T>
class SLINKY_TRIVIAL_ABI ref_count {
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
      std::swap(value, v);
      if (value) value->add_ref();
      if (v) v->release();
    }
    return *this;
  }

  ref_count& operator=(const ref_count& other) { return operator=(other.value); }

  ref_count& operator=(ref_count&& other) noexcept {
    std::swap(value, other.value);
    other = nullptr;
    return *this;
  }

  template <typename U>
  operator ref_count<U>() const {
    return ref_count<U>(value);
  }

  SLINKY_INLINE T& operator*() { return *value; }
  SLINKY_INLINE const T& operator*() const { return *value; }
  SLINKY_INLINE T* operator->() { return value; }
  SLINKY_INLINE const T* operator->() const { return value; }

  SLINKY_INLINE operator T*() { return value; }
  SLINKY_INLINE operator const T*() const { return value; }

  // Take ownership of the value, does not change reference count.
  SLINKY_INLINE T* take() {
    T* result = value;
    value = nullptr;
    return result;
  }
  // Assume ownership of the value, does not change the reference count.
  SLINKY_INLINE static ref_count assume(T* value) {
    ref_count result;
    result.value = value;
    return result;
  }
};

}  // namespace slinky

#endif  // SLINKY_BASE_REF_COUNT_H
