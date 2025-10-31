#ifndef SLINKY_BASE_ARITHMETIC_H
#define SLINKY_BASE_ARITHMETIC_H

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <utility>

namespace slinky {

// Signed integer division in C/C++ is terrible because it's not Euclidean. This is bad in general, but especially
// because the remainder is not in [0, |divisor|) for negative dividends. These implementations of Euclidean division
// and mod are taken from:
// https://github.com/halide/Halide/blob/1a0552bb6101273a0e007782c07e8dafe9bc5366/src/CodeGen_Internal.cpp#L358-L408
template <typename T>
T euclidean_div(T a, T b, T define_b_zero = 0) {
  if (b == 0) {
    return define_b_zero;
  }
  T q = a / b;
  T r = a % b;
  // Get the sign of b and r (-1 or 0).
  T bs = b >> (sizeof(T) * 8 - 1);
  T rs = r >> (sizeof(T) * 8 - 1);
  // Adjust the result (which is rounded towards 0) to be rounded down.
  return q - (rs & bs) + (rs & ~bs);
}

template <typename T>
T euclidean_mod(T a, T b, T define_b_zero = 0) {
  if (b == 0) {
    return define_b_zero;
  }
  T r = a % b;
  return r >= 0 ? r : (b < 0 ? r - b : r + b);
}

template <typename T>
T euclidean_div_positive_divisor(T a, T b) {
  assert(b > 0);
  T q = a / b;
  T r = a % b;
  // Get the sign of r (-1 or 0).
  T rs = r >> (sizeof(T) * 8 - 1);
  return q + rs;
}

template <typename T>
T euclidean_mod_positive_modulus(T a, T b) {
  assert(b > 0);
  T r = a % b;
  return r >= 0 ? r : r + b;
}

// Compute a / b, rounding down.
template <typename T>
T floor_div(T a, T b) {
  return euclidean_div(a, b);
}

// Compute a / b, rounding to nearest.
template <typename T>
T round_div(T a, T b) {
  return floor_div(a + (b >> 1), b);
}

// Compute a / b, rounding upwards.
template <typename T>
T ceil_div(T a, T b) {
  return floor_div(a + b - 1, b);
}

// Align x up to the next multiplie of n.
template <typename T>
T align_up(T x, T n) {
  return ceil_div(x, n) * n;
}

// Align x down to the next multiplie of n.
template <typename T>
T align_down(T x, T n) {
  return floor_div(x, n) * n;
}

#if defined(_MSC_VER) && !defined(__clang__)
inline bool __builtin_add_overflow(int32_t a, int32_t b, int32_t* r) { return _add_overflow_i32(0, a, b, r) != 0; }
inline bool __builtin_add_overflow(int64_t a, int64_t b, int64_t* r) { return _add_overflow_i64(0, a, b, r) != 0; }
inline bool __builtin_sub_overflow(int32_t a, int32_t b, int32_t* r) { return _sub_overflow_i32(0, a, b, r) != 0; }
inline bool __builtin_sub_overflow(int64_t a, int64_t b, int64_t* r) { return _sub_overflow_i64(0, a, b, r) != 0; }
inline bool __builtin_mul_overflow(int32_t a, int32_t b, int32_t* r) { return _mul_overflow_i32(a, b, r) != 0; }
inline bool __builtin_mul_overflow(int64_t a, int64_t b, int64_t* r) { return _mul_overflow_i64(a, b, r) != 0; }
#endif  // defined(_MSC_VER) && !defined(__clang__)

template <typename T>
T saturate_add(T a, T b) {
  T result;
  if (!__builtin_add_overflow(a, b, &result)) {
    return result;
  } else {
    return (a >> 1) + (b >> 1) > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
  }
}

template <typename T>
T saturate_sub(T a, T b) {
  T result;
  if (!__builtin_sub_overflow(a, b, &result)) {
    return result;
  } else {
    return (a >> 1) - (b >> 1) > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
  }
}
template <typename T>
T saturate_negate(T x) {
  if (x == std::numeric_limits<T>::min()) {
    return std::numeric_limits<T>::max();
  } else {
    return -x;
  }
}

template <typename T>
int sign(T x) {
  if (x == 0) return 0;
  return x > 0 ? 1 : -1;
}

template <typename T>
T saturate_mul(T a, T b) {
  T result;
  if (!__builtin_mul_overflow(a, b, &result)) {
    return result;
  } else {
    return sign(a) * sign(b) > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
  }
}

template <typename T>
T saturate_div(T a, T b) {
  // This is safe from overflow unless a is max and b is -1.
  if (b == -1 && a == std::numeric_limits<T>::min()) {
    return std::numeric_limits<T>::max();
  } else {
    return euclidean_div(a, b);
  }
}

template <typename T>
T saturate_mod(T a, T b) {
  // Can this overflow...?
  if (b == -1) {
    return 0;
  } else {
    return euclidean_mod(a, b);
  }
}

template <class T>
bool add_overflows(T a, T b) {
  T dummy;
  return __builtin_add_overflow(a, b, &dummy);
}

template <class T>
bool sub_overflows(T a, T b) {
  T dummy;
  return __builtin_sub_overflow(a, b, &dummy);
}

template <class T>
bool mul_overflows(T a, T b) {
  T dummy;
  return __builtin_mul_overflow(a, b, &dummy);
}

/** Routines to perform arithmetic on signed types without triggering signed
 * overflow. If overflow would occur, sets result to zero, and returns
 * true. Otherwise set result to the correct value, and returns false. */
template <class T>
bool add_with_overflow(T a, T b, T& result) {
  bool overflows = __builtin_add_overflow(a, b, &result);
  if (overflows) {
    result = 0;
  }
  return overflows;
}

template <class T>
bool sub_with_overflow(T a, T b, T& result) {
  bool overflows = __builtin_sub_overflow(a, b, &result);
  if (overflows) {
    result = 0;
  }
  return overflows;
}

template <class T>
bool mul_with_overflow(T a, T b, T& result) {
  bool overflows = __builtin_mul_overflow(a, b, &result);
  if (overflows) {
    result = 0;
  }
  return overflows;
}

template <typename T>
T gcd(T a, T b) {
  if (a < b) {
    std::swap(a, b);
  }
  while (b != 0) {
    T tmp = b;
    b = a % b;
    a = tmp;
  }
  return a;
}

template <typename T>
T lcm(T a, T b) {
  return (a * b) / gcd(a, b);
}

// Computes ((x + a)/b)*c
template <typename T>
T staircase(T x, T a, T b, T c) {
  return euclidean_div(x + a, b) * c;
}

template <typename T>
struct interval {
  // The interval is unbounded if the min or max is missing.
  std::optional<T> min, max;

  interval() = default;
  explicit interval(T x) : min(x), max(x) {}
  interval(T min, T max) : min(min), max(max) {}

  bool operator==(const interval<T>& r) const {
    if (!min != !r.min) return false;
    if (!max != !r.max) return false;
    if (min && *min != *r.min) return false;
    if (max && *max != *r.max) return false;
    return true;
  }
};

// Returns the [min, max] interval over all x of ((x + a1)/b1)*c1 + ((x + a2)/b2)*c2
// Returns std::nullopt if unbounded.
interval<int> staircase_sum_bounds(int a1, int b1, int c1, int a2, int b2, int c2);

}  // namespace slinky

#endif  // SLINKY_BASE_ARITHMETIC_H
