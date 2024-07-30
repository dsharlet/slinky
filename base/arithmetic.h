#ifndef SLINKY_BASE_ARITHMETIC_H
#define SLINKY_BASE_ARITHMETIC_H

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>

namespace slinky {

// Signed integer division in C/C++ is terrible because it's not Euclidean. This is bad in general, but especially
// because the remainder is not in [0, |divisor|) for negative dividends. These implementations of Euclidean division
// and mod are taken from:
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
T euclidean_mod_positive_modulus(T a, T b) {
  assert(b > 0);
  T r = a % b;
  return r >= 0 ? r : r + b;
}

template <typename T>
T euclidean_mod(T a, T b) {
  if (b == 0) {
    return 0;
  }
  T r = a % b;
  return r >= 0 ? r : (b < 0 ? r - b : r + b);
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
  if (b == -1 && a == std::numeric_limits<T>::min()) {
    return std::numeric_limits<T>::max();
  } else {
    return euclidean_div(a, b);
  }
}

template <typename T>
inline T saturate_mod(T a, T b) {
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
 * false. Otherwise set result to the correct value, and returns true. */
template <class T>
bool add_with_overflow(T a, T b, T* result) {
  bool overflows = __builtin_add_overflow(a, b, result);
  if (overflows) {
    *result = 0;
  }
  return !overflows;
}

template <class T>
bool sub_with_overflow(T a, T b, T* result) {
  bool overflows = __builtin_sub_overflow(a, b, result);
  if (overflows) {
    *result = 0;
  }
  return !overflows;
}

template <class T>
bool mul_with_overflow(T a, T b, T* result) {
  bool overflows = __builtin_mul_overflow(a, b, result);
  if (overflows) {
    *result = 0;
  }
  return !overflows;
}

template <typename T>
inline T gcd(T a, T b) {
  while (a != b) {
    if (a > b) {
      a -= b;
    } else {
      b -= a;
    }
  }
  return a;
}

template <typename T>
inline T lcm(T a, T b) {
  return (a * b) / gcd(a, b);
}

}  // namespace slinky

#endif  // SLINKY_BASE_ARITHMETIC_H
