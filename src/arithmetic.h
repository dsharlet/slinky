#ifndef SLINKY_EUCLIDEAN_DIVISION_H
#define SLINKY_EUCLIDEAN_DIVISION_H

#include <cmath>
#include <limits>

namespace slinky {

// Signed integer division in C/C++ is terrible. These implementations
// of Euclidean division and mod are taken from:
// https://github.com/halide/Halide/blob/1a0552bb6101273a0e007782c07e8dafe9bc5366/src/CodeGen_Internal.cpp#L358-L408
template <typename T>
T euclidean_div(T a, T b) {
  if (b == 0) { return 0; }
  T q = a / b;
  T r = a - q * b;
  T bs = b >> (sizeof(T) * 8 - 1);
  T rs = r >> (sizeof(T) * 8 - 1);
  return q - (rs & bs) + (rs & ~bs);
}

template <typename T>
T euclidean_mod(T a, T b) {
  if (b == 0) { return 0; }
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
inline int sign(T x) { return x >= 0 ? 1 : -1; }

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
  // This is safe from overflow unless a is +inf and b is -1.
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


}  // namespace slinky

#endif  // SLINKY_EUCLIDEAN_DIVISION_H
