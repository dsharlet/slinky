#ifndef SLINKY_EUCLIDEAN_DIVISION_H
#define SLINKY_EUCLIDEAN_DIVISION_H

#include <cmath>

namespace slinky {

using index_t = std::ptrdiff_t;

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

}  // namespace slinky

#endif  // SLINKY_EUCLIDEAN_DIVISION_H
