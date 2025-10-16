#ifndef SLINKY_BASE_MODULUS_REMAINDER_H
#define SLINKY_BASE_MODULUS_REMAINDER_H

#include <cassert>
#include <cstdint>

#include "slinky/base/arithmetic.h"

namespace slinky {

// Based on
// https://github.com/halide/Halide/blob/423df3c50b4b5c9b05b6b2cae16aff535dcf81c0/src/ModulusRemainder.cpp#L390-L589
/** The result of modulus_remainder analysis. These represent strided
 * subsets of the integers. A ModulusRemainder object m represents all
 * integers x such that there exists y such that x == m.modulus * y +
 * m.remainder. Note that under this definition a set containing a
 * single integer (a constant) is represented using a modulus of
 * zero. These sets can be combined with several mathematical
 * operators in the obvious way. E.g. m1 + m2 contains (at least) all
 * integers x1 + x2 such that x1 belongs to m1 and x2 belongs to
 * m2. These combinations are conservative. If some internal math
 * would overflow, it defaults to all of the integers (modulus == 1,
 * remainder == 0). */
template <typename T>
struct modulus_remainder {
  modulus_remainder() = default;
  modulus_remainder(T m, T r) : modulus(m), remainder(r) {}

  T modulus = 1, remainder = 0;

  bool operator==(const modulus_remainder<T>& other) const {
    return (modulus == other.modulus) && (remainder == other.remainder);
  }
};

template <typename T>
modulus_remainder<T> operator+(const modulus_remainder<T>& a, const modulus_remainder<T>& b) {
  T m = 1, r = 0;
  if (!add_with_overflow(a.remainder, b.remainder, r)) {
    m = gcd(a.modulus, b.modulus);
    r = euclidean_mod(r, m, r);
  }
  return {m, r};
}

template <typename T>
modulus_remainder<T> operator-(const modulus_remainder<T>& a, const modulus_remainder<T>& b) {
  T m = 1, r = 0;
  if (!sub_with_overflow(a.remainder, b.remainder, r)) {
    m = gcd(a.modulus, b.modulus);
    r = euclidean_mod(r, m, r);
  }
  return {m, r};
}

template <typename T>
modulus_remainder<T> operator*(const modulus_remainder<T>& a, const modulus_remainder<T>& b) {
  T m, r;
  if (a.modulus == 0) {
    // a is constant
    if (!mul_with_overflow(a.remainder, b.modulus, m) && !mul_with_overflow(a.remainder, b.remainder, r)) {
      return {m, r};
    }
  } else if (b.modulus == 0) {
    // b is constant
    if (!mul_with_overflow(a.modulus, b.remainder, m) && !mul_with_overflow(a.remainder, b.remainder, r)) {
      return {m, r};
    }
  } else if (a.remainder == 0 && b.remainder == 0) {
    // multiple times multiple
    if (!mul_with_overflow(a.modulus, b.modulus, m)) {
      return {m, 0};
    }
  } else if (a.remainder == 0) {
    T g = gcd(b.modulus, b.remainder);
    if (!mul_with_overflow(a.modulus, g, m)) {
      return {m, 0};
    }
  } else if (b.remainder == 0) {
    T g = gcd(a.modulus, a.remainder);
    if (!mul_with_overflow(b.modulus, g, m)) {
      return {m, 0};
    }
  } else {
    // Convert them to the same modulus and multiply
    if (!mul_with_overflow(a.remainder, b.remainder, r)) {
      m = gcd(a.modulus, b.modulus);
      r = euclidean_mod(r, m, r);
      return {m, r};
    }
  }

  return modulus_remainder<T>{};
}

template <typename T>
modulus_remainder<T> operator/(const modulus_remainder<T>& a, const modulus_remainder<T>& b) {
  // What can we say about:
  // floor((m1 * x + r1) / (m2 * y + r2))

  // If m2 is zero and m1 is a multiple of r2, then we can pull the
  // varying term out of the floor div and the expression simplifies
  // to:
  // (m1 / r2) * x + floor(r1 / r2)
  // E.g. (8x + 3) / 2 -> (4x + 1)

  if (b.modulus == 0 && b.remainder != 0) {
    if ((euclidean_mod(a.modulus, b.remainder, a.modulus)) == 0) {
      T m = a.modulus / b.remainder;
      T r = euclidean_div(a.remainder, b.remainder);
      r = euclidean_mod(r, m, r);
      return {m, r};
    }
  }

  return modulus_remainder<T>{};
}

template <typename T>
modulus_remainder<T> operator|(const modulus_remainder<T>& a, const modulus_remainder<T>& b) {
  // We don't know if we're going to get a or b, so we'd better find
  // a single modulus remainder that works for both.

  // For example:
  // max(30*_ + 13, 40*_ + 27) ->
  // max(10*_ + 3, 10*_ + 7) ->
  // max(2*_ + 1, 2*_ + 1) ->
  // 2*_ + 1

  if (b.remainder > a.remainder) {
    return b | a;
  }

  // Reduce them to the same modulus and the same remainder
  T modulus = gcd(a.modulus, b.modulus);

  T r;
  if (sub_with_overflow(a.remainder, b.remainder, r)) {
    // The modulus is not representable as an int64.
    return modulus_remainder<T>{};
  }

  T diff = a.remainder - b.remainder;

  modulus = gcd(diff, modulus);

  T ra = euclidean_mod(a.remainder, modulus, a.remainder);

  assert(ra == (euclidean_mod(b.remainder, modulus, b.remainder)));

  return {modulus, ra};
}

template <typename T>
modulus_remainder<T> operator&(const modulus_remainder<T>& a, const modulus_remainder<T>& b) {
  if (a.modulus == 0) return a;
  if (b.modulus == 0) return b;
  if (a.remainder == 0 && b.remainder == 0) return {lcm(a.modulus, b.modulus), 0};
  // We have x == ma * y + ra == mb * z + rb

  // We want to synthesize these two facts into one modulus
  // remainder relationship. We are permitted to be
  // conservatively-large, so it's OK if some elements of the result
  // only satisfy one of the two constraints.

  // For coprime ma and mb you want to use the Chinese remainder
  // theorem. In our case, the moduli will almost always be
  // powers of two, so we should just return the smaller of the two
  // sets (usually the one with the larger modulus).
  if (a.modulus > b.modulus) {
    return a;
  }
  return b;
}

template <typename T>
modulus_remainder<T> operator%(const modulus_remainder<T>& a, const modulus_remainder<T>& b) {
  // For non-zero y, we can treat x mod y as x + z*y, where we know
  // nothing about z.
  // (ax + b) + z (cx + d) ->
  // ax + b + zcx + dz ->
  // gcd(a, c, d) * w + b

  // E.g:
  // (8x + 5) mod (6x + 2) ->
  // (8x + 5) + z (6x + 2) ->
  // (8x + 6zx + 2x) + 5 ->
  // 2(4x + 3zx + x) + 5 ->
  // 2w + 1
  T modulus = gcd(a.modulus, b.modulus);
  modulus = gcd(modulus, b.remainder);
  T remainder = euclidean_mod(a.remainder, modulus, a.remainder);

  if (b.remainder == 0 && remainder != 0) {
    // b could be zero, so the result could also just be zero.
    if (modulus == 0) {
      remainder = 0;
    } else {
      // This can no longer be expressed as ax + b
      remainder = 0;
      modulus = 1;
    }
  }

  return {modulus, remainder};
}

template <typename T>
modulus_remainder<T> operator+(const modulus_remainder<T>& a, T b) {
  return a + modulus_remainder<T>(0, b);
}

template <typename T>
modulus_remainder<T> operator-(const modulus_remainder<T>& a, T b) {
  return a - modulus_remainder<T>(0, b);
}

template <typename T>
modulus_remainder<T> operator*(const modulus_remainder<T>& a, T b) {
  return a * modulus_remainder<T>(0, b);
}

template <typename T>
modulus_remainder<T> operator/(const modulus_remainder<T>& a, T b) {
  return a / modulus_remainder<T>(0, b);
}

template <typename T>
modulus_remainder<T> operator%(const modulus_remainder<T>& a, T b) {
  return a % modulus_remainder<T>(0, b);
}

}  // namespace slinky

#endif  // SLINKY_BASE_MODULUS_REMAINDER_H
