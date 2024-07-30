#include "base/modulus_remainder.h"

// Based on https://github.com/halide/Halide/blob/main/src/ModulusRemainder.cpp.
namespace slinky {

namespace {

int64_t div_imp(int64_t a, int64_t b) {
  int64_t ia = a;
  int64_t ib = b;
  int64_t a_neg = ia >> 63;
  int64_t b_neg = ib >> 63;
  int64_t b_zero = (ib == 0) ? -1 : 0;
  ib -= b_zero;
  ia -= a_neg;
  int64_t q = ia / ib;
  q += a_neg & (~b_neg - b_neg);
  q &= ~b_zero;
  return q;
}

// A version of mod where a % 0 == a
int64_t mod_imp(int64_t a, int64_t b) {
    if (b == 0) {
        return a;
    } else {
      int64_t ia = a;
      int64_t ib = b;
      int64_t a_neg = ia >> 63;
      int64_t b_neg = ib >> 63;
      int64_t b_zero = (ib == 0) ? -1 : 0;
      ia -= a_neg;
      int64_t r = ia % (ib | b_zero);
      r += (a_neg & ((ib ^ b_neg) + ~b_neg));
      r &= ~b_zero;

      return r;
    }
}

int64_t gcd(int64_t a, int64_t b) {
    if (a < b) {
        std::swap(a, b);
    }
    while (b != 0) {
        int64_t tmp = b;
        b = a % b;
        a = tmp;
    }
    return a;
}

} // namespace

modulus_remainder operator+(const modulus_remainder &a, const modulus_remainder &b) {
    int64_t m = 1, r = a.remainder + b.remainder;
  
    m = gcd(a.modulus, b.modulus);
    r = mod_imp(r, m);

    return {m, r};
}

modulus_remainder operator-(const modulus_remainder &a, const modulus_remainder &b) {
    int64_t m = 1, r = a.remainder - b.remainder;

    m = gcd(a.modulus, b.modulus);
    r = mod_imp(r, m);

    return {m, r};
}

modulus_remainder operator*(const modulus_remainder &a, const modulus_remainder &b) {
    int64_t m, r;
    if (a.modulus == 0) {
        // a is constant
        m = a.remainder * b.modulus;
        r = a.remainder * b.remainder;
        return {m, r};
    } else if (b.modulus == 0) {
        // b is constant
        m = a.modulus * b.remainder;
        r = a.remainder * b.remainder;
        return {m, r};
    } else if (a.remainder == 0 && b.remainder == 0) {
        // multiple times multiple
        m = a.modulus * b.modulus;
        return {m, 0};
    } else if (a.remainder == 0) {
        int64_t g = gcd(b.modulus, b.remainder);
        m = a.modulus * g;
        return {m, 0};
    } else if (b.remainder == 0) {
        int64_t g = gcd(a.modulus, a.remainder);
        m = b.modulus * g;
        return {m, 0};
    } else {
        // Convert them to the same modulus and multiply
        r = a.remainder * b.remainder;
        m = gcd(a.modulus, b.modulus);
        r = mod_imp(r, m);
        return {m, r};
    }

    return modulus_remainder{};
}

modulus_remainder operator/(const modulus_remainder &a, const modulus_remainder &b) {
    // What can we say about:
    // floor((m1 * x + r1) / (m2 * y + r2))

    // If m2 is zero and m1 is a multiple of r2, then we can pull the
    // varying term out of the floor div and the expression simplifies
    // to:
    // (m1 / r2) * x + floor(r1 / r2)
    // E.g. (8x + 3) / 2 -> (4x + 1)

    if (b.modulus == 0 && b.remainder != 0) {
        if (mod_imp(a.modulus, b.remainder) == 0) {
            int64_t m = a.modulus / b.remainder;
            int64_t r = mod_imp(div_imp(a.remainder, b.remainder), m);
            return {m, r};
        }
    }

    return modulus_remainder{};
}

modulus_remainder modulus_remainder::unify(const modulus_remainder &a, const modulus_remainder &b) {
    // We don't know if we're going to get a or b, so we'd better find
    // a single modulus remainder that works for both.

    // For example:
    // max(30*_ + 13, 40*_ + 27) ->
    // max(10*_ + 3, 10*_ + 7) ->
    // max(2*_ + 1, 2*_ + 1) ->
    // 2*_ + 1

    if (b.remainder > a.remainder) {
        return unify(b, a);
    }

    // Reduce them to the same modulus and the same remainder
    int64_t modulus = gcd(a.modulus, b.modulus);

    int64_t diff = a.remainder - b.remainder;

    modulus = gcd(diff, modulus);

    int64_t ra = mod_imp(a.remainder, modulus);

    assert(ra == mod_imp(b.remainder, modulus));

    return {modulus, ra};
}

modulus_remainder modulus_remainder::intersect(const modulus_remainder &a, const modulus_remainder &b) {
    // We have x == ma * y + ra == mb * z + rb

    // We want to synthesize these two facts into one modulus
    // remainder relationship. We are permitted to be
    // conservatively-large, so it's OK if some elements of the result
    // only satisfy one of the two constraints.

    // For coprime ma and mb you want to use the Chinese remainder
    // theorem. In our case, the moduli will almost always be
    // powers of two, so we should just return the smaller of the two
    // sets (usually the one with the larger modulus).
    if (a.modulus == 0) {
        return a;
    }
    if (b.modulus == 0) {
        return b;
    }
    if (a.modulus > b.modulus) {
        return a;
    }
    return b;
}

modulus_remainder operator%(const modulus_remainder &a, const modulus_remainder &b) {
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
    int64_t modulus = gcd(a.modulus, b.modulus);
    modulus = gcd(modulus, b.remainder);
    int64_t remainder = mod_imp(a.remainder, modulus);

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

modulus_remainder operator+(const modulus_remainder &a, int64_t b) {
    return a + modulus_remainder(0, b);
}

modulus_remainder operator-(const modulus_remainder &a, int64_t b) {
    return a - modulus_remainder(0, b);
}

modulus_remainder operator*(const modulus_remainder &a, int64_t b) {
    return a * modulus_remainder(0, b);
}

modulus_remainder operator/(const modulus_remainder &a, int64_t b) {
    return a / modulus_remainder(0, b);
}

modulus_remainder operator%(const modulus_remainder &a, int64_t b) {
    return a % modulus_remainder(0, b);
}

} // namespace slinky
