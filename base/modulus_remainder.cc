#include "base/modulus_remainder.h"

#include "base/arithmetic.h"

// Based on https://github.com/halide/Halide/blob/main/src/modulus_remainder.cpp.
namespace slinky {

modulus_remainder operator+(const modulus_remainder &a, const modulus_remainder &b) {
    int64_t m = 1, r = 0;
    if (add_with_overflow(a.remainder, b.remainder, &r)) {
        m = gcd(a.modulus, b.modulus);
        r = euclidean_mod(r, m);
    }
    return {m, r};
}

modulus_remainder operator-(const modulus_remainder &a, const modulus_remainder &b) {
    int64_t m = 1, r = 0;
    if (sub_with_overflow(a.remainder, b.remainder, &r)) {
        m = gcd(a.modulus, b.modulus);
        r = euclidean_mod(r, m);
    }
    return {m, r};
}

modulus_remainder operator*(const modulus_remainder &a, const modulus_remainder &b) {
    int64_t m, r;
    if (a.modulus == 0) {
        // a is constant
        if (mul_with_overflow(a.remainder, b.modulus, &m) &&
            mul_with_overflow(a.remainder, b.remainder, &r)) {
            return {m, r};
        }
    } else if (b.modulus == 0) {
        // b is constant
        if (mul_with_overflow(a.modulus, b.remainder, &m) &&
            mul_with_overflow(a.remainder, b.remainder, &r)) {
            return {m, r};
        }
    } else if (a.remainder == 0 && b.remainder == 0) {
        // multiple times multiple
        if (mul_with_overflow(a.modulus, b.modulus, &m)) {
            return {m, 0};
        }
    } else if (a.remainder == 0) {
        int64_t g = gcd(b.modulus, b.remainder);
        if (mul_with_overflow(a.modulus, g, &m)) {
            return {m, 0};
        }
    } else if (b.remainder == 0) {
        int64_t g = gcd(a.modulus, a.remainder);
        if (mul_with_overflow(b.modulus, g, &m)) {
            return {m, 0};
        }
    } else {
        // Convert them to the same modulus and multiply
        if (mul_with_overflow(a.remainder, b.remainder, &r)) {
            m = gcd(a.modulus, b.modulus);
            r = euclidean_mod(r, m);
            return {m, r};
        }
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
        if (euclidean_mod(a.modulus, b.remainder) == 0) {
            int64_t m = a.modulus / b.remainder;
            int64_t r = euclidean_mod(euclidean_div(a.remainder, b.remainder), m);
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

    int64_t r;
    if (!sub_with_overflow(a.remainder, b.remainder, &r)) {
        // The modulus is not representable as an int64.
        return modulus_remainder{};
    }

    int64_t diff = a.remainder - b.remainder;

    modulus = gcd(diff, modulus);

    int64_t ra = euclidean_mod(a.remainder, modulus);

    assert(ra == euclidean_mod(b.remainder, modulus));

    return {modulus, ra};
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
    int64_t remainder = euclidean_mod(a.remainder, modulus);

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
