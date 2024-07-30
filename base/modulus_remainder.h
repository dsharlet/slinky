#ifndef SLINKY_BUILDER_MODULUS_REMAINDER_H
#define SLINKY_BUILDER_MODULUS_REMAINDER_H

#include <algorithm>
#include <cassert>
#include <cstdint>


namespace slinky {

// Taken from https://github.com/halide/Halide/blob/main/src/ModulusRemainder.h
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
struct modulus_remainder {
    modulus_remainder() = default;
    modulus_remainder(int64_t m, int64_t r)
        : modulus(m), remainder(r) {
    }

    int64_t modulus = 1, remainder = 0;

    // Take a conservatively-large union of two sets. Contains all
    // elements from both sets, and maybe some more stuff.
    static modulus_remainder unify(const modulus_remainder &a, const modulus_remainder &b);

    // Take a conservatively-large intersection. Everything in the
    // result is in at least one of the two sets, but not always both.
    static modulus_remainder intersect(const modulus_remainder &a, const modulus_remainder &b);

    bool operator==(const modulus_remainder &other) const {
        return (modulus == other.modulus) && (remainder == other.remainder);
    }
};

modulus_remainder operator+(const modulus_remainder &a, const modulus_remainder &b);
modulus_remainder operator-(const modulus_remainder &a, const modulus_remainder &b);
modulus_remainder operator*(const modulus_remainder &a, const modulus_remainder &b);
modulus_remainder operator/(const modulus_remainder &a, const modulus_remainder &b);
modulus_remainder operator%(const modulus_remainder &a, const modulus_remainder &b);

modulus_remainder operator+(const modulus_remainder &a, int64_t b);
modulus_remainder operator-(const modulus_remainder &a, int64_t b);
modulus_remainder operator*(const modulus_remainder &a, int64_t b);
modulus_remainder operator/(const modulus_remainder &a, int64_t b);
modulus_remainder operator%(const modulus_remainder &a, int64_t b);

}  // namespace slinky

#endif  // SLINKY_BUILDER_MODULUS_REMAINDER_H