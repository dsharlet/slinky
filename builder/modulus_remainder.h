#ifndef SLINKY_BUILDER_MODULUS_REMAINDER_H
#define SLINKY_BUILDER_MODULUS_REMAINDER_H

#include "runtime/expr.h"
#include "runtime/stmt.h"

namespace slinky {

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