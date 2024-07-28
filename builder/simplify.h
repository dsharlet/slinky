#ifndef SLINKY_BUILDER_SIMPLIFY_H
#define SLINKY_BUILDER_SIMPLIFY_H

#include "runtime/expr.h"
#include "runtime/stmt.h"

namespace slinky {

using bounds_map = symbol_map<interval_expr>;

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

// Try to simplify an expr or stmt.
expr simplify(const expr& e, const bounds_map& bounds = bounds_map());
stmt simplify(const stmt& s, const bounds_map& bounds = bounds_map());
interval_expr simplify(const interval_expr& e, const bounds_map& bounds = bounds_map());

// Determine an interval such that x is always inside the interval.
interval_expr bounds_of(const expr& x, const bounds_map& bounds = bounds_map());
interval_expr bounds_of(const interval_expr& x, const bounds_map& bounds = bounds_map());

// Determine a lower or upper bound of x that is conservative if the bound can be made constant.
expr constant_lower_bound(const expr& x);
expr constant_upper_bound(const expr& x);

// Attempts to determine if e can be proven to be always true or false.
std::optional<bool> attempt_to_prove(const expr& condition, const bounds_map& bounds = bounds_map());
bool prove_true(const expr& condition, const bounds_map& bounds = bounds_map());
bool prove_false(const expr& condition, const bounds_map& bounds = bounds_map());

// Helpers for producing simplified versions of ops. These do not recursively simplify their
// operands. `op` is an existing node that may be returned if op is equivalent. `op` may be null.
expr simplify(const class min* op, expr a, expr b);
expr simplify(const class max* op, expr a, expr b);
expr simplify(const add* op, expr a, expr b);
expr simplify(const sub* op, expr a, expr b);
expr simplify(const mul* op, expr a, expr b);
expr simplify(const div* op, expr a, expr b);
expr simplify(const mod* op, expr a, expr b);
expr simplify(const less* op, expr a, expr b);
expr simplify(const less_equal* op, expr a, expr b);
expr simplify(const equal* op, expr a, expr b);
expr simplify(const not_equal* op, expr a, expr b);
expr simplify(const logical_and* op, expr a, expr b);
expr simplify(const logical_or* op, expr a, expr b);
expr simplify(const logical_not* op, expr a);
expr simplify(const class select* op, expr c, expr t, expr f);
expr simplify(const call* op, intrinsic fn, std::vector<expr> args);

// Helpers for producing the bounds of ops.
interval_expr bounds_of(const class min* op, interval_expr a, interval_expr b);
interval_expr bounds_of(const class max* op, interval_expr a, interval_expr b);
interval_expr bounds_of(const add* op, interval_expr a, interval_expr b);
interval_expr bounds_of(const sub* op, interval_expr a, interval_expr b);
interval_expr bounds_of(const mul* op, interval_expr a, interval_expr b);
interval_expr bounds_of(const div* op, interval_expr a, interval_expr b);
interval_expr bounds_of(const mod* op, interval_expr a, interval_expr b);
interval_expr bounds_of(const less* op, interval_expr a, interval_expr b);
interval_expr bounds_of(const less_equal* op, interval_expr a, interval_expr b);
interval_expr bounds_of(const equal* op, interval_expr a, interval_expr b);
interval_expr bounds_of(const not_equal* op, interval_expr a, interval_expr b);
interval_expr bounds_of(const logical_and* op, interval_expr a, interval_expr b);
interval_expr bounds_of(const logical_or* op, interval_expr a, interval_expr b);
interval_expr bounds_of(const logical_not* op, interval_expr a);
interval_expr bounds_of(const class select* op, interval_expr c, interval_expr t, interval_expr f);
interval_expr bounds_of(const call* op, std::vector<interval_expr> args);

}  // namespace slinky

#endif  // SLINKY_BUILDER_SIMPLIFY_H
