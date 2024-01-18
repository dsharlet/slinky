#ifndef SLINKY_SIMPLIFY_H
#define SLINKY_SIMPLIFY_H

#include "src/expr.h"
#include "src/symbol_map.h"

namespace slinky {

using bounds_map = symbol_map<interval_expr>;

// Try to simplify an expr or stmt.
expr simplify(const expr& e, const bounds_map& bounds = bounds_map());
stmt simplify(const stmt& s, const bounds_map& bounds = bounds_map());
interval_expr simplify(const interval_expr& e, const bounds_map& bounds = bounds_map());

// Determine an interval such that e is always inside the interval.
interval_expr bounds_of(const expr& x, const bounds_map& bounds = bounds_map());

// Attempts to determine if e can be proven to be always true or false.
std::optional<bool> attempt_to_prove(const expr& condition, const bounds_map& bounds = bounds_map());
bool prove_true(const expr& condition, const bounds_map& bounds = bounds_map());
bool prove_false(const expr& condition, const bounds_map& bounds = bounds_map());

// Find the interval for `var` that makes `e` true.
interval_expr where_true(const expr& condition, symbol_id var);

// Compute the derivative of f with respect to x.
expr differentiate(const expr& f, symbol_id x);

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
expr simplify(const logical_not* op, expr x);
expr simplify(const class select* op, expr c, expr t, expr f);
expr simplify(const call* op, std::vector<expr> args);

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
interval_expr bounds_of(const logical_not* op, interval_expr x);
interval_expr bounds_of(const class select* op, interval_expr c, interval_expr t, interval_expr f);
interval_expr bounds_of(const call* op, std::vector<interval_expr> args);

}  // namespace slinky

#endif  // SLINKY_SIMPLIFY_H
