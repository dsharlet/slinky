#ifndef SLINKY_SIMPLIFY_H
#define SLINKY_SIMPLIFY_H

#include "expr.h"
#include "symbol_map.h"

namespace slinky {

using bounds_map = symbol_map<interval_expr>;

expr simplify(const expr& e, const bounds_map& bounds = bounds_map());
stmt simplify(const stmt& s, const bounds_map& bounds = bounds_map());

std::optional<bool> attempt_prove(const expr& e, const bounds_map& bounds = bounds_map());
bool can_prove(const expr& e, const bounds_map& bounds = bounds_map());
bool can_disprove(const expr& e, const bounds_map& bounds = bounds_map());

interval_expr bounds_of(const expr& e, const bounds_map& bounds = bounds_map());

// Helpers for producing simplified versions of ops. These do not recursively simplify their
// operands. `op` is an existing node that may be returned if op is equivalent. It may be null.
expr simplify(const class min* op, expr a, expr b);
expr simplify(const class max* op, expr a, expr b);
expr simplify(const add* op, expr a, expr b);
expr simplify(const sub* op, expr a, expr b);
expr simplify(const mul* op, expr a, expr b);
expr simplify(const div* op, expr a, expr b);
expr simplify(const mod* op, expr a, expr b);
expr simplify(const logical_and* op, expr a, expr b);
expr simplify(const logical_or* op, expr a, expr b);
expr simplify(const call* op, std::vector<expr> args);

}  // namespace slinky

#endif  // SLINKY_SIMPLIFY_H
