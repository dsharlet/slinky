#ifndef SLINKY_SIMPLIFY_H
#define SLINKY_SIMPLIFY_H

#include "expr.h"
#include "symbol_map.h"

namespace slinky {

expr simplify(const expr& e);
stmt simplify(const stmt& s);

bool can_prove(const expr& e);

interval_expr bounds_of(const expr& e, const symbol_map<interval_expr>& bounds = symbol_map<interval_expr>());

bool depends_on(const expr& e, symbol_id var);

}  // namespace slinky

#endif  // SLINKY_SIMPLIFY_H
