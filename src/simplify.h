#ifndef SLINKY_SIMPLIFY_H
#define SLINKY_SIMPLIFY_H

#include "expr.h"
#include "symbol_map.h"

namespace slinky {

expr simplify(const expr& e);
stmt simplify(const stmt& s);

bool can_prove(const expr& e);

interval bounds_of(const expr& e, const symbol_map<interval>& bounds = symbol_map<interval>());

}  // namespace slinky

#endif  // SLINKY_SIMPLIFY_H
