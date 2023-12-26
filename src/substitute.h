#ifndef SLINKY_SUBSTITUTE_H
#define SLINKY_SUBSTITUTE_H

#include <map>

#include "expr.h"

namespace slinky {

// Test if `e` matches `p`, providing a mapping of variables in `p` to expressions in `e`.
// For example, `match(max(x, y), max(1, z), matches)` returns true, and matches will be `{ {x, 1},
// {y, z} }`.
bool match(const expr& p, const expr& e, std::map<symbol_id, expr>& matches);
bool match(const expr& a, const expr& b);
bool match(const stmt& a, const stmt& b);

expr substitute(const expr& e, const std::map<symbol_id, expr>& replacements);
stmt substitute(const stmt& s, const std::map<symbol_id, expr>& replacements);

}  // namespace slinky

#endif  // SLINKY_SUBSTITUTE_H
