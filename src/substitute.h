#ifndef SLINKY_SUBSTITUTE_H
#define SLINKY_SUBSTITUTE_H

#include <map>

#include "expr.h"

namespace slinky {

expr substitute(const expr& e, const std::map<symbol_id, expr>& replacements);
stmt substitute(const stmt& s, const std::map<symbol_id, expr>& replacements);

}  // namespace slinky

#endif  // SLINKY_SUBSTITUTE_H
