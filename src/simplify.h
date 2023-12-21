#ifndef SLINKY_SIMPLIFY_H
#define SLINKY_SIMPLIFY_H

#include "expr.h"

namespace slinky {

expr simplify(const expr& e);
stmt simplify(const stmt& s);

bool can_prove(const expr& e);

}  // namespace slinky

#endif  // SLINKY_SIMPLIFY_H
