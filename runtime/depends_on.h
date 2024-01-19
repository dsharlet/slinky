#ifndef SLINKY_RUNTIME_DEPENDS_ON_H
#define SLINKY_RUNTIME_DEPENDS_ON_H

#include "runtime/expr.h"

namespace slinky {

// Check if `e` depends on a variable `var` or buffer `buf`.
bool depends_on(const expr& e, symbol_id var);
bool depends_on(const interval_expr& e, symbol_id var);
bool depends_on(const stmt& s, symbol_id var);
bool depends_on(const stmt& s, span<const symbol_id> vars);
bool depends_on_variable(const expr& e, symbol_id var);
bool depends_on_buffer(const expr& e, symbol_id buf);

}  // namespace slinky

#endif  // SLINKY_RUNTIME_DEPENDS_ON_H
