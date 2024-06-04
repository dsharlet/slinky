#ifndef SLINKY_BUILDER_CSE_H
#define SLINKY_BUILDER_CSE_H

#include "runtime/expr.h"
#include "runtime/stmt.h"

namespace slinky {

// Replace each common sub-expression in the argument with a
// variable, and wrap the resulting expr in a let statement giving a
// value to that variable.
expr common_subexpression_elimination(const expr&, node_context& ctx);

// Do common-subexpression-elimination across all the exprs in the
// given stmt.
stmt common_subexpression_elimination(const stmt&, node_context& ctx);

}  // namespace slinky

#endif  // SLINKY_BUILDER_CSE_H
