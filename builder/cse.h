#ifndef SLINKY_BUILDER_CSE_H
#define SLINKY_BUILDER_CSE_H

#include "runtime/expr.h"
#include "runtime/stmt.h"

namespace slinky {

// Replace each common sub-expression in the argument with a
// variable, and wrap the resulting expr in a let statement giving a
// value to that variable.
//
// The last parameter determines whether all common subexpressions are
// lifted, or only those that the simplifier would not substitute back
// in (e.g. addition of a constant).
//
expr common_subexpression_elimination(const expr&, node_context& ctx, bool lift_all = false);

// Do common-subexpression-elimination on each expression in a
// statement. Does not introduce let statements.
stmt common_subexpression_elimination(const stmt&, node_context& ctx, bool lift_all = false);

}  // namespace slinky

#endif  // SLINKY_BUILDER_CSE_H
