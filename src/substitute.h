#ifndef SLINKY_SUBSTITUTE_H
#define SLINKY_SUBSTITUTE_H

#include "expr.h"
#include "symbol_map.h"

namespace slinky {

// Test if `e` matches `p`, providing a mapping of variables in `p` to expressions in `e`.
// For example, `match(max(x, y), max(1, z), matches)` returns true, and matches will be `{ {x, 1},
// {y, z} }`.
bool match(const expr& p, const expr& e, symbol_map<expr>& matches);
bool match(const expr& a, const expr& b);
bool match(const stmt& a, const stmt& b);
bool match(const interval_expr& a, const interval_expr& b);
bool match(const dim_expr& a, const dim_expr& b);

expr substitute(const expr& e, const symbol_map<expr>& replacements);
stmt substitute(const stmt& s, const symbol_map<expr>& replacements);
expr substitute(const expr& e, symbol_id target, const expr& replacement);
stmt substitute(const stmt& s, symbol_id target, const expr& replacement);
expr substitute(const expr& e, const expr& target, const expr& replacement);
stmt substitute(const stmt& s, const expr& target, const expr& replacement);
expr substitute_bounds(const expr& e, symbol_id buffer, const box_expr& bounds);
stmt substitute_bounds(const stmt& s, symbol_id buffer, const box_expr& bounds);
expr substitute_bounds(const expr& e, symbol_id buffer, int dim, const interval_expr& bounds);
stmt substitute_bounds(const stmt& s, symbol_id buffer, int dim, const interval_expr& bounds);

// Check if `e` depends on a variable `var` or buffer `buf`.
bool depends_on(const expr& e, symbol_id var);
bool depends_on_variable(const expr& e, symbol_id var);
bool depends_on_buffer(const expr& e, symbol_id buf);

// Compute a sort ordering of two nodes based on their structure (not their values).
int compare(const expr& a, const expr& b);
int compare(const stmt& a, const stmt& b);

// A comparator suitable for using expr/stmt as keys in an std::map/std::set.
struct node_less {
  bool operator()(const expr& a, const expr& b) const { return compare(a, b) < 0; }
  bool operator()(const stmt& a, const stmt& b) const { return compare(a, b) < 0; }
};

}  // namespace slinky

#endif  // SLINKY_SUBSTITUTE_H
