#ifndef SLINKY_BUILDER_SUBSTITUTE_H
#define SLINKY_BUILDER_SUBSTITUTE_H

#include "runtime/expr.h"
#include "runtime/stmt.h"

namespace slinky {

bool match(const expr& a, const expr& b);
bool match(const stmt& a, const stmt& b);
bool match(const interval_expr& a, const interval_expr& b);
bool match(const dim_expr& a, const dim_expr& b);

expr substitute(const expr& e, const symbol_map<expr>& replacements);
stmt substitute(const stmt& s, const symbol_map<expr>& replacements);
expr substitute(const expr& e, var target, const expr& replacement);
stmt substitute(const stmt& s, var target, const expr& replacement);
expr substitute(const expr& e, const expr& target, const expr& replacement);
stmt substitute(const stmt& s, const expr& target, const expr& replacement);
expr substitute_bounds(const expr& e, var buffer, const box_expr& bounds);
stmt substitute_bounds(const stmt& s, var buffer, const box_expr& bounds);
expr substitute_bounds(const expr& e, var buffer, int dim, const interval_expr& bounds);
stmt substitute_bounds(const stmt& s, var buffer, int dim, const interval_expr& bounds);

// Compute a sort ordering of two nodes based on their structure (not their values).
int compare(const var& a, const var& b);
int compare(const expr& a, const expr& b);
int compare(const base_expr_node* a, const base_expr_node* b);
int compare(const stmt& a, const stmt& b);

// Update buffer metadata expressions to account for a slice that has occurred.
expr update_sliced_buffer_metadata(const expr& e, var buf, span<const int> slices);

// A comparator suitable for using expr/stmt as keys in an std::map/std::set.
struct node_less {
  bool operator()(const var& a, const var& b) const { return compare(a, b) < 0; }
  bool operator()(const expr& a, const expr& b) const { return compare(a, b) < 0; }
  bool operator()(const stmt& a, const stmt& b) const { return compare(a, b) < 0; }
};

}  // namespace slinky

#endif  // SLINKY_BUILDER_SUBSTITUTE_H
