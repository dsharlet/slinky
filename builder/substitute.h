#ifndef SLINKY_BUILDER_SUBSTITUTE_H
#define SLINKY_BUILDER_SUBSTITUTE_H

#include "runtime/expr.h"
#include "runtime/stmt.h"

namespace slinky {

bool match(expr_ref a, expr_ref b);
SLINKY_ALWAYS_INLINE SLINKY_UNIQUE bool match(var a, var b) { return a == b; }
SLINKY_ALWAYS_INLINE SLINKY_UNIQUE bool match(var a, expr_ref b) { return is_variable(b, a); }
SLINKY_ALWAYS_INLINE SLINKY_UNIQUE bool match(var a, index_t b) { return false; }
SLINKY_ALWAYS_INLINE SLINKY_UNIQUE bool match(expr_ref a, var b) { return is_variable(a, b); }
SLINKY_ALWAYS_INLINE SLINKY_UNIQUE bool match(expr_ref a, index_t b) { return is_constant(a, b); }
SLINKY_ALWAYS_INLINE SLINKY_UNIQUE bool match(index_t a, expr_ref b) { return is_constant(b, a); }
SLINKY_ALWAYS_INLINE SLINKY_UNIQUE bool match(index_t a, var b) { return false; }
bool match(stmt_ref a, stmt_ref b);
bool match(const interval_expr& a, const interval_expr& b);
bool match(const dim_expr& a, const dim_expr& b);
const call* match_call(expr_ref x, intrinsic fn, var a);
const call* match_call(expr_ref x, intrinsic fn, var a, index_t b);

expr substitute(const expr& e, var target, const expr& replacement);
stmt substitute(const stmt& s, var target, const expr& replacement);
interval_expr substitute(const interval_expr& x, var target, const expr& replacement);
expr substitute(const expr& e, const expr& target, const expr& replacement);
stmt substitute(const stmt& s, const expr& target, const expr& replacement);

// Substitute `elem_size` in for buffer_elem_size(buffer) and the other buffer metadata in `dims` for per-dimension
// metadata.
expr substitute_buffer(const expr& e, var buffer, const expr& elem_size, const std::vector<dim_expr>& dims);
stmt substitute_buffer(const stmt& s, var buffer, const expr& elem_size, const std::vector<dim_expr>& dims);
expr substitute_bounds(const expr& e, var buffer, const box_expr& bounds);
stmt substitute_bounds(const stmt& s, var buffer, const box_expr& bounds);
expr substitute_bounds(const expr& e, var buffer, int dim, const interval_expr& bounds);
stmt substitute_bounds(const stmt& s, var buffer, int dim, const interval_expr& bounds);

// Compute a sort ordering of two nodes based on their structure (not their values).
int compare(const var& a, const var& b);
int compare(expr_ref a, expr_ref b);
int compare(stmt_ref a, stmt_ref b);

// Update buffer metadata expressions to account for a slice that has occurred.
expr update_sliced_buffer_metadata(const expr& e, var buf, span<const int> slices);
interval_expr update_sliced_buffer_metadata(const interval_expr& x, var buf, span<const int> slices);
dim_expr update_sliced_buffer_metadata(const dim_expr& x, var buf, span<const int> slices);

// A comparator suitable for using expr/stmt as keys in an std::map/std::set.
struct node_less {
  bool operator()(const var& a, const var& b) const { return compare(a, b) < 0; }
  bool operator()(expr_ref a, expr_ref b) const { return compare(a, b) < 0; }
  bool operator()(stmt_ref a, stmt_ref b) const { return compare(a, b) < 0; }
};

}  // namespace slinky

#endif  // SLINKY_BUILDER_SUBSTITUTE_H
