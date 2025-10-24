#ifndef SLINKY_BUILDER_SUBSTITUTE_H
#define SLINKY_BUILDER_SUBSTITUTE_H

#include "slinky/builder/node_mutator.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

namespace slinky {

bool match(expr_ref a, expr_ref b);
SLINKY_INLINE bool match(var a, var b) { return a == b; }
SLINKY_INLINE bool match(var a, expr_ref b) { return is_variable(b, a); }
SLINKY_INLINE bool match(var a, index_t b) { return false; }
SLINKY_INLINE bool match(expr_ref a, var b) { return is_variable(a, b); }
SLINKY_INLINE bool match(expr_ref a, index_t b) { return is_constant(a, b); }
SLINKY_INLINE bool match(index_t a, expr_ref b) { return is_constant(b, a); }
SLINKY_INLINE bool match(index_t a, var b) { return false; }
bool match(stmt_ref a, stmt_ref b);
bool match(const interval_expr& a, const interval_expr& b);
bool match(const dim_expr& a, const dim_expr& b);
bool match(const box_expr& a, const box_expr& b);

// Compute a sort ordering of two nodes based on their structure (not their values).
int compare(const var& a, const var& b);
int compare(expr_ref a, expr_ref b);
int compare(stmt_ref a, stmt_ref b);

// A comparator suitable for using expr/stmt as keys in an std::map/std::set.
struct node_less {
  bool operator()(const var& a, const var& b) const { return compare(a, b) < 0; }
  bool operator()(expr_ref a, expr_ref b) const { return compare(a, b) < 0; }
  bool operator()(stmt_ref a, stmt_ref b) const { return compare(a, b) < 0; }
};

// This base class helps substitute implementations handle shadowing correctly.
class substitutor : public node_mutator {
public:
  // Implementation of substitution for vars.
  virtual var visit_symbol(var x) { return x; }

  // Implementation of substitution for buffer fields.
  virtual expr mutate_variable(const variable* op, var buf, buffer_field field, int dim) { return expr(op); }

  // The implementation must provide the maximum rank of any substitution of buffer metadata for x.
  virtual std::size_t get_target_buffer_rank(var x) { return 0; }

  virtual var enter_decl(var sym) { return sym; }
  virtual void exit_decls(int n = 1) {}

  void visit(const variable* op) override;
  void visit(const let* op) override;
  void visit(const call* op) override;

  void visit(const let_stmt* op) override;
  void visit(const loop* op) override;
  void visit(const allocate* op) override;
  void visit(const make_buffer* op) override;
  void visit(const constant_buffer* op) override;
  void visit(const slice_buffer* op) override;
  void visit(const slice_dim* op) override;
  void visit(const crop_buffer* op) override;
  void visit(const crop_dim* op) override;
  void visit(const transpose* op) override;
  void visit(const call_stmt* op) override;
  void visit(const copy_stmt* op) override;
  void visit(const clone_buffer* op) override;
  void visit(const async* op) override;

  // Silences a weird warning on clang. It seems like this should be in the base class (and it is).
  using node_mutator::visit;
};

// Replace the var `target` with a `replacement` expr. Respects shadowing and implicit buffer metadata.
expr substitute(const expr& e, var target, const expr& replacement);
interval_expr substitute(const interval_expr& x, var target, const expr& replacement);
stmt substitute(const stmt& s, var target, const expr& replacement);

// Substitute `elem_size` in for buffer_elem_size(buffer) and the other buffer metadata in `dims` for per-dimension
// metadata.
// If `def` is defined, then buffer metadata that is undefined in `dims` will be replaced with a read of `def`'s
// metadata.
expr substitute_buffer(const expr& e, var buffer, const std::vector<dim_expr>& dims, var def = var());
expr substitute_buffer(
    const expr& e, var buffer, const expr& elem_size, const std::vector<dim_expr>& dims, var def = var());
interval_expr substitute_buffer(const interval_expr& e, var buffer, const std::vector<dim_expr>& dims, var def = var());
interval_expr substitute_buffer(
    const interval_expr& e, var buffer, const expr& elem_size, const std::vector<dim_expr>& dims, var def = var());

// Helpers to make dims for use with `substitute_buffer` for bounds.
std::vector<dim_expr> make_dims_from_bounds(const box_expr& bounds);
std::vector<dim_expr> make_dims_from_bounds(int dim, const interval_expr& bounds);

// Find `target` and replace it with `replacement`. Does not respect shadowing or implicit buffer metadata.
expr substitute(const expr& e, const expr& target, const expr& replacement);

}  // namespace slinky

#endif  // SLINKY_BUILDER_SUBSTITUTE_H
