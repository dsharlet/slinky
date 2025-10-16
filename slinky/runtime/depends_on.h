#ifndef SLINKY_RUNTIME_DEPENDS_ON_H
#define SLINKY_RUNTIME_DEPENDS_ON_H

#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

namespace slinky {

// The various ways a node might depend on a buffer or variable.
struct depends_on_result {
  // True if the node depends on the symbol as a variable.
  bool var = false;

  // True if the buffer is used as a call input or output, respectively.
  bool buffer_input = false;
  bool buffer_output = false;
  // True if the buffer is used as a copy source or destination, respectively.
  bool buffer_src = false;
  bool buffer_dst = false;
  // True if the buffer's base pointer is read.
  bool buffer_base = false;

  // True if the buffer metadata is read.
  bool buffer_dims = false;
  bool buffer_bounds = false;

  bool buffer_data() const { return buffer_input || buffer_output || buffer_src || buffer_dst || buffer_base; }
  bool buffer() const { return buffer_data() || buffer_dims || buffer_bounds; }

  bool buffer_read() const { return buffer_input || buffer_src || buffer_base; }
  bool buffer_write() const { return buffer_output || buffer_dst || buffer_base; }

  bool any() const { return var || buffer(); }
};

// Check if the node depends on a symbol or set of symbols.
void depends_on(expr_ref e, span<const std::pair<var, depends_on_result&>> var_deps);
void depends_on(stmt_ref s, span<const std::pair<var, depends_on_result&>> var_deps);
void depends_on(expr_ref e, var x, depends_on_result& deps);
void depends_on(stmt_ref s, var x, depends_on_result& deps);
depends_on_result depends_on(expr_ref e, var x);
depends_on_result depends_on(const interval_expr& e, var x);
depends_on_result depends_on(stmt_ref s, var x);
depends_on_result depends_on(expr_ref e, span<const var> xs);
depends_on_result depends_on(stmt_ref s, span<const var> xs);

// Check if buffer can be safely substituted.
bool can_substitute_buffer(const depends_on_result& r);

// Check if the node depends on anything that may change value.
bool is_pure(expr_ref x);

// Find all the variables used by a node.
std::vector<var> find_dependencies(expr_ref e);
std::vector<var> find_dependencies(stmt_ref s);

// Find a single buffer data dependency in e.
var find_buffer_data_dependency(expr_ref e);

// Find the buffers used by a stmt or expr. Returns the vars accessed in sorted order.
std::vector<var> find_buffer_dependencies(stmt_ref s);
std::vector<var> find_buffer_dependencies(stmt_ref s, bool input, bool output);
std::vector<var> find_buffer_dependencies(stmt_ref s, bool input, bool src, bool output, bool dst);

}  // namespace slinky

#endif  // SLINKY_RUNTIME_DEPENDS_ON_H
