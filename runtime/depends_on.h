#ifndef SLINKY_RUNTIME_DEPENDS_ON_H
#define SLINKY_RUNTIME_DEPENDS_ON_H

#include "runtime/expr.h"
#include "runtime/stmt.h"

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

  // True if the buffer metadata is read or written.
  bool buffer_meta_read = false;
  bool buffer_meta_mutated = false;

  // How many references there are.
  int ref_count = 0;

  bool buffer_data() const { return buffer_input || buffer_output || buffer_src || buffer_dst; }
  bool buffer_meta() const { return buffer_meta_read || buffer_meta_mutated; }
  bool buffer() const { return buffer_data() || buffer_meta(); }

  bool any() const { return var || buffer(); }
};

// Check if the node depends on a symbol or set of symbols.

void depends_on(const expr& e, span<const std::pair<var, depends_on_result&>> var_deps);
void depends_on(const stmt& s, span<const std::pair<var, depends_on_result&>> var_deps);
void depends_on(const expr& e, var x, depends_on_result& deps);
void depends_on(const stmt& s, var x, depends_on_result& deps);
depends_on_result depends_on(const expr& e, var x);
depends_on_result depends_on(const interval_expr& e, var x);
depends_on_result depends_on(const stmt& s, var x);
depends_on_result depends_on(const expr& e, span<const var> xs);
depends_on_result depends_on(const stmt& s, span<const var> xs);

}  // namespace slinky

#endif  // SLINKY_RUNTIME_DEPENDS_ON_H
