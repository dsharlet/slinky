#ifndef SLINKY_RUNTIME_DEPENDS_ON_H
#define SLINKY_RUNTIME_DEPENDS_ON_H

#include "runtime/expr.h"

namespace slinky {

// The various ways a node might depend on a buffer or variable.
struct depends_on_result {
  // True if the node depends on the symbol as a variable.
  bool var = false;

  // The remaining fields all indicate the symbol is used as a buffer.
  bool buffer = false;
  // True if the buffer's base pointer is used.
  bool buffer_base = false;
  // True if the buffer is used as a call input or output, respectively.
  bool buffer_input = false;
  bool buffer_output = false;
  // True if the buffer is used as a copy source or destination, respectively.
  bool buffer_src = false;
  bool buffer_dst = false;

  bool any() const { return var || buffer; }
};

// Check if the node depends on a symbol or set of symbols.
depends_on_result depends_on(const expr& e, symbol_id var);
depends_on_result depends_on(const interval_expr& e, symbol_id var);
depends_on_result depends_on(const stmt& s, symbol_id var);
depends_on_result depends_on(const expr& e, span<const symbol_id> vars);
depends_on_result depends_on(const stmt& s, span<const symbol_id> vars);

}  // namespace slinky

#endif  // SLINKY_RUNTIME_DEPENDS_ON_H
