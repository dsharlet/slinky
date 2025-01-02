#ifndef SLINKY_RUNTIME_VALIDATE_H
#define SLINKY_RUNTIME_VALIDATE_H

#include "base/span.h"
#include "runtime/expr.h"
#include "runtime/stmt.h"

namespace slinky {

bool is_valid(const expr& e, span<var> external, const node_context* symbols);
bool is_valid(const stmt& s, span<var> external, const node_context* symbols);

}  // namespace slinky

#endif  // SLINKY_RUNTIME_VALIDATE_H
