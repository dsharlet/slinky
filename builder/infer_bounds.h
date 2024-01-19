#ifndef SLINKY_BUILDER_INFER_BOUNDS_H
#define SLINKY_BUILDER_INFER_BOUNDS_H

#include <vector>

#include "runtime/expr.h"

namespace slinky {

stmt infer_bounds(const stmt& s, node_context& ctx, const std::vector<symbol_id>& inputs);

}  // namespace slinky

#endif  // SLINKY_BUILDER_INFER_BOUNDS_H
