#ifndef SLINKY_INFER_ALLOCATE_BOUNDS_H
#define SLINKY_INFER_ALLOCATE_BOUNDS_H

#include "symbol_map.h"

namespace slinky {

stmt infer_allocate_bounds(const stmt& s, node_context& ctx, symbol_map<std::vector<dim_expr>>& buffers);

}  // namespace slinky

#endif  // SLINKY_INFER_ALLOCATE_BOUNDS_H
