#ifndef SLINKY_BUILDER_INFER_BOUNDS_H
#define SLINKY_BUILDER_INFER_BOUNDS_H

#include <vector>

#include "runtime/stmt.h"

namespace slinky {

stmt slide_and_fold_storage(const stmt& s, node_context& ctx);

}  // namespace slinky

#endif  // SLINKY_BUILDER_INFER_BOUNDS_H
