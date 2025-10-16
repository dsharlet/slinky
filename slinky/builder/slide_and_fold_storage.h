#ifndef SLINKY_BUILDER_SLIDE_AND_FOLD_STORAGE_H
#define SLINKY_BUILDER_SLIDE_AND_FOLD_STORAGE_H

#include "slinky/runtime/stmt.h"

namespace slinky {

stmt slide_and_fold_storage(const stmt& s, node_context& ctx);

}  // namespace slinky

#endif  // SLINKY_BUILDER_SLIDE_AND_FOLD_STORAGE_H
