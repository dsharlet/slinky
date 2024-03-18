#ifndef SLINKY_BUILDER_OPTIMIZATIONS_H
#define SLINKY_BUILDER_OPTIMIZATIONS_H

#include "runtime/stmt.h"

namespace slinky {

// Where possible, rewrite copies as buffer metadata rewrites.
stmt alias_buffers(const stmt& s);

// Given a copy_stmt, produce an implementation that calls `slinky::copy`, possibly inside loops that implement copy
// operations that `slinky::copy` cannot express.
stmt implement_copy(const copy_stmt* c, node_context& ctx);

// Replace every `copy_stmt` with the result of `implement_copy`
stmt implement_copies(const stmt& s, node_context& ctx);

// We can't modify buffers allocated outside parallel loops inside parallel loops. To avoid this, this mutation will
// insert `clone_buffer` operations that clone buffers inside parallel loops.
stmt fix_buffer_races(const stmt& s);

}  // namespace slinky

#endif  // SLINKY_BUILDER_OPTIMIZATIONS_H
