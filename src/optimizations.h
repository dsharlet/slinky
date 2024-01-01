#ifndef SLINKY_OPTIMIZATIONS_H
#define SLINKY_OPTIMIZATIONS_H

#include "expr.h"

namespace slinky {

// Where possible, rewrite copies as buffer metadata rewrites.
stmt alias_buffers(const stmt& s);

// Lower copy operations to the necessary calls to implement the copy.
stmt implement_copies(const stmt& s, node_context& ctx);

// Find buffers that can be re-used.
stmt optimize_allocations(const stmt& s);

}  // namespace slinky

#endif  // SLINKY_OPTIMIZATIONS_H
