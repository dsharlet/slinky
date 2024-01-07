#ifndef SLINKY_OPTIMIZATIONS_H
#define SLINKY_OPTIMIZATIONS_H

#include "expr.h"

namespace slinky {

// Where possible, rewrite copies as buffer metadata rewrites.
stmt alias_buffers(const stmt& s);

// Find copy operations that can be implemented with calls to copy.
stmt optimize_copies(const stmt& s);

// Attempt to reduce the scope of statements to only the operations required.
stmt reduce_scopes(const stmt& s);

// Find buffers that can be re-used.
stmt optimize_allocations(const stmt& s);

}  // namespace slinky

#endif  // SLINKY_OPTIMIZATIONS_H
