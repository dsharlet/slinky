#ifndef SLINKY_OPTIMIZATIONS_H
#define SLINKY_OPTIMIZATIONS_H

#include "src/expr.h"

namespace slinky {

// Where possible, rewrite copies as buffer metadata rewrites.
stmt alias_buffers(const stmt& s);

// Find copy operations that can be implemented with calls to copy.
stmt optimize_copies(const stmt& s);

// Attempt to reduce the scope of statements to only the operations required.
stmt reduce_scopes(const stmt& s);

// We can't modify buffers allocated outside parallel loops inside parallel loops. To avoid this, this mutation will
// insert `clone_buffer` operations that clone buffers inside parallel loops.
stmt fix_buffer_races(const stmt& s);

}  // namespace slinky

#endif  // SLINKY_OPTIMIZATIONS_H
