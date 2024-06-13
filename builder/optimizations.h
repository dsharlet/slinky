#ifndef SLINKY_BUILDER_OPTIMIZATIONS_H
#define SLINKY_BUILDER_OPTIMIZATIONS_H

#include "builder/pipeline.h"
#include "runtime/stmt.h"

namespace slinky {

// Where possible, replace `allocate` with `make_buffer` referring to another buffer with appropriate metadata:
// - `copy_stmt`s that only do simple copies or broadcasts in each dimension.
// - `call_stmt`s that can be in-place may be able to replace an allocation for the output with an alias of an input.
stmt alias_buffers(const stmt& s, node_context& ctx, const std::vector<buffer_expr_ptr>& inputs,
    const std::vector<buffer_expr_ptr>& outputs);

// Given a copy_stmt, produce an implementation that calls `slinky::copy`, possibly inside loops that implement copy
// operations that `slinky::copy` cannot express.
stmt implement_copy(const copy_stmt* c, node_context& ctx);

// Replace every `copy_stmt` with the result of `implement_copy`
stmt implement_copies(const stmt& s, node_context& ctx);

// Find allocate nodes and try to insert free into them.
stmt insert_early_free(const stmt& s);

// The simplifier can't handle shadowed symbols. This mutator rewrites all declarations to avoid any shadowing.
stmt deshadow(const stmt& s, node_context& ctx);

// We can improve `evaluate`'s performance and memory usage if:
// - Buffer mutators are self-shadowing, so they can be performed in-place on existing buffers.
// - Symbols are indexed such that there are no unused symbol indices.
stmt optimize_symbols(const stmt& s, node_context& ctx);

}  // namespace slinky

#endif  // SLINKY_BUILDER_OPTIMIZATIONS_H
