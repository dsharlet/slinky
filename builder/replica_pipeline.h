#ifndef SLINKY_BUILDER_REPLICA_PIPELINE_H
#define SLINKY_BUILDER_REPLICA_PIPELINE_H

#include <vector>

#include "builder/pipeline.h"

namespace slinky {

// These take the same arguments as build_pipeline(), but generate C++ code that defines a pipeline of
// the same structure instead of the pipeline itself.
std::string define_replica_pipeline(node_context& ctx, const std::vector<var>& args,
    const std::vector<buffer_expr_ptr>& inputs, const std::vector<buffer_expr_ptr>& outputs,
    const build_options& options = build_options());
std::string define_replica_pipeline(node_context& ctx, const std::vector<buffer_expr_ptr>& inputs,
    const std::vector<buffer_expr_ptr>& outputs, const build_options& options = build_options());

namespace internal {

index_t replica_pipeline_handler(span<const buffer<const void>*> inputs, span<const buffer<void>*> outputs,
    span<func::input> fins, span<std::vector<var>> fout_dims);

}  // namespace internal

}  // namespace slinky

#endif  // SLINKY_BUILDER_REPLICA_PIPELINE_H
