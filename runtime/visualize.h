#ifndef SLINKY_RUNTIME_VISUALIZE_H
#define SLINKY_RUNTIME_VISUALIZE_H

#include <iostream>

#include "runtime/pipeline.h"

namespace slinky {

// Generate an HTML file that visualizes the evaluation of a pipeline given a set of scalar `args`, and buffers `inputs`
// and `outputs`.
void visualize(std::ostream& dst, const pipeline& p, pipeline::scalars args, pipeline::buffers inputs,
    pipeline::buffers outputs, const node_context* ctx = nullptr);
void visualize(std::ostream& dst, const pipeline& p, pipeline::buffers inputs, pipeline::buffers outputs,
    const node_context* ctx = nullptr);

}  // namespace slinky

#endif  // SLINKY_RUNTIME_VISUALIZE_H
