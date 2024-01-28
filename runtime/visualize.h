#ifndef SLINKY_RUNTIME_VISUALIZE_H
#define SLINKY_RUNTIME_VISUALIZE_H

#include "runtime/pipeline.h"

namespace slinky {

void visualize(const char* filename, const pipeline& p, pipeline::buffers inputs, pipeline::buffers outputs, const node_context* ctx = nullptr);

}  // namespace slinky

#endif  // SLINKY_RUNTIME_VISUALIZE_H
