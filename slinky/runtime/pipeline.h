#ifndef SLINKY_RUNTIME_PIPELINE_H
#define SLINKY_RUNTIME_PIPELINE_H

#include <vector>

#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

namespace slinky {

class eval_context;

// This object essentially only stores the mapping of arguments to symbols.
class pipeline {
public:
  std::vector<var> args;
  std::vector<var> inputs;
  std::vector<var> outputs;

  stmt body;

  using scalars = span<const index_t>;
  using buffers = span<const raw_buffer*>;

  // Set up the context to run the pipeline, but do not run it.
  void setup(scalars args, buffers inputs, buffers outputs, eval_context& ctx) const;
  void setup(buffers inputs, buffers outputs, eval_context& ctx) const;

  // Run an already set up pipeline.
  index_t evaluate(eval_context& ctx) const;

  // Combines setup + evaluate.
  index_t evaluate(scalars args, buffers inputs, buffers outputs, eval_context& ctx) const;
  index_t evaluate(buffers inputs, buffers outputs, eval_context& ctx) const;
  index_t evaluate(scalars args, buffers inputs, buffers outputs) const;
  index_t evaluate(buffers inputs, buffers outputs) const;
};

}  // namespace slinky

#endif  // SLINKY_RUNTIME_PIPELINE_H
