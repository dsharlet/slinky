#ifndef SLINKY_RUNTIME_PIPELINE_H
#define SLINKY_RUNTIME_PIPELINE_H

#include <vector>

#include "runtime/expr.h"
#include "runtime/stmt.h"

namespace slinky {

class eval_context;

// This object essentially only stores the mapping of arguments to symbols.
class pipeline {
public:
  std::vector<var> args;
  std::vector<var> inputs;
  std::vector<var> outputs;
  std::vector<std::pair<symbol_id, const_raw_buffer_ptr>> constants;

  stmt body;

  using scalars = span<const index_t>;
  using buffers = span<const raw_buffer*>;

  index_t evaluate(scalars args, buffers inputs, buffers outputs, eval_context& ctx) const;
  index_t evaluate(buffers inputs, buffers outputs, eval_context& ctx) const;
  index_t evaluate(scalars args, buffers inputs, buffers outputs) const;
  index_t evaluate(buffers inputs, buffers outputs) const;
};

}  // namespace slinky

#endif  // SLINKY_RUNTIME_PIPELINE_H
