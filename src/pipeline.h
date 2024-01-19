#ifndef SLINKY_PIPELINE_H
#define SLINKY_PIPELINE_H

#include <vector>

#include "src/expr.h"

namespace slinky {

class eval_context;

class pipeline {
  std::vector<var> args_;
  std::vector<var> inputs_;
  std::vector<var> outputs_;

  stmt body_;

public:
  pipeline(std::vector<var> args, std::vector<var> inputs, std::vector<var> outputs, stmt body);
  pipeline(std::vector<var> inputs, std::vector<var> outputs, stmt body);

  using scalars = span<const index_t>;
  using buffers = span<const raw_buffer*>;

  index_t evaluate(scalars args, buffers inputs, buffers outputs, eval_context& ctx) const;
  index_t evaluate(buffers inputs, buffers outputs, eval_context& ctx) const;
  index_t evaluate(scalars args, buffers inputs, buffers outputs) const;
  index_t evaluate(buffers inputs, buffers outputs) const;

  const std::vector<var>& args() const { return args_; }
  const std::vector<var>& inputs() const { return inputs_; }
  const std::vector<var>& outputs() const { return outputs_; }
  const stmt& body() const { return body_; }
};

}  // namespace slinky

#endif  // SLINKY_PIPELINE_H
