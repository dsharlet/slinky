#include "runtime/pipeline.h"

#include <vector>

#include "runtime/evaluate.h"
#include "runtime/expr.h"

namespace slinky {

void pipeline::setup(scalars args, buffers inputs, buffers outputs, eval_context& ctx) const {
  assert(args.size() == this->args.size());
  assert(inputs.size() == this->inputs.size());
  assert(outputs.size() == this->outputs.size());

  for (std::size_t i = 0; i < args.size(); ++i) {
    ctx[this->args[i]] = args[i];
  }
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    ctx[this->inputs[i]] = reinterpret_cast<index_t>(inputs[i]);
  }
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    ctx[this->outputs[i]] = reinterpret_cast<index_t>(outputs[i]);
  }
}

void pipeline::setup(buffers inputs, buffers outputs, eval_context& ctx) const {
  setup({}, inputs, outputs, ctx);
}

index_t pipeline::evaluate(eval_context& ctx) const { return slinky::evaluate(body, ctx); }

index_t pipeline::evaluate(scalars args, buffers inputs, buffers outputs, eval_context& ctx) const {
  setup(args, inputs, outputs, ctx);
  return slinky::evaluate(body, ctx);
}

index_t pipeline::evaluate(buffers inputs, buffers outputs, eval_context& ctx) const {
  setup(inputs, outputs, ctx);
  return slinky::evaluate(body, ctx);
}

index_t pipeline::evaluate(scalars args, buffers inputs, buffers outputs) const {
  eval_context ctx;
  return evaluate(args, inputs, outputs, ctx);
}

index_t pipeline::evaluate(buffers inputs, buffers outputs) const {
  eval_context ctx;
  return evaluate(scalars(), inputs, outputs, ctx);
}

}  // namespace slinky