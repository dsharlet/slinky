#include "runtime/pipeline.h"

#include <vector>

#include "runtime/evaluate.h"
#include "runtime/expr.h"

namespace slinky {

pipeline::pipeline(std::vector<var> args, std::vector<var> inputs, std::vector<var> outputs, stmt body)
    : args_(std::move(args)), inputs_(std::move(inputs)), outputs_(std::move(outputs)), body_(std::move(body)) {}

pipeline::pipeline(std::vector<var> inputs, std::vector<var> outputs, stmt body)
    : pipeline({}, std::move(inputs), std::move(outputs), std::move(body)) {}

index_t pipeline::evaluate(scalars args, buffers inputs, buffers outputs, eval_context& ctx) const {
  assert(args.size() == args_.size());
  assert(inputs.size() == inputs_.size());
  assert(outputs.size() == outputs_.size());

  for (std::size_t i = 0; i < args.size(); ++i) {
    ctx.symbols()[args_[i]] = args[i];
  }
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    ctx.symbols()[inputs_[i]] = reinterpret_cast<index_t>(inputs[i]);
  }
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    ctx.symbols()[outputs_[i]] = reinterpret_cast<index_t>(outputs[i]);
  }

  return slinky::evaluate(body_, ctx);
}

index_t pipeline::evaluate(buffers inputs, buffers outputs, eval_context& ctx) const {
  return evaluate({}, inputs, outputs, ctx);
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