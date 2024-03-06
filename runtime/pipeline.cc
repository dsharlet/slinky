#include "runtime/pipeline.h"

#include <vector>

#include "runtime/evaluate.h"
#include "runtime/expr.h"

namespace slinky {

pipeline::pipeline(std::vector<var> args, std::vector<var> inputs, std::vector<var> outputs,
    std::vector<std::pair<symbol_id, const_raw_buffer_ptr>> constants, stmt body)
    : args_(std::move(args)), inputs_(std::move(inputs)), outputs_(std::move(outputs)), constants_(std::move(constants)), body_(std::move(body)) {}

index_t pipeline::evaluate(scalars args, buffers inputs, buffers outputs, eval_context& ctx) const {
  assert(args.size() == args_.size());
  assert(inputs.size() == inputs_.size());
  assert(outputs.size() == outputs_.size());

  for (std::size_t i = 0; i < args.size(); ++i) {
    ctx[args_[i]] = args[i];
  }
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    ctx[inputs_[i]] = reinterpret_cast<index_t>(inputs[i]);
  }
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    ctx[outputs_[i]] = reinterpret_cast<index_t>(outputs[i]);
  }
  for (const auto& i : constants_) {
    ctx[i.first] = reinterpret_cast<index_t>(i.second.get());
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