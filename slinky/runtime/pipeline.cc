#include "slinky/runtime/pipeline.h"

#include <vector>

#include "slinky/runtime/evaluate.h"
#include "slinky/runtime/expr.h"

namespace slinky {

namespace {

const let_stmt* as_constant_let(const stmt& s) {
  const let_stmt* let = s.as<let_stmt>();
  return let && let->is_constant ? let : nullptr;
}

}  // namespace

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

  const stmt* s = &body;
  while (const let_stmt* let = as_constant_let(*s)) {
    ctx.reserve(let->max_symbol_id + 1);
    for (const auto& i : let->lets) {
      switch (i.second.type()) {
      case expr_node_type::constant: ctx[i.first] = i.second.as<constant>()->value; break;
      case expr_node_type::constant_buffer:
        ctx[i.first] = reinterpret_cast<index_t>(i.second.as<constant_buffer>()->value.get());
        break;
      default: SLINKY_UNREACHABLE;
      }
    }
    s = &let->body;
  }
}

void pipeline::setup(buffers inputs, buffers outputs, eval_context& ctx) const { setup({}, inputs, outputs, ctx); }

index_t pipeline::evaluate(eval_context& ctx, bool is_set_up) const {
  const stmt* s = &body;
  if (is_set_up) {
    // This context has already been set up, we can skip the constant lets.
    while (const let_stmt* let = as_constant_let(*s)) {
      s = &let->body;
    }
  }
  return slinky::evaluate(*s, ctx);
}

index_t pipeline::evaluate(scalars args, buffers inputs, buffers outputs, eval_context& ctx) const {
  setup(args, inputs, outputs, ctx);
  return evaluate(ctx, /*is_set_up=*/true);
}

index_t pipeline::evaluate(buffers inputs, buffers outputs, eval_context& ctx) const {
  setup(inputs, outputs, ctx);
  return evaluate(ctx, /*is_set_up=*/true);
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
