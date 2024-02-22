#include "runtime/depends_on.h"

#include <cassert>

#include "runtime/expr.h"
#include "runtime/util.h"

namespace slinky {

namespace {

class dependencies : public recursive_node_visitor {
public:
  span<const symbol_id> vars;
  depends_on_result result;

  dependencies(span<const symbol_id> vars) : vars(vars) {}

  bool vars_contains(symbol_id i) const { return std::find(vars.begin(), vars.end(), i) != vars.end(); }

  void accept_buffer(const expr& e, bool uses_base) {
    bool old_found_var = result.var;
    result.var = false;
    e.accept(this);
    result.buffer = result.buffer || result.var;
    result.buffer_base = result.buffer_base || (uses_base && result.var);
    result.var = old_found_var;
  }

  void visit(const variable* op) override { result.var = result.var || vars_contains(op->sym); }
  void visit(const call* op) override {
    if (is_buffer_intrinsic(op->intrinsic)) {
      assert(op->args.size() >= 1);
      accept_buffer(op->args[0], op->intrinsic == intrinsic::buffer_at);

      for (std::size_t i = 1; i < op->args.size(); ++i) {
        if (op->args[i].defined()) op->args[i].accept(this);
      }
    } else {
      recursive_node_visitor::visit(op);
    }
  }

  void visit(const call_stmt* op) override {
    for (symbol_id i : op->inputs) {
      if (vars_contains(i)) {
        result.buffer = true;
        result.buffer_input = true;
      }
    }
    for (symbol_id i : op->outputs) {
      if (vars_contains(i)) {
        result.buffer = true;
        result.buffer_output = true;
      }
    }
  }

  void visit(const copy_stmt* op) override {
    if (vars_contains(op->src)) {
      result.buffer = true;
      result.buffer_src = true;
    }
    if (vars_contains(op->dst)) {
      result.buffer = true;
      result.buffer_dst = true;
    }
  }

  void visit(const clone_buffer* op) override {
    if (vars_contains(op->src)) {
      result.buffer = true;
      result.buffer_base = true;
    }
    recursive_node_visitor::visit(op);
  }
};

}  // namespace

depends_on_result depends_on(const expr& e, symbol_id var) {
  symbol_id vars[] = {var};
  dependencies v(vars);
  if (e.defined()) e.accept(&v);
  return v.result;
}

depends_on_result depends_on(const interval_expr& e, symbol_id var) {
  symbol_id vars[] = {var};
  dependencies v(vars);
  if (e.min.defined()) e.min.accept(&v);
  if (e.max.defined()) e.max.accept(&v);
  return v.result;
}

depends_on_result depends_on(const stmt& s, symbol_id var) {
  symbol_id vars[] = {var};
  dependencies v(vars);
  if (s.defined()) s.accept(&v);
  return v.result;
}

depends_on_result depends_on(const expr& e, span<const symbol_id> vars) {
  dependencies v(vars);
  if (e.defined()) e.accept(&v);
  return v.result;
}

depends_on_result depends_on(const stmt& s, span<const symbol_id> vars) {
  dependencies v(vars);
  if (s.defined()) s.accept(&v);
  return v.result;
}

}  // namespace slinky
