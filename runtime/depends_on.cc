#include "runtime/depends_on.h"

#include <cassert>

#include "runtime/expr.h"
#include "runtime/util.h"

namespace slinky {

namespace {

class dependencies : public recursive_node_visitor {
public:
  span<const symbol_id> vars;
  bool found_var = false;
  bool found_buf = false;

  dependencies(span<const symbol_id> vars) : vars(vars) {}

  void accept_buffer(const expr& e) {
    bool old_found_var = found_var;
    found_var = false;
    e.accept(this);
    found_buf = found_buf || found_var;
    found_var = old_found_var;
  }

  void visit_var(symbol_id sym) {
    for (symbol_id i : vars) {
      if (i == sym) {
        found_var = true;
        return;
      }
    }
  }

  void visit_buf(symbol_id sym) {
    for (symbol_id i : vars) {
      if (i == sym) {
        found_buf = true;
        return;
      }
    }
  }

  void visit(const variable* op) override { visit_var(op->sym); }
  void visit(const wildcard* op) override { visit_var(op->sym); }
  void visit(const call* op) override {
    if (is_buffer_intrinsic(op->intrinsic)) {
      assert(op->args.size() >= 1);
      accept_buffer(op->args[0]);

      for (std::size_t i = 1; i < op->args.size(); ++i) {
        if (op->args[i].defined()) op->args[i].accept(this);
      }
    } else {
      recursive_node_visitor::visit(op);
    }
  }

  void visit(const call_stmt* op) override {
    for (symbol_id i : op->inputs) {
      visit_buf(i);
    }
    for (symbol_id i : op->outputs) {
      visit_buf(i);
    }
  }
  void visit(const copy_stmt* op) override {
    visit_buf(op->src);
    visit_buf(op->dst);
  }
};

}  // namespace

bool depends_on(const expr& e, symbol_id var) {
  if (!e.defined()) return false;
  symbol_id vars[] = {var};
  dependencies v(vars);
  e.accept(&v);
  return v.found_var || v.found_buf;
}

bool depends_on(const interval_expr& e, symbol_id var) {
  symbol_id vars[] = {var};
  dependencies v(vars);
  if (e.min.defined()) e.min.accept(&v);
  if (e.max.defined()) e.max.accept(&v);
  return v.found_var || v.found_buf;
}

bool depends_on(const stmt& s, symbol_id var) {
  if (!s.defined()) return false;
  symbol_id vars[] = {var};
  dependencies v(vars);
  s.accept(&v);
  return v.found_var || v.found_buf;
}

bool depends_on(const stmt& s, span<const symbol_id> vars) {
  if (!s.defined()) return false;
  dependencies v(vars);
  s.accept(&v);
  return v.found_var || v.found_buf;
}

bool depends_on_variable(const expr& e, symbol_id var) {
  if (!e.defined()) return false;
  symbol_id vars[] = {var};
  dependencies v(vars);
  e.accept(&v);
  return v.found_var;
}

bool depends_on_buffer(const expr& e, symbol_id buf) {
  if (!e.defined()) return false;
  symbol_id bufs[] = {buf};
  dependencies v(bufs);
  e.accept(&v);
  return v.found_buf;
}

}  // namespace slinky
