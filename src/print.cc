#include "evaluate.h"

#include <cassert>
#include <iostream>

namespace slinky {
 
class printer : public node_visitor {
public:
  std::ostream& os;
  const node_context* context;

  printer(std::ostream& os, const node_context* context) : os(os), context(context) {}

  void print_symbol_id(symbol_id id) {
    if (context) {
      os << context->name(id);
    } else {
      os << "<" << id << ">";
    }
  }
  
  void print(const expr& e) {
    if (e.defined()) {
      e.accept(this);
    } else {
      os << "<>";
    }
  }

  void visit(const variable* v) override {
    print_symbol_id(v->name);
  }
  void visit(const constant* c) override {
    os << c->value;
  }

  void visit(const let* l) override { 
    os << "let ";
    print_symbol_id(l->name);
    os << " = ";
    print(l->value);
    os << " in ";
    print(l->body);
  }
  
  template <typename T>
  void visit_bin_op(const T* op, const char* s) {
    os << "(";
    print(op->a);
    os << s;
    print(op->b);
    os << ")";
  }

  void visit(const add* x) override { visit_bin_op(x, " + "); }
  void visit(const sub* x) override { visit_bin_op(x, " - "); }
  void visit(const mul* x) override { visit_bin_op(x, " * "); }
  void visit(const div* x) override { visit_bin_op(x, " / "); }
  void visit(const mod* x) override { visit_bin_op(x, " % "); }
  void visit(const equal* x) override { visit_bin_op(x, " == "); }
  void visit(const not_equal* x) override { visit_bin_op(x, " != "); }
  void visit(const less* x) override { visit_bin_op(x, " < "); }
  void visit(const less_equal* x) override { visit_bin_op(x, " <= "); }
  void visit(const bitwise_and* x) override { visit_bin_op(x, " & "); }
  void visit(const bitwise_or* x) override { visit_bin_op(x, " | "); }
  void visit(const bitwise_xor* x) override { visit_bin_op(x, " ^ "); }  
  void visit(const logical_and* x) override { visit_bin_op(x, " && "); }
  void visit(const logical_or* x) override { visit_bin_op(x, " || "); }

  void visit(const class min* op) override {
    os << "min(";
    print(op->a);
    os << ", ";
    print(op->b);
    os << ")";
  }

  void visit(const class max* op) override {
    os << "max(";
    print(op->a);
    os << ", ";
    print(op->b);
    os << ")";
  }
};

void print(std::ostream& os, const expr& e, const node_context* ctx = nullptr) {
  printer p(os, ctx);
  e.accept(&p);
}

std::ostream& operator<<(std::ostream& os, const expr& e) {
  print(os, e);
  return os;
}

}  // namespace slinky