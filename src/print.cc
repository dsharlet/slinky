#include "evaluate.h"

#include <cassert>
#include <iostream>

namespace slinky {
 
std::ostream& operator<<(std::ostream& os, memory_type type) {
  switch (type) {
  case memory_type::stack: return os << "stack";
  case memory_type::heap: return os << "heap";
  }
}

class printer : public node_visitor {
public:
  int depth = 0;
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

  void print(const stmt& s) {
    s.accept(this);
  }

  std::string indent() const {
    return std::string(depth, ' ');
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

  void visit(const let_stmt* l) override {
    os << indent() << "let ";
    print_symbol_id(l->name);
    os << " = ";
    print(l->value);
    os << " in " << std::endl;
    ++depth;
    indent();
    print(l->body);
    --depth;
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
  void visit(const shift_left* x) override { visit_bin_op(x, " << "); }
  void visit(const shift_right* x) override { visit_bin_op(x, " >> "); }

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

  void visit(const block* b) override {
    print(b->a);
    print(b->b);
  }

  void visit(const loop* l) override {
    os << indent() << "loop(";
    print_symbol_id(l->name);
    os << " in [0, ";
    print(l->n);
    os << ")) {" << std::endl;
    ++depth;
    print(l->body);
    --depth;
    os << indent() << "}" << std::endl;
  }

  void visit(const if_then_else* n) override {
    os << indent() << "if(";
    print(n->condition);
    os << ") {" << std::endl;
    ++depth;
    print(n->true_body);
    --depth;
    os << indent() << "} else {";
    if (n->false_body.defined()) {
      os << std::endl;
      ++depth;
      print(n->false_body);
      --depth;
      os << indent();
    }
    os << "}" << std::endl;
  }

  void visit(const call* n) override {
    os << indent() << "call(<fn>, {";
    for (const expr& e : n->scalar_args) {
      print(e);
      if (&e != &n->scalar_args.back()) {
        os << ", ";
      }
    }
    os << "}, {";
    for (const expr& e : n->buffer_args) {
      print(e);
      if (&e != &n->scalar_args.back()) {
        os << ", ";
      }
    }
    os << "})" << std::endl;
  }

  void visit(const allocate* n) override {
    os << indent();
    print_symbol_id(n->name);
    os << " = allocate({";
    for (const dim_expr& d : n->dims) {
      os << "{";
      print(d.min);
      os << ", ";
      print(d.extent);
      os << ", ";
      print(d.stride_bytes);
      os << ", ";
      print(d.fold_factor);
      os << "}";
      if (&d != &n->dims.back()) {
        os << ", ";
      }
    }
    os << "} on " << n->type << ") {" << std::endl;
    ++depth;
    print(n->body);
    --depth;
    os << indent() << "}" << std::endl;
  }

  void visit(const check* n) override {
    os << indent();
    os << "check(";
    print(n->condition);
    os << ")";
  }
};

void print(std::ostream& os, const expr& e, const node_context* ctx = nullptr) {
  printer p(os, ctx);
  e.accept(&p);
}

void print(std::ostream& os, const stmt& s, const node_context* ctx = nullptr) {
  printer p(os, ctx);
  s.accept(&p);
}

std::ostream& operator<<(std::ostream& os, const expr& e) {
  print(os, e);
  return os;
}

std::ostream& operator<<(std::ostream& os, const stmt& s) {
  print(os, s);
  return os;
}

}  // namespace slinky
