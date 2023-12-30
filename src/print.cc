#include "print.h"

#include <cassert>
#include <iostream>

namespace slinky {

std::ostream& operator<<(std::ostream& os, memory_type type) {
  switch (type) {
  case memory_type::stack: return os << "stack";
  case memory_type::heap: return os << "heap";
  default: return os << "<invalid memory_type>";
  }
}

std::ostream& operator<<(std::ostream& os, buffer_meta meta) {
  switch (meta) {
  case buffer_meta::rank: return os << "rank";
  case buffer_meta::base: return os << "base";
  case buffer_meta::elem_size: return os << "elem_size";
  case buffer_meta::min: return os << "min";
  case buffer_meta::max: return os << "max";
  case buffer_meta::extent: return os << "extent";
  case buffer_meta::stride: return os << "stride";
  case buffer_meta::fold_factor: return os << "fold_factor";
  default: return os << "<invalid buffer_meta>";
  }
}

std::ostream& operator<<(std::ostream& os, intrinsic i) {
  switch (i) {
  case intrinsic::positive_infinity: return os << "oo";
  case intrinsic::negative_infinity: return os << "-oo";
  case intrinsic::indeterminate: return os << "indeterminate";
  case intrinsic::abs: return os << "abs";
  default: return os << "<invalid intrinsic>";
  }
}

std::ostream& operator<<(std::ostream& os, const interval_expr& i) {
  return os << "[" << i.min << ", " << i.max << "]";
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

  void print(const interval_expr& e) {
    os << "[";
    print(e.min);
    os << ", ";
    print(e.max);
    os << "]";
  }

  void print(const dim_expr& d) {
    os << "{";
    print(d.bounds);
    os << ", ";
    print(d.stride);
    os << ", ";
    print(d.fold_factor);
    os << "}";
  }

  void print(const stmt& s) { s.accept(this); }

  std::string indent() const { return std::string(depth, ' '); }

  void visit(const variable* v) override { print_symbol_id(v->name); }
  void visit(const wildcard* w) override { print_symbol_id(w->name); }
  void visit(const constant* c) override { os << c->value; }

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
    os << " { " << std::endl;
    ++depth;
    indent();
    print(l->body);
    --depth;
    os << indent() << "}" << std::endl;
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

  void visit(const class select* op) override {
    os << "select(";
    print(op->condition);
    os << ", ";
    print(op->true_value);
    os << ", ";
    print(op->false_value);
    os << ")";
  }

  void visit(const load_buffer_meta* x) override {
    os << "buffer_" << x->meta << "(";
    print(x->buffer);
    switch (x->meta) {
    case buffer_meta::base:
    case buffer_meta::rank:
    case buffer_meta::elem_size: break;
    default:
      os << ", ";
      print(x->dim);
      break;
    }
    os << ")";
  }

  void visit(const call* x) override {
    os << x->intrinsic << "(";
    for (const expr& i : x->args) {
      print(i);
      if (!i.same_as(x->args.back())) {
        os << ", ";
      }
    }
    os << ")";
  }

  void visit(const block* b) override {
    if (b->a.defined()) {
      print(b->a);
    }
    if (b->b.defined()) {
      print(b->b);
    }
  }

  void visit(const loop* l) override {
    os << indent() << "loop(";
    print_symbol_id(l->name);
    os << " in ";
    print(l->bounds);
    os << ") {" << std::endl;
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
    if (n->false_body.defined()) {
      os << indent() << "} else {" << std::endl;
      ++depth;
      print(n->false_body);
      --depth;
    }
    os << indent() << "}" << std::endl;
  }

  void visit(const call_func* n) override {
    os << indent() << "call(<fn>, {";
    for (const expr& e : n->scalar_args) {
      print(e);
      if (&e != &n->scalar_args.back()) {
        os << ", ";
      }
    }
    os << "}, {";
    for (symbol_id id : n->buffer_args) {
      print_symbol_id(id);
      if (id != n->buffer_args.back()) {
        os << ", ";
      }
    }
    os << "})" << std::endl;
  }

  void visit(const allocate* n) override {
    os << indent();
    print_symbol_id(n->name);
    os << " = allocate<" << n->elem_size << ">({" << std::endl;
    ++depth;
    for (const dim_expr& d : n->dims) {
      os << indent();
      print(d);
      if (&d != &n->dims.back()) {
        os << ", ";
      }
      os << std::endl;
    }
    --depth;
    os << indent() << "} on " << n->type << ") {" << std::endl;
    ++depth;
    print(n->body);
    --depth;
    os << indent() << "}" << std::endl;
  }

  void visit(const make_buffer* n) override {
    os << indent();
    print_symbol_id(n->name);
    os << " = make_buffer<" << n->elem_size << ">(";
    print(n->base);
    os << ", {" << std::endl;
    ++depth;
    for (const dim_expr& d : n->dims) {
      os << indent();
      print(d);
      if (&d != &n->dims.back()) {
        os << ", ";
      }
      os << std::endl;
    }
    --depth;
    os << indent() << "}) {" << std::endl;
    ++depth;
    print(n->body);
    --depth;
    os << indent() << "}" << std::endl;
  }

  void visit(const crop_buffer* n) override {
    os << indent();
    os << "crop_buffer(";
    print_symbol_id(n->name);
    os << ", {" << std::endl;
    ++depth;
    for (const interval_expr& d : n->bounds) {
      os << indent();
      print(d);
      if (&d != &n->bounds.back()) {
        os << ", ";
      }
      os << std::endl;
    }
    --depth;
    os << indent() << "}) {" << std::endl;
    ++depth;
    print(n->body);
    --depth;
    os << indent() << "}" << std::endl;
  }

  void visit(const crop_dim* n) override {
    os << indent();
    os << "crop_dim<" << n->dim << ">(";
    print_symbol_id(n->name);
    os << ", ";
    print(n->min);
    os << ", ";
    print(n->extent);
    os << ") {" << std::endl;
    ++depth;
    print(n->body);
    --depth;
    os << indent() << "}" << std::endl;
  }

  void visit(const check* n) override {
    os << indent();
    os << "check(";
    print(n->condition);
    os << ")" << std::endl;
  }
};

void print(std::ostream& os, const expr& e, const node_context* ctx) {
  printer p(os, ctx);
  p.print(e);
}

void print(std::ostream& os, const stmt& s, const node_context* ctx) {
  printer p(os, ctx);
  p.print(s);
}

std::ostream& operator<<(std::ostream& os, const expr& e) {
  print(os, e);
  return os;
}

std::ostream& operator<<(std::ostream& os, const stmt& s) {
  print(os, s);
  return os;
}

std::ostream& operator<<(std::ostream& os, const std::tuple<const expr&, const node_context&>& e) {
  print(os, std::get<0>(e), &std::get<1>(e));
  return os;
}

std::ostream& operator<<(std::ostream& os, const std::tuple<const stmt&, const node_context&>& s) {
  print(os, std::get<0>(s), &std::get<1>(s));
  return os;
}

}  // namespace slinky
