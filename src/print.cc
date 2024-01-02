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

std::ostream& operator<<(std::ostream& os, intrinsic i) {
  switch (i) {
  case intrinsic::positive_infinity: return os << "oo";
  case intrinsic::negative_infinity: return os << "-oo";
  case intrinsic::indeterminate: return os << "indeterminate";
  case intrinsic::abs: return os << "abs";
  case intrinsic::buffer_rank: return os << "buffer_rank";
  case intrinsic::buffer_base: return os << "buffer_base";
  case intrinsic::buffer_elem_size: return os << "buffer_elem_size";
  case intrinsic::buffer_size_bytes: return os << "buffer_size_bytes";
  case intrinsic::buffer_min: return os << "buffer_min";
  case intrinsic::buffer_max: return os << "buffer_max";
  case intrinsic::buffer_stride: return os << "buffer_stride";
  case intrinsic::buffer_fold_factor: return os << "buffer_fold_factor";
  case intrinsic::buffer_extent: return os << "buffer_extent";
  case intrinsic::buffer_at: return os << "buffer_at";

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

  void print(symbol_id id) {
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

  template <typename T>
  void print_vector(const std::vector<T>& v, const std::string& sep = ", ") {
    for (std::size_t i = 0; i < v.size(); ++i) {
      print(v[i]);
      if (i + 1 < v.size()) {
        os << sep;
      }
    }
  }

  void print(const stmt& s) { s.accept(this); }

  std::string indent() const { return std::string(depth, ' '); }

  void visit(const variable* v) override { print(v->name); }
  void visit(const wildcard* w) override { print(w->name); }
  void visit(const constant* c) override { os << c->value; }

  void visit(const let* l) override {
    os << "let ";
    print(l->name);
    os << " = ";
    print(l->value);
    os << " in ";
    print(l->body);
  }

  void visit(const let_stmt* l) override {
    os << indent() << "let ";
    print(l->name);
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
  void visit(const logical_not* x) override {
    os << "!";
    print(x->x);
  }

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

  void visit(const call* x) override {
    os << x->intrinsic << "(";
    print_vector(x->args);
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
    print(l->name);
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
    print_vector(n->scalar_args);
    os << "}, {";
    print_vector(n->buffer_args);
    os << "})" << std::endl;
  }

  void visit(const allocate* n) override {
    os << indent();
    print(n->name);
    os << " = allocate<" << n->elem_size << ">({" << std::endl;
    ++depth;
    os << indent();
    print_vector(n->dims, ",\n" + indent());
    os << std::endl;
    --depth;
    os << indent() << "} on " << n->storage << ") {" << std::endl;
    ++depth;
    print(n->body);
    --depth;
    os << indent() << "}" << std::endl;
  }

  void visit(const make_buffer* n) override {
    os << indent();
    print(n->name);
    os << " = make_buffer(";
    print(n->base);
    os << ", ";
    print(n->elem_size);
    os << ", {";
    if (!n->dims.empty()) {
      os << std::endl;
      ++depth;
      os << indent();
      print_vector(n->dims, ",\n" + indent());
      os << std::endl;
      --depth;
      os << indent();
    }
    os << "}) {" << std::endl;
    ++depth;
    print(n->body);
    --depth;
    os << indent() << "}" << std::endl;
  }

  void visit(const crop_buffer* n) override {
    os << indent();
    os << "crop_buffer(";
    print(n->name);
    os << ", {";
    if (!n->bounds.empty()) {
      os << std::endl;
      ++depth;
      os << indent();
      print_vector(n->bounds, ",\n" + indent());
      os << std::endl;
      --depth;
      os << indent();
    }
    os << "}) {" << std::endl;
    ++depth;
    print(n->body);
    --depth;
    os << indent() << "}" << std::endl;
  }

  void visit(const crop_dim* n) override {
    os << indent();
    os << "crop_dim<" << n->dim << ">(";
    print(n->name);
    os << ", ";
    print(n->bounds);
    os << ") {" << std::endl;
    ++depth;
    print(n->body);
    --depth;
    os << indent() << "}" << std::endl;
  }

  void visit(const slice_buffer* n) override {
    os << indent();
    os << "slice_buffer(";
    print(n->name);
    os << ", {";
    if (!n->at.empty()) {
      os << std::endl;
      ++depth;
      os << indent();
      print_vector(n->at, ",\n" + indent());
      os << std::endl;
      --depth;
      os << indent();
    }
    os << "}) {" << std::endl;
    ++depth;
    print(n->body);
    --depth;
    os << indent() << "}" << std::endl;
  }

  void visit(const slice_dim* n) override {
    os << indent();
    os << "slice_dim<" << n->dim << ">(";
    print(n->name);
    os << ", ";
    print(n->at);
    os << ") {" << std::endl;
    ++depth;
    print(n->body);
    --depth;
    os << indent() << "}" << std::endl;
  }

  void visit(const truncate_rank* n) override {
    os << indent();
    os << "truncate_rank<" << n->rank << ">(";
    print(n->name);
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
