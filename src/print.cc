#include "src/print.h"

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

std::ostream& operator<<(std::ostream& os, loop_mode mode) {
  switch (mode) {
  case loop_mode::serial: return os << "serial";
  case loop_mode::parallel: return os << "parallel";
  default: return os << "<invalid loop_mode>";
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
  int depth = -1;
  std::ostream& os;
  const node_context* context;

  printer(std::ostream& os, const node_context* context) : os(os), context(context) {}

  template <typename T>
  printer& operator<<(const T& op) {
    os << op;
    return *this;
  }

  printer& operator<<(symbol_id sym) {
    if (context) {
      os << context->name(sym);
    } else {
      os << "<" << sym << ">";
    }
    return *this;
  }

  printer& operator<<(const expr& e) {
    if (e.defined()) {
      e.accept(this);
    } else {
      os << "<>";
    }
    return *this;
  }

  printer& operator<<(const interval_expr& e) { return *this << "[" << e.min << ", " << e.max << "]"; }

  printer& operator<<(const dim_expr& d) {
    return *this << "{" << d.bounds << ", " << d.stride << ", " << d.fold_factor << "}";
  }

  template <typename T>
  void print_vector(const std::vector<T>& v, const std::string& sep = ", ") {
    for (std::size_t i = 0; i < v.size(); ++i) {
      *this << v[i];
      if (i + 1 < v.size()) {
        *this << sep;
      }
    }
  }

  void print_vector(const std::vector<char>& v, const std::string& sep = ", ") {
    *this << std::hex;
    for (std::size_t i = 0; i < v.size(); ++i) {
      *this << (int)v[i];
      if (i + 1 < v.size()) {
        *this << sep;
      }
    }
    *this << std::dec;
  }

  template <typename T>
  printer& operator<<(const std::vector<T>& v) {
    print_vector(v);
    return *this;
  }

  printer& operator<<(const stmt& s) {
    if (s.defined()) {
      ++depth;
      s.accept(this);
      --depth;
    }
    return *this;
  }

  std::string indent(int extra = 0) const { return std::string(depth + extra, ' '); }

  void visit(const variable* v) override { *this << v->sym; }
  void visit(const wildcard* w) override { *this << w->sym; }
  void visit(const constant* c) override { *this << c->value; }

  void visit(const let* l) override { *this << "let " << l->sym << " = " << l->value << " in " << l->body; }

  void visit(const let_stmt* l) override {
    *this << indent() << "let " << l->sym << " = " << l->value << " { \n";
    *this << l->body;
    *this << indent() << "}\n";
  }

  template <typename T>
  void visit_bin_op(const T* op, const char* s) {
    *this << "(" << op->a << s << op->b << ")";
  }

  void visit(const add* op) override { visit_bin_op(op, " + "); }
  void visit(const sub* op) override { visit_bin_op(op, " - "); }
  void visit(const mul* op) override { visit_bin_op(op, " * "); }
  void visit(const div* op) override { visit_bin_op(op, " / "); }
  void visit(const mod* op) override { visit_bin_op(op, " % "); }
  void visit(const equal* op) override { visit_bin_op(op, " == "); }
  void visit(const not_equal* op) override { visit_bin_op(op, " != "); }
  void visit(const less* op) override { visit_bin_op(op, " < "); }
  void visit(const less_equal* op) override { visit_bin_op(op, " <= "); }
  void visit(const logical_and* op) override { visit_bin_op(op, " && "); }
  void visit(const logical_or* op) override { visit_bin_op(op, " || "); }
  void visit(const logical_not* op) override { *this << "!" << op->a; }

  void visit(const class min* op) override { *this << "min(" << op->a << ", " << op->b << ")"; }
  void visit(const class max* op) override { *this << "max(" << op->a << ", " << op->b << ")"; }

  void visit(const class select* op) override {
    *this << "select(" << op->condition << ", " << op->true_value << ", " << op->false_value << ")";
  }

  void visit(const call* op) override { *this << op->intrinsic << "(" << op->args << ")"; }

  void visit(const block* b) override {
    if (b->a.defined()) {
      b->a.accept(this);
    }
    if (b->b.defined()) {
      b->b.accept(this);
    }
  }

  void visit(const loop* l) override {
    *this << indent() << l->mode << " loop(" << l->sym << " in " << l->bounds;
    if (l->step.defined()) {
      *this << ", " << l->step;
    }
    *this << ") {\n";
    *this << l->body;
    *this << indent() << "}\n";
  }

  void visit(const if_then_else* n) override {
    *this << indent() << "if(" << n->condition << ") {\n";
    *this << n->true_body;
    if (n->false_body.defined()) {
      *this << indent() << "} else {\n";
      *this << n->false_body;
    }
    *this << indent() << "}\n";
  }

  void visit(const call_stmt* n) override {
    *this << indent() << "call(<fn>, {" << n->inputs << "}, {" << n->outputs << "})\n";
  }

  void visit(const copy_stmt* n) override {
    *this << indent() << "copy(" << n->src << ", {" << n->src_x << "}, " << n->dst << ", {" << n->dst_x << "}, {"
          << n->padding << "})\n";
  }

  void visit(const allocate* n) override {
    *this << indent() << n->sym << " = allocate(" << n->storage << ", " << static_cast<index_t>(n->elem_size) << ", {\n";
    *this << indent(2);
    print_vector(n->dims, ",\n" + indent(2));
    *this << "\n";
    *this << indent() << "}) {\n";
    *this << n->body;
    *this << indent() << "}\n";
  }

  void visit(const make_buffer* n) override {
    *this << indent() << n->sym << " = make_buffer(" << n->base << ", " << n->elem_size << ", {";
    if (!n->dims.empty()) {
      *this << "\n";
      *this << indent(2);
      print_vector(n->dims, ",\n" + indent(2));
      *this << "\n";
      *this << indent();
    }
    *this << "}) {\n";
    *this << n->body;
    *this << indent() << "}\n";
  }

  void visit(const clone_buffer* n) override {
    *this << indent() << n->sym << " = clone_buffer(" << n->src << ") {\n";
    *this << n->body;
    *this << indent() << "}\n";
  }

  void visit(const crop_buffer* n) override {
    *this << indent() << "crop_buffer(" << n->sym << ", {";
    if (!n->bounds.empty()) {
      *this << "\n";
      *this << indent(2);
      print_vector(n->bounds, ",\n" + indent(2));
      *this << "\n";
      *this << indent();
    }
    *this << "}) {\n";
    *this << n->body;
    *this << indent() << "}\n";
  }

  void visit(const crop_dim* n) override {
    *this << indent() << "crop_dim(" << n->sym << ", " << n->dim << ", " << n->bounds << ") {\n";
    *this << n->body;
    *this << indent() << "}\n";
  }

  void visit(const slice_buffer* n) override {
    *this << indent() << "slice_buffer(" << n->sym << ", {" << n->at << "}) {\n";
    *this << n->body;
    *this << indent() << "}\n";
  }

  void visit(const slice_dim* n) override {
    *this << indent() << "slice_dim(" << n->sym << ", " << n->dim << ", " << n->at << ") {\n";
    *this << n->body;
    *this << indent() << "}\n";
  }

  void visit(const truncate_rank* n) override {
    *this << indent() << "truncate_rank(" << n->sym << ", " << n->rank << ") {\n";
    *this << n->body;
    *this << indent() << "}\n";
  }

  void visit(const check* n) override { *this << indent() << "check(" << n->condition << ")\n"; }
};

void print(std::ostream& os, const expr& e, const node_context* ctx) {
  printer p(os, ctx);
  p << e;
}

void print(std::ostream& os, const stmt& s, const node_context* ctx) {
  printer p(os, ctx);
  p << s;
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
