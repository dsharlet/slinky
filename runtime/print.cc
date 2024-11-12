#include "runtime/print.h"

#include <cassert>
#include <string>

#include "runtime/expr.h"
#include "runtime/stmt.h"

namespace slinky {

std::string to_string(var sym) { return "<" + std::to_string(sym.id) + ">"; }

const char* to_string(memory_type type) {
  switch (type) {
  case memory_type::stack: return "stack";
  case memory_type::heap: return "heap";
  default: return "<invalid memory_type>";
  }
}

const char* to_string(intrinsic fn) {
  switch (fn) {
  case intrinsic::positive_infinity: return "oo";
  case intrinsic::negative_infinity: return "-oo";
  case intrinsic::indeterminate: return "indeterminate";
  case intrinsic::abs: return "abs";
  case intrinsic::gcd: return "gcd";
  case intrinsic::lcm: return "lcm";
  case intrinsic::and_then: return "and_then";
  case intrinsic::or_else: return "or_else";
  case intrinsic::buffer_rank: return "buffer_rank";
  case intrinsic::buffer_elem_size: return "buffer_elem_size";
  case intrinsic::buffer_size_bytes: return "buffer_size_bytes";
  case intrinsic::buffer_min: return "buffer_min";
  case intrinsic::buffer_max: return "buffer_max";
  case intrinsic::buffer_stride: return "buffer_stride";
  case intrinsic::buffer_fold_factor: return "buffer_fold_factor";
  case intrinsic::buffer_at: return "buffer_at";
  case intrinsic::semaphore_init: return "semaphore_init";
  case intrinsic::semaphore_signal: return "semaphore_signal";
  case intrinsic::semaphore_wait: return "semaphore_wait";
  case intrinsic::trace_begin: return "trace_begin";
  case intrinsic::trace_end: return "trace_end";
  case intrinsic::free: return "free";

  default: return "<invalid intrinsic>";
  }
}

std::ostream& operator<<(std::ostream& os, var sym) { return os << to_string(sym); }
std::ostream& operator<<(std::ostream& os, memory_type type) { return os << to_string(type); }
std::ostream& operator<<(std::ostream& os, intrinsic fn) { return os << to_string(fn); }

std::ostream& operator<<(std::ostream& os, const interval_expr& i) {
  return os << "[" << i.min << ", " << i.max << "]";
}

std::ostream& operator<<(std::ostream& os, const box_expr& b) {
  os << "{";
  for (std::size_t i = 0; i < b.size(); ++i) {
    os << b[i];
    if (i + 1 < b.size()) {
      os << ", ";
    }
  }
  os << "}";

  return os;
}

class printer : public expr_visitor, public stmt_visitor {
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

  printer& operator<<(var sym) {
    if (context) {
      os << context->name(sym);
    } else {
      os << sym;
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

  printer& operator<<(const std::pair<var, expr>& let) { return *this << let.first << " = " << let.second; }

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
  void visit(const constant* c) override { *this << c->value; }

  void visit(const let* l) override {
    *this << "let {";
    print_vector(l->lets);
    *this << "} in " << l->body;
  }

  void visit(const let_stmt* l) override {
    if (l->lets.size() == 1) {
      *this << indent() << "let " << l->lets.front().first << " = " << l->lets.front().second << " in {\n";
    } else {
      *this << indent() << "let {\n";
      *this << indent(2);
      print_vector(l->lets, ",\n" + indent(2));
      *this << "\n" << indent() << "} in {\n";
    }
    *this << l->body;
    *this << indent() << "}\n";
  }

  void visit_bin_op(const expr& a, const char* s, const expr& b) {
    *this << "(" << a << s << b << ")";
  }

  void visit(const add* op) override { visit_bin_op(op->a, " + ", op->b); }
  void visit(const sub* op) override { visit_bin_op(op->a, " - ", op->b); }
  void visit(const mul* op) override { visit_bin_op(op->a, " * ", op->b); }
  void visit(const div* op) override { visit_bin_op(op->a, " / ", op->b); }
  void visit(const mod* op) override { visit_bin_op(op->a, " % ", op->b); }
  void visit(const equal* op) override { visit_bin_op(op->a, " == ", op->b); }
  void visit(const not_equal* op) override { visit_bin_op(op->a, " != ", op->b); }
  void visit(const less* op) override { visit_bin_op(op->a, " < ", op->b); }
  void visit(const less_equal* op) override { visit_bin_op(op->a, " <= ", op->b); }
  void visit(const logical_and* op) override { visit_bin_op(op->a, " && ", op->b); }
  void visit(const logical_or* op) override { visit_bin_op(op->a, " || ", op->b); }
  void visit(const logical_not* op) override { *this << "!" << op->a; }

  void visit(const class min* op) override { *this << "min(" << op->a << ", " << op->b << ")"; }
  void visit(const class max* op) override { *this << "max(" << op->a << ", " << op->b << ")"; }

  void visit(const class select* op) override {
    *this << "select(" << op->condition << ", " << op->true_value << ", " << op->false_value << ")";
  }

  void visit(const call* op) override { *this << op->intrinsic << "(" << op->args << ")"; }

  void visit(const block* b) override {
    for (const auto& s : b->stmts) {
      s.accept(this);
    }
  }

  void visit(const loop* l) override {
    *this << indent() << l->sym << " = loop(";
    switch (l->max_workers) {
    case loop::serial: *this << "serial"; break;
    case loop::parallel: *this << "parallel"; break;
    default: *this << l->max_workers; break;
    }
    *this << ", " << l->bounds;
    if (l->step.defined()) {
      *this << ", " << l->step;
    }
    *this << ") {\n";
    *this << l->body;
    *this << indent() << "}\n";
  }

  void visit(const call_stmt* n) override {
    *this << indent() << "call(";
    if (!n->attrs.name.empty()) {
      *this << n->attrs.name;
    } else if (n->target) {
      *this << "<anonymous target>";
    } else {
      *this << "<null target>";
    }
    *this << ", {" << n->inputs << "}, {" << n->outputs << "})\n";
  }

  void visit(const copy_stmt* n) override {
    *this << indent() << "copy(" << n->src << ", {" << n->src_x << "}, " << n->dst << ", {" << n->dst_x << "}";
    if (n->padding) {
      *this << ", {" << *n->padding << "}";
    }
    *this << ")\n";
  }

  void visit(const allocate* n) override {
    *this << indent() << n->sym << " = allocate(" << n->storage << ", " << n->elem_size << ", {";
    if (!n->dims.empty()) {
      *this << "\n" << indent(2);
      print_vector(n->dims, ",\n" + indent(2));
      *this << "\n" << indent();
    }
    *this << "}) {\n";
    *this << n->body;
    *this << indent() << "}\n";
  }

  void visit(const make_buffer* n) override {
    *this << indent() << n->sym << " = make_buffer(" << n->base << ", " << n->elem_size << ", {";
    if (!n->dims.empty()) {
      *this << "\n" << indent(2);
      print_vector(n->dims, ",\n" + indent(2));
      *this << "\n" << indent();
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
    *this << indent() << n->sym << " = crop_buffer(" << n->src << ", {";
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
    *this << indent() << n->sym << " = crop_dim(" << n->src << ", " << n->dim << ", " << n->bounds << ") {\n";
    *this << n->body;
    *this << indent() << "}\n";
  }

  void visit(const slice_buffer* n) override {
    *this << indent() << n->sym << " = slice_buffer(" << n->src << ", {" << n->at << "}) {\n";
    *this << n->body;
    *this << indent() << "}\n";
  }

  void visit(const slice_dim* n) override {
    *this << indent() << n->sym << " = slice_dim(" << n->src << ", " << n->dim << ", " << n->at << ") {\n";
    *this << n->body;
    *this << indent() << "}\n";
  }

  void visit(const transpose* n) override {
    *this << indent() << n->sym << " = transpose(" << n->src << ", {" << n->dims << "}) {\n";
    *this << n->body;
    *this << indent() << "}\n";
  }

  void visit(const check* n) override { *this << indent() << "check(" << n->condition << ")\n"; }
};

namespace {

thread_local const node_context* default_context = nullptr;

}  // namespace

const node_context* set_default_print_context(const node_context* ctx) {
  const node_context* old = default_context;
  default_context = ctx;
  return old;
}

void print(std::ostream& os, const expr& e, const node_context* ctx) {
  printer p(os, ctx ? ctx : default_context);
  p << e;
}

void print(std::ostream& os, const stmt& s, const node_context* ctx) {
  printer p(os, ctx ? ctx : default_context);
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

std::ostream& operator<<(std::ostream& os, const raw_buffer& buf) {
  os << "{base=" << buf.base << ", elem_size=" << buf.elem_size << ", dims={";
  for (std::size_t d = 0; d < buf.rank; ++d) {
    os << buf.dims[d];
    if (d + 1 < buf.rank) {
      os << ",";
    }
  }
  os << "}";
  return os;
}

std::ostream& operator<<(std::ostream& os, const dim& d) {
  os << "{min=" << d.min() << ", max=" << d.max() << ", extent=" << d.extent() << ", stride=" << d.stride();
  if (d.fold_factor() != dim::unfolded) {
    os << ", fold_factor=" << d.fold_factor();
  }
  os << "}";
  return os;
}

}  // namespace slinky
