#include "runtime/print.h"

#include <cassert>
#include <string>
#include <sstream>

#include "runtime/expr.h"
#include "runtime/stmt.h"

namespace slinky {

const char* to_string(memory_type type) {
  switch (type) {
  case memory_type::stack: return "stack";
  case memory_type::heap: return "heap";
  default: return "<invalid memory_type>";
  }
}

const char* to_string(buffer_field m) {
  switch (m) {
  case buffer_field::rank: return "rank";
  case buffer_field::elem_size: return "elem_size";
  case buffer_field::min: return "min";
  case buffer_field::max: return "max";
  case buffer_field::stride: return "stride";
  case buffer_field::fold_factor: return "fold_factor";

  default: return "<invalid buffer_field>";
  }
}

const char* to_string(intrinsic fn) {
  switch (fn) {
  case intrinsic::positive_infinity: return "oo";
  case intrinsic::negative_infinity: return "-oo";
  case intrinsic::indeterminate: return "indeterminate";
  case intrinsic::abs: return "abs";
  case intrinsic::and_then: return "and_then";
  case intrinsic::or_else: return "or_else";
  case intrinsic::buffer_size_bytes: return "buffer_size_bytes";
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

const char* to_string(stmt_node_type type) {
  switch (type) {
  case stmt_node_type::none: return "none";
  case stmt_node_type::call_stmt: return "call_stmt";
  case stmt_node_type::copy_stmt: return "copy_stmt";
  case stmt_node_type::let_stmt: return "let_stmt";
  case stmt_node_type::block: return "block";
  case stmt_node_type::loop: return "loop";
  case stmt_node_type::allocate: return "allocate";
  case stmt_node_type::make_buffer: return "make_buffer";
  case stmt_node_type::constant_buffer: return "constant_buffer";
  case stmt_node_type::clone_buffer: return "clone_buffer";
  case stmt_node_type::crop_buffer: return "crop_buffer";
  case stmt_node_type::crop_dim: return "crop_dim";
  case stmt_node_type::slice_buffer: return "slice_buffer";
  case stmt_node_type::slice_dim: return "slice_dim";
  case stmt_node_type::transpose: return "transpose";
  case stmt_node_type::check: return "check";

  default: return "<invalid stmt_node_type>";
  }
}

const char* to_string(expr_node_type type) {
  switch (type) {
  case expr_node_type::none: return "none";
  case expr_node_type::variable: return "variable";
  case expr_node_type::let: return "let";
  case expr_node_type::add: return "add";
  case expr_node_type::sub: return "sub";
  case expr_node_type::mul: return "mul";
  case expr_node_type::div: return "div";
  case expr_node_type::mod: return "mod";
  case expr_node_type::min: return "min";
  case expr_node_type::max: return "max";
  case expr_node_type::equal: return "equal";
  case expr_node_type::not_equal: return "not_equal";
  case expr_node_type::less: return "less";
  case expr_node_type::less_equal: return "less_equal";
  case expr_node_type::logical_and: return "logical_and";
  case expr_node_type::logical_or: return "logical_or";
  case expr_node_type::logical_not: return "logical_not";
  case expr_node_type::select: return "select";
  case expr_node_type::call: return "call";
  case expr_node_type::constant: return "constant";

  default: return "<invalid expr_node_type>";
  }
}

std::ostream& operator<<(std::ostream& os, memory_type type) { return os << to_string(type); }
std::ostream& operator<<(std::ostream& os, intrinsic fn) { return os << to_string(fn); }
std::ostream& operator<<(std::ostream& os, buffer_field f) { return os << to_string(f); }
std::ostream& operator<<(std::ostream& os, stmt_node_type t) { return os << to_string(t); }
std::ostream& operator<<(std::ostream& os, expr_node_type t) { return os << to_string(t); }

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
      os << "<" + std::to_string(sym.id) << ">";
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
  void print_vector(span<const T> v, const std::string& sep = ", ") {
    for (std::size_t i = 0; i < v.size(); ++i) {
      *this << v[i];
      if (i + 1 < v.size()) {
        *this << sep;
      }
    }
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

  void visit(const variable* v) override { 
    switch (v->field) {
    case buffer_field::none: *this << v->sym; return;
    case buffer_field::rank: *this << "buffer_rank(" << v->sym << ")"; return;
    case buffer_field::elem_size: *this << "buffer_elem_size(" << v->sym << ")"; return;
    case buffer_field::min: *this << "buffer_min(" << v->sym << ", " << v->dim << ")"; return;
    case buffer_field::max: *this << "buffer_max(" << v->sym << ", " << v->dim << ")"; return;
    case buffer_field::stride: *this << "buffer_stride(" << v->sym << ", " << v->dim << ")"; return;
    case buffer_field::fold_factor: *this << "buffer_fold_factor(" << v->sym << ", " << v->dim << ")"; return;
    default: SLINKY_UNREACHABLE << "unknown buffer_field " << to_string(v->field);
    }
  }
  void visit(const constant* c) override { *this << c->value; }

  void visit(const let* l) override {
    *this << "let {";
    print_vector(l->lets);
    *this << "} in " << l->body;
  }

  void visit(const let_stmt* l) override {
    const char* tag = l->is_closure ? "closure" : "let";
    if (l->lets.size() == 1) {
      *this << indent() << tag << " " << l->lets.front().first << " = " << l->lets.front().second << " in {\n";
    } else {
      *this << indent() << tag << " {\n";
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

  void visit(const constant_buffer* n) override {
    const raw_buffer& buf = *n->value;
    *this << indent() << n->sym << " = constant_buffer(" << buf.base << ", " << buf.elem_size << ", {";
    if (buf.rank > 0) {
      *this << "\n" << indent(2);
      print_vector(span<const dim>{buf.dims, buf.rank}, ",\n" + indent(2));
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

void print(std::ostream& os, var x, const node_context* ctx) {
  printer p(os, ctx ? ctx : default_context);
  p << x;
}

void print(std::ostream& os, const expr& e, const node_context* ctx) {
  printer p(os, ctx ? ctx : default_context);
  p << e;
}

void print(std::ostream& os, const stmt& s, const node_context* ctx) {
  printer p(os, ctx ? ctx : default_context);
  p << s;
}

std::string to_string(var x) { 
  std::stringstream ss;
  printer p(ss, default_context);
  p << x;
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, var sym) { 
  print(os, sym);
  return os;
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
