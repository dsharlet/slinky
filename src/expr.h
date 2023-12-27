#ifndef SLINKY_EXPR_H
#define SLINKY_EXPR_H

#include "buffer.h"

#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace slinky {

using symbol_id = std::size_t;

// We don't want to be doing string lookups in the inner loops. A node_context
// uniquely maps strings to symbol_id.
class node_context {
  std::map<std::string, symbol_id> name_to_id;
  std::vector<std::string> id_to_name;

public:
  // Get the name of a symbol_id.
  std::string name(symbol_id i) const;

  // Get or insert a new symbol_id for a name.
  symbol_id insert(const std::string& name);
  symbol_id insert();
  symbol_id lookup(const std::string& name) const;
};

enum class node_type {
  variable,
  wildcard,
  constant,
  let,
  add,
  sub,
  mul,
  div,
  mod,
  min,
  max,
  equal,
  not_equal,
  less,
  less_equal,
  bitwise_and,
  bitwise_or,
  bitwise_xor,
  logical_and,
  logical_or,
  shift_left,
  shift_right,
  select,
  load_buffer_meta,

  call,
  let_stmt,
  block,
  loop,
  if_then_else,
  allocate,
  make_buffer,
  crop_buffer,
  crop_dim,
  check,
};

enum class memory_type {
  stack,
  heap,
};

enum class buffer_meta {
  rank,
  base,
  elem_size,
  min,
  max,
  extent,
  stride_bytes,
  fold_factor,
};

class node_visitor;

class base_node {
public:
  base_node(node_type type) : type(type) {}
  virtual ~base_node() {}

  virtual void accept(node_visitor* v) const = 0;

  node_type type;

  template <typename T>
  const T* as() const {
    if (type == T::static_type) {
      return reinterpret_cast<const T*>(this);
    } else {
      return nullptr;
    }
  }
};

class base_expr_node : public base_node, public std::enable_shared_from_this<base_expr_node> {
public:
  base_expr_node(node_type type) : base_node(type) {}
};

class base_stmt_node : public base_node, public std::enable_shared_from_this<base_stmt_node> {
public:
  base_stmt_node(node_type type) : base_node(type) {}
};

template <typename T>
class expr_node : public base_expr_node {
public:
  expr_node() : base_expr_node(T::static_type) {}
};

template <typename T>
class stmt_node : public base_stmt_node {
public:
  stmt_node() : base_stmt_node(T::static_type) {}
};

class expr;

expr operator+(expr a, expr b);
expr operator-(expr a, expr b);
expr operator*(expr a, expr b);
expr operator/(expr a, expr b);
expr operator%(expr a, expr b);
expr operator&(expr a, expr b);
expr operator|(expr a, expr b);
expr operator^(expr a, expr b);
expr operator<<(expr a, expr b);
expr operator>>(expr a, expr b);

class expr {
public:
  std::shared_ptr<const base_expr_node> e;

  expr() = default;
  expr(const expr&) = default;
  expr(expr&&) = default;
  expr(index_t x);
  expr(int x) : expr(static_cast<index_t>(x)) {}

  // Unfortunately, std::enable_shared_from_this doesn't mean we can just do this:
  // T* n = new T();
  // std::shared_ptr<T> shared(n);
  // Instead, we have to use shared_from_this().
  // This also means all the initializations in expr.cc are a mess.
  // TODO(https://github.com/dsharlet/slinky/issues/5): Maybe we should just roll our own
  // smart pointer, this sucks.
  expr(const base_expr_node* e) : e(e ? e->shared_from_this() : nullptr) {}

  expr& operator=(const expr&) = default;
  expr& operator=(expr&&) = default;

  void accept(node_visitor* v) const {
    assert(defined());
    e->accept(v);
  }

  bool defined() const { return e != nullptr; }
  bool same_as(const expr& other) const { return e == other.e; }

  template <typename T>
  const T* as() const {
    if (e && e->type == T::static_type) {
      return reinterpret_cast<const T*>(e.get());
    } else {
      return nullptr;
    }
  }

  expr operator-() const { return 0 - *this; }

  expr& operator+=(const expr& r) {
    *this = *this + r;
    return *this;
  }
  expr& operator-=(const expr& r) {
    *this = *this - r;
    return *this;
  }
  expr& operator*=(const expr& r) {
    *this = *this * r;
    return *this;
  }
  expr& operator/=(const expr& r) {
    *this = *this / r;
    return *this;
  }
  expr& operator%=(const expr& r) {
    *this = *this % r;
    return *this;
  }
  expr& operator&=(const expr& r) {
    *this = *this & r;
    return *this;
  }
  expr& operator^=(const expr& r) {
    *this = *this ^ r;
    return *this;
  }
  expr& operator|=(const expr& r) {
    *this = *this | r;
    return *this;
  }
  expr& operator<<=(const expr& r) {
    *this = *this << r;
    return *this;
  }
  expr& operator>>=(const expr& r) {
    *this = *this >> r;
    return *this;
  }
};

expr operator==(expr a, expr b);
expr operator!=(expr a, expr b);
expr operator<(expr a, expr b);
expr operator<=(expr a, expr b);
expr operator>(expr a, expr b);
expr operator>=(expr a, expr b);
expr operator&&(expr a, expr b);
expr operator||(expr a, expr b);
expr min(expr a, expr b);
expr max(expr a, expr b);

struct interval {
  expr min, max;

  interval() {}
  explicit interval(const expr& point) : min(point), max(point) {}
  interval(expr min, expr max) : min(std::move(min)), max(std::move(max)) {}

  bool same_as(const interval& r) { return min.same_as(r.min) && max.same_as(r.max); }

  static interval all();
  static interval none();
  // An interval x such that x | y == y
  static interval union_identity();
  // An interval x such that x & y == y
  static interval intersection_identity();

  expr extent() const { return max - min + 1; }
  void set_extent(expr extent) { max = min + extent - 1; }

  interval& operator*=(expr scale) {
    min *= scale;
    max *= scale;
    return *this;
  }

  interval& operator/=(expr scale) {
    min /= scale;
    max /= scale;
    return *this;
  }

  interval& operator+=(expr offset) {
    min += offset;
    max += offset;
    return *this;
  }

  interval& operator-=(expr offset) {
    min -= offset;
    max -= offset;
    return *this;
  }

  // This is the union operator. I don't really like this, but
  // I also don't like that I can't name a function `union`.
  // It does kinda make sense...
  interval& operator|=(const interval& r) {
    min = slinky::min(min, r.min);
    max = slinky::max(max, r.max);
    return *this;
  }

  // This is intersection, just to be consistent with union.
  interval& operator&=(const interval& r) {
    min = slinky::min(min, r.min);
    max = slinky::max(max, r.max);
    return *this;
  }

  interval operator*(expr scale) const {
    interval result(*this);
    result *= scale;
    return result;
  }

  interval operator/(expr scale) const {
    interval result(*this);
    result /= scale;
    return result;
  }

  interval operator+(expr offset) const {
    interval result(*this);
    result += offset;
    return result;
  }

  interval operator-(expr offset) const {
    interval result(*this);
    result -= offset;
    return result;
  }

  interval operator|(const interval& r) const {
    interval result(*this);
    result |= r;
    return result;
  }

  interval operator&(const interval& r) const {
    interval result(*this);
    result &= r;
    return result;
  }
};

inline interval operator*(const expr& a, const interval& b) { return b * a; }
inline interval operator+(const expr& a, const interval& b) { return b + a; }

using box = std::vector<interval>;

box operator|(box a, const box& b);
box operator&(box a, const box& b);

class stmt {
public:
  std::shared_ptr<const base_stmt_node> s;

  stmt() = default;
  stmt(const stmt&) = default;
  stmt(stmt&&) = default;
  stmt(const base_stmt_node* s) : s(s->shared_from_this()) {}
  
  stmt& operator=(const stmt&) = default;
  stmt& operator=(stmt&&) = default;

  void accept(node_visitor* v) const {
    assert(defined());
    s->accept(v);
  }

  bool defined() const { return s != nullptr; }
  bool same_as(const stmt& other) const { return s == other.s; }

  template <typename T>
  const T* as() const {
    if (s && s->type == T::static_type) {
      return reinterpret_cast<const T*>(s.get());
    } else {
      return nullptr;
    }
  }
};

class let : public expr_node<let> {
public:
  symbol_id name;
  expr value;
  expr body;

  void accept(node_visitor* v) const;

  static expr make(symbol_id name, expr value, expr body);

  static constexpr node_type static_type = node_type::let;
};

class variable : public expr_node<variable> {
public:
  symbol_id name;

  void accept(node_visitor* v) const;

  static expr make(symbol_id name);

  static constexpr node_type static_type = node_type::variable;
};

// Similar to a variable, designed for use in pattern matching. A match with x is only
// accepted if matches(x) returns true.
// TODO(https://github.com/dsharlet/slinky/issues/6): This is pretty ugly. We should be
// able to contain this kind of logic to pattern matching only, it shouldn't be polluting
// the expression mechanism.
class wildcard : public expr_node<wildcard> {
public:
  symbol_id name;
  std::function<bool(const expr&)> matches;

  void accept(node_visitor* v) const;

  static expr make(symbol_id name, std::function<bool(const expr&)> matches);

  static constexpr node_type static_type = node_type::wildcard;
};

class constant : public expr_node<constant> {
public:
  index_t value;

  void accept(node_visitor* v) const;

  static expr make(index_t value);
  static expr make(const void* value);

  static constexpr node_type static_type = node_type::constant;
};

#define DECLARE_BINARY_OP(op)                                                                                          \
  class op : public expr_node<class op> {                                                                              \
  public:                                                                                                              \
    expr a, b;                                                                                                         \
    void accept(node_visitor* v) const;                                                                                \
    static expr make(expr a, expr b);                                                                                  \
    static constexpr node_type static_type = node_type::op;                                                            \
  };

DECLARE_BINARY_OP(add)
DECLARE_BINARY_OP(sub)
DECLARE_BINARY_OP(mul)
DECLARE_BINARY_OP(div)
DECLARE_BINARY_OP(mod)
DECLARE_BINARY_OP(min)
DECLARE_BINARY_OP(max)
DECLARE_BINARY_OP(equal)
DECLARE_BINARY_OP(not_equal)
DECLARE_BINARY_OP(less)
DECLARE_BINARY_OP(less_equal)
DECLARE_BINARY_OP(bitwise_and)
DECLARE_BINARY_OP(bitwise_or)
DECLARE_BINARY_OP(bitwise_xor)
DECLARE_BINARY_OP(logical_and)
DECLARE_BINARY_OP(logical_or)
DECLARE_BINARY_OP(shift_left)
DECLARE_BINARY_OP(shift_right)

#undef DECLARE_BINARY_OP

class select : public expr_node<select> {
public:
  expr condition;
  expr true_value;
  expr false_value;

  void accept(node_visitor* v) const;

  static expr make(expr condition, expr true_value, expr false_value);

  static constexpr node_type static_type = node_type::select;
};

// This expression loads buffer->base or a field from buffer->dims.
class load_buffer_meta : public expr_node<load_buffer_meta> {
public:
  // TODO(https://github.com/dsharlet/slinky/issues/6): These should not be exprs, they are only
  // because the simplifier wants to put wildcards here. A better pattern matching engine or just
  // not using patterns to simplify these would eliminate this requirement.
  expr buffer;
  buffer_meta meta;
  expr dim;

  void accept(node_visitor* v) const;

  static expr make(expr buffer, buffer_meta meta, expr dim = expr());

  static constexpr node_type static_type = node_type::load_buffer_meta;
};

class func;

class call : public stmt_node<call> {
public:
  typedef index_t (*callable_t)(std::span<const index_t>, std::span<buffer_base*>);
  using callable = std::function<index_t(std::span<const index_t>, std::span<buffer_base*>)>;

  callable target;
  std::vector<expr> scalar_args;
  std::vector<symbol_id> buffer_args;
  const func* fn;

  void accept(node_visitor* v) const;

  static stmt make(callable target, std::vector<expr> scalar_args, std::vector<symbol_id> buffer_args, const func* fn);

  static constexpr node_type static_type = node_type::call;
};

class let_stmt : public stmt_node<let_stmt> {
public:
  symbol_id name;
  expr value;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(symbol_id name, expr value, stmt body);

  static constexpr node_type static_type = node_type::let_stmt;
};

class block : public stmt_node<block> {
public:
  stmt a, b;

  void accept(node_visitor* v) const;

  static stmt make(stmt a, stmt b);
  static stmt make(std::span<stmt> stmts) { return make(stmts.begin(), stmts.end()); }
  static stmt make(std::initializer_list<stmt> stmts) { return make(stmts.begin(), stmts.end()); }
  template <typename It>
  static stmt make(It begin, It end) {
    stmt result;
    for (It i = begin; i != end; ++i) {
      if (result.defined()) {
        result = block::make(result, *i);
      } else {
        result = *i;
      }
    }
    return result;
  }

  static constexpr node_type static_type = node_type::block;
};

class loop : public stmt_node<loop> {
public:
  symbol_id name;
  expr begin;
  expr end;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(symbol_id name, expr begin, expr end, stmt body);

  static constexpr node_type static_type = node_type::loop;
};

class if_then_else : public stmt_node<if_then_else> {
public:
  expr condition;
  stmt true_body;
  stmt false_body;

  void accept(node_visitor* v) const;

  static stmt make(expr condition, stmt true_body, stmt false_body = stmt());

  static constexpr node_type static_type = node_type::if_then_else;
};

struct dim_expr {
  expr min;
  expr extent;
  expr stride_bytes;
  expr fold_factor;

  expr max() const { return min + extent - 1; }

  bool same_as(const dim_expr& r) {
    return min.same_as(r.min) && extent.same_as(r.extent) && stride_bytes.same_as(r.stride_bytes) &&
           fold_factor.same_as(r.fold_factor);
  }
};

class allocate : public stmt_node<allocate> {
public:
  memory_type type;
  symbol_id name;
  std::size_t elem_size;
  std::vector<dim_expr> dims;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(memory_type type, symbol_id name, std::size_t elem_size, std::vector<dim_expr> dims, stmt body);

  static constexpr node_type static_type = node_type::allocate;
};

// Make a new buffer from raw fields.
class make_buffer : public stmt_node<make_buffer> {
public:
  symbol_id name;
  expr base;
  std::size_t elem_size;
  std::vector<dim_expr> dims;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(symbol_id name, expr base, std::size_t elem_size, std::vector<dim_expr> dims, stmt body);

  static constexpr node_type static_type = node_type::make_buffer;
};

// This node is equivalent to the following:
// 1. Crop `name` to the interval `min, max` in-place in each dimension.
// 2. Evaluate `body`
// 3. Restore the original buffer
class crop_buffer : public stmt_node<make_buffer> {
public:
  symbol_id name;
  std::vector<interval> bounds;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(symbol_id name, std::vector<interval> bounds, stmt body);

  static constexpr node_type static_type = node_type::crop_buffer;
};

// This node is equivalent to the following:
// 1. Crop `name` to the interval `min, max` in-place
// 2. Evaluate `body`
// 3. Restore the original buffer
class crop_dim : public stmt_node<crop_dim> {
public:
  symbol_id name;
  int dim;
  expr min;
  expr extent;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(symbol_id name, int dim, expr min, expr extent, stmt body);

  static constexpr node_type static_type = node_type::crop_dim;
};

class check : public stmt_node<check> {
public:
  expr condition;

  void accept(node_visitor* v) const;

  static stmt make(expr condition);

  static constexpr node_type static_type = node_type::check;
};

class node_visitor {
public:
  virtual ~node_visitor() {}

  virtual void visit(const variable*) = 0;
  virtual void visit(const wildcard*) = 0;
  virtual void visit(const constant*) = 0;
  virtual void visit(const let*) = 0;
  virtual void visit(const add*) = 0;
  virtual void visit(const sub*) = 0;
  virtual void visit(const mul*) = 0;
  virtual void visit(const div*) = 0;
  virtual void visit(const mod*) = 0;
  virtual void visit(const class min*) = 0;
  virtual void visit(const class max*) = 0;
  virtual void visit(const equal*) = 0;
  virtual void visit(const not_equal*) = 0;
  virtual void visit(const less*) = 0;
  virtual void visit(const less_equal*) = 0;
  virtual void visit(const bitwise_and*) = 0;
  virtual void visit(const bitwise_or*) = 0;
  virtual void visit(const bitwise_xor*) = 0;
  virtual void visit(const logical_and*) = 0;
  virtual void visit(const logical_or*) = 0;
  virtual void visit(const shift_left*) = 0;
  virtual void visit(const shift_right*) = 0;
  virtual void visit(const select*) = 0;
  virtual void visit(const load_buffer_meta*) = 0;

  virtual void visit(const let_stmt*) = 0;
  virtual void visit(const block*) = 0;
  virtual void visit(const loop*) = 0;
  virtual void visit(const if_then_else*) = 0;
  virtual void visit(const call*) = 0;
  virtual void visit(const allocate*) = 0;
  virtual void visit(const make_buffer*) = 0;
  virtual void visit(const crop_buffer*) = 0;
  virtual void visit(const crop_dim*) = 0;
  virtual void visit(const check*) = 0;
};

inline void variable::accept(node_visitor* v) const { v->visit(this); }
inline void wildcard::accept(node_visitor* v) const { v->visit(this); }
inline void constant::accept(node_visitor* v) const { v->visit(this); }
inline void let::accept(node_visitor* v) const { v->visit(this); }
inline void add::accept(node_visitor* v) const { v->visit(this); }
inline void sub::accept(node_visitor* v) const { v->visit(this); }
inline void mul::accept(node_visitor* v) const { v->visit(this); }
inline void div::accept(node_visitor* v) const { v->visit(this); }
inline void mod::accept(node_visitor* v) const { v->visit(this); }
inline void min::accept(node_visitor* v) const { v->visit(this); }
inline void max::accept(node_visitor* v) const { v->visit(this); }
inline void equal::accept(node_visitor* v) const { v->visit(this); }
inline void not_equal::accept(node_visitor* v) const { v->visit(this); }
inline void less::accept(node_visitor* v) const { v->visit(this); }
inline void less_equal::accept(node_visitor* v) const { v->visit(this); }
inline void bitwise_and::accept(node_visitor* v) const { v->visit(this); }
inline void bitwise_or::accept(node_visitor* v) const { v->visit(this); }
inline void bitwise_xor::accept(node_visitor* v) const { v->visit(this); }
inline void logical_and::accept(node_visitor* v) const { v->visit(this); }
inline void logical_or::accept(node_visitor* v) const { v->visit(this); }
inline void shift_left::accept(node_visitor* v) const { v->visit(this); }
inline void shift_right::accept(node_visitor* v) const { v->visit(this); }
inline void select::accept(node_visitor* v) const { v->visit(this); }
inline void load_buffer_meta::accept(node_visitor* v) const { v->visit(this); }

inline void let_stmt::accept(node_visitor* v) const { v->visit(this); }
inline void block::accept(node_visitor* v) const { v->visit(this); }
inline void loop::accept(node_visitor* v) const { v->visit(this); }
inline void if_then_else::accept(node_visitor* v) const { v->visit(this); }
inline void call::accept(node_visitor* v) const { v->visit(this); }
inline void allocate::accept(node_visitor* v) const { v->visit(this); }
inline void make_buffer::accept(node_visitor* v) const { v->visit(this); }
inline void crop_buffer::accept(node_visitor* v) const { v->visit(this); }
inline void crop_dim::accept(node_visitor* v) const { v->visit(this); }
inline void check::accept(node_visitor* v) const { v->visit(this); }

expr make_variable(node_context& ctx, const std::string& name);

inline const index_t* as_constant(const expr& x) {
  const constant* cx = x.as<constant>();
  return cx ? &cx->value : nullptr;
}

inline const symbol_id* as_variable(const expr& x) {
  const variable* vx = x.as<variable>();
  return vx ? &vx->name : nullptr;
}

inline bool is_zero(const expr& x) {
  const constant* cx = x.as<constant>();
  return cx ? cx->value == 0 : false;
}

inline bool is_true(const expr& x) {
  const constant* cx = x.as<constant>();
  return cx ? cx->value != 0 : false;
}

inline bool is_false(const expr& x) {
  return is_zero(x);
}

}  // namespace slinky

#endif  // SLINKY_EXPR_H
