#ifndef SLINKY_EXPR_H
#define SLINKY_EXPR_H

#include "buffer.h"
#include "ref_count.h"

#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <optional>

namespace slinky {

using symbol_id = std::size_t;

// We don't want to be doing string lookups in the inner loops. A node_context
// uniquely maps strings to symbol_id.
class node_context {
  std::vector<std::string> id_to_name;

public:
  // Get the name of a symbol_id.
  std::string name(symbol_id i) const;

  // Get or insert a new symbol_id for a name.
  symbol_id insert(const std::string& name);
  symbol_id insert_unique(const std::string& prefix = "_");
  std::optional<symbol_id> lookup(const std::string& name) const;
};

enum class node_type {
  none,

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
  logical_and,
  logical_or,
  logical_not,
  select,
  call,

  call_func,
  let_stmt,
  block,
  loop,
  if_then_else,
  allocate,
  make_buffer,
  crop_buffer,
  crop_dim,
  slice_buffer,
  slice_dim,
  truncate_rank,
  check,
};

enum class memory_type {
  stack,
  heap,
};

enum class intrinsic {
  negative_infinity,
  positive_infinity,
  indeterminate,
  abs,

  buffer_rank,
  buffer_base,
  buffer_elem_size,
  buffer_size_bytes,

  buffer_min,
  buffer_max,
  buffer_stride,
  buffer_fold_factor,
  buffer_extent,

  buffer_at,
};

class node_visitor;

class base_node : public ref_counted {
public:
  base_node(node_type type) : type(type) {}

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

class base_expr_node : public base_node {
public:
  base_expr_node(node_type type) : base_node(type) {}
};

class base_stmt_node : public base_node {
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

class expr {
  ref_count<const base_expr_node> n_;
  node_type type_ = node_type::none;

public:

  expr() = default;
  expr(const expr&) = default;
  expr(expr&&) = default;
  expr(index_t x);
  expr(int x) : expr(static_cast<index_t>(x)) {}
  expr(const base_expr_node* n) : n_(n), type_(n ? n->type : node_type::none) {}

  expr& operator=(const expr&) = default;
  expr& operator=(expr&&) = default;

  void accept(node_visitor* v) const {
    assert(defined());
    n_->accept(v);
  }

  bool defined() const { return n_ != nullptr; }
  bool same_as(const expr& other) const { return n_ == other.n_; }
  node_type type() const { return type_; }
  const base_expr_node* get() const { return n_; }

  template <typename T>
  const T* as() const {
    if (n_ && type_ == T::static_type) {
      return reinterpret_cast<const T*>(&*n_);
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
};

expr operator==(expr a, expr b);
expr operator!=(expr a, expr b);
expr operator<(expr a, expr b);
expr operator<=(expr a, expr b);
expr operator>(expr a, expr b);
expr operator>=(expr a, expr b);
expr operator&&(expr a, expr b);
expr operator||(expr a, expr b);
expr operator!(expr x);
expr min(expr a, expr b);
expr max(expr a, expr b);
expr clamp(expr x, expr a, expr b);
expr select(expr c, expr t, expr f);
expr min(std::span<expr> x);
expr max(std::span<expr> x);

struct interval_expr {
  expr min, max;

  interval_expr() {}
  explicit interval_expr(const expr& point) : min(point), max(point) {}
  interval_expr(expr min, expr max) : min(std::move(min)), max(std::move(max)) {}

  bool same_as(const interval_expr& r) { return min.same_as(r.min) && max.same_as(r.max); }

  bool is_single_point() const { return min.same_as(max); }

  static const interval_expr& all();
  static const interval_expr& none();
  // An interval_expr x such that x | y == y
  static const interval_expr& union_identity();
  // An interval_expr x such that x & y == y
  static const interval_expr& intersection_identity();

  const expr& begin() const { return min; }
  expr end() const { return max + 1; }
  expr extent() const { return max - min + 1; }

  interval_expr& operator*=(const expr& scale);
  interval_expr& operator/=(const expr& scale);
  interval_expr& operator+=(const expr& offset);
  interval_expr& operator-=(const expr& offset);
  interval_expr operator*(const expr& scale) const;
  interval_expr operator/(const expr& scale) const;
  interval_expr operator+(const expr& offset) const;
  interval_expr operator-(const expr& offset) const;
  interval_expr operator-() const;

  // This is the union operator. I don't really like this, but
  // I also don't like that I can't name a function `union`.
  // It does kinda make sense...
  interval_expr& operator|=(const interval_expr& r);
  // This is intersection, just to be consistent with union.
  interval_expr& operator&=(const interval_expr& r);
  interval_expr operator|(const interval_expr& r) const;
  interval_expr operator&(const interval_expr& r) const;
};

// Make an interval of the region [begin, end) (like python's range).
inline interval_expr range(expr begin, const expr& end) { return {std::move(begin), end - 1}; }
// Make an interval of the region [min, max].
inline interval_expr bounds(expr min, expr max) { return {std::move(min), std::move(max)}; }
// Make an interval of the region [min, min + extent).
inline interval_expr min_extent(const expr& min, const expr& extent) { return {min, min + extent - 1}; }
// Make a interval of the region [x, x].
inline interval_expr point(const expr& x) { return {x, x}; }

inline interval_expr operator*(const expr& a, const interval_expr& b) { return b * a; }
inline interval_expr operator+(const expr& a, const interval_expr& b) { return b + a; }

using box_expr = std::vector<interval_expr>;
box_expr operator|(box_expr a, const box_expr& b);
box_expr operator&(box_expr a, const box_expr& b);

class stmt {
  ref_count<const base_stmt_node> n_;
  node_type type_ = node_type::none;

public:
  stmt() = default;
  stmt(const stmt&) = default;
  stmt(stmt&&) = default;
  stmt(const base_stmt_node* n) : n_(n), type_(n ? n->type : node_type::none) {}

  stmt& operator=(const stmt&) = default;
  stmt& operator=(stmt&&) = default;

  void accept(node_visitor* v) const {
    assert(defined());
    n_->accept(v);
  }

  bool defined() const { return n_ != nullptr; }
  bool same_as(const stmt& other) const { return n_ == other.n_; }
  node_type type() const { return type_; }
  const base_stmt_node* get() const { return n_; }

  template <typename T>
  const T* as() const {
    if (type_ == T::static_type) {
      return reinterpret_cast<const T*>(&*n_);
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
DECLARE_BINARY_OP(logical_and)
DECLARE_BINARY_OP(logical_or)

#undef DECLARE_BINARY_OP

class logical_not : public expr_node<logical_not> {
public:
  expr x;

  void accept(node_visitor* v) const;

  static expr make(expr x);

  static constexpr node_type static_type = node_type::logical_not;
};

class select : public expr_node<class select> {
public:
  expr condition;
  expr true_value;
  expr false_value;

  void accept(node_visitor* v) const;

  static expr make(expr condition, expr true_value, expr false_value);

  static constexpr node_type static_type = node_type::select;
};

class call : public expr_node<call> {
public:
  slinky::intrinsic intrinsic;
  std::vector<expr> args;

  void accept(node_visitor* v) const;

  static expr make(slinky::intrinsic i, std::vector<expr> args);

  static constexpr node_type static_type = node_type::call;
};

class func;

class call_func : public stmt_node<call_func> {
public:
  typedef index_t (*callable_t)(std::span<const index_t>, std::span<raw_buffer*>);
  using callable = std::function<index_t(std::span<const index_t>, std::span<raw_buffer*>)>;

  callable target;
  std::vector<expr> scalar_args;
  std::vector<symbol_id> buffer_args;
  const func* fn;

  void accept(node_visitor* v) const;

  static stmt make(callable target, std::vector<expr> scalar_args, std::vector<symbol_id> buffer_args, const func* fn);

  static constexpr node_type static_type = node_type::call_func;
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
  interval_expr bounds;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(symbol_id name, interval_expr bounds, stmt body);

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
  interval_expr bounds;
  expr stride;
  expr fold_factor;

  const expr& min() const { return bounds.min; }
  const expr& max() const { return bounds.max; }
  expr extent() const { return bounds.extent(); }

  bool same_as(const dim_expr& r) {
    return bounds.same_as(r.bounds) && stride.same_as(r.stride) && fold_factor.same_as(r.fold_factor);
  }
};

class allocate : public stmt_node<allocate> {
public:
  memory_type storage;
  symbol_id name;
  std::size_t elem_size;
  std::vector<dim_expr> dims;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(memory_type storage, symbol_id name, std::size_t elem_size, std::vector<dim_expr> dims, stmt body);

  static constexpr node_type static_type = node_type::allocate;
};

// Make a new buffer from raw fields.
class make_buffer : public stmt_node<make_buffer> {
public:
  symbol_id name;
  expr base;
  expr elem_size;
  std::vector<dim_expr> dims;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(symbol_id name, expr base, expr elem_size, std::vector<dim_expr> dims, stmt body);

  static constexpr node_type static_type = node_type::make_buffer;
};

// This node is equivalent to the following:
// 1. Crop `name` to the interval_expr `min, max` in-place in each dimension.
// 2. Evaluate `body`
// 3. Restore the original buffer
class crop_buffer : public stmt_node<crop_buffer> {
public:
  symbol_id name;
  std::vector<interval_expr> bounds;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(symbol_id name, std::vector<interval_expr> bounds, stmt body);

  static constexpr node_type static_type = node_type::crop_buffer;
};

// This node is equivalent to the following:
// 1. Crop the `dim`th dimension of `name` to `bounds` in-place
// 2. Evaluate `body`
// 3. Restore the original buffer
class crop_dim : public stmt_node<crop_dim> {
public:
  symbol_id name;
  int dim;
  interval_expr bounds;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(symbol_id name, int dim, interval_expr bounds, stmt body);

  static constexpr node_type static_type = node_type::crop_dim;
};

class slice_buffer : public stmt_node<slice_buffer> {
public:
  symbol_id name;
  std::vector<expr> at;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(symbol_id name, std::vector<expr> at, stmt body);

  static constexpr node_type static_type = node_type::slice_buffer;
};

class slice_dim : public stmt_node<slice_dim> {
public:
  symbol_id name;
  int dim;
  expr at;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(symbol_id name, int dim, expr at, stmt body);

  static constexpr node_type static_type = node_type::slice_dim;
};

class truncate_rank : public stmt_node<truncate_rank> {
public:
  symbol_id name;
  int rank;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(symbol_id name, int rank, stmt body);

  static constexpr node_type static_type = node_type::truncate_rank;
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
  virtual void visit(const logical_and*) = 0;
  virtual void visit(const logical_or*) = 0;
  virtual void visit(const logical_not*) = 0;
  virtual void visit(const class select*) = 0;
  virtual void visit(const call*) = 0;

  virtual void visit(const let_stmt*) = 0;
  virtual void visit(const block*) = 0;
  virtual void visit(const loop*) = 0;
  virtual void visit(const if_then_else*) = 0;
  virtual void visit(const call_func*) = 0;
  virtual void visit(const allocate*) = 0;
  virtual void visit(const make_buffer*) = 0;
  virtual void visit(const crop_buffer*) = 0;
  virtual void visit(const crop_dim*) = 0;
  virtual void visit(const slice_buffer*) = 0;
  virtual void visit(const slice_dim*) = 0;
  virtual void visit(const truncate_rank*) = 0;
  virtual void visit(const check*) = 0;
};

class recursive_node_visitor : public node_visitor {
public:
  virtual void visit(const variable*) override {}
  virtual void visit(const wildcard*) override {}
  virtual void visit(const constant*) override {}
  virtual void visit(const let* x) override {
    x->value.accept(this);
    x->body.accept(this);
  }

  template <typename T>
  void visit_binary(const T* x) {
    x->a.accept(this);
    x->b.accept(this);
  }

  virtual void visit(const add* x) override { visit_binary(x); }
  virtual void visit(const sub* x) override { visit_binary(x); }
  virtual void visit(const mul* x) override { visit_binary(x); }
  virtual void visit(const div* x) override { visit_binary(x); }
  virtual void visit(const mod* x) override { visit_binary(x); }
  virtual void visit(const class min* x) override { visit_binary(x); }
  virtual void visit(const class max* x) override { visit_binary(x); }
  virtual void visit(const equal* x) override { visit_binary(x); }
  virtual void visit(const not_equal* x) override { visit_binary(x); }
  virtual void visit(const less* x) override { visit_binary(x); }
  virtual void visit(const less_equal* x) override { visit_binary(x); }
  virtual void visit(const logical_and* x) override { visit_binary(x); }
  virtual void visit(const logical_or* x) override { visit_binary(x); }
  virtual void visit(const logical_not* x) override { x->x.accept(this); }
  virtual void visit(const class select* x) override {
    x->condition.accept(this);
    x->true_value.accept(this);
    x->false_value.accept(this);
  }
  virtual void visit(const call* x) override {
    for (const expr& i : x->args) {
      if (i.defined()) i.accept(this);
    }
  }

  virtual void visit(const let_stmt* x) override {
    x->value.accept(this);
    x->body.accept(this);
  }
  virtual void visit(const block* x) override {
    if (x->a.defined()) x->a.accept(this);
    if (x->b.defined()) x->b.accept(this);
  }
  virtual void visit(const loop* x) override {
    x->bounds.min.accept(this);
    x->bounds.max.accept(this);
    x->body.accept(this);
  }
  virtual void visit(const if_then_else* x) override {
    x->condition.accept(this);
    if (x->true_body.defined()) x->true_body.accept(this);
    if (x->false_body.defined()) x->false_body.accept(this);
  }
  virtual void visit(const call_func* x) override {
    for (const expr& i : x->scalar_args) {
      i.accept(this);
    }
  }
  virtual void visit(const allocate* x) override {
    for (const dim_expr& i : x->dims) {
      i.bounds.min.accept(this);
      i.bounds.max.accept(this);
      i.stride.accept(this);
      i.fold_factor.accept(this);
    }
    x->body.accept(this);
  }
  virtual void visit(const make_buffer* x) override {
    x->base.accept(this);
    x->elem_size.accept(this);
    for (const dim_expr& i : x->dims) {
      i.bounds.min.accept(this);
      i.bounds.max.accept(this);
      i.stride.accept(this);
      i.fold_factor.accept(this);
    }
    x->body.accept(this);
  }
  virtual void visit(const crop_buffer* x) override {
    for (const interval_expr& i : x->bounds) {
      if (i.min.defined()) i.min.accept(this);
      if (i.max.defined()) i.max.accept(this);
    }
    x->body.accept(this);
  }
  virtual void visit(const crop_dim* x) override {
    x->bounds.min.accept(this);
    x->bounds.max.accept(this);
    x->body.accept(this);
  }
  virtual void visit(const slice_buffer* x) override {
    for (const expr& i : x->at) {
      if (i.defined()) i.accept(this);
    }
    x->body.accept(this);
  }
  virtual void visit(const slice_dim* x) override {
    x->at.accept(this);
    x->body.accept(this);
  }
  virtual void visit(const truncate_rank* x) override {
    x->body.accept(this);
  }
  virtual void visit(const check* x) override { x->condition.accept(this); }
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
inline void logical_and::accept(node_visitor* v) const { v->visit(this); }
inline void logical_or::accept(node_visitor* v) const { v->visit(this); }
inline void logical_not::accept(node_visitor* v) const { v->visit(this); }
inline void select::accept(node_visitor* v) const { v->visit(this); }
inline void call::accept(node_visitor* v) const { v->visit(this); }

inline void let_stmt::accept(node_visitor* v) const { v->visit(this); }
inline void block::accept(node_visitor* v) const { v->visit(this); }
inline void loop::accept(node_visitor* v) const { v->visit(this); }
inline void if_then_else::accept(node_visitor* v) const { v->visit(this); }
inline void call_func::accept(node_visitor* v) const { v->visit(this); }
inline void allocate::accept(node_visitor* v) const { v->visit(this); }
inline void make_buffer::accept(node_visitor* v) const { v->visit(this); }
inline void crop_buffer::accept(node_visitor* v) const { v->visit(this); }
inline void crop_dim::accept(node_visitor* v) const { v->visit(this); }
inline void slice_buffer::accept(node_visitor* v) const { v->visit(this); }
inline void slice_dim::accept(node_visitor* v) const { v->visit(this); }
inline void truncate_rank::accept(node_visitor* v) const { v->visit(this); }
inline void check::accept(node_visitor* v) const { v->visit(this); }


inline const index_t* as_constant(const expr& x) {
  const constant* cx = x.as<constant>();
  return cx ? &cx->value : nullptr;
}

inline const symbol_id* as_variable(const expr& x) {
  const variable* vx = x.as<variable>();
  return vx ? &vx->name : nullptr;
}

inline bool is_constant(const expr& x, index_t value) {
  const constant* cx = x.as<constant>();
  return cx ? cx->value == value : false;
}

inline bool is_zero(const expr& x) { return is_constant(x, 0); }
inline bool is_true(const expr& x) {
  const constant* cx = x.as<constant>();
  return cx ? cx->value != 0 : false;
}

inline bool is_false(const expr& x) { return is_zero(x); }

inline bool is_intrinsic(const expr& x, intrinsic i) {
  const call* c = x.as<call>();
  return c ? c->intrinsic == i : false;
}

inline bool is_positive_infinity(const expr& x) { return is_intrinsic(x, intrinsic::positive_infinity); }
inline bool is_negative_infinity(const expr& x) { return is_intrinsic(x, intrinsic::negative_infinity); }
inline bool is_indeterminate(const expr& x) { return is_intrinsic(x, intrinsic::indeterminate); }
inline bool is_infinity(const expr& x) { return is_positive_infinity(x) || is_negative_infinity(x); }
inline bool is_finite(const expr& x) { return !is_infinity(x) && !is_indeterminate(x); }

const expr& positive_infinity();
const expr& negative_infinity();
const expr& indeterminate();

inline bool is_positive(const expr& x) {
  if (is_positive_infinity(x)) return true;
  const index_t* c = as_constant(x);
  return c ? *c > 0 : false;
}

inline bool is_non_negative(const expr& x) {
  if (is_positive_infinity(x)) return true;
  const index_t* c = as_constant(x);
  return c ? *c >= 0 : false;
}

inline bool is_negative(const expr& x) {
  if (is_negative_infinity(x)) return true;
  const index_t* c = as_constant(x);
  return c ? *c < 0 : false;
}

inline bool is_non_positive(const expr& x) {
  if (is_negative_infinity(x)) return true;
  const index_t* c = as_constant(x);
  return c ? *c <= 0 : false;
}

// This is an expr-like wrapper for use where only a `variable` expr is allowed.
class var {
  expr e_;

public:
  var();
  var(symbol_id name);
  var(node_context& ctx, const std::string& name);

  symbol_id name() const {
    assert(e_.defined());
    return *as_variable(e_);
  }

  operator const expr&() const { return e_; }

  expr operator-() const { return -e_; }
};

expr abs(expr x);

expr buffer_rank(expr buf);
expr buffer_base(expr buf);
expr buffer_elem_size(expr buf);
expr buffer_min(expr buf, expr dim);
expr buffer_max(expr buf, expr dim);
expr buffer_extent(expr buf, expr dim);
expr buffer_stride(expr buf, expr dim);
expr buffer_fold_factor(expr buf, expr dim);
expr buffer_at(expr buf, const std::vector<expr>& at);
expr buffer_at(expr buf, const std::vector<var>& at);

interval_expr buffer_bounds(const expr& buf, const expr& dim);
dim_expr buffer_dim(const expr& buf, const expr& dim);

bool is_buffer_intrinsic(intrinsic fn);


}  // namespace slinky

#endif  // SLINKY_EXPR_H
