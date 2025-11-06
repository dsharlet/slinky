#ifndef SLINKY_RUNTIME_EXPR_H
#define SLINKY_RUNTIME_EXPR_H

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "slinky/base/arithmetic.h"
#include "slinky/base/ref_count.h"
#include "slinky/base/span.h"
#include "slinky/runtime/buffer.h"

namespace slinky {

class node_context;
class expr;

class var {
public:
  using type = std::size_t;

  static constexpr type invalid = static_cast<size_t>(-1);

  type id;

  var() : id(invalid) {}
  explicit var(type id) : id(id) {}
  var(node_context& ctx, const std::string& name);

  bool defined() const { return id != invalid; }

  expr operator-() const;

  // TODO: This are risky. Someone could write var(x) == var(y) expecting to get an `expr` that compares the values of
  // the variables when evaluated, but this compares the variable ids instead.
  bool operator==(var r) const { return id == r.id; }
  bool operator!=(var r) const { return id != r.id; }
  bool operator<(var r) const { return id < r.id; }
};

// We don't want to be doing string lookups in the inner loops. A node_context
// uniquely maps strings to var.
class node_context {
  std::vector<std::string> sym_to_name;
  std::map<std::string, var> name_to_sym;

public:
  // Get the name of a var.
  std::string name(var i) const;

  // Get or insert a new var for a name.
  var insert(const std::string& name);
  var insert_unique(const std::string& prefix = "_");
  std::optional<var> lookup(const std::string& name) const;
};

enum class expr_node_type {
  none,

  variable,
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
  constant,
};

enum class intrinsic {
  none,

  // Some mathematical constants are expressed as calls to functions with no arguments.
  negative_infinity,
  positive_infinity,
  indeterminate,

  // Common arithmetic functions.
  abs,

  // These are short circuiting logical operations, unlike the && and || operators which are commutative and do not
  // short circuit.
  and_then,
  or_else,

  // This function returns the address of the element x in (buf, x_0, x_1, ...). x can be any rank, including 0.
  buffer_at,

  // These functions implement counting semaphores.
  // The first argument of all of these semaphore helpers is a pointer to an index_t that will be used as the semaphore,
  // and the second argument is a count.
  semaphore_init,
  // wait and signal can take multiple semaphores as a sequence of (sem, count) pairs of arguments.
  semaphore_signal,
  semaphore_wait,

  // Wait for tasks created by an `async` node to complete.
  wait_for,

  // Calls the tracing callback with the first argument. Returns a token that should be passed to end_trace.
  trace_begin,
  trace_end,

  // Free a buffer.
  free,
};

enum class buffer_field : unsigned {
  none = 0,

  rank,
  elem_size,
  size_bytes,

  min,
  max,
  stride,
  fold_factor,
};

class expr_visitor;

// The next few classes are the base of the expression (`expr`) and statement (`stmt`) mechanism.
// `base_expr_node` is the base of `expr`s, and always produce an `index_t`-sized result when evaluated.
// `base_stmt_node` is the base of `stmt`s, and do not produce any result.
// Both are immutable.
template <typename NodeType, typename VisitorType>
class base_node : public ref_counted<base_node<NodeType, VisitorType>> {
public:
  base_node(NodeType type) : type(type) {}

  virtual void accept(VisitorType* v) const = 0;

  NodeType type;

  template <typename T>
  const T* as() const {
    if (type == T::static_type) {
      return reinterpret_cast<const T*>(this);
    } else {
      return nullptr;
    }
  }

  static void destroy(base_node* p) { delete p; }
};

using base_expr_node = base_node<expr_node_type, expr_visitor>;

// These next two are just helpers for constructing the type information.
template <typename T>
class expr_node : public base_expr_node {
public:
  expr_node() : base_expr_node(T::static_type) {}
};

class expr;

// `expr_ref` is a non-owning reference to a `base_expr_node`.
class expr_ref {
  const base_expr_node* n_;

public:
  SLINKY_INLINE expr_ref(const expr_ref&) = default;
  SLINKY_INLINE expr_ref(expr_ref&&) = default;
  SLINKY_INLINE expr_ref& operator=(const expr_ref&) = default;
  SLINKY_INLINE expr_ref& operator=(expr_ref&&) = default;

  SLINKY_INLINE expr_ref(const expr& e);
  SLINKY_INLINE expr_ref(const base_expr_node* n) : n_(n) {}

  SLINKY_INLINE void accept(expr_visitor* v) const {
    assert(defined());
    n_->accept(v);
  }

  SLINKY_INLINE bool defined() const { return n_ != nullptr; }
  SLINKY_INLINE expr_node_type type() const { return n_ ? n_->type : expr_node_type::none; }
  SLINKY_INLINE const base_expr_node* get() const { return n_; }

  template <typename T>
  SLINKY_INLINE const T* as() const {
    if (n_ && type() == T::static_type) {
      return static_cast<const T*>(&*n_);
    } else {
      return nullptr;
    }
  }
};

// `expr` is an owner of a reference counted pointer to a `base_expr_node`.
// Operations that appear to mutate these objects are actually just reassigning this reference counted pointer.
class expr {
  ref_count<const base_expr_node> n_;

public:
  SLINKY_INLINE expr() = default;
  SLINKY_INLINE expr(const expr&) = default;
  SLINKY_INLINE expr(expr&&) = default;
  SLINKY_INLINE expr& operator=(const expr&) = default;
  SLINKY_INLINE expr& operator=(expr&&) = default;

  // Make a new constant expression.
  expr(std::int64_t x);
  SLINKY_INLINE expr(std::int32_t x) : expr(static_cast<std::int64_t>(x)) {}
  SLINKY_INLINE expr(std::size_t x) : expr(static_cast<std::int64_t>(x)) {}
  expr(var sym);

  // Make an `expr` referencing an existing node.
  SLINKY_INLINE expr(expr_ref e) : n_(e.get()) {}
  SLINKY_INLINE explicit expr(const base_expr_node* n) : n_(n) {}

  SLINKY_INLINE void accept(expr_visitor* v) const {
    assert(defined());
    n_->accept(v);
  }

  SLINKY_INLINE bool defined() const { return n_ != nullptr; }
  SLINKY_INLINE bool same_as(expr_ref other) const { return n_ == other.get(); }
  SLINKY_INLINE expr_node_type type() const { return n_ ? n_->type : expr_node_type::none; }
  SLINKY_INLINE const base_expr_node* get() const { return n_; }

  template <typename T>
  SLINKY_INLINE const T* as() const {
    if (n_ && type() == T::static_type) {
      return reinterpret_cast<const T*>(&*n_);
    } else {
      return nullptr;
    }
  }

  expr operator-() const;

  expr& operator+=(expr r);
  expr& operator-=(expr r);
  expr& operator*=(expr r);
  expr& operator/=(expr r);
  expr& operator%=(expr r);
};

SLINKY_INLINE expr_ref::expr_ref(const expr& e) : n_(e.get()) {}

expr operator+(expr a, expr b);
expr operator-(expr a, expr b);
expr operator*(expr a, expr b);
expr operator/(expr a, expr b);
expr operator%(expr a, expr b);
expr euclidean_div(expr a, expr b);
expr euclidean_mod(expr a, expr b);
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
expr min(span<expr> x);
expr max(span<expr> x);

// Check if a and b should be commuted.
SLINKY_INLINE constexpr bool should_commute(expr_node_type a, expr_node_type b) { return a > b; }
inline bool should_commute(const expr& a, const expr& b) { return should_commute(a.type(), b.type()); }

struct interval_expr {
  expr min, max;

  interval_expr() = default;
  explicit interval_expr(expr point) : min(std::move(point)), max(min) {}
  interval_expr(expr min, expr max) : min(std::move(min)), max(std::move(max)) {}

  bool same_as(const interval_expr& r) const { return min.same_as(r.min) && max.same_as(r.max); }

  bool is_point() const { return min.defined() && min.same_as(max); }
  bool is_point(expr_ref x) const { return min.same_as(x) && max.same_as(x); }

  expr_ref as_point() const { return is_point() ? expr_ref(min) : expr_ref(nullptr); }

  static const interval_expr& all();
  static const interval_expr& none();
  // An interval_expr x such that x | y == y
  static const interval_expr& union_identity();
  // An interval_expr x such that x & y == y
  static const interval_expr& intersection_identity();

  const expr& begin() const;
  expr end() const;
  expr extent() const;
  expr empty() const;
  expr contains(expr_ref x) const;

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
  interval_expr& operator|=(interval_expr r);
  // This is intersection, just to be consistent with union.
  interval_expr& operator&=(interval_expr r);
  interval_expr operator|(interval_expr r) const;
  interval_expr operator&(interval_expr r) const;
};

// Make an interval of the region [begin, end) (like python's range).
interval_expr range(expr begin, expr end);
// Make an interval of the region [min, max].
interval_expr bounds(expr min, expr max);
// Make an interval of the region [min, min + extent).
interval_expr min_extent(const expr& min, expr extent);
// Make a interval of the region [x, x].
inline interval_expr point(expr x) { return interval_expr(std::move(x)); }

interval_expr operator*(const expr& a, const interval_expr& b);
interval_expr operator+(const expr& a, const interval_expr& b);
interval_expr operator-(const expr& a, const interval_expr& b);

expr clamp(expr x, interval_expr b);
interval_expr select(const expr& c, interval_expr t, interval_expr f);

// A box is a multidimensional interval.
using box_expr = std::vector<interval_expr>;
box_expr operator|(box_expr a, const box_expr& b);
box_expr operator&(box_expr a, const box_expr& b);

// Allows lifting a list of common subexpressions (specified by var/expr pairs)
// out of another expression.
class let : public expr_node<let> {
public:
  // Conceptually, these are evaluated and placed on the stack in order, i.e. lets later in this
  // list can use the values defined by earlier lets in the list.
  std::vector<std::pair<var, expr>> lets;
  expr body;

  void accept(expr_visitor* v) const override;

  static expr make(std::vector<std::pair<var, expr>> lets, expr body);

  static expr make(var sym, expr value, expr body);

  static constexpr expr_node_type static_type = expr_node_type::let;
};

class variable : public expr_node<variable> {
public:
  // The variable being referenced.
  var sym;

  // The field of the variable being referenced, if any (field != buffer_field::none).
  buffer_field field;

  // If `field` is a per-dimension field, which dimension being referenced.
  int dim;

  void accept(expr_visitor* v) const override;

  // Make a scalar variable reference.
  static expr make(var sym);

  // Make a reference to a field of a buffer.
  static expr make(var sym, buffer_field field, int dim = -1);

  static constexpr expr_node_type static_type = expr_node_type::variable;
};

class constant : public expr_node<constant> {
public:
  index_t value;

  void accept(expr_visitor* v) const override;

  static expr_ref get(index_t value);
  static expr make(index_t value);
  static expr make(const void* value);

  static constexpr expr_node_type static_type = expr_node_type::constant;
};

class binary_op : public base_expr_node {
public:
  expr a, b;

  binary_op(expr_node_type type) : base_expr_node(type) {}
};

#define DECLARE_BINARY_OP(op, c)                                                                                       \
  class op : public binary_op {                                                                                        \
  public:                                                                                                              \
    op() : binary_op(static_type) {}                                                                                   \
    void accept(expr_visitor* v) const override;                                                                       \
    static expr make(expr a, expr b);                                                                                  \
    static constexpr expr_node_type static_type = expr_node_type::op;                                                  \
    static constexpr bool commutative = c;                                                                             \
  };

DECLARE_BINARY_OP(add, true)
DECLARE_BINARY_OP(sub, false)
DECLARE_BINARY_OP(mul, true)
DECLARE_BINARY_OP(div, false)
DECLARE_BINARY_OP(mod, false)
DECLARE_BINARY_OP(min, true)
DECLARE_BINARY_OP(max, true)
DECLARE_BINARY_OP(equal, true)
DECLARE_BINARY_OP(not_equal, true)
DECLARE_BINARY_OP(less, false)
DECLARE_BINARY_OP(less_equal, false)
DECLARE_BINARY_OP(logical_and, true)
DECLARE_BINARY_OP(logical_or, true)

#undef DECLARE_BINARY_OP

// Helpers that do compile-time constant folding based on type.
template <typename T>
expr make_binary(expr a, expr b) {
  return T::make(std::move(a), std::move(b));
}

// clang-format off
template <typename T> SLINKY_INLINE index_t make_binary(index_t a, index_t b);
template <> SLINKY_INLINE index_t make_binary<add>(index_t a, index_t b) { return a + b; }
template <> SLINKY_INLINE index_t make_binary<sub>(index_t a, index_t b) { return a - b; }
template <> SLINKY_INLINE index_t make_binary<mul>(index_t a, index_t b) { return a * b; }
template <> SLINKY_INLINE index_t make_binary<div>(index_t a, index_t b) { return euclidean_div(a, b); }
template <> SLINKY_INLINE index_t make_binary<mod>(index_t a, index_t b) { return euclidean_mod(a, b); }
template <> SLINKY_INLINE index_t make_binary<class min>(index_t a, index_t b) { return std::min(a, b); }
template <> SLINKY_INLINE index_t make_binary<class max>(index_t a, index_t b) { return std::max(a, b); }
template <> SLINKY_INLINE index_t make_binary<equal>(index_t a, index_t b) { return a == b ? 1 : 0; }
template <> SLINKY_INLINE index_t make_binary<not_equal>(index_t a, index_t b) { return a != b ? 1 : 0; }
template <> SLINKY_INLINE index_t make_binary<less>(index_t a, index_t b) { return a < b ? 1 : 0; }
template <> SLINKY_INLINE index_t make_binary<less_equal>(index_t a, index_t b) { return a <= b ? 1 : 0; }
template <> SLINKY_INLINE index_t make_binary<logical_and>(index_t a, index_t b) { return a && b ? 1 : 0; }
template <> SLINKY_INLINE index_t make_binary<logical_or>(index_t a, index_t b) { return a || b ? 1 : 0; }
// clang-format on

// clang-format off
template <typename T> SLINKY_INLINE bool binary_overflows(index_t a, index_t b) { return false; }
template <> SLINKY_INLINE bool binary_overflows<add>(index_t a, index_t b) { return add_overflows(a, b); }
template <> SLINKY_INLINE bool binary_overflows<sub>(index_t a, index_t b) { return sub_overflows(a, b); }
template <> SLINKY_INLINE bool binary_overflows<mul>(index_t a, index_t b) { return mul_overflows(a, b); }
// clang-format on

template <typename T>
expr make_or_eval_binary(index_t a, index_t b) {
  if (binary_overflows<T>(a, b)) {
    return make_binary<T>(expr(a), expr(b));
  } else {
    return make_binary<T>(a, b);
  }
}

class logical_not : public expr_node<logical_not> {
public:
  expr a;

  void accept(expr_visitor* v) const override;

  static expr make(expr a);

  static constexpr expr_node_type static_type = expr_node_type::logical_not;
};

// Similar to the C++ ternary operator. `true_value` or `false_value` are only evaluated when the `condition` is true or
// false, respectively.
class select : public expr_node<class select> {
public:
  expr condition;
  expr true_value;
  expr false_value;

  void accept(expr_visitor* v) const override;

  static expr make(expr condition, expr true_value, expr false_value);

  static constexpr expr_node_type static_type = expr_node_type::select;
};

class eval_context;

class call : public expr_node<call> {
public:
  using callable = std::function<index_t(const call*, eval_context&)>;

  // If the call is an intrinsic operation, which intrinsic it is.
  slinky::intrinsic intrinsic;

  // Implementation of the call as a user defined function. This function must be a pure function of the inputs.
  callable target;

  // Arguments to the function.
  std::vector<expr> args;

  void accept(expr_visitor* v) const override;

  static expr make(slinky::intrinsic i, callable target, std::vector<expr> args);
  static expr make(slinky::intrinsic i, std::vector<expr> args);
  static expr make(callable target, std::vector<expr> args);

  static constexpr expr_node_type static_type = expr_node_type::call;
};

// A helper containing sub-expressions that describe a dimension of a buffer, corresponding to `dim`.
struct dim_expr {
  interval_expr bounds;
  expr stride;
  expr fold_factor;

  dim_expr() = default;
  dim_expr(interval_expr bounds, expr stride = expr(), expr fold_factor = expr())
      : bounds(std::move(bounds)), stride(std::move(stride)), fold_factor(std::move(fold_factor)) {}
  dim_expr(const dim& d) : bounds(d.min(), d.max()), stride(d.stride()), fold_factor(d.fold_factor()) {}

  const expr& min() const { return bounds.min; }
  const expr& max() const { return bounds.max; }
  expr extent() const { return bounds.extent(); }

  bool same_as(const dim_expr& r) const {
    return bounds.same_as(r.bounds) && stride.same_as(r.stride) && fold_factor.same_as(r.fold_factor);
  }

  const expr& get_field(buffer_field field) const;

  expr is_broadcast() const {
    return bounds.min == 0 && bounds.max == 0 && stride == 0 && fold_factor == 0;
  }
};

class expr_visitor {
public:
  virtual ~expr_visitor() = default;

  virtual void visit(const variable*) = 0;
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
};

inline void variable::accept(expr_visitor* v) const { v->visit(this); }
inline void constant::accept(expr_visitor* v) const { v->visit(this); }
inline void let::accept(expr_visitor* v) const { v->visit(this); }
inline void add::accept(expr_visitor* v) const { v->visit(this); }
inline void sub::accept(expr_visitor* v) const { v->visit(this); }
inline void mul::accept(expr_visitor* v) const { v->visit(this); }
inline void div::accept(expr_visitor* v) const { v->visit(this); }
inline void mod::accept(expr_visitor* v) const { v->visit(this); }
inline void min::accept(expr_visitor* v) const { v->visit(this); }
inline void max::accept(expr_visitor* v) const { v->visit(this); }
inline void equal::accept(expr_visitor* v) const { v->visit(this); }
inline void not_equal::accept(expr_visitor* v) const { v->visit(this); }
inline void less::accept(expr_visitor* v) const { v->visit(this); }
inline void less_equal::accept(expr_visitor* v) const { v->visit(this); }
inline void logical_and::accept(expr_visitor* v) const { v->visit(this); }
inline void logical_or::accept(expr_visitor* v) const { v->visit(this); }
inline void logical_not::accept(expr_visitor* v) const { v->visit(this); }
inline void select::accept(expr_visitor* v) const { v->visit(this); }
inline void call::accept(expr_visitor* v) const { v->visit(this); }

// If `x` is a constant, returns the value of the constant, otherwise `nullopt`.
SLINKY_INLINE std::optional<index_t> as_constant(expr_ref x) {
  const constant* cx = x.as<constant>();
  if (cx) {
    return cx->value;
  } else {
    return std::nullopt;
  }
}

// If `x` is a variable, returns the `var` of the variable, otherwise `nullopt`.
SLINKY_INLINE std::optional<var> as_variable(expr_ref x, buffer_field field = buffer_field::none) {
  const variable* vx = x.as<variable>();
  if (vx && vx->field == field) {
    return vx->sym;
  } else {
    return std::nullopt;
  }
}

// Check if `x` is a variable equal to the symbol `sym`.
SLINKY_INLINE bool is_variable(expr_ref x, var sym, buffer_field field = buffer_field::none) {
  const variable* vx = x.as<variable>();
  return vx ? vx->sym == sym && vx->field == field : false;
}

bool is_variable(expr_ref x, var b, buffer_field field, int dim);

// Check if `x` is equal to the constant `value`.
SLINKY_INLINE bool is_constant(expr_ref x, index_t value) {
  const constant* cx = x.as<constant>();
  return cx ? cx->value == value : false;
}
SLINKY_INLINE bool is_zero(expr_ref x) { return is_constant(x, 0); }
SLINKY_INLINE bool is_one(expr_ref x) { return is_constant(x, 1); }
SLINKY_INLINE bool is_true(expr_ref x) {
  const constant* cx = x.as<constant>();
  return cx ? cx->value != 0 : false;
}
SLINKY_INLINE bool is_false(expr_ref x) { return is_zero(x); }

// Check if `x` is a call to the intrinsic `fn`.
SLINKY_INLINE const call* as_intrinsic(expr_ref x, intrinsic fn) {
  const call* c = x.as<call>();
  return c && c->intrinsic == fn ? c : nullptr;
}

bool is_positive_infinity(expr_ref x);
bool is_negative_infinity(expr_ref x);
bool is_indeterminate(expr_ref x);
int is_infinity(expr_ref x);
bool is_finite(expr_ref x);

SLINKY_INLINE index_t boolean(index_t x) { return x != 0 ? 1 : 0; }
expr boolean(const expr& x);
bool is_boolean(expr_ref x);

inline constexpr bool is_boolean_node(expr_node_type t) {
  switch (t) {
  case expr_node_type::less: return true;
  case expr_node_type::less_equal: return true;
  case expr_node_type::equal: return true;
  case expr_node_type::not_equal: return true;
  case expr_node_type::logical_and: return true;
  case expr_node_type::logical_or: return true;
  case expr_node_type::logical_not: return true;
  default: return false;
  }
}

// Get an expression representing non-numerical constants.
const expr& positive_infinity();
const expr& negative_infinity();
const expr& infinity(int sign = 1);
const expr& indeterminate();

bool is_positive(expr_ref x);
bool is_non_negative(expr_ref x);
bool is_negative(expr_ref x);
bool is_non_positive(expr_ref x);

expr abs(expr x);
expr align_down(expr x, const expr& a);
expr align_up(expr x, const expr& a);
// Expand the interval to have a min and extent aligned to a multiple of a.
interval_expr align(interval_expr x, const expr& a);

// Short circuiting `and` and `or` ops.
expr and_then(expr a, expr b);
expr or_else(expr a, expr b);

expr buffer_rank(var buf);
expr buffer_elem_size(var buf);
expr buffer_min(var buf, int dim);
expr buffer_max(var buf, int dim);
expr buffer_extent(var buf, int dim);
expr buffer_stride(var buf, int dim);
expr buffer_fold_factor(var buf, int dim);

interval_expr buffer_bounds(var buf, int dim);
dim_expr buffer_dim(var buf, int dim);
std::vector<dim_expr> buffer_dims(var buf, int rank);
std::vector<dim_expr> buffer_dims(const raw_buffer& buf);

expr buffer_at(expr buf, span<const expr> at);
expr buffer_at(expr buf, span<const var> at);
expr buffer_at(expr buf);

template <typename... Args>
expr buffer_at(expr buf, expr at0, Args... at) {
  expr args[] = {at0, at...};
  return buffer_at(buf, args);
}

box_expr dims_bounds(span<const dim_expr> dims);

expr semaphore_init(expr sem, expr count = expr());
expr semaphore_signal(expr sem, expr count = expr());
expr semaphore_signal(span<const expr> sems, span<const expr> counts = {});
expr semaphore_wait(expr sem, expr count = expr());
expr semaphore_wait(span<const expr> sems, span<const expr> counts = {});

expr wait_for(expr task);
expr wait_for(std::vector<expr> tasks);

template <typename T>
class symbol_map {
  std::vector<std::optional<T>> values;

  void grow(std::size_t size) {
    if (size >= values.size()) {
      values.resize(std::max(values.size() * 2, size + 1));
    }
  }

  // Allow this to access values without calling `grow`.
  template <typename T2>
  friend class scoped_value_in_symbol_map;

public:
  symbol_map() = default;
  symbol_map(std::size_t reserve) : values(reserve) {}
  symbol_map(std::initializer_list<std::pair<var, T>> init) {
    for (const std::pair<var, T>& i : init) {
      operator[](i.first) = i.second;
    }
  }

  std::optional<T> lookup(var v) const {
    if (v.id < values.size()) {
      return values[v.id];
    }
    return std::nullopt;
  }

  const T& lookup(var v, const T& def) const {
    if (v.id < values.size() && values[v.id]) {
      return *values[v.id];
    }
    return def;
  }

  std::optional<T> operator[](var v) const { return lookup(v); }
  std::optional<T>& operator[](var v) {
    grow(v.id);
    return values[v.id];
  }

  bool contains(var v) const {
    if (v.id >= values.size()) {
      return false;
    }
    return !!values[v.id];
  }

  std::optional<T> operator[](std::size_t i) const {
    assert(i < values.size());
    return values[i];
  }
  std::optional<T>& operator[](std::size_t i) {
    assert(i < values.size());
    return values[i];
  }

  void erase(var v) {
    if (v.id < values.size()) {
      values[v.id] = std::nullopt;
    }
  }

  std::size_t size() const { return values.size(); }
  void reserve(std::size_t size) { values.resize(std::max(values.size(), size)); }
  auto begin() { return values.begin(); }
  auto end() { return values.end(); }
  auto begin() const { return values.begin(); }
  auto end() const { return values.end(); }
  void clear() { values.clear(); }
};

// Set a value in an eval_context upon construction, and restore the old value upon destruction.
template <typename T>
class scoped_value_in_symbol_map {
  symbol_map<T>* context_;
  var sym_;
  std::optional<T> old_value_;

public:
  scoped_value_in_symbol_map(symbol_map<T>& context, var sym) : context_(&context), sym_(sym) {
    std::optional<T>& ctx_value = context[sym];
    old_value_ = std::move(ctx_value);
  }
  scoped_value_in_symbol_map(symbol_map<T>& context, var sym, T value) : context_(&context), sym_(sym) {
    std::optional<T>& ctx_value = context[sym];
    old_value_ = std::move(ctx_value);
    ctx_value = std::move(value);
  }
  scoped_value_in_symbol_map(symbol_map<T>& context, var sym, std::optional<T> value) : context_(&context), sym_(sym) {
    std::optional<T>& ctx_value = context[sym];
    old_value_ = std::move(ctx_value);
    ctx_value = std::move(value);
  }

  scoped_value_in_symbol_map(scoped_value_in_symbol_map&& other) noexcept
      : context_(other.context_), sym_(other.sym_), old_value_(std::move(other.old_value_)) {
    // Don't let other.~scoped_value() unset this value.
    other.context_ = nullptr;
  }
  scoped_value_in_symbol_map(const scoped_value_in_symbol_map&) = delete;
  scoped_value_in_symbol_map& operator=(const scoped_value_in_symbol_map&) = delete;
  scoped_value_in_symbol_map& operator=(scoped_value_in_symbol_map&& other) noexcept {
    std::swap(context_, other.context_);
    std::swap(sym_, other.sym_);
    std::swap(old_value_, other.old_value_);
    return *this;
  }

  var sym() const { return sym_; }
  const std::optional<T>& old_value() const { return old_value_; }

  void exit_scope() {
    if (context_) {
      // We know this context is big enough because we got old_value_ from it.
      context_->values[sym_.id] = std::move(old_value_);
      context_ = nullptr;
    }
  }

  ~scoped_value_in_symbol_map() { exit_scope(); }
};

template <typename T>
scoped_value_in_symbol_map<T> set_value_in_scope(symbol_map<T>& context, var sym, T value) {
  return scoped_value_in_symbol_map<T>(context, sym, std::move(value));
}
template <typename T>
scoped_value_in_symbol_map<T> set_value_in_scope(symbol_map<T>& context, var sym, std::optional<T> value) {
  return scoped_value_in_symbol_map<T>(context, sym, std::move(value));
}
template <typename T>
scoped_value_in_symbol_map<T> set_value_in_scope(symbol_map<T>& context, var sym, std::nullopt_t) {
  return scoped_value_in_symbol_map<T>(context, sym, std::nullopt);
}

}  // namespace slinky

#endif  // SLINKY_RUNTIME_EXPR_H
