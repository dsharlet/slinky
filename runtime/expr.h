#ifndef SLINKY_RUNTIME_EXPR_H
#define SLINKY_RUNTIME_EXPR_H

#include "runtime/buffer.h"
#include "runtime/util.h"

#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace slinky {

using symbol_id = std::size_t;

// We don't want to be doing string lookups in the inner loops. A node_context
// uniquely maps strings to symbol_id.
class node_context {
  std::vector<std::string> sym_to_name;

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

  call_stmt,
  copy_stmt,
  let_stmt,
  block,
  loop,
  if_then_else,
  allocate,
  make_buffer,
  clone_buffer,
  crop_buffer,
  crop_dim,
  slice_buffer,
  slice_dim,
  truncate_rank,
  check,
};

enum class loop_mode {
  serial,
  parallel,
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

// The next few classes are the base of the expression (`expr`) and statement (`stmt`) mechanism.
// `base_expr_node` is the base of `expr`s, and always produce an `index_t`-sized result when evaluated.
// `base_stmt_node` is the base of `stmt`s, and do not produce any result.
// Both are immutable.
class base_node : public ref_counted<base_node> {
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

  static void destroy(base_node* p) { delete p; }
};

class base_expr_node : public base_node {
public:
  base_expr_node(node_type type) : base_node(type) {}
};

class base_stmt_node : public base_node {
public:
  base_stmt_node(node_type type) : base_node(type) {}
};

// These next two are just helpers for constructing the type information.
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

// These are the same as operator/ and operator% for expr, but having these allows overloaded calls to work properly.
expr euclidean_div(expr a, expr b);
expr euclidean_mod(expr a, expr b);

// `expr` is an owner of a reference counted pointer to a `base_expr_node`, `stmt` similarly owns a `base_stmt_node`
// pointer. Operations that appear to mutate these objects are actually just reassigning this reference counted pointer.
class expr {
  ref_count<const base_expr_node> n_;

public:
  expr() = default;
  expr(const expr&) = default;
  expr(expr&&) = default;
  expr& operator=(const expr&) = default;
  expr& operator=(expr&&) = default;

  // Make a new constant expression.
  expr(index_t x);
  expr(int x) : expr(static_cast<index_t>(x)) {}

  // Make an `expr` referencing an existing node.
  expr(const base_expr_node* n) : n_(n) {}

  void accept(node_visitor* v) const {
    assert(defined());
    n_->accept(v);
  }

  bool defined() const { return n_ != nullptr; }
  bool same_as(const expr& other) const { return n_ == other.n_; }
  bool same_as(const base_expr_node* other) const { return n_ == other; }
  node_type type() const { return n_ ? n_->type : node_type::none; }
  const base_expr_node* get() const { return n_; }

  template <typename T>
  const T* as() const {
    if (n_ && type() == T::static_type) {
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
expr min(span<expr> x);
expr max(span<expr> x);

struct interval_expr {
  expr min, max;

  interval_expr() {}
  explicit interval_expr(const expr& point) : min(point), max(point) {}
  interval_expr(expr min, expr max) : min(std::move(min)), max(std::move(max)) {}

  bool same_as(const interval_expr& r) const { return min.same_as(r.min) && max.same_as(r.max); }

  bool is_point() const { return min.same_as(max); }

  static const interval_expr& all();
  static const interval_expr& none();
  // An interval_expr x such that x | y == y
  static const interval_expr& union_identity();
  // An interval_expr x such that x & y == y
  static const interval_expr& intersection_identity();

  const expr& begin() const { return min; }
  expr end() const { return max + 1; }
  expr extent() const { return max - min + 1; }
  expr empty() const { return min > max; }

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

// A box is a multidimensional interval.
using box_expr = std::vector<interval_expr>;
box_expr operator|(box_expr a, const box_expr& b);
box_expr operator&(box_expr a, const box_expr& b);

class stmt {
  ref_count<const base_stmt_node> n_;

public:
  stmt() = default;
  stmt(const stmt&) = default;
  stmt(stmt&&) = default;
  stmt(const base_stmt_node* n) : n_(n) {}

  stmt& operator=(const stmt&) = default;
  stmt& operator=(stmt&&) = default;

  void accept(node_visitor* v) const {
    assert(defined());
    n_->accept(v);
  }

  bool defined() const { return n_ != nullptr; }
  bool same_as(const stmt& other) const { return n_ == other.n_; }
  bool same_as(const base_stmt_node* other) const { return n_ == other; }
  node_type type() const { return n_ ? n_->type : node_type::none; }
  const base_stmt_node* get() const { return n_; }

  template <typename T>
  const T* as() const {
    if (n_ && type() == T::static_type) {
      return reinterpret_cast<const T*>(&*n_);
    } else {
      return nullptr;
    }
  }
};

// Allows lifting a list of common subexpressions (specified by symbol_id/expr pairs)
// out of another expression.
class let : public expr_node<let> {
public:
  // Conceptually, these are evaluated and placed on the stack in order, i.e. lets later in this
  // list can use the values defined by earlier lets in the list.
  std::vector<std::pair<symbol_id, expr>> lets;
  expr body;

  void accept(node_visitor* v) const;

  static expr make(std::vector<std::pair<symbol_id, expr>> lets, expr body);

  static expr make(symbol_id sym, expr value, expr body) { return make({{sym, std::move(value)}}, std::move(body)); }

  static constexpr node_type static_type = node_type::let;
};

class variable : public expr_node<variable> {
public:
  symbol_id sym;

  void accept(node_visitor* v) const;

  static expr make(symbol_id sym);

  static constexpr node_type static_type = node_type::variable;
};

// Similar to a variable, designed for use in pattern matching. A match with x is only
// accepted if matches(x) returns true.
// TODO(https://github.com/dsharlet/slinky/issues/6): This is pretty ugly. We should be
// able to contain this kind of logic to pattern matching only, it shouldn't be polluting
// the expression mechanism.
class wildcard : public expr_node<wildcard> {
public:
  symbol_id sym;
  std::function<bool(const expr&)> matches;

  void accept(node_visitor* v) const;

  static expr make(symbol_id sym, std::function<bool(const expr&)> matches);

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
  expr a;

  void accept(node_visitor* v) const;

  static expr make(expr a);

  static constexpr node_type static_type = node_type::logical_not;
};

// Similar to the C++ ternary operator. `true_value` or `false_value` are only evaluated when the `condition` is true or
// false, respectively.
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

class eval_context;

// Call `target`.
class call_stmt : public stmt_node<call_stmt> {
public:
  using callable = std::function<index_t(eval_context&)>;
  using symbol_list = std::vector<symbol_id>;

  callable target;
  // These are not actually used during evaluation. They are only here for analyzing the IR, so we can know what will be
  // accessed (and how) by the callable.
  symbol_list inputs;
  symbol_list outputs;

  void accept(node_visitor* v) const;

  static stmt make(callable target, symbol_list inputs, symbol_list outputs);

  static constexpr node_type static_type = node_type::call_stmt;
};

class copy_stmt : public stmt_node<copy_stmt> {
public:
  symbol_id src;
  std::vector<expr> src_x;
  symbol_id dst;
  std::vector<symbol_id> dst_x;
  std::vector<char> padding;

  void accept(node_visitor* v) const;

  static stmt make(
      symbol_id src, std::vector<expr> src_x, symbol_id dst, std::vector<symbol_id> dst_x, std::vector<char> padding);

  static constexpr node_type static_type = node_type::copy_stmt;
};

// Allows lifting a list of common subexpressions (specified by symbol_id/stmt pairs)
// out of another stmt.
class let_stmt : public stmt_node<let_stmt> {
public:
  // Conceptually, these are evaluated and placed on the stack in order, i.e. lets later in this
  // list can use the values defined by earlier lets in the list.
  std::vector<std::pair<symbol_id, expr>> lets;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(std::vector<std::pair<symbol_id, expr>> lets, stmt body);

  static stmt make(symbol_id sym, expr value, stmt body) { return make({{sym, std::move(value)}}, std::move(body)); }

  static constexpr node_type static_type = node_type::let_stmt;
};

class block : public stmt_node<block> {
public:
  std::vector<stmt> stmts;

  void accept(node_visitor* v) const;

  // Create a single block to contain all of the `stmts`.
  // Nested block statements are flattened, and undef stmts are removed.
  // Note that this may not produce a block at all if `stmts` contains < 2 items.
  static stmt make(std::vector<stmt> stmts);

  // Convenience for the not-uncommon case that we have a vector of stmts
  // (eg checks) followed by a result.
  static stmt make(std::vector<stmt> stmts, stmt tail_stmt);

  static constexpr node_type static_type = node_type::block;
};

// Runs `body` for each value i in the interval `bounds` with `sym` set to i.
class loop : public stmt_node<loop> {
public:
  symbol_id sym;
  loop_mode mode;
  interval_expr bounds;
  expr step;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(symbol_id sym, loop_mode mode, interval_expr bounds, expr step, stmt body);

  static constexpr node_type static_type = node_type::loop;
};

// Run `true_body` if `condition` is true, or `false_body` otherwise. Either body can be undefined, indicating that
// nothing should happen in that case.
class if_then_else : public stmt_node<if_then_else> {
public:
  expr condition;
  stmt true_body;
  stmt false_body;

  void accept(node_visitor* v) const;

  static stmt make(expr condition, stmt true_body, stmt false_body = stmt());

  static constexpr node_type static_type = node_type::if_then_else;
};

// A helper containing sub-expressions that describe a dimension of a buffer, corresponding to `dim`.
struct dim_expr {
  interval_expr bounds;
  expr stride;
  expr fold_factor;

  const expr& min() const { return bounds.min; }
  const expr& max() const { return bounds.max; }
  expr extent() const { return bounds.extent(); }

  bool same_as(const dim_expr& r) const {
    return bounds.same_as(r.bounds) && stride.same_as(r.stride) && fold_factor.same_as(r.fold_factor);
  }
};

// Allocates memory and creates a buffer pointing to that memory. When control flow exits `body`, the buffer is freed.
// `sym` refers to a pointer to a `raw_buffer` object, the fields are initialized by the corresponding expressions in
// this node (`rank` is the size of `dims`).
class allocate : public stmt_node<allocate> {
public:
  memory_type storage;
  symbol_id sym;
  std::size_t elem_size;
  std::vector<dim_expr> dims;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(symbol_id sym, memory_type storage, std::size_t elem_size, std::vector<dim_expr> dims, stmt body);

  static constexpr node_type static_type = node_type::allocate;
};

// Make a `raw_buffer` object around an existing pointer `base` with fields corresponding to the expressions in this
// node (`rank` is the size of `dims`).
class make_buffer : public stmt_node<make_buffer> {
public:
  symbol_id sym;
  expr base;
  expr elem_size;
  std::vector<dim_expr> dims;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(symbol_id sym, expr base, expr elem_size, std::vector<dim_expr> dims, stmt body);

  static constexpr node_type static_type = node_type::make_buffer;
};

// Makes a clone of an existing buffer.
// TODO: This basically only exists because we cannot use `make_buffer` to clone a buffer of unknown rank. Maybe there's
// a better way to do this.
class clone_buffer : public stmt_node<clone_buffer> {
public:
  symbol_id sym;
  symbol_id src;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(symbol_id sym, symbol_id src, stmt body);

  static constexpr node_type static_type = node_type::clone_buffer;
};

// For the `body` scope, crops the buffer `sym` to `bounds`. If the expressions in `bounds` are undefined, they default
// to their original values in the existing buffer. The rank of the buffer is unchanged. If the size of `bounds` is less
// than the rank, the missing values are considered undefined.
class crop_buffer : public stmt_node<crop_buffer> {
public:
  symbol_id sym;
  std::vector<interval_expr> bounds;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(symbol_id sym, std::vector<interval_expr> bounds, stmt body);

  static constexpr node_type static_type = node_type::crop_buffer;
};

// Similar to `crop_buffer`, but only crops the dimension `dim`.
class crop_dim : public stmt_node<crop_dim> {
public:
  symbol_id sym;
  int dim;
  interval_expr bounds;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(symbol_id sym, int dim, interval_expr bounds, stmt body);

  static constexpr node_type static_type = node_type::crop_dim;
};

// For the `body` scope, slices the buffer `sym` at the coordinate `at`. The `at` expressions can be undefined,
// indicating that the corresponding dimension is preserved in the sliced buffer. The sliced buffer will have `rank`
// equal to the rank of the existing buffer, less the number of sliced dimensions. If `at` is smaller than the rank
// of the buffer, the missing values are considered undefined.
class slice_buffer : public stmt_node<slice_buffer> {
public:
  symbol_id sym;
  std::vector<expr> at;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(symbol_id sym, std::vector<expr> at, stmt body);

  static constexpr node_type static_type = node_type::slice_buffer;
};

// Similar to `slice_buffer`, but only slices one dimension `dim`. The rank of the result is one less than the original
// buffer.
class slice_dim : public stmt_node<slice_dim> {
public:
  symbol_id sym;
  int dim;
  expr at;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(symbol_id sym, int dim, expr at, stmt body);

  static constexpr node_type static_type = node_type::slice_dim;
};

// Within `body`, remove the dimensions of the buffer `sym` above `rank`.
class truncate_rank : public stmt_node<truncate_rank> {
public:
  symbol_id sym;
  int rank;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(symbol_id sym, int rank, stmt body);

  static constexpr node_type static_type = node_type::truncate_rank;
};

// Basically an assert.
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
  virtual void visit(const call_stmt*) = 0;
  virtual void visit(const copy_stmt*) = 0;
  virtual void visit(const allocate*) = 0;
  virtual void visit(const make_buffer*) = 0;
  virtual void visit(const clone_buffer*) = 0;
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
  virtual void visit(const let* op) override {
    for (const auto& p : op->lets) {
      p.second.accept(this);
    }
    op->body.accept(this);
  }

  template <typename T>
  void visit_binary(const T* op) {
    op->a.accept(this);
    op->b.accept(this);
  }

  virtual void visit(const add* op) override { visit_binary(op); }
  virtual void visit(const sub* op) override { visit_binary(op); }
  virtual void visit(const mul* op) override { visit_binary(op); }
  virtual void visit(const div* op) override { visit_binary(op); }
  virtual void visit(const mod* op) override { visit_binary(op); }
  virtual void visit(const class min* op) override { visit_binary(op); }
  virtual void visit(const class max* op) override { visit_binary(op); }
  virtual void visit(const equal* op) override { visit_binary(op); }
  virtual void visit(const not_equal* op) override { visit_binary(op); }
  virtual void visit(const less* op) override { visit_binary(op); }
  virtual void visit(const less_equal* op) override { visit_binary(op); }
  virtual void visit(const logical_and* op) override { visit_binary(op); }
  virtual void visit(const logical_or* op) override { visit_binary(op); }
  virtual void visit(const logical_not* op) override { op->a.accept(this); }
  virtual void visit(const class select* op) override {
    op->condition.accept(this);
    op->true_value.accept(this);
    op->false_value.accept(this);
  }
  virtual void visit(const call* op) override {
    for (const expr& i : op->args) {
      if (i.defined()) i.accept(this);
    }
  }

  virtual void visit(const let_stmt* op) override {
    for (const auto& p : op->lets) {
      p.second.accept(this);
    }
    if (op->body.defined()) op->body.accept(this);
  }
  virtual void visit(const block* op) override {
    for (const auto& s : op->stmts) {
      s.accept(this);
    }
  }
  virtual void visit(const loop* op) override {
    op->bounds.min.accept(this);
    op->bounds.max.accept(this);
    if (op->step.defined()) op->step.accept(this);
    if (op->body.defined()) op->body.accept(this);
  }
  virtual void visit(const if_then_else* op) override {
    op->condition.accept(this);
    if (op->true_body.defined()) op->true_body.accept(this);
    if (op->false_body.defined()) op->false_body.accept(this);
  }
  virtual void visit(const call_stmt* op) override {}
  virtual void visit(const copy_stmt* op) override {
    for (const expr& i : op->src_x) {
      i.accept(this);
    }
  }
  virtual void visit(const allocate* op) override {
    for (const dim_expr& i : op->dims) {
      i.bounds.min.accept(this);
      i.bounds.max.accept(this);
      i.stride.accept(this);
      if (i.fold_factor.defined()) i.fold_factor.accept(this);
    }
    if (op->body.defined()) op->body.accept(this);
  }
  virtual void visit(const make_buffer* op) override {
    op->base.accept(this);
    op->elem_size.accept(this);
    for (const dim_expr& i : op->dims) {
      i.bounds.min.accept(this);
      i.bounds.max.accept(this);
      i.stride.accept(this);
      if (i.fold_factor.defined()) i.fold_factor.accept(this);
    }
    if (op->body.defined()) op->body.accept(this);
  }
  virtual void visit(const clone_buffer* op) override {
    if (op->body.defined()) op->body.accept(this);
  }
  virtual void visit(const crop_buffer* op) override {
    for (const interval_expr& i : op->bounds) {
      if (i.min.defined()) i.min.accept(this);
      if (i.max.defined()) i.max.accept(this);
    }
    if (op->body.defined()) op->body.accept(this);
  }
  virtual void visit(const crop_dim* op) override {
    if (op->bounds.min.defined()) op->bounds.min.accept(this);
    if (op->bounds.max.defined()) op->bounds.max.accept(this);
    if (op->body.defined()) op->body.accept(this);
  }
  virtual void visit(const slice_buffer* op) override {
    for (const expr& i : op->at) {
      if (i.defined()) i.accept(this);
    }
    if (op->body.defined()) op->body.accept(this);
  }
  virtual void visit(const slice_dim* op) override {
    op->at.accept(this);
    if (op->body.defined()) op->body.accept(this);
  }
  virtual void visit(const truncate_rank* op) override { op->body.accept(this); }
  virtual void visit(const check* op) override { op->condition.accept(this); }
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
inline void call_stmt::accept(node_visitor* v) const { v->visit(this); }
inline void copy_stmt::accept(node_visitor* v) const { v->visit(this); }
inline void allocate::accept(node_visitor* v) const { v->visit(this); }
inline void make_buffer::accept(node_visitor* v) const { v->visit(this); }
inline void clone_buffer::accept(node_visitor* v) const { v->visit(this); }
inline void crop_buffer::accept(node_visitor* v) const { v->visit(this); }
inline void crop_dim::accept(node_visitor* v) const { v->visit(this); }
inline void slice_buffer::accept(node_visitor* v) const { v->visit(this); }
inline void slice_dim::accept(node_visitor* v) const { v->visit(this); }
inline void truncate_rank::accept(node_visitor* v) const { v->visit(this); }
inline void check::accept(node_visitor* v) const { v->visit(this); }

// If `x` is a constant, returns the value of the constant, otherwise `nullptr`.
inline const index_t* as_constant(const expr& x) {
  const constant* cx = x.as<constant>();
  return cx ? &cx->value : nullptr;
}

// If `x` is a variable, returns the `symbol_id` of the variable, otherwise `nullptr`.
inline const symbol_id* as_variable(const expr& x) {
  const variable* vx = x.as<variable>();
  return vx ? &vx->sym : nullptr;
}

// Check if `x` is a variable equal to the symbol `sym`.
inline bool is_variable(const expr& x, symbol_id sym) {
  const variable* vx = x.as<variable>();
  return vx ? vx->sym == sym : false;
}

// Check if `x` is equal to the constant `value`.
inline bool is_constant(const expr& x, index_t value) {
  const constant* cx = x.as<constant>();
  return cx ? cx->value == value : false;
}
inline bool is_zero(const expr& x) { return is_constant(x, 0); }
inline bool is_one(const expr& x) { return is_constant(x, 1); }
inline bool is_true(const expr& x) {
  const constant* cx = x.as<constant>();
  return cx ? cx->value != 0 : false;
}
inline bool is_false(const expr& x) { return is_zero(x); }

// Check if `x` is a call to the intrinsic `fn`.
inline bool is_intrinsic(const expr& x, intrinsic fn) {
  const call* c = x.as<call>();
  return c ? c->intrinsic == fn : false;
}
bool is_buffer_min(const expr& x, symbol_id sym, int dim);
bool is_buffer_max(const expr& x, symbol_id sym, int dim);
bool is_buffer_intrinsic(intrinsic fn);

inline bool is_positive_infinity(const expr& x) { return is_intrinsic(x, intrinsic::positive_infinity); }
inline bool is_negative_infinity(const expr& x) { return is_intrinsic(x, intrinsic::negative_infinity); }
inline bool is_indeterminate(const expr& x) { return is_intrinsic(x, intrinsic::indeterminate); }
inline bool is_infinity(const expr& x) { return is_positive_infinity(x) || is_negative_infinity(x); }
bool is_finite(const expr& x);

// Get an expression representing non-numerical constants.
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
  symbol_id sym_;

public:
  var();
  var(symbol_id sym);
  var(node_context& ctx, const std::string& sym);

  bool defined() const;
  symbol_id sym() const;

  operator expr() const;
  expr operator-() const { return -static_cast<expr>(*this); }
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
expr buffer_at(expr buf, span<const expr> at);
expr buffer_at(expr buf, span<const var> at);

interval_expr buffer_bounds(const expr& buf, const expr& dim);
dim_expr buffer_dim(const expr& buf, const expr& dim);
std::vector<dim_expr> buffer_dims(const expr& buf, int rank);

box_expr dims_bounds(span<const dim_expr> dims);

template <typename T>
class symbol_map {
  std::vector<std::optional<T>> values;

  void grow(std::size_t size) {
    if (size >= values.size()) {
      values.resize(std::max(values.size() * 2, size + 1));
    }
  }

public:
  symbol_map() {}
  symbol_map(std::initializer_list<std::pair<symbol_id, T>> init) {
    for (const std::pair<symbol_id, T>& i : init) {
      operator[](i.first) = i.second;
    }
  }

  std::optional<T> lookup(symbol_id sym) const {
    if (sym < values.size()) {
      return values[sym];
    }
    return std::nullopt;
  }
  std::optional<T> lookup(const var& v) const { return lookup(v.sym()); }

  const T& lookup(symbol_id sym, const T& def) const {
    if (sym < values.size() && values[sym]) {
      return *values[sym];
    }
    return def;
  }
  const T& lookup(const var& v, const T& def) const { return lookup(v.sym(), def); }

  std::optional<T> operator[](symbol_id sym) const { return lookup(sym); }
  std::optional<T> operator[](const var& v) const { return lookup(v.sym()); }
  std::optional<T>& operator[](symbol_id sym) {
    grow(sym);
    return values[sym];
  }
  std::optional<T>& operator[](const var& v) { return operator[](v.sym()); }

  bool contains(symbol_id sym) const {
    if (sym >= values.size()) {
      return false;
    }
    return !!values[sym];
  }
  bool contains(const var& v) const { return contains(v.sym()); }

  std::size_t size() const { return values.size(); }
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
  symbol_id sym_;
  std::optional<T> old_value_;

public:
  scoped_value_in_symbol_map(symbol_map<T>& context, symbol_id sym, T value) : context_(&context), sym_(sym) {
    std::optional<T>& ctx_value = context[sym];
    old_value_ = std::move(ctx_value);
    ctx_value = std::move(value);
  }
  scoped_value_in_symbol_map(symbol_map<T>& context, symbol_id sym, std::optional<T> value)
      : context_(&context), sym_(sym) {
    std::optional<T>& ctx_value = context[sym];
    old_value_ = std::move(ctx_value);
    ctx_value = std::move(value);
  }

  scoped_value_in_symbol_map(scoped_value_in_symbol_map&& other)
      : context_(other.context_), sym_(other.sym_), old_value_(std::move(other.old_value_)) {
    // Don't let other.~scoped_value() unset this value.
    other.context_ = nullptr;
  }
  scoped_value_in_symbol_map(const scoped_value_in_symbol_map&) = delete;
  scoped_value_in_symbol_map& operator=(const scoped_value_in_symbol_map&) = delete;
  scoped_value_in_symbol_map& operator=(scoped_value_in_symbol_map&& other) {
    context_ = other.context_;
    sym_ = other.sym_;
    old_value_ = std::move(other.old_value_);
    // Don't let other.~scoped_value_in_symbol_map() unset this value.
    other.context_ = nullptr;
  }

  const std::optional<T>& old_value() const { return old_value_; }

  void exit_scope() {
    if (context_) {
      (*context_)[sym_] = std::move(old_value_);
      context_ = nullptr;
    }
  }

  ~scoped_value_in_symbol_map() { exit_scope(); }
};

template <typename T>
scoped_value_in_symbol_map<T> set_value_in_scope(symbol_map<T>& context, symbol_id sym, T value) {
  return scoped_value_in_symbol_map<T>(context, sym, value);
}
template <typename T>
scoped_value_in_symbol_map<T> set_value_in_scope(symbol_map<T>& context, symbol_id sym, std::optional<T> value) {
  return scoped_value_in_symbol_map<T>(context, sym, value);
}

}  // namespace slinky

#endif  // SLINKY_RUNTIME_EXPR_H
