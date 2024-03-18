#ifndef SLINKY_RUNTIME_STMT_H
#define SLINKY_RUNTIME_STMT_H

#include "runtime/expr.h"

#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace slinky {

enum class stmt_node_type {
  none,

  call_stmt,
  copy_stmt,
  let_stmt,
  block,
  loop,
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

class stmt_visitor;

using base_stmt_node = base_node<stmt_node_type, stmt_visitor>;

class stmt {
  ref_count<const base_stmt_node> n_;

public:
  stmt() = default;
  stmt(const stmt&) = default;
  stmt(stmt&&) = default;
  stmt(const base_stmt_node* n) : n_(n) {}

  stmt& operator=(const stmt&) = default;
  stmt& operator=(stmt&&) noexcept = default;

  SLINKY_ALWAYS_INLINE void accept(stmt_visitor* v) const {
    assert(defined());
    n_->accept(v);
  }

  SLINKY_ALWAYS_INLINE bool defined() const { return n_ != nullptr; }
  SLINKY_ALWAYS_INLINE bool same_as(const stmt& other) const { return n_ == other.n_; }
  SLINKY_ALWAYS_INLINE bool same_as(const base_stmt_node* other) const { return n_ == other; }
  SLINKY_ALWAYS_INLINE stmt_node_type type() const { return n_ ? n_->type : stmt_node_type::none; }
  SLINKY_ALWAYS_INLINE const base_stmt_node* get() const { return n_; }

  template <typename T>
  SLINKY_ALWAYS_INLINE const T* as() const {
    if (n_ && type() == T::static_type) {
      return reinterpret_cast<const T*>(&*n_);
    } else {
      return nullptr;
    }
  }
};

template <typename T>
class stmt_node : public base_stmt_node {
public:
  stmt_node() : base_stmt_node(T::static_type) {}
};

class eval_context;

// Call `target`.
class call_stmt : public stmt_node<call_stmt> {
public:
  // TODO: I think it would be cleaner to pass two spans for the input and output symbol lists here, but the overhead
  // might be significant.
  using callable = std::function<index_t(const call_stmt*, eval_context&)>;
  using symbol_list = std::vector<symbol_id>;

  struct callable_attrs {
    // Allow inputs and outputs to this call to be aliased to the same buffer.
    bool allow_in_place = false;
  };

  callable target;
  // These are not actually used during evaluation. They are only here for analyzing the IR, so we can know what will be
  // accessed (and how) by the callable.
  symbol_list inputs;
  symbol_list outputs;
  callable_attrs attrs;

  void accept(stmt_visitor* v) const override;

  static stmt make(callable target, symbol_list inputs, symbol_list outputs, callable_attrs attrs);

  static constexpr stmt_node_type static_type = stmt_node_type::call_stmt;
};

class copy_stmt : public stmt_node<copy_stmt> {
public:
  symbol_id src;
  std::vector<expr> src_x;
  symbol_id dst;
  std::vector<symbol_id> dst_x;
  // padding = nullopt => no padding
  // padding = {} => padding is uninitialized
  // padding = <elem_size bytes> => value to put in padding
  std::optional<std::vector<char>> padding;

  void accept(stmt_visitor* v) const override;

  static stmt make(symbol_id src, std::vector<expr> src_x, symbol_id dst, std::vector<symbol_id> dst_x,
      std::optional<std::vector<char>> padding);

  static constexpr stmt_node_type static_type = stmt_node_type::copy_stmt;
};

// Allows lifting a list of common subexpressions (specified by symbol_id/stmt pairs)
// out of another stmt.
class let_stmt : public stmt_node<let_stmt> {
public:
  // Conceptually, these are evaluated and placed on the stack in order, i.e. lets later in this
  // list can use the values defined by earlier lets in the list.
  std::vector<std::pair<symbol_id, expr>> lets;
  stmt body;

  void accept(stmt_visitor* v) const override;

  static stmt make(std::vector<std::pair<symbol_id, expr>> lets, stmt body);

  static stmt make(symbol_id sym, expr value, stmt body) { return make({{sym, std::move(value)}}, std::move(body)); }

  static constexpr stmt_node_type static_type = stmt_node_type::let_stmt;
};

class block : public stmt_node<block> {
public:
  std::vector<stmt> stmts;

  void accept(stmt_visitor* v) const override;

  // Create a single block to contain all of the `stmts`.
  // Nested block statements are flattened, and undef stmts are removed.
  // Note that this may not produce a block at all if `stmts` contains < 2 items.
  static stmt make(std::vector<stmt> stmts);

  // Convenience for the not-uncommon case that we have a vector of stmts
  // (eg checks) followed by a result.
  static stmt make(std::vector<stmt> stmts, stmt tail_stmt);

  static constexpr stmt_node_type static_type = stmt_node_type::block;
};

// Runs `body` for each value i in the interval `bounds` with `sym` set to i.
class loop : public stmt_node<loop> {
public:
  symbol_id sym;
  loop_mode mode;
  interval_expr bounds;
  expr step;
  stmt body;

  void accept(stmt_visitor* v) const override;

  static stmt make(symbol_id sym, loop_mode mode, interval_expr bounds, expr step, stmt body);

  static constexpr stmt_node_type static_type = stmt_node_type::loop;
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

  void accept(stmt_visitor* v) const override;

  static stmt make(symbol_id sym, memory_type storage, std::size_t elem_size, std::vector<dim_expr> dims, stmt body);

  static constexpr stmt_node_type static_type = stmt_node_type::allocate;
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

  void accept(stmt_visitor* v) const override;

  static stmt make(symbol_id sym, expr base, expr elem_size, std::vector<dim_expr> dims, stmt body);

  static constexpr stmt_node_type static_type = stmt_node_type::make_buffer;
};

// Makes a clone of an existing buffer.
// TODO: This basically only exists because we cannot use `make_buffer` to clone a buffer of unknown rank. Maybe there's
// a better way to do this.
class clone_buffer : public stmt_node<clone_buffer> {
public:
  symbol_id sym;
  symbol_id src;
  stmt body;

  void accept(stmt_visitor* v) const override;

  static stmt make(symbol_id sym, symbol_id src, stmt body);

  static constexpr stmt_node_type static_type = stmt_node_type::clone_buffer;
};

// For the `body` scope, crops the buffer `sym` to `bounds`. If the expressions in `bounds` are undefined, they default
// to their original values in the existing buffer. The rank of the buffer is unchanged. If the size of `bounds` is less
// than the rank, the missing values are considered undefined.
class crop_buffer : public stmt_node<crop_buffer> {
public:
  symbol_id sym;
  std::vector<interval_expr> bounds;
  stmt body;

  void accept(stmt_visitor* v) const override;

  static stmt make(symbol_id sym, std::vector<interval_expr> bounds, stmt body);

  static constexpr stmt_node_type static_type = stmt_node_type::crop_buffer;
};

// Similar to `crop_buffer`, but only crops the dimension `dim`.
class crop_dim : public stmt_node<crop_dim> {
public:
  symbol_id sym;
  int dim;
  interval_expr bounds;
  stmt body;

  void accept(stmt_visitor* v) const override;

  static stmt make(symbol_id sym, int dim, interval_expr bounds, stmt body);

  static constexpr stmt_node_type static_type = stmt_node_type::crop_dim;
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

  void accept(stmt_visitor* v) const override;

  static stmt make(symbol_id sym, std::vector<expr> at, stmt body);

  static constexpr stmt_node_type static_type = stmt_node_type::slice_buffer;
};

// Similar to `slice_buffer`, but only slices one dimension `dim`. The rank of the result is one less than the original
// buffer.
class slice_dim : public stmt_node<slice_dim> {
public:
  symbol_id sym;
  int dim;
  expr at;
  stmt body;

  void accept(stmt_visitor* v) const override;

  static stmt make(symbol_id sym, int dim, expr at, stmt body);

  static constexpr stmt_node_type static_type = stmt_node_type::slice_dim;
};

// Within `body`, remove the dimensions of the buffer `sym` above `rank`.
class truncate_rank : public stmt_node<truncate_rank> {
public:
  symbol_id sym;
  int rank;
  stmt body;

  void accept(stmt_visitor* v) const override;

  static stmt make(symbol_id sym, int rank, stmt body);

  static constexpr stmt_node_type static_type = stmt_node_type::truncate_rank;
};

// Basically an assert.
class check : public stmt_node<check> {
public:
  expr condition;

  void accept(stmt_visitor* v) const override;

  static stmt make(expr condition);

  static constexpr stmt_node_type static_type = stmt_node_type::check;
};

class stmt_visitor {
public:
  virtual ~stmt_visitor() = default;

  virtual void visit(const let_stmt*) = 0;
  virtual void visit(const block*) = 0;
  virtual void visit(const loop*) = 0;
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

class recursive_node_visitor : public expr_visitor, public stmt_visitor {
public:
  void visit(const variable*) override;
  void visit(const constant*) override;
  void visit(const let* op) override;

  void visit(const add* op) override;
  void visit(const sub* op) override;
  void visit(const mul* op) override;
  void visit(const div* op) override;
  void visit(const mod* op) override;
  void visit(const class min* op) override;
  void visit(const class max* op) override;
  void visit(const equal* op) override;
  void visit(const not_equal* op) override;
  void visit(const less* op) override;
  void visit(const less_equal* op) override;
  void visit(const logical_and* op) override;
  void visit(const logical_or* op) override;
  void visit(const logical_not* op) override;
  void visit(const class select* op) override;
  void visit(const call* op) override;

  void visit(const let_stmt* op) override;
  void visit(const block* op) override;
  void visit(const loop* op) override;
  void visit(const call_stmt* op) override;
  void visit(const copy_stmt* op) override;
  void visit(const allocate* op) override;
  void visit(const make_buffer* op) override;
  void visit(const clone_buffer* op) override;
  void visit(const crop_buffer* op) override;
  void visit(const crop_dim* op) override;
  void visit(const slice_buffer* op) override;
  void visit(const slice_dim* op) override;
  void visit(const truncate_rank* op) override;
  void visit(const check* op) override;
};

inline void let_stmt::accept(stmt_visitor* v) const { v->visit(this); }
inline void block::accept(stmt_visitor* v) const { v->visit(this); }
inline void loop::accept(stmt_visitor* v) const { v->visit(this); }
inline void call_stmt::accept(stmt_visitor* v) const { v->visit(this); }
inline void copy_stmt::accept(stmt_visitor* v) const { v->visit(this); }
inline void allocate::accept(stmt_visitor* v) const { v->visit(this); }
inline void make_buffer::accept(stmt_visitor* v) const { v->visit(this); }
inline void clone_buffer::accept(stmt_visitor* v) const { v->visit(this); }
inline void crop_buffer::accept(stmt_visitor* v) const { v->visit(this); }
inline void crop_dim::accept(stmt_visitor* v) const { v->visit(this); }
inline void slice_buffer::accept(stmt_visitor* v) const { v->visit(this); }
inline void slice_dim::accept(stmt_visitor* v) const { v->visit(this); }
inline void truncate_rank::accept(stmt_visitor* v) const { v->visit(this); }
inline void check::accept(stmt_visitor* v) const { v->visit(this); }

}  // namespace slinky

#endif  // SLINKY_RUNTIME_STMT_H
