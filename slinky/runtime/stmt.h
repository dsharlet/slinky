#ifndef SLINKY_RUNTIME_STMT_H
#define SLINKY_RUNTIME_STMT_H

#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "slinky/runtime/expr.h"

namespace slinky {

// This file defines `stmt`, slinky's statement IR. Like `expr` (see runtime/expr.h), a `stmt` is an immutable,
// reference-counted node (`base_stmt_node`); unlike an `expr`, evaluating a `stmt` produces no value, only effects
// (running callbacks, allocating buffers, looping, etc.). Most of these nodes manipulate "symbolic" buffers (as opposed
// to concrete buffers described in buffer.h).
//
// Almost every node that introduces a name (`allocate`, `make_buffer`, `let_stmt`, `loop`, `crop_*`, `slice_*`,
// `transpose`, ...) does so for the duration of a nested `body` statement only. The bound value is live while `body`
// runs and is torn down when control flow leaves `body`. This nesting maps directly onto a call/value stack: entering a
// node pushes its binding, and exiting pops it, so buffers and `let` values can be stack-allocated and buffer metadata
// can be mutated in place and restored on exit.
//
// Buffers are referenced by `var`, and the buffer-view nodes (`crop_buffer`, `slice_buffer`, `clone_buffer`,
// `transpose`, ...) produce a new view of an existing buffer that shares its storage.
//
// Like `raw_buffer` and `buffer<>`, buffers manipulated by these operators are conceptually infinite-dimensional,
// these nodes treat dimension indices at or beyond a buffer's rank as broadcasts, so e.g. cropping or slicing such
// a dimension is well-defined.

enum class stmt_node_type {
  none,

  call_stmt,
  copy_stmt,
  let_stmt,
  block,
  loop,
  allocate,
  make_buffer,
  constant_buffer,
  clone_buffer,
  crop_buffer,
  crop_dim,
  slice_buffer,
  slice_dim,
  transpose,
  async,
  check,
};

enum class memory_type {
  // Automatically place small stack allocations on the stack, as determined by `eval_config::auto_stack_threshold`.
  automatic,
  stack,
  heap,
};

class stmt_visitor;

using base_stmt_node = base_node<stmt_node_type, stmt_visitor>;

class stmt;

// `stmt_ref` is a non-owning reference to a `base_stmt_node`.
class stmt_ref {
  const base_stmt_node* n_;

public:
  SLINKY_INLINE stmt_ref(const stmt_ref&) = default;
  SLINKY_INLINE stmt_ref(stmt_ref&&) = default;
  SLINKY_INLINE stmt_ref& operator=(const stmt_ref&) = default;
  SLINKY_INLINE stmt_ref& operator=(stmt_ref&&) = default;

  SLINKY_INLINE stmt_ref(const stmt& e);
  SLINKY_INLINE stmt_ref(const base_stmt_node* n) : n_(n) {}

  SLINKY_INLINE void accept(stmt_visitor* v) const {
    assert(defined());
    n_->accept(v);
  }

  SLINKY_INLINE bool defined() const { return n_ != nullptr; }
  SLINKY_INLINE stmt_node_type type() const { return n_ ? n_->type : stmt_node_type::none; }
  SLINKY_INLINE const base_stmt_node* get() const { return n_; }

  template <typename T>
  SLINKY_INLINE const T* as() const {
    if (n_ && type() == T::static_type) {
      return static_cast<const T*>(&*n_);
    } else {
      return nullptr;
    }
  }
};

// `stmt` is an owner of a reference counted pointer to a `base_stmt_node`.
class stmt {
  ref_count<const base_stmt_node> n_;

public:
  SLINKY_INLINE stmt() = default;
  SLINKY_INLINE stmt(const stmt&) = default;
  SLINKY_INLINE stmt(stmt&&) = default;
  SLINKY_INLINE stmt(stmt_ref s) : n_(s.get()) {}
  SLINKY_INLINE explicit stmt(const base_stmt_node* n) : n_(n) {}

  SLINKY_INLINE stmt& operator=(const stmt&) = default;
  SLINKY_INLINE stmt& operator=(stmt&&) noexcept = default;

  SLINKY_INLINE void accept(stmt_visitor* v) const {
    assert(defined());
    n_->accept(v);
  }

  SLINKY_INLINE bool defined() const { return n_ != nullptr; }
  SLINKY_INLINE bool same_as(const stmt& other) const { return n_ == other.n_; }
  SLINKY_INLINE bool same_as(const base_stmt_node* other) const { return n_ == other; }
  SLINKY_INLINE stmt_node_type type() const { return n_ ? n_->type : stmt_node_type::none; }
  SLINKY_INLINE const base_stmt_node* get() const { return n_; }

  template <typename T>
  SLINKY_INLINE const T* as() const {
    if (n_ && type() == T::static_type) {
      return reinterpret_cast<const T*>(&*n_);
    } else {
      return nullptr;
    }
  }
};

SLINKY_INLINE stmt_ref::stmt_ref(const stmt& s) : n_(s.get()) {}

template <typename T>
class stmt_node : public base_stmt_node {
public:
  stmt_node() : base_stmt_node(T::static_type) {}
};

class eval_context;

// Calls a user-provided callback. This is how all actual computation in a pipeline happens; everything else in the IR
// exists to set up the buffers a `call_stmt` reads and writes.
class call_stmt : public stmt_node<call_stmt> {
public:
  // TODO: I think it would be cleaner to pass two spans for the input and output symbol lists here, but the overhead
  // might be significant.
  using callable = std::function<index_t(const call_stmt*, eval_context&)>;
  using symbol_list = std::vector<var>;

  struct attributes {
    // A bit mask where the bit `o * inputs.size() + i` being set to 1 indicates that the input `i` may be computed in
    // place with the output `o`.
    // TODO: This being a bitmask limits the number of inputs and outputs that can indicate `allow_in_place` is true...
    int allow_in_place = 0;

    // The dimensions greater than min_rank are assumed to be "shift invariant"
    // for the callback. If such a dimension can be proved to have extent 1 it
    // can be sliced off without changing the semantics of the computation.
    int min_rank = std::numeric_limits<int>::max();

    // A name for the callable. This is only a tag that is passed through slinky and used for printing, it doesn't
    // affect any slinky logic, *except* for if this is named `memcpy`, slinky assumes that the implementation of this
    // callback is:
    // assert(inputs[0]->size_bytes() == outputs[0]->size_bytes());
    // memcpy(outputs[0]->base(), inputs[0]->base(), outputs[0]->size_bytes());
    std::string name;
  };

  // The callback to invoke. It is passed the buffers bound to `inputs` and `outputs` (looked up by `var` in the
  // `eval_context`). It must be a pure function of its `inputs` and `scalars`, and must only write to its `outputs`.
  callable target;
  // These are not actually used during evaluation. They are only here for analyzing the IR, so we can know what will be
  // accessed (and how) by the callable.
  symbol_list inputs;
  symbol_list outputs;
  std::vector<expr> scalars;
  attributes attrs;

  void accept(stmt_visitor* v) const override;

  static stmt make(
      callable target, symbol_list inputs, symbol_list outputs, std::vector<expr> scalars, attributes attrs);

  static constexpr stmt_node_type static_type = stmt_node_type::call_stmt;
};

// Copies from buffer `src` to buffer `dst`. This is a distinct node from `call_stmt` (rather than just another
// callback) so the builder can reason about and optimize copies, e.g. folding them into the producer.
class copy_stmt : public stmt_node<copy_stmt> {
public:
  using callable = std::function<void(const raw_buffer&, const raw_buffer&, const raw_buffer& pad)>;

  // Conceptually, the copy iterates over the domain of `dst` with the loop variables `dst_x` (one per dimension of
  // `dst`), and for each point writes `dst[dst_x...] = src[src_x...]`. Each `src_x` is an expression in terms of
  // `dst_x`; expressing the source coordinates as functions of the destination coordinates allows a single node to
  // describe copies, broadcasts (a `src_x` that doesn't depend on `dst_x`), transposes, flips, and other coordinate
  // remappings.
  var src;
  std::vector<expr> src_x;
  var dst;
  std::vector<var> dst_x;
  // If defined, the copy will be padded with the values from this buffer where `src` is out of bounds of `dst`.
  var pad;

  // This function implements the copy operation. `slinky::copy` is always a suitable implementation of this.
  // The implementation must only perform a copy and no other operations.
  callable impl;

  void accept(stmt_visitor* v) const override;

  static stmt make(callable impl, var src, std::vector<expr> src_x, var dst, std::vector<var> dst_x, var pad);

  static constexpr stmt_node_type static_type = stmt_node_type::copy_stmt;
};

// Binds each `var` in `lets` to the value of the corresponding `expr` while running `body`, allowing common
// subexpressions to be computed once and reused. The bindings are only in scope within `body`. (This is the statement
// analogue of the `let` expression in expr.h.)
class let_stmt : public stmt_node<let_stmt> {
public:
  // Conceptually, these are evaluated and placed on the stack in order, i.e. lets later in this
  // list can use the values defined by earlier lets in the list.
  std::vector<std::pair<var, expr>> lets;
  stmt body;

  // If this is true, then the body does not access any symbols outside of those defined by `lets`.
  // The values of every let must be a `variable` expression.
  bool is_closure;

  void accept(stmt_visitor* v) const override;

  static stmt make(std::vector<std::pair<var, expr>> lets, stmt body, bool is_closure = false);

  static stmt make(var sym, expr value, stmt body);

  static constexpr stmt_node_type static_type = stmt_node_type::let_stmt;
};

// Runs each statement in `stmts` in order. This is the IR's sequencing construct.
class block : public stmt_node<block> {
public:
  std::vector<stmt> stmts;

  void accept(stmt_visitor* v) const override;

  // Create a single block to contain all of the `stmts`.
  // Nested block statements are flattened, and undef stmts are removed.
  // Note that this may not produce a block at all if `stmts` contains < 2 items.
  static stmt make(std::vector<stmt> stmts);
  static stmt make(stmt a, stmt b);

  // Convenience for the not-uncommon case that we have a vector of stmts
  // (eg checks) followed by a result.
  static stmt make(std::vector<stmt> stmts, stmt tail_stmt);

  static constexpr stmt_node_type static_type = stmt_node_type::block;
};

// Runs `body` once for each value of `sym` in a range. Each iteration gets its own scope for `sym`.
class loop : public stmt_node<loop> {
public:
  var sym;
  // Controls parallelism: `serial` runs the iterations sequentially, `parallel` allows any number to run concurrently,
  // and an integer in between caps the number of concurrent workers. When run in parallel, `body` must be safe to
  // execute concurrently across iterations.
  expr max_workers;
  // The (inclusive) range of values taken by `sym`.
  interval_expr bounds;
  // The increment between successive values of `sym`, i.e. `sym` takes the values `bounds.min, bounds.min + step, ...`
  // up to and including `bounds.max`. When undefined it defaults to 1.
  expr step;
  stmt body;

  static constexpr int serial = 1;
  static constexpr int parallel = std::numeric_limits<int>::max();

  void accept(stmt_visitor* v) const override;

  static stmt make(var sym, expr max_workers, interval_expr bounds, expr step, stmt body);

  static constexpr stmt_node_type static_type = stmt_node_type::loop;
};

// Enqueues `task` to run in parallel, storing the task handle in `sym`, which can be waited on using `wait_for`.
// `body` then runs, and after it is complete, waits for `task` to complete.
class async : public stmt_node<async> {
public:
  var sym;
  stmt task;
  stmt body;

  void accept(stmt_visitor* v) const override;

  static stmt make(var sym, stmt task, stmt body);

  static constexpr stmt_node_type static_type = stmt_node_type::async;
};

// Allocates memory and creates a buffer pointing to that memory, bound to `sym` as a pointer to a `raw_buffer` object.
// When control flow exits `body`, the buffer is freed. The buffer's fields are initialized by the expressions in this
// node.
class allocate : public stmt_node<allocate> {
public:
  var sym;
  // Whether the allocation is placed on the stack, the heap, or chosen automatically based on its size.
  memory_type storage;
  expr elem_size;
  // The dimensions of the allocated buffer; its `rank` is the size of `dims`.
  std::vector<dim_expr> dims;
  stmt body;

  void accept(stmt_visitor* v) const override;

  static stmt make(var sym, memory_type storage, expr elem_size, std::vector<dim_expr> dims, stmt body);

  static constexpr stmt_node_type static_type = stmt_node_type::allocate;
};

// Makes a `raw_buffer` object around an existing pointer, bound to `sym` for the duration of `body`. Unlike `allocate`,
// it does not allocate any storage; the buffer's fields are initialized by the expressions in this node.
class make_buffer : public stmt_node<make_buffer> {
public:
  var sym;
  // A pointer to the existing memory the buffer points at.
  expr base;
  expr elem_size;
  // The dimensions of the buffer; its `rank` is the size of `dims`.
  std::vector<dim_expr> dims;
  stmt body;

  void accept(stmt_visitor* v) const override;

  static stmt make(var sym, expr base, expr elem_size, std::vector<dim_expr> dims, stmt body);

  static constexpr stmt_node_type static_type = stmt_node_type::make_buffer;
};

// Similar to `make_buffer`, but takes its buffer parameters from a pointer to a `raw_buffer` object.
class constant_buffer : public stmt_node<constant_buffer> {
public:
  var sym;
  const_raw_buffer_ptr value;
  stmt body;

  void accept(stmt_visitor* v) const override;

  static stmt make(var sym, const_raw_buffer_ptr value, stmt body);

  static constexpr stmt_node_type static_type = stmt_node_type::constant_buffer;
};

// Makes a new buffer `sym` that is a copy of the metadata (base pointer, dims, etc.) of `src`, sharing the same
// underlying storage. This gives `body` an independent buffer object it can mutate (e.g. crop or slice in place)
// without affecting `src`. Most crop/slice/transpose nodes already produce a fresh view, so this is mainly needed when
// the same buffer must be modified along two different paths.
class clone_buffer : public stmt_node<clone_buffer> {
public:
  var sym;
  var src;
  stmt body;

  void accept(stmt_visitor* v) const override;

  static stmt make(var sym, var src, stmt body);

  static constexpr stmt_node_type static_type = stmt_node_type::clone_buffer;
};

// Makes a new buffer `sym` that is a cropped view of the buffer `src`. The rank of the buffer is unchanged.
class crop_buffer : public stmt_node<crop_buffer> {
public:
  var sym;
  var src;
  // The new bounds for each dimension. An undefined bound (min or max) defaults to its original value in `src`. If
  // `bounds` has fewer entries than the rank, the missing dimensions are left at their original bounds.
  std::vector<interval_expr> bounds;
  stmt body;

  void accept(stmt_visitor* v) const override;

  static stmt make(var sym, var src, std::vector<interval_expr> bounds, stmt body);

  static constexpr stmt_node_type static_type = stmt_node_type::crop_buffer;
};

// Similar to `crop_buffer`, but only crops the dimension `dim`.
class crop_dim : public stmt_node<crop_dim> {
public:
  var sym;
  var src;
  int dim;
  interval_expr bounds;
  stmt body;

  void accept(stmt_visitor* v) const override;

  static stmt make(var sym, var src, int dim, interval_expr bounds, stmt body);

  static constexpr stmt_node_type static_type = stmt_node_type::crop_dim;
};

// Makes a new buffer `sym` that is a sliced view of the buffer `src`. The sliced buffer has `rank` equal to that of
// `src`, less the number of sliced dimensions.
class slice_buffer : public stmt_node<slice_buffer> {
public:
  var sym;
  var src;
  // The coordinate at which to slice each dimension. A defined `at[d]` removes dimension `d`, fixing it at that
  // coordinate; an undefined `at[d]` preserves dimension `d` in the result. Dimensions beyond the size of `at` are
  // preserved as well.
  std::vector<expr> at;
  stmt body;

  void accept(stmt_visitor* v) const override;

  static stmt make(var sym, var src, std::vector<expr> at, stmt body);

  static constexpr stmt_node_type static_type = stmt_node_type::slice_buffer;
};

// Similar to `slice_buffer`, but only slices one dimension `dim`. The rank of the result is one less than the original
// buffer.
class slice_dim : public stmt_node<slice_dim> {
public:
  var sym;
  var src;
  int dim;
  expr at;
  stmt body;

  void accept(stmt_visitor* v) const override;

  static stmt make(var sym, var src, int dim, expr at, stmt body);

  static constexpr stmt_node_type static_type = stmt_node_type::slice_dim;
};

// Makes a new buffer `sym` that is a reordered and/or reduced view of `src`, sharing the same storage. This can both
// reorder dimensions (a transpose) and select a subset of them.
class transpose : public stmt_node<transpose> {
public:
  // TODO: We might want a placeholder dim index that indicates any un-selected dims appear there.

  var sym;
  var src;
  // Dimension `i` of the result is dimension `dims[i]` of `src`, so the result has rank `dims.size()`. An entry equal
  // to `new_dim` inserts a new broadcast dimension that is not present in `src`.
  std::vector<int> dims;
  stmt body;

  static constexpr int new_dim = std::numeric_limits<int>::max();

  // A transpose is a "truncate" when `dims` is `{0, 1, ..., n-1}`, i.e. it just keeps the first `n` dimensions in order
  // (lowering the rank without reordering); `is_truncate` detects this case and `make_truncate` builds it.
  static bool is_truncate(span<const int> dims);
  bool is_truncate() const;
  static stmt make_truncate(var sym, var src, int rank, stmt body);

  void accept(stmt_visitor* v) const override;

  static stmt make(var sym, var src, std::vector<int> dims, stmt body);

  static constexpr stmt_node_type static_type = stmt_node_type::transpose;
};

// Asserts that `condition` evaluates to a nonzero value, aborting evaluation otherwise. These can provide information
// that can be used to simplify the smts that follow a check.
class check : public stmt_node<check> {
public:
  expr condition;

  void accept(stmt_visitor* v) const override;

  static stmt make(expr condition);

  static constexpr stmt_node_type static_type = stmt_node_type::check;
};

SLINKY_INLINE const let_stmt* is_closure(const stmt& s) {
  const let_stmt* let = s.as<let_stmt>();
  return let && let->is_closure ? let : nullptr;
}

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
  virtual void visit(const constant_buffer*) = 0;
  virtual void visit(const clone_buffer*) = 0;
  virtual void visit(const crop_buffer*) = 0;
  virtual void visit(const crop_dim*) = 0;
  virtual void visit(const slice_buffer*) = 0;
  virtual void visit(const slice_dim*) = 0;
  virtual void visit(const transpose*) = 0;
  virtual void visit(const async*) = 0;
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
  void visit(const constant_buffer* op) override;
  void visit(const clone_buffer* op) override;
  void visit(const crop_buffer* op) override;
  void visit(const crop_dim* op) override;
  void visit(const slice_buffer* op) override;
  void visit(const slice_dim* op) override;
  void visit(const transpose* op) override;
  void visit(const async* op) override;
  void visit(const check* op) override;
};

inline void let_stmt::accept(stmt_visitor* v) const { v->visit(this); }
inline void block::accept(stmt_visitor* v) const { v->visit(this); }
inline void loop::accept(stmt_visitor* v) const { v->visit(this); }
inline void call_stmt::accept(stmt_visitor* v) const { v->visit(this); }
inline void copy_stmt::accept(stmt_visitor* v) const { v->visit(this); }
inline void allocate::accept(stmt_visitor* v) const { v->visit(this); }
inline void make_buffer::accept(stmt_visitor* v) const { v->visit(this); }
inline void constant_buffer::accept(stmt_visitor* v) const { v->visit(this); }
inline void clone_buffer::accept(stmt_visitor* v) const { v->visit(this); }
inline void crop_buffer::accept(stmt_visitor* v) const { v->visit(this); }
inline void crop_dim::accept(stmt_visitor* v) const { v->visit(this); }
inline void slice_buffer::accept(stmt_visitor* v) const { v->visit(this); }
inline void slice_dim::accept(stmt_visitor* v) const { v->visit(this); }
inline void transpose::accept(stmt_visitor* v) const { v->visit(this); }
inline void async::accept(stmt_visitor* v) const { v->visit(this); }
inline void check::accept(stmt_visitor* v) const { v->visit(this); }

}  // namespace slinky

#endif  // SLINKY_RUNTIME_STMT_H
