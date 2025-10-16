#ifndef SLINKY_BUILDER_NODE_MUTATOR_H
#define SLINKY_BUILDER_NODE_MUTATOR_H

#include "slinky/base/function_ref.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

namespace slinky {

class stmt_mutator : public stmt_visitor {
  stmt s_;

public:
  // We need to be careful not to allow derived classes to release these while
  // they might still be in used.
  void set_result(stmt s) {
    assert(!s_.defined());
    s_ = std::move(s);
  }
  void set_result(const base_stmt_node* s) { set_result(stmt(s)); }
  const stmt& mutated_stmt() const { return s_; }

  virtual stmt mutate(const stmt& s) {
    assert(!s_.defined());
    if (s.defined()) {
      s.accept(this);
    }
    return std::move(s_);
  }

  void visit(const let_stmt*) override;
  void visit(const block*) override;
  void visit(const loop*) override;
  void visit(const call_stmt*) override;
  void visit(const copy_stmt*) override;
  void visit(const allocate*) override;
  void visit(const make_buffer*) override;
  void visit(const constant_buffer*) override;
  void visit(const clone_buffer*) override;
  void visit(const crop_buffer*) override;
  void visit(const crop_dim*) override;
  void visit(const slice_buffer*) override;
  void visit(const slice_dim*) override;
  void visit(const transpose*) override;
  void visit(const async*) override;
  void visit(const check*) override;
};

class node_mutator : public expr_visitor, public stmt_mutator {
  expr e_;

public:
  // We need to be careful not to allow derived classes to release these while
  // they might still be in used.
  void set_result(expr e) {
    assert(!e_.defined());
    e_ = std::move(e);
  }
  void set_result(const base_expr_node* e) { set_result(expr(e)); }
  using stmt_mutator::set_result;
  const expr& mutated_expr() const { return e_; }
  using stmt_mutator::mutated_stmt;

  virtual expr mutate(const expr& e) {
    assert(!e_.defined());
    if (e.defined()) {
      switch (e.type()) {
      case expr_node_type::variable: visit(static_cast<const variable*>(e.get())); break;
      case expr_node_type::constant: visit(static_cast<const constant*>(e.get())); break;
      default: e.accept(this);
      }
    }
    return std::move(e_);
  }
  using stmt_mutator::mutate;

  virtual interval_expr mutate(const interval_expr& x) {
    if (x.is_point()) {
      return point(mutate(x.min));
    } else {
      return {mutate(x.min), mutate(x.max)};
    }
  }

  void visit(const variable* op) override;
  void visit(const constant* op) override;

  void visit(const let*) override;
  void visit(const add*) override;
  void visit(const sub*) override;
  void visit(const mul*) override;
  void visit(const div*) override;
  void visit(const mod*) override;
  void visit(const class min*) override;
  void visit(const class max*) override;
  void visit(const equal*) override;
  void visit(const not_equal*) override;
  void visit(const less*) override;
  void visit(const less_equal*) override;
  void visit(const logical_and*) override;
  void visit(const logical_or*) override;
  void visit(const logical_not*) override;
  void visit(const class select*) override;
  void visit(const call*) override;

  void visit(const let_stmt*) override;
  void visit(const loop*) override;
  void visit(const call_stmt*) override;
  void visit(const copy_stmt*) override;
  void visit(const allocate*) override;
  void visit(const make_buffer*) override;
  void visit(const crop_buffer*) override;
  void visit(const crop_dim*) override;
  void visit(const slice_buffer*) override;
  void visit(const slice_dim*) override;
  void visit(const check*) override;
  using stmt_mutator::visit;
};

// This is helpful for writing templated mutators.
stmt clone_with(const loop* op, stmt new_body);
stmt clone_with(const let_stmt* op, stmt new_body);
stmt clone_with(const allocate* op, stmt new_body);
stmt clone_with(const make_buffer* op, stmt new_body);
stmt clone_with(const constant_buffer* op, stmt new_body);
stmt clone_with(const clone_buffer* op, stmt new_body);
stmt clone_with(const crop_buffer* op, stmt new_body);
stmt clone_with(const crop_dim* op, stmt new_body);
stmt clone_with(const slice_buffer* op, stmt new_body);
stmt clone_with(const slice_dim* op, stmt new_body);
stmt clone_with(const transpose* op, stmt new_body);
stmt clone_with(const async* op, stmt new_body);

stmt clone_with(const loop* op, var sym, stmt new_body);
stmt clone_with(const allocate* op, var sym, stmt new_body);
stmt clone_with(const make_buffer* op, var sym, stmt new_body);
stmt clone_with(const constant_buffer* op, var sym, stmt new_body);
stmt clone_with(const clone_buffer* op, var sym, stmt new_body);
stmt clone_with(const crop_buffer* op, var sym, stmt new_body);
stmt clone_with(const crop_dim* op, var sym, stmt new_body);
stmt clone_with(const slice_buffer* op, var sym, stmt new_body);
stmt clone_with(const slice_dim* op, var sym, stmt new_body);
stmt clone_with(const transpose* op, var sym, stmt new_body);
stmt clone_with(const async* op, var sym, stmt new_body);

// Helper for single statement mutators.
template <typename T>
stmt recursive_mutate(const stmt& s, function_ref<stmt(const T*)> mutator) {
  using mutator_fn = function_ref<stmt(const T*)>;
  class impl : public stmt_mutator {
  public:
    mutator_fn mutator;
    impl(mutator_fn mutator) : mutator(mutator) {}
    stmt mutate(const stmt& s) override {
      if (const T* t = s.as<T>()) {
        return mutator(t);
      } else {
        return stmt_mutator::mutate(s);
      }
    }
  };

  return impl(mutator).mutate(s);
}

}  // namespace slinky

#endif  // SLINKY_BUILDER_NODE_MUTATOR_H