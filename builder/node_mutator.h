#ifndef SLINKY_BUILDER_NODE_MUTATOR_H
#define SLINKY_BUILDER_NODE_MUTATOR_H

#include "runtime/expr.h"
#include "runtime/stmt.h"

namespace slinky {

class node_mutator : public expr_visitor, public stmt_visitor {
  expr e_;
  stmt s_;

public:
  // We need to be careful not to allow derived classes to release these while
  // they might still be in used.
  void set_result(expr e) {
    assert(!e_.defined());
    e_ = std::move(e);
  }
  void set_result(stmt s) {
    assert(!s_.defined());
    s_ = std::move(s);
  }
  const expr& mutated_expr() const { return e_; }
  const stmt& mutated_stmt() const { return s_; }

  virtual expr mutate(const expr& e) {
    if (e.defined()) {
      e.accept(this);
      return std::move(e_);
    } else {
      return expr();
    }
  }
  virtual stmt mutate(const stmt& s) {
    if (s.defined()) {
      s.accept(this);
      return std::move(s_);
    } else {
      return stmt();
    }
  }

  virtual interval_expr mutate(const interval_expr& x) {
    if (x.is_point()) {
      return point(mutate(x.min));
    } else {
      return {mutate(x.min), mutate(x.max)};
    }
  }

  void visit(const variable* op) override { set_result(op); }
  void visit(const constant* op) override { set_result(op); }

  void visit(const let*) override;
  void visit(const let_stmt*) override;
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

  void visit(const block*) override;
  void visit(const loop*) override;
  void visit(const call_stmt*) override;
  void visit(const copy_stmt*) override;
  void visit(const allocate*) override;
  void visit(const make_buffer*) override;
  void visit(const clone_buffer*) override;
  void visit(const crop_buffer*) override;
  void visit(const crop_dim*) override;
  void visit(const slice_buffer*) override;
  void visit(const slice_dim*) override;
  void visit(const transpose*) override;
  void visit(const check*) override;
};

// This is helpful for writing templated mutators.
stmt clone_with(const loop* op, stmt new_body);
stmt clone_with(const let_stmt* op, stmt new_body);
stmt clone_with(const allocate* op, stmt new_body);
stmt clone_with(const make_buffer* op, stmt new_body);
stmt clone_with(const clone_buffer* op, stmt new_body);
stmt clone_with(const crop_buffer* op, stmt new_body);
stmt clone_with(const crop_dim* op, stmt new_body);
stmt clone_with(const slice_buffer* op, stmt new_body);
stmt clone_with(const slice_dim* op, stmt new_body);
stmt clone_with(const transpose* op, stmt new_body);

stmt clone_with(const loop* op, var sym, stmt new_body);
stmt clone_with(const allocate* op, var sym, stmt new_body);
stmt clone_with(const make_buffer* op, var sym, stmt new_body);
stmt clone_with(const clone_buffer* op, var sym, stmt new_body);
stmt clone_with(const crop_buffer* op, var sym, stmt new_body);
stmt clone_with(const crop_dim* op, var sym, stmt new_body);
stmt clone_with(const slice_buffer* op, var sym, stmt new_body);
stmt clone_with(const slice_dim* op, var sym, stmt new_body);
stmt clone_with(const transpose* op, var sym, stmt new_body);

// Helper for single statement mutators.
template <typename T>
stmt recursive_mutate(const stmt& s, const std::function<stmt(const T*)>& mutator) {
  using mutator_fn = std::function<stmt(const T*)>;
  class impl : public node_mutator {
  public:
    const mutator_fn& mutator;
    impl(const mutator_fn& mutator) : mutator(mutator) {}
    stmt mutate(const stmt& s) override {
      if (const T* t = s.as<T>()) {
        return mutator(t);
      } else {
        return node_mutator::mutate(s);
      }
    }
  };

  return impl(mutator).mutate(s);
}

}  // namespace slinky

#endif  // SLINKY_BUILDER_NODE_MUTATOR_H