#ifndef SLINKY_BUILDER_NODE_MUTATOR_H
#define SLINKY_BUILDER_NODE_MUTATOR_H

#include "runtime/expr.h"

namespace slinky {

class node_mutator : public node_visitor {
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

  virtual void visit(const variable* op) override { set_result(op); }
  virtual void visit(const wildcard* op) override { set_result(op); }
  virtual void visit(const constant* op) override { set_result(op); }

  virtual void visit(const let*) override;
  virtual void visit(const let_stmt*) override;
  virtual void visit(const add*) override;
  virtual void visit(const sub*) override;
  virtual void visit(const mul*) override;
  virtual void visit(const div*) override;
  virtual void visit(const mod*) override;
  virtual void visit(const class min*) override;
  virtual void visit(const class max*) override;
  virtual void visit(const equal*) override;
  virtual void visit(const not_equal*) override;
  virtual void visit(const less*) override;
  virtual void visit(const less_equal*) override;
  virtual void visit(const logical_and*) override;
  virtual void visit(const logical_or*) override;
  virtual void visit(const logical_not*) override;
  virtual void visit(const select_expr*) override;
  virtual void visit(const call*) override;

  virtual void visit(const block*) override;
  virtual void visit(const loop*) override;
  virtual void visit(const if_then_else*) override;
  virtual void visit(const call_stmt*) override;
  virtual void visit(const copy_stmt*) override;
  virtual void visit(const allocate*) override;
  virtual void visit(const make_buffer*) override;
  virtual void visit(const clone_buffer*) override;
  virtual void visit(const crop_buffer*) override;
  virtual void visit(const crop_dim*) override;
  virtual void visit(const slice_buffer*) override;
  virtual void visit(const slice_dim*) override;
  virtual void visit(const truncate_rank*) override;
  virtual void visit(const check*) override;
};

// This is helpful for writing templated mutators.
stmt clone_with_new_body(const let_stmt* op, stmt new_body);
stmt clone_with_new_body(const allocate* op, stmt new_body);
stmt clone_with_new_body(const make_buffer* op, stmt new_body);
stmt clone_with_new_body(const clone_buffer* op, stmt new_body);
stmt clone_with_new_body(const crop_buffer* op, stmt new_body);
stmt clone_with_new_body(const crop_dim* op, stmt new_body);
stmt clone_with_new_body(const slice_buffer* op, stmt new_body);
stmt clone_with_new_body(const slice_dim* op, stmt new_body);
stmt clone_with_new_body(const truncate_rank* op, stmt new_body);

}  // namespace slinky

#endif  // SLINKY_BUILDER_NODE_MUTATOR_H