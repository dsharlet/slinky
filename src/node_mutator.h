#ifndef SLINKY_NODE_MUTATOR_H
#define SLINKY_NODE_MUTATOR_H

#include "expr.h"

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

  virtual expr mutate(const expr& x) {
    if (x.defined()) {
      x.accept(this);
      return std::move(e_);
    } else {
      return expr();
    }
  }
  virtual stmt mutate(const stmt& x) {
    if (x.defined()) {
      x.accept(this);
      return std::move(s_);
    } else {
      return stmt();
    }
  }

  virtual void visit(const variable* x) override { set_result(x); }
  virtual void visit(const wildcard* x) override { set_result(x); }
  virtual void visit(const constant* x) override { set_result(x); }

  virtual void visit(const let* x) override;
  virtual void visit(const let_stmt* x) override;
  virtual void visit(const add* x) override;
  virtual void visit(const sub* x) override;
  virtual void visit(const mul* x) override;
  virtual void visit(const div* x) override;
  virtual void visit(const mod* x) override;
  virtual void visit(const class min* x) override;
  virtual void visit(const class max* x) override;
  virtual void visit(const equal* x) override;
  virtual void visit(const not_equal* x) override;
  virtual void visit(const less* x) override;
  virtual void visit(const less_equal* x) override;
  virtual void visit(const logical_and* x) override;
  virtual void visit(const logical_or* x) override;
  virtual void visit(const logical_not* x) override;
  virtual void visit(const class select* x) override;
  virtual void visit(const call* x) override;

  virtual void visit(const block* x) override;
  virtual void visit(const loop* x) override;
  virtual void visit(const if_then_else* x) override;
  virtual void visit(const call_stmt* x) override;
  virtual void visit(const copy_stmt* x) override;
  virtual void visit(const allocate* x) override;
  virtual void visit(const make_buffer* x) override;
  virtual void visit(const crop_buffer* x) override;
  virtual void visit(const crop_dim* x) override;
  virtual void visit(const slice_buffer* x) override;
  virtual void visit(const slice_dim* x) override;
  virtual void visit(const truncate_rank* x) override;
  virtual void visit(const check* x) override;
};

}  // namespace slinky

#endif  // SLINKY_NODE_MUTATOR_H