#ifndef SLINKY_NODE_MUTATOR_H
#define SLINKY_NODE_MUTATOR_H

#include "expr.h"

namespace slinky {

class node_mutator : public node_visitor {
public:
  expr e;
  stmt s;

  expr mutate(const expr& x) {
    x.accept(this);
    return e;
  }
  stmt mutate(const stmt& x) {
    x.accept(this);
    return s;
  }

  template <typename T>
  auto mutate_let(const T* x) {
    expr value = mutate(x->value);
    auto body = mutate(x->body);
    if (value.same_as(x->value) && body.same_as(x->body)) {
      return decltype(body){x};
    } else {
      return T::make(x->name, std::move(value), std::move(body));
    }
  }

  template <typename T>
  expr mutate_binary(const T* x) {
    expr a = mutate(x->a);
    expr b = mutate(x->b);
    if (a.same_as(x->a) && b.same_as(x->b)) {
      return x;
    } else {
      return T::make(std::move(a), std::move(b));
    }
  }

  virtual void visit(const variable* x) { e = x; }
  virtual void visit(const constant* x) { e = x; }

  virtual void visit(const let* x) { e = mutate_let(x); }
  virtual void visit(const add* x) { e = mutate_binary(x); }
  virtual void visit(const sub* x) { e = mutate_binary(x); }
  virtual void visit(const mul* x) { e = mutate_binary(x); }
  virtual void visit(const div* x) { e = mutate_binary(x); }
  virtual void visit(const mod* x) { e = mutate_binary(x); }
  virtual void visit(const class min* x) { e = mutate_binary(x); }
  virtual void visit(const class max* x) { e = mutate_binary(x); }
  virtual void visit(const equal* x) { e = mutate_binary(x); }
  virtual void visit(const not_equal* x) { e = mutate_binary(x); }
  virtual void visit(const less* x) { e = mutate_binary(x); }
  virtual void visit(const less_equal* x) { e = mutate_binary(x); }
  virtual void visit(const bitwise_and* x) { e = mutate_binary(x); }
  virtual void visit(const bitwise_or* x) { e = mutate_binary(x); }
  virtual void visit(const bitwise_xor* x) { e = mutate_binary(x); }
  virtual void visit(const logical_and* x) { e = mutate_binary(x); }
  virtual void visit(const logical_or* x) { e = mutate_binary(x); }
  virtual void visit(const shift_left* x) { e = mutate_binary(x); }
  virtual void visit(const shift_right* x) { e = mutate_binary(x); }
  
  virtual void visit(const load_buffer_meta* x) {
    expr buffer = mutate(x->buffer);
    expr dim = mutate(x->dim);
    if (buffer.same_as(x->buffer) && dim.same_as(x->dim)) {
      e = x;
    } else {
      e = load_buffer_meta::make(std::move(buffer), x->meta, std::move(dim));
    }
  }

  virtual void visit(const let_stmt* x) { s = mutate_let(x); }
  virtual void visit(const block* x) {
    stmt a = mutate(x->a);
    stmt b = mutate(x->b);
    if (a.defined() && b.defined()) {
      if (a.same_as(x->a) && b.same_as(x->b)) {
        s = x;
      } else {
        s = block::make(std::move(a), std::move(b));
      }
    } else if (a.defined()) {
      s = a;
    } else {
      s = b;
    }
  }
  virtual void visit(const loop* x) {
    expr n = mutate(x->n);
    stmt body = mutate(x->body);
    if (n.same_as(x->n) && body.same_as(x->body)) {
      s = x;
    } else {
      s = loop::make(x->name, std::move(x->n), std::move(x->body));
    }
  }
  virtual void visit(const if_then_else* x) {
    expr cond = mutate(x->condition);
    stmt true_body = mutate(x->true_body);
    stmt false_body = mutate(x->false_body);
    if (cond.same_as(x->condition) && true_body.same_as(x->true_body) && false_body.same_as(x->false_body)) {
      s = x;
    } else {
      s = if_then_else::make(std::move(cond), std::move(true_body), std::move(false_body));
    }
  }
  virtual void visit(const call* x) {
    std::vector<expr> scalar_args;
    scalar_args.reserve(x->scalar_args.size());
    for (const expr& i : x->scalar_args) {
      scalar_args.push_back(mutate(i));
    }
    s = call::make(x->target, std::move(scalar_args), x->buffer_args, x->fn);
  }
  virtual void visit(const allocate* x) {
    std::vector<dim_expr> dims;
    dims.reserve(x->dims.size());
    for (const dim_expr& i : x->dims) {
      dims.emplace_back(mutate(i.min), mutate(i.extent), mutate(i.stride_bytes), mutate(i.fold_factor));
    }
    stmt body = mutate(x->body);
    s = allocate::make(x->type, x->name, x->elem_size, std::move(dims), std::move(body));
  }
  virtual void visit(const check* x) {
    expr condition = mutate(x->condition);
    if (condition.same_as(x->condition)) {
      s = x;
    } else {
      s = check::make(std::move(condition));
    }
  }
};

}  // namespace slinky

#endif  // SLINKY_NODE_MUTATOR_H