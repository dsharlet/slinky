#ifndef SLINKY_NODE_MUTATOR_H
#define SLINKY_NODE_MUTATOR_H

#include "expr.h"

namespace slinky {

class node_mutator : public node_visitor {
public:
  expr e;
  stmt s;

  virtual expr mutate(const expr& x) {
    if (x.defined()) {
      x.accept(this);
      return e;
    } else {
      return expr();
    }
  }
  virtual stmt mutate(const stmt& x) {
    if (x.defined()) {
      x.accept(this);
      return s;
    } else {
      return stmt();
    }
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

  virtual void visit(const variable* x) override { e = x; }
  virtual void visit(const wildcard* x) override { e = x; }
  virtual void visit(const constant* x) override { e = x; }

  virtual void visit(const let* x) override { e = mutate_let(x); }
  virtual void visit(const add* x) override { e = mutate_binary(x); }
  virtual void visit(const sub* x) override { e = mutate_binary(x); }
  virtual void visit(const mul* x) override { e = mutate_binary(x); }
  virtual void visit(const div* x) override { e = mutate_binary(x); }
  virtual void visit(const mod* x) override { e = mutate_binary(x); }
  virtual void visit(const class min* x) override { e = mutate_binary(x); }
  virtual void visit(const class max* x) override { e = mutate_binary(x); }
  virtual void visit(const equal* x) override { e = mutate_binary(x); }
  virtual void visit(const not_equal* x) override { e = mutate_binary(x); }
  virtual void visit(const less* x) override { e = mutate_binary(x); }
  virtual void visit(const less_equal* x) override { e = mutate_binary(x); }
  virtual void visit(const bitwise_and* x) override { e = mutate_binary(x); }
  virtual void visit(const bitwise_or* x) override { e = mutate_binary(x); }
  virtual void visit(const bitwise_xor* x) override { e = mutate_binary(x); }
  virtual void visit(const logical_and* x) override { e = mutate_binary(x); }
  virtual void visit(const logical_or* x) override { e = mutate_binary(x); }
  virtual void visit(const shift_left* x) override { e = mutate_binary(x); }
  virtual void visit(const shift_right* x) override { e = mutate_binary(x); }

  virtual void visit(const class select* x) override {
    expr c = mutate(x->condition);
    expr t = mutate(x->true_value);
    expr f = mutate(x->false_value);
    if (c.same_as(x->condition) && t.same_as(x->true_value) && f.same_as(x->false_value)) {
      e = x;
    } else {
      e = select::make(std::move(c), std::move(t), std::move(f));
    }
  }

  virtual void visit(const load_buffer_meta* x) override {
    expr buffer = mutate(x->buffer);
    expr dim = mutate(x->dim);
    if (buffer.same_as(x->buffer) && dim.same_as(x->dim)) {
      e = x;
    } else {
      e = load_buffer_meta::make(std::move(buffer), x->meta, std::move(dim));
    }
  }

  virtual void visit(const call* x) override { 
    std::vector<expr> args;
    args.reserve(x->args.size());
    bool changed = false;
    for (const expr& i : x->args) {
      args.emplace_back(mutate(i));
      changed = changed || !args.back().same_as(i);
    }
    if (!changed) {
      e = x;
    } else {
      e = call::make(x->intrinsic, std::move(args));
    }
  }

  virtual void visit(const let_stmt* x) override { s = mutate_let(x); }
  virtual void visit(const block* x) override {
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
  virtual void visit(const loop* x) override {
    expr begin = mutate(x->begin);
    expr end = mutate(x->end);
    stmt body = mutate(x->body);
    if (begin.same_as(x->begin) && end.same_as(x->end) && body.same_as(x->body)) {
      s = x;
    } else {
      s = loop::make(x->name, std::move(begin), std::move(end), std::move(body));
    }
  }
  virtual void visit(const if_then_else* x) override {
    expr cond = mutate(x->condition);
    stmt true_body = mutate(x->true_body);
    stmt false_body = mutate(x->false_body);
    if (cond.same_as(x->condition) && true_body.same_as(x->true_body) && false_body.same_as(x->false_body)) {
      s = x;
    } else {
      s = if_then_else::make(std::move(cond), std::move(true_body), std::move(false_body));
    }
  }
  virtual void visit(const call_func* x) override {
    std::vector<expr> scalar_args;
    scalar_args.reserve(x->scalar_args.size());
    bool changed = false;
    for (const expr& i : x->scalar_args) {
      scalar_args.push_back(mutate(i));
      changed = changed || !scalar_args.back().same_as(i);
    }
    if (!changed) {
      s = x;
    } else {
      s = call_func::make(x->target, std::move(scalar_args), x->buffer_args, x->fn);
    }
  }
  virtual void visit(const allocate* x) override {
    std::vector<dim_expr> dims;
    dims.reserve(x->dims.size());
    bool changed = false;
    for (const dim_expr& i : x->dims) {
      interval_expr bounds = {mutate(i.bounds.min), mutate(i.bounds.max)};
      dims.emplace_back(std::move(bounds), mutate(i.stride_bytes), mutate(i.fold_factor));
      changed = changed || !dims.back().same_as(i);
    }
    stmt body = mutate(x->body);
    if (!changed && body.same_as(x->body)) {
      s = x;
    } else {
      s = allocate::make(x->type, x->name, x->elem_size, std::move(dims), std::move(body));
    }
  }
  virtual void visit(const make_buffer* x) override {
    expr base = mutate(x->base);
    std::vector<dim_expr> dims;
    dims.reserve(x->dims.size());
    bool changed = false;
    for (const dim_expr& i : x->dims) {
      interval_expr bounds = {mutate(i.bounds.min), mutate(i.bounds.max)};
      dims.emplace_back(std::move(bounds), mutate(i.stride_bytes), mutate(i.fold_factor));
      changed = changed || dims.back().same_as(i);
    }
    stmt body = mutate(x->body);
    if (!changed && base.same_as(x->base) && body.same_as(x->body)) {
      s = x;
    } else {
      s = make_buffer::make(x->name, std::move(base), x->elem_size, std::move(dims), std::move(body));
    }
  }
  virtual void visit(const crop_buffer* x) override {
    std::vector<interval_expr> bounds;
    bounds.reserve(x->bounds.size());
    bool changed = false;
    for (const interval_expr& i : x->bounds) {
      bounds.emplace_back(mutate(i.min), mutate(i.max));
      changed = changed || bounds.back().same_as(i);
    }
    stmt body = mutate(x->body);
    if (!changed && body.same_as(x->body)) {
      s = x;
    } else {
      s = crop_buffer::make(x->name, std::move(bounds), std::move(body));
    }
  }
  virtual void visit(const crop_dim* x) override {
    expr min = mutate(x->min);
    expr extent = mutate(x->extent);
    stmt body = mutate(x->body);
    if (min.same_as(x->min) && extent.same_as(x->extent) && body.same_as(x->body)) {
      s = x;
    } else {
      s = crop_dim::make(x->name, x->dim, std::move(min), std::move(extent), std::move(body));
    }
  }
  virtual void visit(const check* x) override {
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