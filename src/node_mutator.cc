#include "node_mutator.h"

namespace slinky {

namespace {

template <typename T>
auto mutate_let(node_mutator* this_, const T* x) {
  expr value = this_->mutate(x->value);
  auto body = this_->mutate(x->body);
  if (value.same_as(x->value) && body.same_as(x->body)) {
    return decltype(body){x};
  } else {
    return T::make(x->sym, std::move(value), std::move(body));
  }
}

template <typename T>
expr mutate_binary(node_mutator* this_, const T* x) {
  expr a = this_->mutate(x->a);
  expr b = this_->mutate(x->b);
  if (a.same_as(x->a) && b.same_as(x->b)) {
    return x;
  } else {
    return T::make(std::move(a), std::move(b));
  }
}

}  // namespace

void node_mutator::visit(const let* x) { set_result(mutate_let(this, x)); }
void node_mutator::visit(const let_stmt* x) { set_result(mutate_let(this, x)); }
void node_mutator::visit(const add* x) { set_result(mutate_binary(this, x)); }
void node_mutator::visit(const sub* x) { set_result(mutate_binary(this, x)); }
void node_mutator::visit(const mul* x) { set_result(mutate_binary(this, x)); }
void node_mutator::visit(const div* x) { set_result(mutate_binary(this, x)); }
void node_mutator::visit(const mod* x) { set_result(mutate_binary(this, x)); }
void node_mutator::visit(const class min* x) { set_result(mutate_binary(this, x)); }
void node_mutator::visit(const class max* x) { set_result(mutate_binary(this, x)); }
void node_mutator::visit(const equal* x) { set_result(mutate_binary(this, x)); }
void node_mutator::visit(const not_equal* x) { set_result(mutate_binary(this, x)); }
void node_mutator::visit(const less* x) { set_result(mutate_binary(this, x)); }
void node_mutator::visit(const less_equal* x) { set_result(mutate_binary(this, x)); }
void node_mutator::visit(const logical_and* x) { set_result(mutate_binary(this, x)); }
void node_mutator::visit(const logical_or* x) { set_result(mutate_binary(this, x)); }
void node_mutator::visit(const logical_not* x) {
  expr new_x = mutate(x->x);
  if (new_x.same_as(x->x)) {
    set_result(x);
  } else {
    set_result(logical_not::make(std::move(new_x)));
  }
}

void node_mutator::visit(const class select* x) {
  expr c = mutate(x->condition);
  expr t = mutate(x->true_value);
  expr f = mutate(x->false_value);
  if (c.same_as(x->condition) && t.same_as(x->true_value) && f.same_as(x->false_value)) {
    set_result(x);
  } else {
    set_result(select::make(std::move(c), std::move(t), std::move(f)));
  }
}

void node_mutator::visit(const call* x) {
  std::vector<expr> args;
  args.reserve(x->args.size());
  bool changed = false;
  for (const expr& i : x->args) {
    args.emplace_back(mutate(i));
    changed = changed || !args.back().same_as(i);
  }
  if (!changed) {
    set_result(x);
  } else {
    set_result(call::make(x->intrinsic, std::move(args)));
  }
}

void node_mutator::visit(const block* x) {
  stmt a = mutate(x->a);
  stmt b = mutate(x->b);
  if (a.defined() && b.defined()) {
    if (a.same_as(x->a) && b.same_as(x->b)) {
      set_result(x);
    } else {
      set_result(block::make(std::move(a), std::move(b)));
    }
  } else if (a.defined()) {
    set_result(a);
  } else {
    set_result(b);
  }
}
void node_mutator::visit(const loop* x) {
  interval_expr bounds = {mutate(x->bounds.min), mutate(x->bounds.max)};
  expr step = mutate(x->step);
  stmt body = mutate(x->body);
  if (bounds.same_as(x->bounds) && step.same_as(x->step) && body.same_as(x->body)) {
    set_result(x);
  } else {
    set_result(loop::make(x->sym, std::move(bounds), std::move(step), std::move(body)));
  }
}
void node_mutator::visit(const if_then_else* x) {
  expr cond = mutate(x->condition);
  stmt true_body = mutate(x->true_body);
  stmt false_body = mutate(x->false_body);
  if (cond.same_as(x->condition) && true_body.same_as(x->true_body) && false_body.same_as(x->false_body)) {
    set_result(x);
  } else {
    set_result(if_then_else::make(std::move(cond), std::move(true_body), std::move(false_body)));
  }
}
void node_mutator::visit(const call_stmt* x) { set_result(x); }
void node_mutator::visit(const copy_stmt* x) {
  std::vector<expr> src_x;
  src_x.reserve(x->src_x.size());
  bool changed = false;
  for (const expr& i : x->src_x) {
    src_x.push_back(mutate(i));
    changed = changed || !src_x.back().same_as(i);
  }
  if (!changed) {
    set_result(x);
  } else {
    set_result(copy_stmt::make(x->src, std::move(src_x), x->dst, x->dst_x, x->padding));
  }
}
void node_mutator::visit(const allocate* x) {
  std::vector<dim_expr> dims;
  dims.reserve(x->dims.size());
  bool changed = false;
  for (const dim_expr& i : x->dims) {
    interval_expr bounds = {mutate(i.bounds.min), mutate(i.bounds.max)};
    dims.emplace_back(std::move(bounds), mutate(i.stride), mutate(i.fold_factor));
    changed = changed || !dims.back().same_as(i);
  }
  stmt body = mutate(x->body);
  if (!changed && body.same_as(x->body)) {
    set_result(x);
  } else {
    set_result(allocate::make(x->storage, x->sym, x->elem_size, std::move(dims), std::move(body)));
  }
}
void node_mutator::visit(const make_buffer* x) {
  expr base = mutate(x->base);
  expr elem_size = mutate(x->elem_size);
  std::vector<dim_expr> dims;
  dims.reserve(x->dims.size());
  bool changed = false;
  for (const dim_expr& i : x->dims) {
    interval_expr bounds = {mutate(i.bounds.min), mutate(i.bounds.max)};
    dims.emplace_back(std::move(bounds), mutate(i.stride), mutate(i.fold_factor));
    changed = changed || dims.back().same_as(i);
  }
  stmt body = mutate(x->body);
  if (!changed && base.same_as(x->base) && elem_size.same_as(x->elem_size) && body.same_as(x->body)) {
    set_result(x);
  } else {
    set_result(make_buffer::make(x->sym, std::move(base), std::move(elem_size), std::move(dims), std::move(body)));
  }
}
void node_mutator::visit(const crop_buffer* x) {
  std::vector<interval_expr> bounds;
  bounds.reserve(x->bounds.size());
  bool changed = false;
  for (const interval_expr& i : x->bounds) {
    bounds.emplace_back(mutate(i.min), mutate(i.max));
    changed = changed || bounds.back().same_as(i);
  }
  stmt body = mutate(x->body);
  if (!changed && body.same_as(x->body)) {
    set_result(x);
  } else {
    set_result(crop_buffer::make(x->sym, std::move(bounds), std::move(body)));
  }
}
void node_mutator::visit(const crop_dim* x) {
  interval_expr bounds = {mutate(x->bounds.min), mutate(x->bounds.max)};
  stmt body = mutate(x->body);
  if (bounds.same_as(x->bounds) && body.same_as(x->body)) {
    set_result(x);
  } else {
    set_result(crop_dim::make(x->sym, x->dim, std::move(bounds), std::move(body)));
  }
}
void node_mutator::visit(const slice_buffer* x) {
  std::vector<expr> at;
  at.reserve(x->at.size());
  bool changed = false;
  for (const expr& i : x->at) {
    at.emplace_back(mutate(i));
    changed = changed || at.back().same_as(i);
  }
  stmt body = mutate(x->body);
  if (!changed && body.same_as(x->body)) {
    set_result(x);
  } else {
    set_result(slice_buffer::make(x->sym, std::move(at), std::move(body)));
  }
}
void node_mutator::visit(const slice_dim* x) {
  expr at = mutate(x->at);
  stmt body = mutate(x->body);
  if (at.same_as(x->at) && body.same_as(x->body)) {
    set_result(x);
  } else {
    set_result(slice_dim::make(x->sym, x->dim, std::move(at), std::move(body)));
  }
}
void node_mutator::visit(const truncate_rank* x) {
  stmt body = mutate(x->body);
  if (body.same_as(x->body)) {
    set_result(x);
  } else {
    set_result(truncate_rank::make(x->sym, x->rank, std::move(body)));
  }
}

void node_mutator::visit(const check* x) {
  expr condition = mutate(x->condition);
  if (condition.same_as(x->condition)) {
    set_result(x);
  } else {
    set_result(check::make(std::move(condition)));
  }
}

}  // namespace slinky
