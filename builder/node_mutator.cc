#include "builder/node_mutator.h"

#include <utility>
#include <vector>

#include "runtime/expr.h"

namespace slinky {

namespace {

template <typename T>
auto mutate_let(node_mutator* this_, const T* op) {
  std::vector<std::pair<symbol_id, expr>> lets;
  lets.reserve(op->lets.size());
  bool changed = false;
  for (const auto& s : op->lets) {
    lets.emplace_back(s.first, this_->mutate(s.second));
    changed = changed || !lets.back().second.same_as(s.second);
  }
  auto body = this_->mutate(op->body);
  changed = changed || !body.same_as(op->body);
  if (!changed) {
    return decltype(body){op};
  } else {
    return T::make(std::move(lets), std::move(body));
  }
}

template <typename T>
expr mutate_binary(node_mutator* this_, const T* op) {
  expr a = this_->mutate(op->a);
  expr b = this_->mutate(op->b);
  if (a.same_as(op->a) && b.same_as(op->b)) {
    return op;
  } else {
    return T::make(std::move(a), std::move(b));
  }
}

}  // namespace

stmt clone_with_new_body(const loop* op, stmt new_body) {
  return loop::make(op->sym, op->mode, op->bounds, op->step, std::move(new_body));
}
stmt clone_with_new_body(const let_stmt* op, stmt new_body) { return let_stmt::make(op->lets, std::move(new_body)); }
stmt clone_with_new_body(const allocate* op, stmt new_body) {
  return allocate::make(op->sym, op->storage, op->elem_size, op->dims, std::move(new_body));
}
stmt clone_with_new_body(const make_buffer* op, stmt new_body) {
  return make_buffer::make(op->sym, op->base, op->elem_size, op->dims, std::move(new_body));
}
stmt clone_with_new_body(const clone_buffer* op, stmt new_body) {
  return clone_buffer::make(op->sym, op->src, std::move(new_body));
}
stmt clone_with_new_body(const crop_buffer* op, stmt new_body) {
  return crop_buffer::make(op->sym, op->bounds, std::move(new_body));
}
stmt clone_with_new_body(const crop_dim* op, stmt new_body) {
  return crop_dim::make(op->sym, op->dim, op->bounds, std::move(new_body));
}
stmt clone_with_new_body(const slice_buffer* op, stmt new_body) {
  return slice_buffer::make(op->sym, op->at, std::move(new_body));
}
stmt clone_with_new_body(const slice_dim* op, stmt new_body) {
  return slice_dim::make(op->sym, op->dim, op->at, std::move(new_body));
}
stmt clone_with_new_body(const truncate_rank* op, stmt new_body) {
  return truncate_rank::make(op->sym, op->rank, std::move(new_body));
}

void node_mutator::visit(const let* op) { set_result(mutate_let(this, op)); }
void node_mutator::visit(const let_stmt* op) { set_result(mutate_let(this, op)); }
void node_mutator::visit(const add* op) { set_result(mutate_binary(this, op)); }
void node_mutator::visit(const sub* op) { set_result(mutate_binary(this, op)); }
void node_mutator::visit(const mul* op) { set_result(mutate_binary(this, op)); }
void node_mutator::visit(const div* op) { set_result(mutate_binary(this, op)); }
void node_mutator::visit(const mod* op) { set_result(mutate_binary(this, op)); }
void node_mutator::visit(const class min* op) { set_result(mutate_binary(this, op)); }
void node_mutator::visit(const class max* op) { set_result(mutate_binary(this, op)); }
void node_mutator::visit(const equal* op) { set_result(mutate_binary(this, op)); }
void node_mutator::visit(const not_equal* op) { set_result(mutate_binary(this, op)); }
void node_mutator::visit(const less* op) { set_result(mutate_binary(this, op)); }
void node_mutator::visit(const less_equal* op) { set_result(mutate_binary(this, op)); }
void node_mutator::visit(const logical_and* op) { set_result(mutate_binary(this, op)); }
void node_mutator::visit(const logical_or* op) { set_result(mutate_binary(this, op)); }
void node_mutator::visit(const logical_not* op) {
  expr a = mutate(op->a);
  if (a.same_as(op->a)) {
    set_result(op);
  } else {
    set_result(logical_not::make(std::move(a)));
  }
}

void node_mutator::visit(const class select* op) {
  expr c = mutate(op->condition);
  expr t = mutate(op->true_value);
  expr f = mutate(op->false_value);
  if (c.same_as(op->condition) && t.same_as(op->true_value) && f.same_as(op->false_value)) {
    set_result(op);
  } else {
    set_result(select::make(std::move(c), std::move(t), std::move(f)));
  }
}

void node_mutator::visit(const call* op) {
  std::vector<expr> args;
  args.reserve(op->args.size());
  bool changed = false;
  for (const expr& i : op->args) {
    args.push_back(mutate(i));
    changed = changed || !args.back().same_as(i);
  }
  if (!changed) {
    set_result(op);
  } else {
    set_result(call::make(op->intrinsic, std::move(args)));
  }
}

void node_mutator::visit(const block* op) {
  std::vector<stmt> stmts;
  stmts.reserve(op->stmts.size());
  bool changed = false;
  for (const stmt& s : op->stmts) {
    stmts.push_back(mutate(s));
    changed = changed || !stmts.back().same_as(s);
  }
  if (!changed) {
    set_result(op);
  } else {
    set_result(block::make(std::move(stmts)));
  }
}
void node_mutator::visit(const loop* op) {
  interval_expr bounds = {mutate(op->bounds.min), mutate(op->bounds.max)};
  expr step = mutate(op->step);
  stmt body = mutate(op->body);
  if (bounds.same_as(op->bounds) && step.same_as(op->step) && body.same_as(op->body)) {
    set_result(op);
  } else {
    set_result(loop::make(op->sym, op->mode, std::move(bounds), std::move(step), std::move(body)));
  }
}
void node_mutator::visit(const call_stmt* op) { set_result(op); }
void node_mutator::visit(const copy_stmt* op) {
  std::vector<expr> src_x;
  src_x.reserve(op->src_x.size());
  bool changed = false;
  for (const expr& i : op->src_x) {
    src_x.push_back(mutate(i));
    changed = changed || !src_x.back().same_as(i);
  }
  if (!changed) {
    set_result(op);
  } else {
    set_result(copy_stmt::make(op->src, std::move(src_x), op->dst, op->dst_x, op->padding));
  }
}
void node_mutator::visit(const allocate* op) {
  std::vector<dim_expr> dims;
  dims.reserve(op->dims.size());
  bool changed = false;
  for (const dim_expr& i : op->dims) {
    interval_expr bounds = {mutate(i.bounds.min), mutate(i.bounds.max)};
    dims.push_back({std::move(bounds), mutate(i.stride), mutate(i.fold_factor)});
    changed = changed || !dims.back().same_as(i);
  }
  stmt body = mutate(op->body);
  if (!changed && body.same_as(op->body)) {
    set_result(op);
  } else {
    set_result(allocate::make(op->sym, op->storage, op->elem_size, std::move(dims), std::move(body)));
  }
}
void node_mutator::visit(const make_buffer* op) {
  expr base = mutate(op->base);
  expr elem_size = mutate(op->elem_size);
  std::vector<dim_expr> dims;
  dims.reserve(op->dims.size());
  bool changed = false;
  for (const dim_expr& i : op->dims) {
    interval_expr bounds = {mutate(i.bounds.min), mutate(i.bounds.max)};
    dims.push_back({std::move(bounds), mutate(i.stride), mutate(i.fold_factor)});
    changed = changed || !dims.back().same_as(i);
  }
  stmt body = mutate(op->body);
  if (!changed && base.same_as(op->base) && elem_size.same_as(op->elem_size) && body.same_as(op->body)) {
    set_result(op);
  } else {
    set_result(make_buffer::make(op->sym, std::move(base), std::move(elem_size), std::move(dims), std::move(body)));
  }
}
void node_mutator::visit(const clone_buffer* op) {
  stmt body = mutate(op->body);
  if (body.same_as(op->body)) {
    set_result(op);
  } else {
    set_result(clone_buffer::make(op->sym, op->src, std::move(body)));
  }
}
void node_mutator::visit(const crop_buffer* op) {
  std::vector<interval_expr> bounds;
  bounds.reserve(op->bounds.size());
  bool changed = false;
  for (const interval_expr& i : op->bounds) {
    bounds.emplace_back(mutate(i.min), mutate(i.max));
    changed = changed || !bounds.back().same_as(i);
  }
  stmt body = mutate(op->body);
  if (!changed && body.same_as(op->body)) {
    set_result(op);
  } else {
    set_result(crop_buffer::make(op->sym, std::move(bounds), std::move(body)));
  }
}
void node_mutator::visit(const crop_dim* op) {
  interval_expr bounds = {mutate(op->bounds.min), mutate(op->bounds.max)};
  stmt body = mutate(op->body);
  if (bounds.same_as(op->bounds) && body.same_as(op->body)) {
    set_result(op);
  } else {
    set_result(crop_dim::make(op->sym, op->dim, std::move(bounds), std::move(body)));
  }
}
void node_mutator::visit(const slice_buffer* op) {
  std::vector<expr> at;
  at.reserve(op->at.size());
  bool changed = false;
  for (const expr& i : op->at) {
    at.push_back(mutate(i));
    changed = changed || !at.back().same_as(i);
  }
  stmt body = mutate(op->body);
  if (!changed && body.same_as(op->body)) {
    set_result(op);
  } else {
    set_result(slice_buffer::make(op->sym, std::move(at), std::move(body)));
  }
}
void node_mutator::visit(const slice_dim* op) {
  expr at = mutate(op->at);
  stmt body = mutate(op->body);
  if (at.same_as(op->at) && body.same_as(op->body)) {
    set_result(op);
  } else {
    set_result(slice_dim::make(op->sym, op->dim, std::move(at), std::move(body)));
  }
}
void node_mutator::visit(const truncate_rank* op) {
  stmt body = mutate(op->body);
  if (body.same_as(op->body)) {
    set_result(op);
  } else {
    set_result(truncate_rank::make(op->sym, op->rank, std::move(body)));
  }
}

void node_mutator::visit(const check* op) {
  expr condition = mutate(op->condition);
  if (condition.same_as(op->condition)) {
    set_result(op);
  } else {
    set_result(check::make(std::move(condition)));
  }
}

}  // namespace slinky
