#include "builder/node_mutator.h"

#include <utility>
#include <vector>

#include "runtime/expr.h"

namespace slinky {

namespace {

template <typename T, typename... Args>
auto mutate_let(node_mutator* this_, const T* op, Args... args) {
  std::vector<std::pair<var, expr>> lets;
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
    return T::make(std::move(lets), std::move(body), args...);
  }
}

template <typename T>
expr mutate_binary(node_mutator* this_, const T* op) {
  expr a = this_->mutate(op->a);
  expr b = this_->mutate(op->b);
  if (a.same_as(op->a) && b.same_as(op->b)) {
    return expr(op);
  } else {
    return T::make(std::move(a), std::move(b));
  }
}

}  // namespace

stmt clone_with(const loop* op, var sym, stmt new_body) {
  return loop::make(sym, op->max_workers, op->bounds, op->step, std::move(new_body));
}
stmt clone_with(const allocate* op, var sym, stmt new_body) {
  return allocate::make(sym, op->storage, op->elem_size, op->dims, std::move(new_body));
}
stmt clone_with(const make_buffer* op, var sym, stmt new_body) {
  return make_buffer::make(sym, op->base, op->elem_size, op->dims, std::move(new_body));
}
stmt clone_with(const constant_buffer* op, var sym, stmt new_body) {
  return constant_buffer::make(sym, op->value, std::move(new_body));
}
stmt clone_with(const clone_buffer* op, var sym, stmt new_body) {
  return clone_buffer::make(sym, op->src, std::move(new_body));
}
stmt clone_with(const crop_buffer* op, var sym, stmt new_body) {
  return crop_buffer::make(sym, op->src, op->bounds, std::move(new_body));
}
stmt clone_with(const crop_dim* op, var sym, stmt new_body) {
  return crop_dim::make(sym, op->src, op->dim, op->bounds, std::move(new_body));
}
stmt clone_with(const slice_buffer* op, var sym, stmt new_body) {
  return slice_buffer::make(sym, op->src, op->at, std::move(new_body));
}
stmt clone_with(const slice_dim* op, var sym, stmt new_body) {
  return slice_dim::make(sym, op->src, op->dim, op->at, std::move(new_body));
}
stmt clone_with(const transpose* op, var sym, stmt new_body) {
  return transpose::make(sym, op->src, op->dims, std::move(new_body));
}
stmt clone_with(const async* op, var sym, stmt new_body) { return async::make(sym, op->task, std::move(new_body)); }

stmt clone_with(const let_stmt* op, stmt new_body) {
  return let_stmt::make(op->lets, std::move(new_body), op->is_closure);
}

stmt clone_with(const loop* op, stmt new_body) { return clone_with(op, op->sym, std::move(new_body)); }
stmt clone_with(const allocate* op, stmt new_body) { return clone_with(op, op->sym, std::move(new_body)); }
stmt clone_with(const make_buffer* op, stmt new_body) { return clone_with(op, op->sym, std::move(new_body)); }
stmt clone_with(const constant_buffer* op, stmt new_body) { return clone_with(op, op->sym, std::move(new_body)); }
stmt clone_with(const clone_buffer* op, stmt new_body) { return clone_with(op, op->sym, std::move(new_body)); }
stmt clone_with(const crop_buffer* op, stmt new_body) { return clone_with(op, op->sym, std::move(new_body)); }
stmt clone_with(const crop_dim* op, stmt new_body) { return clone_with(op, op->sym, std::move(new_body)); }
stmt clone_with(const slice_buffer* op, stmt new_body) { return clone_with(op, op->sym, std::move(new_body)); }
stmt clone_with(const slice_dim* op, stmt new_body) { return clone_with(op, op->sym, std::move(new_body)); }
stmt clone_with(const transpose* op, stmt new_body) { return clone_with(op, op->sym, std::move(new_body)); }
stmt clone_with(const async* op, stmt new_body) { return clone_with(op, op->sym, std::move(new_body)); }

void stmt_mutator::visit(const block* op) {
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

void stmt_mutator::visit(const call_stmt* op) { set_result(op); }
void stmt_mutator::visit(const copy_stmt* op) { set_result(op); }
void stmt_mutator::visit(const check* op) { set_result(op); }

namespace {

template <typename T>
stmt mutate_decl(stmt_mutator* this_, const T* op) {
  stmt body = this_->mutate(op->body);
  if (body.same_as(op->body)) {
    return stmt(op);
  } else {
    return clone_with(op, std::move(body));
  }
}

}  // namespace

void stmt_mutator::visit(const let_stmt* op) { set_result(mutate_decl(this, op)); }
void stmt_mutator::visit(const loop* op) { set_result(mutate_decl(this, op)); }
void stmt_mutator::visit(const allocate* op) { set_result(mutate_decl(this, op)); }
void stmt_mutator::visit(const make_buffer* op) { set_result(mutate_decl(this, op)); }
void stmt_mutator::visit(const constant_buffer* op) { set_result(mutate_decl(this, op)); }
void stmt_mutator::visit(const clone_buffer* op) { set_result(mutate_decl(this, op)); }
void stmt_mutator::visit(const crop_buffer* op) { set_result(mutate_decl(this, op)); }
void stmt_mutator::visit(const crop_dim* op) { set_result(mutate_decl(this, op)); }
void stmt_mutator::visit(const slice_buffer* op) { set_result(mutate_decl(this, op)); }
void stmt_mutator::visit(const slice_dim* op) { set_result(mutate_decl(this, op)); }
void stmt_mutator::visit(const transpose* op) { set_result(mutate_decl(this, op)); }
void stmt_mutator::visit(const async* op) {
  stmt task = mutate(op->task);
  stmt body = mutate(op->body);
  if (body.same_as(op->body) && task.same_as(op->task)) {
    set_result(stmt(op));
  } else {
    set_result(async::make(op->sym, std::move(task), std::move(body)));
  }
}

void node_mutator::visit(const variable* op) { set_result(op); }
void node_mutator::visit(const constant* op) { set_result(op); }
void node_mutator::visit(const let* op) { set_result(mutate_let(this, op)); }
void node_mutator::visit(const let_stmt* op) { set_result(mutate_let(this, op, op->is_closure)); }
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
    set_result(call::make(op->intrinsic, op->target, std::move(args)));
  }
}

void node_mutator::visit(const loop* op) {
  interval_expr bounds = mutate(op->bounds);
  expr step = mutate(op->step);
  expr max_workers = mutate(op->max_workers);
  stmt body = mutate(op->body);
  if (bounds.same_as(op->bounds) && step.same_as(op->step) && max_workers.same_as(op->max_workers) &&
      body.same_as(op->body)) {
    set_result(op);
  } else {
    set_result(loop::make(op->sym, std::move(max_workers), std::move(bounds), std::move(step), std::move(body)));
  }
}
void node_mutator::visit(const call_stmt* op) {
  std::vector<expr> scalars;
  scalars.reserve(op->scalars.size());
  bool changed = false;
  for (const expr& i : op->scalars) {
    scalars.push_back(mutate(i));
    changed = changed || !scalars.back().same_as(i);
  }
  if (!changed) {
    set_result(op);
  } else {
    set_result(call_stmt::make(op->target, op->inputs, op->outputs, std::move(scalars), op->attrs));
  }
}
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
    set_result(copy_stmt::make(op->impl, op->src, std::move(src_x), op->dst, op->dst_x, op->pad));
  }
}
void node_mutator::visit(const allocate* op) {
  expr elem_size = mutate(op->elem_size);
  std::vector<dim_expr> dims;
  dims.reserve(op->dims.size());
  bool changed = false;
  for (const dim_expr& i : op->dims) {
    dims.push_back({mutate(i.bounds), mutate(i.stride), mutate(i.fold_factor)});
    changed = changed || !dims.back().same_as(i);
  }
  stmt body = mutate(op->body);
  if (!changed && elem_size.same_as(op->elem_size) && body.same_as(op->body)) {
    set_result(op);
  } else {
    set_result(allocate::make(op->sym, op->storage, std::move(elem_size), std::move(dims), std::move(body)));
  }
}
void node_mutator::visit(const make_buffer* op) {
  expr base = mutate(op->base);
  expr elem_size = mutate(op->elem_size);
  std::vector<dim_expr> dims;
  dims.reserve(op->dims.size());
  bool changed = false;
  for (const dim_expr& i : op->dims) {
    dims.push_back({mutate(i.bounds), mutate(i.stride), mutate(i.fold_factor)});
    changed = changed || !dims.back().same_as(i);
  }
  stmt body = mutate(op->body);
  if (!changed && base.same_as(op->base) && elem_size.same_as(op->elem_size) && body.same_as(op->body)) {
    set_result(op);
  } else {
    set_result(make_buffer::make(op->sym, std::move(base), std::move(elem_size), std::move(dims), std::move(body)));
  }
}
void node_mutator::visit(const crop_buffer* op) {
  std::vector<interval_expr> bounds;
  bounds.reserve(op->bounds.size());
  bool changed = false;
  for (const interval_expr& i : op->bounds) {
    bounds.emplace_back(mutate(i));
    changed = changed || !bounds.back().same_as(i);
  }
  stmt body = mutate(op->body);
  if (!changed && body.same_as(op->body)) {
    set_result(op);
  } else {
    set_result(crop_buffer::make(op->sym, op->src, std::move(bounds), std::move(body)));
  }
}
void node_mutator::visit(const crop_dim* op) {
  interval_expr bounds = mutate(op->bounds);
  stmt body = mutate(op->body);
  if (bounds.same_as(op->bounds) && body.same_as(op->body)) {
    set_result(op);
  } else {
    set_result(crop_dim::make(op->sym, op->src, op->dim, std::move(bounds), std::move(body)));
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
    set_result(slice_buffer::make(op->sym, op->src, std::move(at), std::move(body)));
  }
}
void node_mutator::visit(const slice_dim* op) {
  expr at = mutate(op->at);
  stmt body = mutate(op->body);
  if (at.same_as(op->at) && body.same_as(op->body)) {
    set_result(op);
  } else {
    set_result(slice_dim::make(op->sym, op->src, op->dim, std::move(at), std::move(body)));
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
