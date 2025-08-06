#include "builder/substitute.h"

#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "base/arithmetic.h"
#include "base/chrome_trace.h"
#include "base/span.h"
#include "builder/node_mutator.h"
#include "runtime/buffer.h"
#include "runtime/depends_on.h"
#include "runtime/expr.h"

namespace slinky {

namespace {

template <typename T, typename... U>
std::uintptr_t get_target(const std::function<T(U...)>& f) {
  typedef T (*fn)(U...);
  return reinterpret_cast<std::uintptr_t>(f.template target<fn>());
}

class matcher : public expr_visitor, public stmt_visitor {
  // In this class, we visit the pattern, and manually traverse the expression being matched.
  union {
    void* self = nullptr;
    const base_expr_node* self_expr;
    const base_stmt_node* self_stmt;
  };

public:
  int match = 0;

  template <typename T>
  bool try_match(T self, T op) {
    assert(match == 0);
    if (self < op) {
      match = -1;
    } else if (op < self) {
      match = 1;
    }
    return match == 0;
  }

  bool try_match(const var& self, const var& op) { return try_match(self.id, op.id); }

  bool try_match(const raw_buffer* self, const raw_buffer* op) {
    assert(match == 0);
    if (self == op) {
    } else if (!self) {
      match = -1;
    } else if (!op) {
      match = 1;
    } else {
      if (!try_match(self->rank, op->rank)) return false;
      if (!try_match(self->elem_size, op->elem_size)) return false;
      if (!try_match(span<const dim>{self->dims, self->rank}, span<const dim>{op->dims, op->rank})) return false;

      assert(self->size_bytes() == op->size_bytes());
      if (self->size_bytes() > 128) {
        return try_match<const raw_buffer*>(self, op);
      }
      const index_t elem_size = self->elem_size;
      for_each_contiguous_slice(
          *self,
          [&](index_t slice_extent, const void* a, const void* b) {
            if (match != 0) return;
            if (a && b) {
              match = sign(std::memcmp(a, b, slice_extent * elem_size));
            } else if (!a && b) {
              match = -1;
            } else if (a && !b) {
              match = 1;
            }
          },
          *op);
    }
    return match == 0;
  }

  // Skip the visitor pattern (two virtual function calls) for a few node types that are very frequently visited.
  void visit(const base_expr_node* op) {
    switch (op->type) {
    case expr_node_type::variable: visit(reinterpret_cast<const variable*>(op)); return;
    case expr_node_type::constant: visit(reinterpret_cast<const constant*>(op)); return;
    case expr_node_type::min: visit(reinterpret_cast<const class min*>(op)); return;
    case expr_node_type::max: visit(reinterpret_cast<const class max*>(op)); return;
    default: op->accept(this);
    }
  }

  bool try_match(const base_expr_node* e, const base_expr_node* op) {
    assert(match == 0);
    if (e == op) {
    } else if (!e) {
      match = -1;
    } else if (!op) {
      match = 1;
    } else if (e->type < op->type) {
      match = -1;
    } else if (e->type > op->type) {
      match = 1;
    } else {
      self_expr = e;
      visit(op);
    }
    return match == 0;
  }
  bool try_match(const expr& e, const expr& op) { return try_match(e.get(), op.get()); }

  bool try_match(const base_stmt_node* s, const base_stmt_node* op) {
    assert(match == 0);
    if (s == op) {
    } else if (!s) {
      match = -1;
    } else if (!op) {
      match = 1;
    } else if (s->type < op->type) {
      match = -1;
    } else if (s->type > op->type) {
      match = 1;
    } else {
      self_stmt = s;
      op->accept(this);
    }
    return match == 0;
  }
  bool try_match(const stmt& s, const stmt& op) { return try_match(s.get(), op.get()); }

  bool try_match(const interval_expr& self, const interval_expr& op) {
    if (!try_match(self.min, op.min)) return false;
    if (!self.min.same_as(self.max) || !op.min.same_as(op.max)) {
      if (!try_match(self.max, op.max)) return false;
    }
    return true;
  }

  bool try_match(const dim_expr& self, const dim_expr& op) {
    if (!try_match(self.bounds, op.bounds)) return false;
    if (!try_match(self.stride, op.stride)) return false;
    if (!try_match(self.fold_factor, op.fold_factor)) return false;
    return true;
  }

  bool try_match(const dim& self, const dim& op) {
    if (!try_match(self.min(), op.min())) return false;
    if (!try_match(self.max(), op.max())) return false;
    if (!try_match(self.stride(), op.stride())) return false;
    if (!try_match(self.fold_factor(), op.fold_factor())) return false;
    return true;
  }

  template <typename A, typename B>
  bool try_match(const std::pair<A, B>& self, const std::pair<A, B>& op) {
    if (!try_match(self.first, op.first)) return false;
    if (!try_match(self.second, op.second)) return false;
    return true;
  }

  template <typename Container>
  bool try_match_container(const Container& self, const Container& op) {
    if (!try_match(self.size(), op.size())) return false;
    for (std::size_t i = 0; i < self.size(); ++i) {
      if (!try_match(self[i], op[i])) return false;
    }

    return true;
  }

  template <typename T>
  bool try_match(const span<T>& self, const span<T>& op) {
    return try_match_container(self, op);
  }

  template <typename T>
  bool try_match(const std::vector<T>& self, const std::vector<T>& op) {
    return try_match_container(self, op);
  }

  template <typename T>
  void match_binary(const T* op) {
    const T* ex = static_cast<const T*>(self);

    if (!try_match(ex->a, op->a)) return;
    if (!try_match(ex->b, op->b)) return;
  }

  void visit(const variable* op) override {
    const variable* ev = static_cast<const variable*>(self);
    if (!try_match(ev->sym, op->sym)) return;
    if (!try_match(ev->field, op->field)) return;
    if (!try_match(ev->dim, op->dim)) return;
  }

  void visit(const constant* op) override {
    const constant* ec = static_cast<const constant*>(self);
    try_match(ec->value, op->value);
  }

  template <typename T>
  void visit_let(const T* op) {
    const T* el = static_cast<const T*>(self);

    if (!try_match(el->lets, op->lets)) return;
    if (!try_match(el->body, op->body)) return;
  }

  void visit(const let* op) override { visit_let(op); }
  void visit(const add* op) override { match_binary(op); }
  void visit(const sub* op) override { match_binary(op); }
  void visit(const mul* op) override { match_binary(op); }
  void visit(const div* op) override { match_binary(op); }
  void visit(const mod* op) override { match_binary(op); }
  void visit(const class min* op) override { match_binary(op); }
  void visit(const class max* op) override { match_binary(op); }
  void visit(const equal* op) override { match_binary(op); }
  void visit(const not_equal* op) override { match_binary(op); }
  void visit(const less* op) override { match_binary(op); }
  void visit(const less_equal* op) override { match_binary(op); }
  void visit(const logical_and* op) override { match_binary(op); }
  void visit(const logical_or* op) override { match_binary(op); }

  void visit(const logical_not* op) override {
    const class logical_not* ne = static_cast<const logical_not*>(self);

    try_match(ne->a, op->a);
  }

  void visit(const class select* op) override {
    const class select* se = static_cast<const class select*>(self);

    if (!try_match(se->condition, op->condition)) return;
    if (!try_match(se->true_value, op->true_value)) return;
    if (!try_match(se->false_value, op->false_value)) return;
  }

  void visit(const call* op) override {
    const call* c = static_cast<const call*>(self);

    if (!try_match(c->intrinsic, op->intrinsic)) return;
    if (!try_match(c->args, op->args)) return;
  }

  void visit(const let_stmt* op) override { visit_let(static_cast<const let_stmt*>(op)); }

  void visit(const block* op) override {
    const block* bs = static_cast<const block*>(self);

    if (!try_match(bs->stmts, op->stmts)) return;
  }

  void visit(const loop* op) override {
    const loop* ls = static_cast<const loop*>(self);

    if (!try_match(ls->sym, op->sym)) return;
    if (!try_match(ls->bounds, op->bounds)) return;
    if (!try_match(ls->step, op->step)) return;
    if (!try_match(ls->body, op->body)) return;
  }

  void visit(const call_stmt* op) override {
    if (match) return;
    const call_stmt* cs = static_cast<const call_stmt*>(self);
    assert(cs);

    if (!try_match(cs->inputs, op->inputs)) return;
    if (!try_match(cs->outputs, op->outputs)) return;
    if (cs->target && op->target) {
      // If std::function-s are defined we can't compare the functions, compare the pointers instead.
      if (!try_match(cs, op)) return;
    } else {
      if (!try_match(!cs->target, !op->target)) return;
    }
  }

  void visit(const copy_stmt* op) override {
    const copy_stmt* cs = static_cast<const copy_stmt*>(self);
    assert(cs);

    if (!try_match(cs->src, op->src)) return;
    if (!try_match(cs->src_x, op->src_x)) return;
    if (!try_match(cs->dst, op->dst)) return;
    if (!try_match(cs->dst_x, op->dst_x)) return;
    if (!try_match(cs->pad, op->pad)) return;
    if (cs->impl && op->impl) {
      // If std::function-s are defined we can't compare the functions, compare the pointers instead.
      if (!try_match(cs, op)) return;
    } else {
      if (!try_match(!cs->impl, !op->impl)) return;
    }
  }

  void visit(const allocate* op) override {
    const allocate* as = static_cast<const allocate*>(self);
    assert(as);

    if (!try_match(as->sym, op->sym)) return;
    if (!try_match(as->elem_size, op->elem_size)) return;
    if (!try_match(as->dims, op->dims)) return;
    if (!try_match(as->body, op->body)) return;
  }

  void visit(const make_buffer* op) override {
    const make_buffer* mbs = static_cast<const make_buffer*>(self);
    assert(mbs);

    if (!try_match(mbs->sym, op->sym)) return;
    if (!try_match(mbs->base, op->base)) return;
    if (!try_match(mbs->elem_size, op->elem_size)) return;
    if (!try_match(mbs->dims, op->dims)) return;
    if (!try_match(mbs->body, op->body)) return;
  }

  void visit(const constant_buffer* op) override {
    const constant_buffer* mbs = static_cast<const constant_buffer*>(self);
    assert(mbs);

    if (!try_match(mbs->sym, op->sym)) return;
    if (!try_match(mbs->value.get(), op->value.get())) return;
    if (!try_match(mbs->body, op->body)) return;
  }

  void visit(const clone_buffer* op) override {
    const clone_buffer* mbs = static_cast<const clone_buffer*>(self);
    assert(mbs);

    if (!try_match(mbs->sym, op->sym)) return;
    if (!try_match(mbs->src, op->src)) return;
    if (!try_match(mbs->body, op->body)) return;
  }

  void visit(const crop_buffer* op) override {
    const crop_buffer* cbs = static_cast<const crop_buffer*>(self);
    assert(cbs);

    if (!try_match(cbs->sym, op->sym)) return;
    if (!try_match(cbs->src, op->src)) return;
    if (!try_match(cbs->bounds, op->bounds)) return;
    if (!try_match(cbs->body, op->body)) return;
  }

  void visit(const crop_dim* op) override {
    const crop_dim* cds = static_cast<const crop_dim*>(self);
    assert(cds);

    if (!try_match(cds->sym, op->sym)) return;
    if (!try_match(cds->src, op->src)) return;
    if (!try_match(cds->dim, op->dim)) return;
    if (!try_match(cds->bounds, op->bounds)) return;
    if (!try_match(cds->body, op->body)) return;
  }

  void visit(const slice_buffer* op) override {
    const slice_buffer* cbs = static_cast<const slice_buffer*>(self);
    assert(cbs);

    if (!try_match(cbs->sym, op->sym)) return;
    if (!try_match(cbs->src, op->src)) return;
    if (!try_match(cbs->at, op->at)) return;
    if (!try_match(cbs->body, op->body)) return;
  }

  void visit(const slice_dim* op) override {
    const slice_dim* cds = static_cast<const slice_dim*>(self);
    assert(cds);

    if (!try_match(cds->sym, op->sym)) return;
    if (!try_match(cds->src, op->src)) return;
    if (!try_match(cds->dim, op->dim)) return;
    if (!try_match(cds->at, op->at)) return;
    if (!try_match(cds->body, op->body)) return;
  }

  void visit(const transpose* op) override {
    const transpose* trs = static_cast<const transpose*>(self);
    assert(trs);

    if (!try_match(trs->sym, op->sym)) return;
    if (!try_match(trs->src, op->src)) return;
    if (!try_match(trs->dims, op->dims)) return;
    if (!try_match(trs->body, op->body)) return;
  }

  void visit(const async* op) override {
    const async* as = static_cast<const async*>(self);
    assert(as);

    if (!try_match(as->sym, op->sym)) return;
    if (!try_match(as->task, op->task)) return;
    if (!try_match(as->body, op->body)) return;
  }

  void visit(const check* op) override {
    const check* cs = static_cast<const check*>(self);
    assert(cs);

    try_match(cs->condition, op->condition);
  }
};

}  // namespace

bool match(expr_ref a, expr_ref b) { return matcher().try_match(a.get(), b.get()); }
bool match(stmt_ref a, stmt_ref b) { return matcher().try_match(a.get(), b.get()); }
bool match(const interval_expr& a, const interval_expr& b) { return matcher().try_match(a, b); }
bool match(const dim_expr& a, const dim_expr& b) { return matcher().try_match(a, b); }
bool match(const box_expr& a, const box_expr& b) { return matcher().try_match(a, b); }

int compare(const var& a, const var& b) {
  matcher m;
  m.try_match(a, b);
  return m.match;
}
int compare(expr_ref a, expr_ref b) {
  matcher m;
  m.try_match(a.get(), b.get());
  return m.match;
}

int compare(stmt_ref a, stmt_ref b) {
  matcher m;
  m.try_match(a.get(), b.get());
  return m.match;
}

namespace {

template <typename T>
auto mutate_let(substitutor* this_, const T* op) {
  std::vector<std::pair<var, expr>> lets = op->lets;
  bool changed = false;
  std::size_t decls_entered = 0;
  for (auto& s : lets) {
    expr value = this_->mutate(s.second);
    changed = changed || !value.same_as(s.second);
    s.second = std::move(value);
    var decl = this_->enter_decl(s.first);
    if (!decl.defined()) {
      this_->exit_decls(decls_entered);
      break;
    }
    changed = changed || decl != s.first;
    s.first = decl;
    ++decls_entered;
  }

  auto body = decls_entered == lets.size() ? this_->mutate(op->body) : op->body;
  changed = changed || !body.same_as(op->body);
  this_->exit_decls(decls_entered);
  if (!changed) {
    return decltype(op->body){op};
  } else {
    return T::make(std::move(lets), std::move(body));
  }
}

}  // namespace

void substitutor::visit(const variable* op) {
  var new_sym = visit_symbol(op->sym);
  if (!new_sym.defined()) {
    set_result(expr());
    return;
  }
  expr result = new_sym != op->sym ? variable::make(new_sym, op->field, op->dim) : expr(op);
  if (op->field != buffer_field::none) {
    set_result(mutate_variable(result.as<variable>(), new_sym, op->field, op->dim));
  } else {
    set_result(std::move(result));
  }
}

void substitutor::visit(const let* op) { set_result(mutate_let(this, op)); }
void substitutor::visit(const let_stmt* op) { set_result(mutate_let(this, op)); }

void substitutor::visit(const loop* op) {
  interval_expr bounds = mutate(op->bounds);
  expr step = mutate(op->step);
  var sym = enter_decl(op->sym);
  stmt body = sym.defined() ? mutate(op->body) : op->body;
  sym = sym.defined() ? sym : op->sym;
  if (sym == op->sym && bounds.same_as(op->bounds) && step.same_as(op->step) && body.same_as(op->body)) {
    set_result(op);
  } else {
    set_result(loop::make(sym, op->max_workers, std::move(bounds), std::move(step), std::move(body)));
  }
  exit_decls();
}
void substitutor::visit(const allocate* op) {
  expr elem_size = mutate(op->elem_size);
  std::vector<dim_expr> dims;
  dims.reserve(op->dims.size());
  bool changed = false;
  for (const dim_expr& i : op->dims) {
    dims.push_back({mutate(i.bounds), mutate(i.stride), mutate(i.fold_factor)});
    changed = changed || !dims.back().same_as(i);
  }
  var sym = enter_decl(op->sym);
  stmt body = sym.defined() ? mutate(op->body) : op->body;
  sym = sym.defined() ? sym : op->sym;
  if (!changed && sym == op->sym && elem_size.same_as(op->elem_size) && body.same_as(op->body)) {
    set_result(op);
  } else {
    set_result(allocate::make(sym, op->storage, std::move(elem_size), std::move(dims), std::move(body)));
  }
  exit_decls();
}
void substitutor::visit(const make_buffer* op) {
  expr base = mutate(op->base);
  expr elem_size = mutate(op->elem_size);
  std::vector<dim_expr> dims;
  dims.reserve(op->dims.size());
  bool changed = false;
  for (const dim_expr& i : op->dims) {
    dims.push_back({mutate(i.bounds), mutate(i.stride), mutate(i.fold_factor)});
    changed = changed || !dims.back().same_as(i);
  }
  var sym = enter_decl(op->sym);
  stmt body = sym.defined() ? mutate(op->body) : op->body;
  sym = sym.defined() ? sym : op->sym;
  if (!changed && sym == op->sym && base.same_as(op->base) && elem_size.same_as(op->elem_size) &&
      body.same_as(op->body)) {
    set_result(op);
  } else {
    set_result(make_buffer::make(sym, std::move(base), std::move(elem_size), std::move(dims), std::move(body)));
  }
  exit_decls();
}
void substitutor::visit(const constant_buffer* op) {
  var sym = enter_decl(op->sym);
  stmt body = sym.defined() ? mutate(op->body) : op->body;
  sym = sym.defined() ? sym : op->sym;
  if (sym == op->sym && body.same_as(op->body)) {
    set_result(op);
  } else {
    set_result(constant_buffer::make(sym, op->value, std::move(body)));
  }
  exit_decls();
}

void substitutor::visit(const slice_buffer* op) {
  var src = visit_symbol(op->src);
  std::vector<expr> at(op->at.size());
  at.reserve(op->at.size());
  bool changed = false;
  std::vector<int> dims;
  for (int d = 0; d < static_cast<int>(op->at.size()); ++d) {
    at[d] = mutate(op->at[d]);
    changed = changed || !at[d].same_as(op->at[d]);
    if (at[d].defined()) {
      dims.push_back(d);
    }
  }
  var sym = enter_decl(op->sym);
  stmt body = sym.defined() ? mutate(op->body) : op->body;
  sym = sym.defined() ? sym : op->sym;
  if (!changed && sym == op->sym && src == op->src && body.same_as(op->body)) {
    set_result(op);
  } else {
    set_result(slice_buffer::make(sym, src, std::move(at), std::move(body)));
  }
  exit_decls();
}
void substitutor::visit(const slice_dim* op) {
  var src = visit_symbol(op->src);
  expr at = mutate(op->at);
  var sym = enter_decl(op->sym);
  stmt body = sym.defined() ? mutate(op->body) : op->body;
  sym = sym.defined() ? sym : op->sym;
  if (sym == op->sym && src == op->src && at.same_as(op->at) && body.same_as(op->body)) {
    set_result(op);
  } else {
    set_result(slice_dim::make(sym, src, op->dim, std::move(at), std::move(body)));
  }
  exit_decls();
}

namespace {

interval_expr substitute_crop_bounds(substitutor* this_, var new_src, var src, int dim, const interval_expr& bounds) {
  // When substituting crop bounds, we need to apply the implicit clamp, which uses buffer_min(src, dim) and
  // buffer_max(src, dim).
  interval_expr result = this_->mutate(bounds);
  if (is_variable(result.min, new_src, buffer_field::min, dim)) {
    result.min = expr();
  } else if (!is_variable(bounds.min, src, buffer_field::min, dim)) {
    expr new_bounds = this_->mutate_variable(nullptr, src, buffer_field::min, dim);
    if (new_bounds.defined() && !is_variable(new_bounds, new_src, buffer_field::min, dim)) {
      // The substitution changed the implicit clamp, include it.
      result.min = max(result.min, new_bounds);
    }
  }
  if (is_variable(result.max, new_src, buffer_field::max, dim)) {
    result.max = expr();
  } else if (!is_variable(bounds.max, src, buffer_field::max, dim)) {
    expr new_bounds = this_->mutate_variable(nullptr, src, buffer_field::max, dim);
    if (new_bounds.defined() && !is_variable(new_bounds, new_src, buffer_field::max, dim)) {
      // The substitution changed the implicit clamp, include it.
      result.max = min(result.max, new_bounds);
    }
  }
  return result;
}

}  // namespace

void substitutor::visit(const crop_buffer* op) {
  var src = visit_symbol(op->src);
  box_expr bounds(op->bounds.size());
  bool changed = false;
  for (std::size_t i = 0; i < op->bounds.size(); ++i) {
    bounds[i] = substitute_crop_bounds(this, src, op->src, i, op->bounds[i]);
    changed = changed || !bounds[i].same_as(op->bounds[i]);
  }
  var sym = enter_decl(op->sym);
  stmt body = sym.defined() ? mutate(op->body) : op->body;
  sym = sym.defined() ? sym : op->sym;
  if (changed || sym != op->sym || src != op->src || !body.same_as(op->body)) {
    set_result(crop_buffer::make(sym, src, std::move(bounds), std::move(body)));
  } else {
    set_result(op);
  }
  exit_decls();
}

void substitutor::visit(const crop_dim* op) {
  var src = visit_symbol(op->src);
  interval_expr bounds = substitute_crop_bounds(this, src, op->src, op->dim, op->bounds);
  var sym = enter_decl(op->sym);
  stmt body = sym.defined() ? mutate(op->body) : op->body;
  sym = sym.defined() ? sym : op->sym;
  if (sym == op->sym && src == op->src && bounds.same_as(op->bounds) && body.same_as(op->body)) {
    set_result(op);
  } else {
    set_result(crop_dim::make(sym, src, op->dim, std::move(bounds), std::move(body)));
  }
  exit_decls();
}

void substitutor::visit(const async* op) {
  var sym = enter_decl(op->sym);
  stmt task = sym.defined() ? mutate(op->task) : op->task;
  stmt body = sym.defined() ? mutate(op->body) : op->body;
  sym = sym.defined() ? sym : op->sym;
  if (sym == op->sym && task.same_as(op->task) && body.same_as(op->body)) {
    set_result(op);
  } else {
    set_result(async::make(sym, std::move(task), std::move(body)));
  }
  exit_decls();
}

void substitutor::visit(const call* op) {
  std::vector<expr> args;
  args.reserve(op->args.size());
  bool changed = false;
  for (const expr& i : op->args) {
    args.push_back(mutate(i));
    changed = changed || !args.back().same_as(i);
  }
  if (op->intrinsic == intrinsic::buffer_at && !args.empty() && args[0].defined()) {
    auto buf = as_variable(args[0]);
    assert(buf);
    const std::size_t buf_rank = std::max(args.size() - 1, get_target_buffer_rank(*buf));
    for (std::size_t d = 0; d < buf_rank; ++d) {
      if (d + 1 >= args.size() || !args[d + 1].defined()) {
        // buffer_at has an implicit buffer_min if it is not defined.
        expr min = mutate_variable(nullptr, *buf, buffer_field::min, d);
        if (min.defined()) {
          assert(!is_variable(min, *buf, buffer_field::min, d));
          args.resize(std::max(args.size(), d + 2));
          args[d + 1] = min;
          changed = true;
        }
      } else if (d + 1 < args.size() && is_variable(args[d + 1], *buf, buffer_field::min, d)) {
        args[d + 1] = expr();
        changed = true;
      }
    }

    while (!args.empty() && !args.back().defined()) {
      args.pop_back();
      changed = true;
    }
  }
  if (changed) {
    set_result(call::make(op->intrinsic, std::move(args)));
  } else {
    set_result(op);
  }
}

void substitutor::visit(const transpose* op) {
  var src = visit_symbol(op->src);
  var sym = enter_decl(op->sym);
  stmt body = sym.defined() ? mutate(op->body) : op->body;
  sym = sym.defined() ? sym : op->sym;
  if (sym != op->sym || src != op->src || !body.same_as(op->body)) {
    set_result(transpose::make(sym, src, op->dims, std::move(body)));
  } else {
    set_result(op);
  }
  exit_decls();
}

void substitutor::visit(const call_stmt* op) {
  call_stmt::symbol_list inputs(op->inputs.size());
  call_stmt::symbol_list outputs(op->outputs.size());
  bool changed = false;
  for (std::size_t i = 0; i < op->inputs.size(); ++i) {
    inputs[i] = visit_symbol(op->inputs[i]);
    changed = changed || inputs[i] != op->inputs[i];
  }
  for (std::size_t i = 0; i < op->outputs.size(); ++i) {
    outputs[i] = visit_symbol(op->outputs[i]);
    changed = changed || outputs[i] != op->outputs[i];
  }
  if (changed) {
    set_result(call_stmt::make(op->target, std::move(inputs), std::move(outputs), op->attrs));
  } else {
    set_result(op);
  }
}

void substitutor::visit(const copy_stmt* op) {
  var src = visit_symbol(op->src);
  var dst = visit_symbol(op->dst);
  var pad = visit_symbol(op->pad);

  std::size_t decls_entered = 0;
  // copy_stmt is effectively a declaration of the dst_x symbols for the src_x expressions.
  std::vector<var> dst_x = op->dst_x;
  bool changed = false;
  for (var& i : dst_x) {
    var new_i = enter_decl(i);
    if (!new_i.defined()) {
      set_result(op);
      exit_decls(decls_entered);
      return;
    } else if (new_i != i) {
      i = new_i;
      changed = true;
    }
    ++decls_entered;
  }
  std::vector<expr> src_x(op->src_x.size());
  for (std::size_t i = 0; i < op->src_x.size(); ++i) {
    src_x[i] = decls_entered == dst_x.size() ? mutate(op->src_x[i]) : op->src_x[i];
    changed = changed || !src_x[i].same_as(op->src_x[i]);
  }
  exit_decls(decls_entered);
  if (changed || src != op->src || dst != op->dst || pad != op->pad) {
    set_result(copy_stmt::make(op->impl, src, std::move(src_x), dst, std::move(dst_x), pad));
  } else {
    set_result(op);
  }
}

void substitutor::visit(const clone_buffer* op) {
  var src = visit_symbol(op->src);
  var sym = enter_decl(op->sym);
  stmt body = sym.defined() ? mutate(op->body) : op->body;
  sym = sym.defined() ? sym : op->sym;
  if (sym != op->sym || src != op->src || !body.same_as(op->body)) {
    set_result(clone_buffer::make(sym, src, std::move(body)));
  } else {
    set_result(op);
  }
  exit_decls();
}

namespace {

// A substutitor implementation for target vars
class var_substitutor : public substitutor {
public:
  var target;
  expr replacement;

public:
  var_substitutor(var target, const expr& replacement) : target(target), replacement(replacement) {}

  var enter_decl(var x) override { return x != target && !depends_on(replacement, x).any() ? x : var(); }

  void visit(const variable* v) override {
    if (v->sym == target) {
      if (v->field != buffer_field::none) {
        // The replacement must be another var.
        var new_sym = replacement_symbol(replacement);
        set_result(new_sym == v->sym ? expr(v) : variable::make(new_sym, v->field, v->dim));
      } else {
        set_result(replacement);
      }
    } else {
      set_result(v);
    }
  }
  using substitutor::visit;

  static var replacement_symbol(const expr& r) {
    auto s = as_variable(r);
    assert(s);
    return *s;
  }

  var visit_symbol(var x) override {
    if (x == target) {
      return replacement_symbol(replacement);
    } else {
      return x;
    }
  }
};

// A substitutor implementation for target buffers.
class buffer_substitutor : public substitutor {
public:
  var target;
  expr elem_size;
  span<const dim_expr> dims;
  var def;

public:
  buffer_substitutor(var target, expr elem_size, span<const dim_expr> dims, var def = var())
      : target(target), elem_size(elem_size), dims(dims), def(def) {}

  var enter_decl(var x) override { return x != target ? x : var(); }

  stmt mutate(const stmt& s) override { SLINKY_UNREACHABLE << "can't substitute buffer into stmt"; }
  dim_expr mutate(const dim_expr& e) { return {mutate(e.bounds), mutate(e.stride), mutate(e.fold_factor)}; }
  using substitutor::mutate;

  std::size_t get_target_buffer_rank(var x) override { return x == target ? dims.size() : 0; }

  expr mutate_variable(const variable* op, var buf, buffer_field field, int dim) override {
    if (buf != target) return expr(op);

    switch (field) {
    case buffer_field::rank: return def.defined() ? buffer_rank(def) : expr(dims.size());
    case buffer_field::elem_size:
      if (elem_size.defined()) return elem_size;
      break;
    case buffer_field::min:
    case buffer_field::max:
    case buffer_field::stride:
    case buffer_field::fold_factor: {
      expr sub = dim < static_cast<index_t>(dims.size()) ? dims[dim].get_field(field) : expr();
      if (sub.defined()) return sub;
      break;
    }
    default: SLINKY_UNREACHABLE << "got scalar var instead of buffer";
    }
    return def.defined() ? variable::make(def, field, dim) : expr();
  }
};

}  // namespace

expr substitute(const expr& e, var target, const expr& replacement) {
  if (is_variable(replacement, target)) return e;
  return var_substitutor(target, replacement).mutate(e);
}
interval_expr substitute(const interval_expr& x, var target, const expr& replacement) {
  if (is_variable(replacement, target)) return x;
  return var_substitutor(target, replacement).mutate(x);
}
stmt substitute(const stmt& s, var target, const expr& replacement) {
  if (is_variable(replacement, target)) return s;
  scoped_trace trace("substitute");
  return var_substitutor(target, replacement).mutate(s);
}

expr substitute_buffer(const expr& e, var buffer, const std::vector<dim_expr>& dims, var def) {
  return substitute_buffer(e, buffer, expr(), dims, def);
}
expr substitute_buffer(const expr& e, var buffer, const expr& elem_size, const std::vector<dim_expr>& dims, var def) {
  return buffer_substitutor(buffer, elem_size, dims, def).mutate(e);
}
interval_expr substitute_buffer(const interval_expr& e, var buffer, const std::vector<dim_expr>& dims, var def) {
  return substitute_buffer(e, buffer, expr(), dims, def);
}
interval_expr substitute_buffer(
    const interval_expr& e, var buffer, const expr& elem_size, const std::vector<dim_expr>& dims, var def) {
  return buffer_substitutor(buffer, elem_size, dims, def).mutate(e);
}

std::vector<dim_expr> make_dims_from_bounds(const box_expr& bounds) {
  std::vector<dim_expr> dims(bounds.size());
  for (index_t d = 0; d < static_cast<index_t>(bounds.size()); ++d) {
    dims[d].bounds = bounds[d];
  }
  return dims;
}
std::vector<dim_expr> make_dims_from_bounds(int dim, const interval_expr& bounds) {
  std::vector<dim_expr> dims(dim + 1);
  dims[dim].bounds = bounds;
  return dims;
}

namespace {

class expr_substitutor : public node_mutator {
public:
  expr_ref target;
  expr_ref replacement;

public:
  expr_substitutor(expr_ref target, expr_ref replacement) : target(target), replacement(replacement) {}

  expr mutate(const expr& op) override {
    if (matcher().try_match(op.get(), target.get())) {
      return replacement;
    }
    return node_mutator::mutate(op);
  }
  stmt mutate(const stmt& op) override {
    // We don't support substituting exprs into stmts.
    SLINKY_UNREACHABLE << "can't substitute expr into stmt";
  }
  using node_mutator::mutate;
};

}  // namespace

expr substitute(const expr& e, const expr& target, const expr& replacement) {
  return expr_substitutor(target, replacement).mutate(e);
}

}  // namespace slinky
