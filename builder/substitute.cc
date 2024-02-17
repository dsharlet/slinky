#include "builder/substitute.h"

#include <cassert>
#include <cstddef>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "builder/node_mutator.h"
#include "runtime/depends_on.h"
#include "runtime/expr.h"
#include "runtime/util.h"

namespace slinky {

class matcher : public node_visitor {
  // In this class, we visit the pattern, and manually traverse the expression being matched.
  const base_node* self = nullptr;
  symbol_map<expr>* matches;

public:
  int match = 0;

  matcher(symbol_map<expr>* matches = nullptr) : matches(matches) {}

  template <typename T>
  bool try_match(T self, T op) {
    assert(match == 0);
    if (self < op) {
      match = -1;
    } else if (self > op) {
      match = 1;
    }
    return match == 0;
  }

  // Skip the visitor pattern (two virtual function calls) for a few node types that are very frequently visited.
  void visit(const base_expr_node* op) {
    switch (op->type) {
    case node_type::variable: visit(reinterpret_cast<const variable*>(op)); return;
    case node_type::add: visit(reinterpret_cast<const add*>(op)); return;
    case node_type::min: visit(reinterpret_cast<const class min*>(op)); return;
    case node_type::max: visit(reinterpret_cast<const class max*>(op)); return;
    default: op->accept(this);
    }
  }

  bool try_match(const base_expr_node* e, const base_expr_node* op) {
    assert(match == 0);
    if (!e && !op) {
    } else if (!e) {
      match = -1;
    } else if (!op) {
      match = 1;
    } else if (matches && op->type == node_type::variable) {
      // When we are matching with variables as wildcards, the type doesn't need to match.
      self = e;
      visit(reinterpret_cast<const variable*>(op));
    } else if (e->type < op->type) {
      match = -1;
    } else if (e->type > op->type) {
      match = 1;
    } else {
      self = e;
      visit(op);
    }
    return match == 0;
  }
  bool try_match(const expr& e, const expr& op) { return try_match(e.get(), op.get()); }

  bool try_match(const base_stmt_node* s, const base_stmt_node* op) {
    assert(match == 0);
    if (!s && !op) {
    } else if (!s) {
      match = -1;
    } else if (!op) {
      match = 1;
    } else if (s->type < op->type) {
      match = -1;
    } else if (s->type > op->type) {
      match = 1;
    } else {
      self = s;
      op->accept(this);
    }
    return match == 0;
  }
  bool try_match(const stmt& s, const stmt& op) { return try_match(s.get(), op.get()); }

  bool try_match(const interval_expr& self, const interval_expr& op) {
    if (!try_match(self.min, op.min)) return false;
    if (!try_match(self.max, op.max)) return false;
    return true;
  }

  bool try_match(const dim_expr& self, const dim_expr& op) {
    if (!try_match(self.bounds, op.bounds)) return false;
    if (!try_match(self.stride, op.stride)) return false;
    if (!try_match(self.fold_factor, op.fold_factor)) return false;
    return true;
  }

  template <typename A, typename B>
  bool try_match(const std::pair<A, B>& self, const std::pair<A, B>& op) {
    if (!try_match(self.first, op.first)) return false;
    if (!try_match(self.second, op.second)) return false;
    return true;
  }

  template <typename T>
  bool try_match(const std::vector<T>& self, const std::vector<T>& op) {
    if (self.size() < op.size()) {
      match = -1;
      return false;
    } else if (self.size() > op.size()) {
      match = 1;
      return false;
    }

    for (std::size_t i = 0; i < self.size(); ++i) {
      if (!try_match(self[i], op[i])) return false;
    }

    return true;
  }

  template <typename T>
  void match_binary(const T* op) {
    const T* ex = static_cast<const T*>(self);

    if (!try_match(ex->a, op->a)) return;
    if (!try_match(ex->b, op->b)) return;
  }

  void visit(const variable* op) override {
    if (matches) {
      std::optional<expr>& matched = (*matches)[op->sym];
      if (matched) {
        // We already matched this variable. The expression must match.
        if (!matched->same_as(static_cast<const base_expr_node*>(self))) {
          symbol_map<expr>* old_matches = matches;
          matches = nullptr;
          try_match(matched->get(), static_cast<const base_expr_node*>(self));
          matches = old_matches;
        }
      } else {
        // This is a new match.
        matched = static_cast<const base_expr_node*>(self);
      }
    } else {
      const variable* ev = static_cast<const variable*>(self);
      try_match(ev->sym, op->sym);
    }
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

  void visit(const let_stmt* op) override { visit_let(static_cast<const let_stmt*>(self)); }

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
  }

  void visit(const copy_stmt* op) override {
    const copy_stmt* cs = static_cast<const copy_stmt*>(self);
    assert(cs);

    if (!try_match(cs->src, op->src)) return;
    if (!try_match(cs->src_x, op->src_x)) return;
    if (!try_match(cs->dst, op->dst)) return;
    if (!try_match(cs->dst_x, op->dst_x)) return;
    if (!try_match(cs->padding, op->padding)) return;
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
    if (!try_match(cbs->bounds, op->bounds)) return;
    if (!try_match(cbs->body, op->body)) return;
  }

  void visit(const crop_dim* op) override {
    const crop_dim* cds = static_cast<const crop_dim*>(self);
    assert(cds);

    if (!try_match(cds->sym, op->sym)) return;
    if (!try_match(cds->dim, op->dim)) return;
    if (!try_match(cds->bounds, op->bounds)) return;
    if (!try_match(cds->body, op->body)) return;
  }

  void visit(const slice_buffer* op) override {
    const slice_buffer* cbs = static_cast<const slice_buffer*>(self);
    assert(cbs);

    if (!try_match(cbs->sym, op->sym)) return;
    if (!try_match(cbs->at, op->at)) return;
    if (!try_match(cbs->body, op->body)) return;
  }

  void visit(const slice_dim* op) override {
    const slice_dim* cds = static_cast<const slice_dim*>(self);
    assert(cds);

    if (!try_match(cds->sym, op->sym)) return;
    if (!try_match(cds->dim, op->dim)) return;
    if (!try_match(cds->at, op->at)) return;
    if (!try_match(cds->body, op->body)) return;
  }

  void visit(const truncate_rank* op) override {
    const truncate_rank* trs = static_cast<const truncate_rank*>(self);
    assert(trs);

    if (!try_match(trs->sym, op->sym)) return;
    if (!try_match(trs->rank, op->rank)) return;
    if (!try_match(trs->body, op->body)) return;
  }

  void visit(const check* op) override {
    const check* cs = static_cast<const check*>(self);
    assert(cs);

    try_match(cs->condition, op->condition);
  }
};

bool match(const expr& p, const expr& e, symbol_map<expr>& matches) {
  matcher m(&matches);
  m.try_match(e, p);
  return m.match == 0;
}

bool match(const expr& a, const expr& b) { return compare(a, b) == 0; }
bool match(const stmt& a, const stmt& b) { return compare(a, b) == 0; }
bool match(const interval_expr& a, const interval_expr& b) { return match(a.min, b.min) && match(a.max, b.max); }
bool match(const dim_expr& a, const dim_expr& b) {
  return match(a.bounds, b.bounds) && match(a.stride, b.stride) && match(a.fold_factor, b.fold_factor);
}

int compare(const expr& a, const expr& b) { return compare(a.get(), b.get()); }

int compare(const base_expr_node* a, const base_expr_node* b) {
  matcher m;
  m.try_match(a, b);
  return m.match;
}

int compare(const stmt& a, const stmt& b) {
  matcher m;
  m.try_match(a, b);
  return m.match;
}

namespace {

symbol_map<expr> empty_replacements;

class substitutor : public node_mutator {
  const symbol_map<expr>& replacements = empty_replacements;
  symbol_id target_var = -1;
  expr target;
  expr replacement;

  // Track newly declared variables that might shadow the variables we want to replace.
  symbol_map<bool> shadowed;

public:
  substitutor(const symbol_map<expr>& replacements) : replacements(replacements) {}
  substitutor(symbol_id target, const expr& replacement) : target_var(target), replacement(replacement) {}
  substitutor(const expr& target, const expr& replacement) : target(target), replacement(replacement) {}

  expr mutate(const expr& op) override {
    if (target.defined() && op.defined() && match(op, target)) {
      return replacement;
    } else {
      return node_mutator::mutate(op);
    }
  }
  using node_mutator::mutate;

  void visit(const variable* v) override {
    if (shadowed.contains(v->sym)) {
      // This variable has been shadowed, don't substitute it.
      set_result(v);
    } else if (v->sym == target_var) {
      set_result(replacement);
    } else {
      std::optional<expr> r = replacements.lookup(v->sym);
      set_result(r ? *r : v);
    }
  }

  template <typename T>
  T mutate_decl_body(symbol_id sym, const T& x) {
    auto s = set_value_in_scope(shadowed, sym, true);
    if (target.defined() && depends_on(target, sym).any()) {
      // If the target expression depends on the symbol we're declaring, don't substitute it because it's a different
      // expression now.
      return x;
    } else {
      return mutate(x);
    }
  }

  template <typename T>
  auto mutate_let(const T* op) {
    std::vector<std::pair<symbol_id, expr>> lets;
    lets.reserve(op->lets.size());
    std::vector<scoped_value_in_symbol_map<bool>> scoped_values;
    scoped_values.reserve(op->lets.size());
    bool changed = false;
    for (const auto& s : op->lets) {
      lets.emplace_back(s.first, mutate(s.second));
      scoped_values.push_back(set_value_in_scope(shadowed, s.first, true));
      changed = changed || !lets.back().second.same_as(s.second);
    }

    symbol_map<bool> old_shadowed = shadowed;
    bool can_substitute = true;
    for (const auto& s : lets) {
      shadowed[s.first] = true;
      if (target.defined() && depends_on(target, s.first).any()) {
        // If the target expression depends on the symbol we're declaring, don't substitute it because it's a different
        // expression now.
        can_substitute = false;
        break;
      }
    }
    auto body = op->body;
    if (can_substitute) {
      body = mutate(body);
    }
    changed = changed || !body.same_as(op->body);

    if (!changed) {
      return decltype(body){op};
    } else {
      return T::make(std::move(lets), std::move(body));
    }
  }

  void visit(const let* op) override { set_result(mutate_let(op)); }
  void visit(const let_stmt* op) override { set_result(mutate_let(op)); }

  void visit(const loop* op) override {
    interval_expr bounds = {mutate(op->bounds.min), mutate(op->bounds.max)};
    expr step = mutate(op->step);
    stmt body = mutate_decl_body(op->sym, op->body);
    if (bounds.same_as(op->bounds) && step.same_as(op->step) && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(loop::make(op->sym, op->mode, std::move(bounds), std::move(step), std::move(body)));
    }
  }
  void visit(const allocate* op) override {
    std::vector<dim_expr> dims;
    dims.reserve(op->dims.size());
    bool changed = false;
    for (const dim_expr& i : op->dims) {
      interval_expr bounds = {mutate(i.bounds.min), mutate(i.bounds.max)};
      dims.push_back({std::move(bounds), mutate(i.stride), mutate(i.fold_factor)});
      changed = changed || !dims.back().same_as(i);
    }
    stmt body = mutate_decl_body(op->sym, op->body);
    if (!changed && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(allocate::make(op->sym, op->storage, op->elem_size, std::move(dims), std::move(body)));
    }
  }
  void visit(const make_buffer* op) override {
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
    stmt body = mutate_decl_body(op->sym, op->body);
    if (!changed && base.same_as(op->base) && elem_size.same_as(op->elem_size) && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(make_buffer::make(op->sym, std::move(base), std::move(elem_size), std::move(dims), std::move(body)));
    }
  }
  void visit(const slice_buffer* op) override {
    std::vector<expr> at;
    at.reserve(op->at.size());
    bool changed = false;
    for (const expr& i : op->at) {
      at.push_back(mutate(i));
      changed = changed || !at.back().same_as(i);
    }
    stmt body = mutate_decl_body(op->sym, op->body);
    if (!changed && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(slice_buffer::make(op->sym, std::move(at), std::move(body)));
    }
  }
  void visit(const slice_dim* op) override {
    expr at = mutate(op->at);
    stmt body = mutate_decl_body(op->sym, op->body);
    if (at.same_as(op->at) && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(slice_dim::make(op->sym, op->dim, std::move(at), std::move(body)));
    }
  }
  // truncate_rank, clone_buffer, crop_buffer, crop_dim not treated here because references to dimensions of these
  // operations are still valid.
  // TODO: This seems sketchy. Shadowed symbols are shadowed symbols. But the simplifier relies on this behavior
  // currently... Another reason this is sketchy: we treat make_buffers as shadowing, but not crop_buffer. But the
  // simplifier will rewrite some make_buffer to crop_buffer, so that means substitute will behave differently before
  // vs. after simplification.
};

template <typename T>
T substitute(T op, span<const std::pair<expr, expr>> subs) {
  for (const std::pair<expr, expr>& i : subs) {
    op = substitutor(i.first, i.second).mutate(op);
  }
  return op;
}

template <typename T>
T substitute_bounds_impl(T op, symbol_id buffer, int dim, const interval_expr& bounds) {
  expr buf_var = variable::make(buffer);
  if (bounds.min.defined()) op = substitute(op, buffer_min(buf_var, dim), bounds.min);
  if (bounds.max.defined()) op = substitute(op, buffer_max(buf_var, dim), bounds.max);
  return op;
}

template <typename T>
T substitute_bounds_impl(T op, symbol_id buffer, const box_expr& bounds) {
  expr buf_var = variable::make(buffer);
  std::vector<std::pair<expr, expr>> subs;
  subs.reserve(bounds.size() * 2);
  for (index_t d = 0; d < static_cast<index_t>(bounds.size()); ++d) {
    if (bounds[d].min.defined()) subs.emplace_back(buffer_min(buf_var, d), bounds[d].min);
    if (bounds[d].max.defined()) subs.emplace_back(buffer_max(buf_var, d), bounds[d].max);
  }
  return substitute(op, subs);
}

}  // namespace

expr substitute(const expr& e, const symbol_map<expr>& replacements) { return substitutor(replacements).mutate(e); }
stmt substitute(const stmt& s, const symbol_map<expr>& replacements) { return substitutor(replacements).mutate(s); }

expr substitute(const expr& e, symbol_id target, const expr& replacement) {
  return substitutor(target, replacement).mutate(e);
}
stmt substitute(const stmt& s, symbol_id target, const expr& replacement) {
  return substitutor(target, replacement).mutate(s);
}

expr substitute(const expr& e, const expr& target, const expr& replacement) {
  return substitutor(target, replacement).mutate(e);
}
stmt substitute(const stmt& s, const expr& target, const expr& replacement) {
  return substitutor(target, replacement).mutate(s);
}

expr substitute_bounds(const expr& e, symbol_id buffer, const box_expr& bounds) {
  return substitute_bounds_impl(e, buffer, bounds);
}
stmt substitute_bounds(const stmt& s, symbol_id buffer, const box_expr& bounds) {
  return substitute_bounds_impl(s, buffer, bounds);
}
expr substitute_bounds(const expr& e, symbol_id buffer, int dim, const interval_expr& bounds) {
  return substitute_bounds_impl(e, buffer, dim, bounds);
}
stmt substitute_bounds(const stmt& s, symbol_id buffer, int dim, const interval_expr& bounds) {
  return substitute_bounds_impl(s, buffer, dim, bounds);
}

}  // namespace slinky
