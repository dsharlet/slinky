#include "substitute.h"

#include <cassert>

#include "node_mutator.h"
#include "symbol_map.h"

namespace slinky {

class matcher : public node_visitor {
  // In this class, we visit the pattern, and manually traverse the expression being matched.
  const base_node* self;
  symbol_map<expr>* matches;

  template <typename T>
  const T* self_as() const {
    if (self && self->type == T::static_type) {
      return static_cast<const T*>(self);
    } else {
      return nullptr;
    }
  }

public:
  int match = 0;

  matcher(const expr& e, symbol_map<expr>* matches = nullptr) : self(e.get()), matches(matches) {}
  matcher(const stmt& s, symbol_map<expr>* matches = nullptr) : self(s.get()), matches(matches) {}

  template <typename T>
  bool try_match(T self, T op) {
    if (self == op) {
      match = 0;
    } else if (self < op) {
      match = -1;
    } else {
      match = 1;
    }
    return match == 0;
  }

  // Skip the visitor pattern (two virtual function calls) for a few node types that are very frequently visited.
  void visit(const expr& op) {
    switch (op.type()) {
    case node_type::add: visit(reinterpret_cast<const add*>(op.get())); return;
    case node_type::min: visit(reinterpret_cast<const class min*>(op.get())); return;
    case node_type::max: visit(reinterpret_cast<const class max*>(op.get())); return;
    default: op.accept(this);
    }
  }

  bool try_match(const expr& e, const expr& op) {
    if (!e.defined() && !op.defined()) {
      match = 0;
    } else if (!e.defined()) {
      match = -1;
    } else if (!op.defined()) {
      match = 1;
    } else {
      self = e.get();
      visit(op);
    }
    return match == 0;
  }

  bool try_match(const stmt& s, const stmt& op) {
    if (!s.defined() && !op.defined()) {
      match = 0;
    } else if (!s.defined()) {
      match = -1;
    } else if (!op.defined()) {
      match = 1;
    } else {
      self = s.get();
      op.accept(this);
    }
    return match == 0;
  }

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
  const T* match_self_as(const T* op) {
    const T* result = self_as<T>();
    if (result) {
      match = 0;
    } else if (!self || self->type < op->type) {
      match = -1;
    } else {
      match = 1;
    }
    return result;
  }

  template <typename T>
  void match_binary(const T* op) {
    if (match) return;
    const T* ex = match_self_as(op);
    if (!ex) return;

    if (!try_match(ex->a, op->a)) return;
    if (!try_match(ex->b, op->b)) return;
  }

  void match_wildcard(symbol_id sym, std::function<bool(const expr&)> predicate) {
    if (match) return;

    std::optional<expr>& matched = (*matches)[sym];
    if (matched) {
      // We already matched this variable. The expression must match.
      symbol_map<expr>* old_matches = matches;
      matches = nullptr;
      matched->accept(this);
      matches = old_matches;
    } else if (!predicate || predicate(static_cast<const base_expr_node*>(self))) {
      // This is a new match.
      matched = static_cast<const base_expr_node*>(self);
      match = 0;
    } else {
      // The predicate failed, we can't match this.
      match = 1;
    }
  }

  void visit(const variable* op) override {
    if (matches) {
      match_wildcard(op->sym, nullptr);
    } else {
      const variable* ev = match_self_as(op);
      if (ev) {
        try_match(ev->sym, op->sym);
      }
    }
  }

  void visit(const wildcard* op) override {
    if (matches) {
      match_wildcard(op->sym, op->matches);
    } else {
      const wildcard* ew = match_self_as(op);
      if (ew) {
        try_match(ew->sym, op->sym);
      }
    }
  }

  void visit(const constant* op) override {
    if (match) return;

    const constant* ec = match_self_as(op);
    if (ec) {
      try_match(ec->value, op->value);
    }
  }

  template <typename T>
  void visit_let(const T* op) {
    if (match) return;
    const T* el = match_self_as(op);
    if (!el) return;

    if (!try_match(el->sym, op->sym)) return;
    if (!try_match(el->value, op->value)) return;
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
    if (match) return;
    const class logical_not* ne = match_self_as(op);
    if (!ne) return;

    try_match(ne->a, op->a);
  }

  void visit(const class select* op) override {
    if (match) return;
    const class select* se = match_self_as(op);
    if (!se) return;

    if (!try_match(se->condition, op->condition)) return;
    if (!try_match(se->true_value, op->true_value)) return;
    if (!try_match(se->false_value, op->false_value)) return;
  }

  void visit(const call* op) override {
    if (match) return;
    const call* c = match_self_as(op);
    if (!c) return;

    if (!try_match(c->intrinsic, op->intrinsic)) return;
    if (!try_match(c->args, op->args)) return;
  }

  void visit(const let_stmt* op) override { visit_let(op); }

  void visit(const block* op) override {
    if (match) return;
    const block* bs = match_self_as(op);
    if (!bs) return;

    if (!try_match(bs->a, op->a)) return;
    if (!try_match(bs->b, op->b)) return;
  }

  void visit(const loop* op) override {
    if (match) return;
    const loop* ls = match_self_as(op);
    if (!ls) return;

    if (!try_match(ls->sym, op->sym)) return;
    if (!try_match(ls->bounds, op->bounds)) return;
    if (!try_match(ls->step, op->step)) return;
    if (!try_match(ls->body, op->body)) return;
  }

  void visit(const if_then_else* op) override {
    if (match) return;
    const if_then_else* is = match_self_as(op);
    if (!is) return;

    if (!try_match(is->condition, op->condition)) return;
    if (!try_match(is->true_body, op->true_body)) return;
    if (!try_match(is->false_body, op->false_body)) return;
  }

  void visit(const call_stmt* op) override {
    if (match) return;
    const call_stmt* cs = match_self_as(op);
    if (!cs) return;

    if (!try_match(cs->inputs, op->inputs)) return;
    if (!try_match(cs->outputs, op->outputs)) return;
  }

  void visit(const copy_stmt* op) override {
    if (match) return;
    const copy_stmt* cs = match_self_as(op);
    if (!cs) return;

    if (!try_match(cs->src, op->src)) return;
    if (!try_match(cs->src_x, op->src_x)) return;
    if (!try_match(cs->dst, op->dst)) return;
    if (!try_match(cs->dst_x, op->dst_x)) return;
    if (!try_match(cs->padding, op->padding)) return;
  }

  void visit(const allocate* op) override {
    if (match) return;
    const allocate* as = match_self_as(op);
    if (!as) return;

    if (!try_match(as->sym, op->sym)) return;
    if (!try_match(as->elem_size, op->elem_size)) return;
    if (!try_match(as->dims, op->dims)) return;
    if (!try_match(as->body, op->body)) return;
  }

  void visit(const make_buffer* op) override {
    if (match) return;
    const make_buffer* mbs = match_self_as(op);
    if (!mbs) return;

    if (!try_match(mbs->sym, op->sym)) return;
    if (!try_match(mbs->base, op->base)) return;
    if (!try_match(mbs->elem_size, op->elem_size)) return;
    if (!try_match(mbs->dims, op->dims)) return;
    if (!try_match(mbs->body, op->body)) return;
  }

  void visit(const crop_buffer* op) override {
    if (match) return;
    const crop_buffer* cbs = match_self_as(op);
    if (!cbs) return;

    if (!try_match(cbs->sym, op->sym)) return;
    if (!try_match(cbs->bounds, op->bounds)) return;
    if (!try_match(cbs->body, op->body)) return;
  }

  void visit(const crop_dim* op) override {
    if (match) return;
    const crop_dim* cds = match_self_as(op);
    if (!cds) return;

    if (!try_match(cds->sym, op->sym)) return;
    if (!try_match(cds->dim, op->dim)) return;
    if (!try_match(cds->bounds, op->bounds)) return;
    if (!try_match(cds->body, op->body)) return;
  }

  void visit(const slice_buffer* op) override {
    if (match) return;
    const slice_buffer* cbs = match_self_as(op);
    if (!cbs) return;

    if (!try_match(cbs->sym, op->sym)) return;
    if (!try_match(cbs->at, op->at)) return;
    if (!try_match(cbs->body, op->body)) return;
  }

  void visit(const slice_dim* op) override {
    if (match) return;
    const slice_dim* cds = match_self_as(op);
    if (!cds) return;

    if (!try_match(cds->sym, op->sym)) return;
    if (!try_match(cds->dim, op->dim)) return;
    if (!try_match(cds->at, op->at)) return;
    if (!try_match(cds->body, op->body)) return;
  }

  void visit(const truncate_rank* op) override {
    if (match) return;
    const truncate_rank* trs = match_self_as(op);
    if (!trs) return;

    if (!try_match(trs->sym, op->sym)) return;
    if (!try_match(trs->rank, op->rank)) return;
    if (!try_match(trs->body, op->body)) return;
  }

  void visit(const check* op) override {
    if (match) return;
    const check* cs = match_self_as(op);
    if (!cs) return;

    try_match(cs->condition, op->condition);
  }
};

bool match(const expr& p, const expr& e, symbol_map<expr>& matches) {
  matcher m(e, &matches);
  p.accept(&m);
  return m.match == 0;
}

bool match(const expr& a, const expr& b) { return compare(a, b) == 0; }
bool match(const stmt& a, const stmt& b) { return compare(a, b) == 0; }
bool match(const interval_expr& a, const interval_expr& b) { return match(a.min, b.min) && match(a.max, b.max); }
bool match(const dim_expr& a, const dim_expr& b) {
  return match(a.bounds, b.bounds) && match(a.stride, b.stride) && match(a.fold_factor, b.fold_factor);
}

int compare(const expr& a, const expr& b) {
  // This should match the behavior of matcher::try_match.
  // TODO: It would be nice if we didn't need to duplicate this tricky logic.
  if (!b.defined()) return a.defined() ? 1 : 0;
  matcher m(a);
  b.accept(&m);
  return m.match;
}

int compare(const stmt& a, const stmt& b) {
  // This should match the behavior of matcher::try_match.
  // TODO: It would be nice if we didn't need to duplicate this tricky logic.
  if (!b.defined()) return a.defined() ? 1 : 0;
  matcher m(a);
  b.accept(&m);
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
    if (target.defined() && match(op, target)) {
      return replacement;
    } else {
      return node_mutator::mutate(op);
    }
  }
  using node_mutator::mutate;

  template <typename T>
  void visit_variable(const T* v) {
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

  void visit(const variable* v) override { visit_variable(v); }
  void visit(const wildcard* v) override { visit_variable(v); }

  template <typename T>
  void visit_decl(const T* op, symbol_id sym) {
    auto s = set_value_in_scope(shadowed, sym, true);
    node_mutator::visit(op);
  }

  void visit(const loop* op) override { visit_decl(op, op->sym); }
  void visit(const let* op) override { visit_decl(op, op->sym); }
  void visit(const let_stmt* op) override { visit_decl(op, op->sym); }
  void visit(const allocate* op) override { visit_decl(op, op->sym); }
  void visit(const make_buffer* op) override { visit_decl(op, op->sym); }
};

template <typename T>
T substitute(T op, std::span<const std::pair<expr, expr>> subs) {
  for (const std::pair<expr, expr>& i : subs) {
    op = substitutor(i.first, i.second).mutate(op);
  }
  return op;
}

template <typename T>
T substitute_bounds_impl(T op, symbol_id buffer, int dim, const interval_expr& bounds) {
  expr buf_var = variable::make(buffer);
  std::pair<expr, expr> subs[] = {
      {buffer_min(buf_var, dim), bounds.min},
      {buffer_max(buf_var, dim), bounds.max},
  };
  return substitute(op, subs);
}

template <typename T>
T substitute_bounds_impl(T op, symbol_id buffer, const box_expr& bounds) {
  expr buf_var = variable::make(buffer);
  std::vector<std::pair<expr, expr>> subs;
  subs.reserve(bounds.size() * 2);
  for (index_t d = 0; d < static_cast<index_t>(bounds.size()); ++d) {
    if (bounds[d].min.defined()) {
      subs.emplace_back(buffer_min(buf_var, d), bounds[d].min);
    }
    if (bounds[d].max.defined()) {
      subs.emplace_back(buffer_max(buf_var, d), bounds[d].max);
    }
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

namespace {

class dependencies : public recursive_node_visitor {
public:
  std::span<const symbol_id> vars;
  bool found_var = false;
  bool found_buf = false;

  dependencies(std::span<const symbol_id> vars) : vars(vars) {}

  void accept_buffer(const expr& e) {
    bool old_found_var = found_var;
    found_var = false;
    e.accept(this);
    found_buf = found_buf || found_var;
    found_var = old_found_var;
  }

  void visit_var(symbol_id sym) {
    for (symbol_id i : vars) {
      if (i == sym) {
        found_var = true;
        return;
      }
    }
  }

  void visit_buf(symbol_id sym) {
    for (symbol_id i : vars) {
      if (i == sym) {
        found_buf = true;
        return;
      }
    }
  }

  void visit(const variable* op) override { visit_var(op->sym); }
  void visit(const wildcard* op) override { visit_var(op->sym); }
  void visit(const call* op) override {
    if (is_buffer_intrinsic(op->intrinsic)) {
      assert(op->args.size() >= 1);
      accept_buffer(op->args[0]);

      for (std::size_t i = 1; i < op->args.size(); ++i) {
        op->args[i].accept(this);
      }
    } else {
      recursive_node_visitor::visit(op);
    }
  }

  void visit(const call_stmt* op) override {
    for (symbol_id i : op->inputs) {
      visit_buf(i);
    }
    for (symbol_id i : op->outputs) {
      visit_buf(i);
    }
  }
  void visit(const copy_stmt* op) override {
    visit_buf(op->src);
    visit_buf(op->dst);
  }
};

}  // namespace

bool depends_on(const expr& e, symbol_id var) {
  if (!e.defined()) return false;
  symbol_id vars[] = {var};
  dependencies v(vars);
  e.accept(&v);
  return v.found_var || v.found_buf;
}

bool depends_on(const interval_expr& e, symbol_id var) {
  symbol_id vars[] = {var};
  dependencies v(vars);
  if (e.min.defined()) e.min.accept(&v);
  if (e.max.defined()) e.max.accept(&v);
  return v.found_var || v.found_buf;
}

bool depends_on(const stmt& s, std::span<const symbol_id> vars) {
  if (!s.defined()) return false;
  dependencies v(vars);
  s.accept(&v);
  return v.found_var || v.found_buf;
}

bool depends_on_variable(const expr& e, symbol_id var) {
  if (!e.defined()) return false;
  symbol_id vars[] = {var};
  dependencies v(vars);
  e.accept(&v);
  return v.found_var;
}

bool depends_on_buffer(const expr& e, symbol_id buf) {
  if (!e.defined()) return false;
  symbol_id bufs[] = {buf};
  dependencies v(bufs);
  e.accept(&v);
  return v.found_buf;
}

}  // namespace slinky
