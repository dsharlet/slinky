#include "substitute.h"

#include <cassert>

#include "node_mutator.h"
#include "symbol_map.h"

namespace slinky {

class matcher : public node_visitor {
  // In this class, we visit the pattern, and manually traverse the expression being matched.
  expr e;
  stmt s;
  std::map<symbol_id, expr>* matches;

public:
  int match = 0;

  matcher(const expr& e, const stmt& s, std::map<symbol_id, expr>* matches = nullptr) : e(e), s(s), matches(matches) {}

  template <typename T>
  bool try_match(T self, T x) {
    if (self == x) {
      match = 0;
    } else if (self < x) {
      match = -1;
    } else {
      match = 1;
    }
    return match == 0;
  }

  bool try_match(const expr& self, const expr& x) {
    if (!self.defined() && !x.defined()) {
      match = 0;
    } else if (!self.defined()) {
      match = -1;
    } else if (!x.defined()) {
      match = 1;
    } else {
      e = self;
      x.accept(this);
    }
    return match == 0;
  }

  bool try_match(const stmt& self, const stmt& x) {
    if (!self.defined() && !x.defined()) {
      match = 0;
    } else if (!self.defined()) {
      match = -1;
    } else if (!x.defined()) {
      match = 1;
    } else {
      s = self;
      x.accept(this);
    }
    return match == 0;
  }

  bool try_match(const interval_expr& self, const interval_expr& x) {
    if (!try_match(self.min, x.min)) return false;
    if (!try_match(self.max, x.max)) return false;
    return true;
  }

  bool try_match(const dim_expr& self, const dim_expr& x) {
    if (!try_match(self.bounds, x.bounds)) return false;
    if (!try_match(self.stride, x.stride)) return false;
    if (!try_match(self.fold_factor, x.fold_factor)) return false;
    return true;
  }

  template <typename T>
  bool try_match(const std::vector<T>& self, const std::vector<T>& x) {
    if (self.size() < x.size()) {
      match = -1;
      return false;
    } else if (self.size() > x.size()) {
      match = 1;
      return false;
    }

    for (std::size_t i = 0; i < self.size(); ++i) {
      if (!try_match(self[i], x[i])) return false;
    }

    return true;
  }

  template <typename T, typename N>
  const T* try_match_node_type(const N& n, const T* x) {
    const T* result = n.template as<T>();
    if (result) {
      match = 0;
    } else if (!n.defined() || n.n->type < x->type) {
      match = -1;
    } else {
      match = 1;
    }
    return result;
  }

  template <typename T>
  void match_binary(const T* x) {
    if (match) return;
    const T* ex = try_match_node_type(e, x);
    if (!ex) return;

    if (!try_match(ex->a, x->a)) return;
    if (!try_match(ex->b, x->b)) return;
  }

  void match_wildcard(symbol_id name, std::function<bool(const expr&)> predicate) {
    if (match) return;

    expr& matched = (*matches)[name];
    if (matched.defined()) {
      // We already matched this variable. The expression must match.
      match = slinky::match(matched, e) ? 0 : 1;
    } else if (!predicate || predicate(e)) {
      // This is a new match.
      matched = e;
      match = 0;
    } else {
      // The predicate failed, we can't match this.
      match = 1;
    }
  }

  void visit(const variable* x) override {
    if (matches) {
      match_wildcard(x->name, nullptr);
    } else {
      const variable* ev = try_match_node_type(e, x);
      if (ev) {
        try_match(ev->name, x->name);
      }
    }
  }

  void visit(const wildcard* x) override {
    if (matches) {
      match_wildcard(x->name, x->matches);
    } else {
      const wildcard* ew = try_match_node_type(e, x);
      if (ew) {
        try_match(ew->name, x->name);
      }
    }
  }

  void visit(const constant* x) override {
    if (match) return;

    const constant* ec = try_match_node_type(e, x);
    if (ec) {
      try_match(ec->value, x->value);
    }
  }

  template <typename T>
  void visit_let(const T* x) {
    if (match) return;
    const T* el = try_match_node_type(e, x);
    if (!el) return;

    if (!try_match(el->name, x->name)) return;
    if (!try_match(el->value, x->value)) return;
    if (!try_match(el->body, x->body)) return;
  }

  void visit(const let* x) override { visit_let(x); }
  void visit(const add* x) override { match_binary(x); }
  void visit(const sub* x) override { match_binary(x); }
  void visit(const mul* x) override { match_binary(x); }
  void visit(const div* x) override { match_binary(x); }
  void visit(const mod* x) override { match_binary(x); }
  void visit(const class min* x) override { match_binary(x); }
  void visit(const class max* x) override { match_binary(x); }
  void visit(const equal* x) override { match_binary(x); }
  void visit(const not_equal* x) override { match_binary(x); }
  void visit(const less* x) override { match_binary(x); }
  void visit(const less_equal* x) override { match_binary(x); }
  void visit(const logical_and* x) override { match_binary(x); }
  void visit(const logical_or* x) override { match_binary(x); }
  void visit(const logical_not* x) override {
    if (match) return;
    const class logical_not* ne = try_match_node_type(e, x);
    if (!ne) return;

    try_match(ne->x, x->x);
  }

  void visit(const class select* x) override {
    if (match) return;
    const class select* se = try_match_node_type(e, x);
    if (!se) return;

    if (!try_match(se->condition, x->condition)) return;
    if (!try_match(se->true_value, x->true_value)) return;
    if (!try_match(se->false_value, x->false_value)) return;
  }

  void visit(const load_buffer_meta* x) override {
    if (match) return;

    const load_buffer_meta* lbme = try_match_node_type(e, x);
    if (!lbme) return;

    if (!try_match(x->meta, lbme->meta)) return;
    if (!try_match(lbme->buffer, x->buffer)) return;
    if (!try_match(lbme->dim, x->dim)) return;
  }

  void visit(const call* x) override {
    if (match) return;
    const call* c = try_match_node_type(e, x);
    if (!c) return;

    if (!try_match(c->intrinsic, x->intrinsic)) return;
    if (!try_match(c->args, x->args)) return;
  }

  void visit(const let_stmt* x) override { visit_let(x); }

  void visit(const block* x) override {
    if (match) return;
    const block* bs = try_match_node_type(s, x);
    if (!bs) return;

    if (!try_match(bs->a, x->a)) return;
    if (!try_match(bs->b, x->b)) return;
  }

  void visit(const loop* x) override {
    if (match) return;
    const loop* ls = try_match_node_type(s, x);
    if (!ls) return;

    if (!try_match(ls->name, x->name)) return;
    if (!try_match(ls->bounds, x->bounds)) return;
    if (!try_match(ls->body, x->body)) return;
  }

  void visit(const if_then_else* x) override {
    if (match) return;
    const if_then_else* is = try_match_node_type(s, x);
    if (!is) return;

    if (!try_match(is->condition, x->condition)) return;
    if (!try_match(is->true_body, x->true_body)) return;
    if (!try_match(is->false_body, x->false_body)) return;
  }

  void visit(const call_func* x) override {
    if (match) return;
    const call_func* cs = try_match_node_type(s, x);
    if (!cs) return;

    if (!try_match(cs->fn, x->fn)) return;
    if (!try_match(cs->scalar_args, x->scalar_args)) return;
    if (!try_match(cs->buffer_args, x->buffer_args)) return;
    // TODO(https://github.com/dsharlet/slinky/issues/11): How to compare callable?
  }

  void visit(const allocate* x) override {
    if (match) return;
    const allocate* as = try_match_node_type(s, x);
    if (!as) return;

    if (!try_match(as->name, x->name)) return;
    if (!try_match(as->elem_size, x->elem_size)) return;
    if (!try_match(as->dims, x->dims)) return;
    if (!try_match(as->body, x->body)) return;
  }

  void visit(const make_buffer* x) override {
    if (match) return;
    const make_buffer* mbs = try_match_node_type(s, x);
    if (!mbs) return;

    if (!try_match(mbs->name, x->name)) return;
    if (!try_match(mbs->elem_size, x->elem_size)) return;
    if (!try_match(mbs->base, x->base)) return;
    if (!try_match(mbs->dims, x->dims)) return;
    if (!try_match(mbs->body, x->body)) return;
  }

  void visit(const crop_buffer* x) override {
    if (match) return;
    const crop_buffer* cbs = try_match_node_type(s, x);
    if (!cbs) return;

    if (!try_match(cbs->name, x->name)) return;
    if (!try_match(cbs->bounds, x->bounds)) return;
    if (!try_match(cbs->body, x->body)) return;
  }

  void visit(const crop_dim* x) override {
    if (match) return;
    const crop_dim* cds = try_match_node_type(s, x);
    if (!cds) return;

    if (!try_match(cds->name, x->name)) return;
    if (!try_match(cds->dim, x->dim)) return;
    if (!try_match(cds->min, x->min)) return;
    if (!try_match(cds->extent, x->extent)) return;
    if (!try_match(cds->body, x->body)) return;
  }

  void visit(const check* x) override {
    if (match) return;
    const check* cs = try_match_node_type(s, x);
    if (!cs) return;

    try_match(cs->condition, x->condition);
  }
};

bool match(const expr& p, const expr& e, std::map<symbol_id, expr>& matches) {
  matcher m(e, stmt(), &matches);
  p.accept(&m);
  return m.match == 0;
}

bool match(const expr& a, const expr& b) { return compare(a, b) == 0; }
bool match(const stmt& a, const stmt& b) { return compare(a, b) == 0; }

int compare(const expr& a, const expr& b) {
  matcher m(a, stmt());
  b.accept(&m);
  return m.match;
}

int compare(const stmt& a, const stmt& b) {
  matcher m(expr(), a);
  b.accept(&m);
  return m.match;
}

namespace {

std::map<symbol_id, expr> empty_replacements;

class substitutor : public node_mutator {
  const std::map<symbol_id, expr>& replacements = empty_replacements;
  symbol_id target_var = -1;
  expr target;
  expr replacement;

  // Track newly declared variables that might shadow the variables we want to replace.
  symbol_map<bool> shadowed;

public:
  substitutor(const std::map<symbol_id, expr>& replacements) : replacements(replacements) {}
  substitutor(symbol_id target, const expr& replacement) : target_var(target), replacement(replacement) {}
  substitutor(const expr& target, const expr& replacement) : target(target), replacement(replacement) {}

  expr mutate(const expr& x) override {
    if (target.defined() && match(x, target)) {
      return replacement;
    } else {
      return node_mutator::mutate(x);
    }
  }
  using node_mutator::mutate;

  template <typename T>
  void visit_variable(const T* v) {
    if (shadowed.contains(v->name)) {
      // This variable has been shadowed, don't substitute it.
      set_result(v);
    } else if (v->name == target_var) {
      set_result(replacement);
    } else {
      auto i = replacements.find(v->name);
      if (i != replacements.end()) {
        set_result(i->second);
      } else {
        set_result(v);
      }
    }
  }

  void visit(const variable* v) override { visit_variable(v); }
  void visit(const wildcard* v) override { visit_variable(v); }

  template <typename T>
  void visit_decl(const T* x, symbol_id name) {
    auto s = set_value_in_scope(shadowed, name, true);
    node_mutator::visit(x);
  }

  void visit(const loop* x) override { visit_decl(x, x->name); }
  void visit(const let* x) override { visit_decl(x, x->name); }
  void visit(const let_stmt* x) override { visit_decl(x, x->name); }
  void visit(const allocate* x) override { visit_decl(x, x->name); }
  void visit(const make_buffer* x) override { visit_decl(x, x->name); }
};

}  // namespace

expr substitute(const expr& e, const std::map<symbol_id, expr>& replacements) {
  return substitutor(replacements).mutate(e);
}
stmt substitute(const stmt& s, const std::map<symbol_id, expr>& replacements) {
  return substitutor(replacements).mutate(s);
}

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

bool depends_on(const expr& e, symbol_id var) {
  class visitor : public recursive_node_visitor {
  public:
    symbol_id var;
    bool result = false;

    visitor(symbol_id var) : var(var) {}

    void visit(const variable* x) override { result = result || x->name == var; }
    void visit(const wildcard* x) override { result = result || x->name == var; }
  };

  visitor v(var);
  e.accept(&v);
  return v.result;
}

}  // namespace slinky
