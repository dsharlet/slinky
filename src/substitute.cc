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
  bool match = true;

  matcher(const expr& e, const stmt& s, std::map<symbol_id, expr>* matches = nullptr) : e(e), s(s), matches(matches) {}

  void fail() { match = false; }

  bool try_match(const expr& self, const expr& x) {
    if (!self.defined() && !x.defined()) { return true; }
    if (!self.defined() || !x.defined()) {
      match = false;
      return false;
    }
    e = self;
    x.accept(this);
    return match;
  }

  bool try_match(symbol_id self, symbol_id x) {
    match = self == x;
    return match;
  }

  bool try_match(const dim_expr& self, const dim_expr& x) {
    return try_match(self.min, x.min) && try_match(self.extent, x.extent) &&
           try_match(self.stride_bytes, x.stride_bytes) && try_match(self.fold_factor, x.fold_factor);
  }

  bool try_match(const interval& self, const interval& x) {
    return try_match(self.min, x.min) && try_match(self.max, x.max);
  }

  template <typename T>
  bool try_match(const std::vector<T>& self, const std::vector<T>& x) {
    if (self.size() != x.size()) {
      match = false;
      return false;
    }

    for (std::size_t i = 0; i < self.size(); ++i) {
      if (!try_match(self[i], x[i])) return false;
    }

    return true;
  }

  bool try_match(const stmt& self, const stmt& x) {
    if (!self.defined() && !x.defined()) { return true; }
    if (!self.defined() || !x.defined()) {
      match = false;
      return false;
    }
    s = self;
    x.accept(this);
    return match;
  }

  template <typename T>
  void match_binary(const T* x) {
    if (!match) return;
    const T* ex = e.as<T>();
    if (!ex) return fail();

    match = try_match(ex->a, x->a) && try_match(ex->b, x->b);
  }

  void match_wildcard(symbol_id name, std::function<bool(const expr&)> predicate) {
    if (!match) return;

    expr& matched = (*matches)[name];
    if (matched.defined()) {
      // We already matched this variable. The expression must match.
      match = slinky::match(matched, e);
    } else if (!predicate || predicate(e)) {
      // This is a new match.
      matched = e;
    } else {
      // The predicate failed, we can't match this.
      match = false;
    }
  }

  virtual void visit(const variable* x) {
    if (matches) {
      match_wildcard(x->name, nullptr);
    } else {
      const variable* ev = e.as<variable>();
      match = ev != nullptr && x->name == ev->name;
    }
  }

  virtual void visit(const wildcard* x) {
    if (matches) {
      match_wildcard(x->name, x->matches);
    } else {
      const wildcard* ew = e.as<wildcard>();
      match = ew != nullptr && x->name == ew->name;
    }
  }

  virtual void visit(const constant* x) {
    if (!match) return;

    const constant* ec = e.as<constant>();
    if (!ec) return fail();

    match = ec->value == x->value;
  }

  template <typename T>
  void visit_let(const T* x) {
    if (!match) return;
    const T* el = e.as<T>();
    if (!el) return fail();

    match = el->name == x->name && try_match(el->value, x->value) && try_match(el->body, x->body);
  }

  virtual void visit(const let* x) { visit_let(x); }
  virtual void visit(const add* x) { match_binary(x); }
  virtual void visit(const sub* x) { match_binary(x); }
  virtual void visit(const mul* x) { match_binary(x); }
  virtual void visit(const div* x) { match_binary(x); }
  virtual void visit(const mod* x) { match_binary(x); }
  virtual void visit(const class min* x) { match_binary(x); }
  virtual void visit(const class max* x) { match_binary(x); }
  virtual void visit(const equal* x) { match_binary(x); }
  virtual void visit(const not_equal* x) { match_binary(x); }
  virtual void visit(const less* x) { match_binary(x); }
  virtual void visit(const less_equal* x) { match_binary(x); }
  virtual void visit(const bitwise_and* x) { match_binary(x); }
  virtual void visit(const bitwise_or* x) { match_binary(x); }
  virtual void visit(const bitwise_xor* x) { match_binary(x); }
  virtual void visit(const logical_and* x) { match_binary(x); }
  virtual void visit(const logical_or* x) { match_binary(x); }
  virtual void visit(const shift_left* x) { match_binary(x); }
  virtual void visit(const shift_right* x) { match_binary(x); }

  virtual void visit(const class select* x) {
    if (!match) return;
    const class select* se = e.as<class select>();
    if (!se) return fail();

    match = try_match(se->condition, x->condition) && try_match(se->true_value, x->true_value) &&
            try_match(se->false_value, x->false_value);
  }

  virtual void visit(const load_buffer_meta* x) {
    if (!match) return;

    const load_buffer_meta* lbme = e.as<load_buffer_meta>();
    if (!lbme) return fail();

    match = x->meta == lbme->meta && try_match(lbme->buffer, x->buffer) && try_match(lbme->dim, x->dim);
  }

  virtual void visit(const call* x) {
    if (!match) return;
    const call* c = e.as<call>();
    if (!c) return fail();
    match = c->intrinsic == x->intrinsic && try_match(c->args, x->args);
  }

  virtual void visit(const let_stmt* x) { visit_let(x); }

  virtual void visit(const block* x) {
    if (!match) return;
    const block* bs = s.as<block>();
    if (!bs) return fail();

    match = try_match(bs->a, x->a) && try_match(bs->b, x->b);
  }

  virtual void visit(const loop* x) {
    if (!match) return;
    const loop* ls = s.as<loop>();
    if (!ls) return fail();

    match = ls->name == x->name && try_match(ls->begin, x->begin) && try_match(ls->end, x->end) &&
            try_match(ls->body, x->body);
  }

  virtual void visit(const if_then_else* x) {
    if (!match) return;
    const if_then_else* is = s.as<if_then_else>();
    if (!is) return fail();

    match = try_match(is->condition, x->condition) && try_match(is->true_body, x->true_body) &&
            try_match(is->false_body, x->false_body);
  }

  virtual void visit(const call_func* x) {
    if (!match) return;
    const call_func* cs = s.as<call_func>();
    if (!cs) return fail();

    match = cs->fn != x->fn && try_match(cs->scalar_args, x->scalar_args) && try_match(cs->buffer_args, x->buffer_args);
    // TODO(https://github.com/dsharlet/slinky/issues/11): How to compare callable?
  }

  virtual void visit(const allocate* x) {
    if (!match) return;
    const allocate* as = s.as<allocate>();
    if (!as) return fail();

    match = as->name == x->name && as->elem_size == x->elem_size && try_match(as->dims, x->dims) &&
            try_match(as->body, x->body);
  }

  virtual void visit(const make_buffer* x) {
    if (!match) return;
    const make_buffer* mbs = s.as<make_buffer>();
    if (!mbs) return fail();

    match = mbs->name == x->name && mbs->elem_size == x->elem_size && try_match(mbs->base, x->base) &&
            try_match(mbs->dims, x->dims) && try_match(mbs->body, x->body);
  }

  virtual void visit(const crop_buffer* x) {
    if (!match) return;
    const crop_buffer* cbs = s.as<crop_buffer>();
    if (!cbs) return fail();

    match = cbs->name == x->name && try_match(cbs->bounds, x->bounds) && try_match(cbs->body, x->body);
  }

  virtual void visit(const crop_dim* x) {
    if (!match) return;
    const crop_dim* cds = s.as<crop_dim>();
    if (!cds) return fail();

    match = cds->name == x->name && cds->dim == x->dim && try_match(cds->min, x->min) &&
            try_match(cds->extent, x->extent) && try_match(cds->body, x->body);
  }

  virtual void visit(const check* x) {
    if (!match) return;
    const check* cs = s.as<check>();
    if (!cs) return fail();

    match = try_match(cs->condition, x->condition);
  }
};

bool match(const expr& p, const expr& e, std::map<symbol_id, expr>& matches) {
  matcher m(e, stmt(), &matches);
  p.accept(&m);
  return m.match;
}

bool match(const expr& a, const expr& b) {
  matcher m(a, stmt());
  b.accept(&m);
  return m.match;
}

bool match(const stmt& a, const stmt& b) {
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
      e = v;
    } else if (v->name == target_var) {
      e = replacement;
    } else {
      auto i = replacements.find(v->name);
      if (i != replacements.end()) {
        e = i->second;
      } else {
        e = v;
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

}  // namespace slinky
