#include "substitute.h"

#include <cassert>
#include <iostream>

#include "node_mutator.h"
#include "print.h"

namespace slinky {

class matcher : public node_visitor {
  // In this class, we visit the pattern, and manually traverse the expression being matched.
  expr e;
  std::map<symbol_id, expr>* matches;

public:
  bool match = true;

  matcher(const expr& e, std::map<symbol_id, expr>* matches = nullptr) : e(e), matches(matches) {}

  void fail() { match = false; }

  template <typename T>
  void match_let(const T* x) {
    if (!match) return;
    const T* el = e.as<T>();
    if (!el) {
      match = false;
      return;
    }

    if (el->name != x->name) {
      match = false;
      return;
    }

    e = el->value;
    x->value.accept(this);
    if (!match) return;

    e = el->body;
    x->body.accept(this);
  }

  template <typename T>
  void match_binary(const T* x) {
    if (!match) return;
    const T* ex = e.as<T>();
    if (!ex) {
      match = false;
      return;
    }

    e = ex->a;
    x->a.accept(this);
    if (!match) return;

    e = ex->b;
    x->b.accept(this);
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
    if (!ec) {
      match = false;
      return;
    }

    match = ec->value == x->value;
  }

  virtual void visit(const let* x) { match_let(x); }
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

  virtual void visit(const load_buffer_meta* x) {
    if (!match) return;

    const load_buffer_meta* lbme = e.as<load_buffer_meta>();
    if (!lbme || x->meta != lbme->meta) {
      match = false;
      return;
    }

    e = lbme->buffer;
    x->buffer.accept(this);
    if (!match) return;

    e = lbme->dim;
    x->dim.accept(this);
  }

  virtual void visit(const let_stmt* x) { std::abort(); }
  virtual void visit(const block* x) { std::abort(); }
  virtual void visit(const loop* x) { std::abort(); }
  virtual void visit(const if_then_else* x) { std::abort(); }
  virtual void visit(const call* x) { std::abort(); }
  virtual void visit(const allocate* x) { std::abort(); }
  virtual void visit(const make_buffer* x) { std::abort(); }
  virtual void visit(const crop_buffer* x) { std::abort(); }
  virtual void visit(const crop_dim* x) { std::abort(); }
  virtual void visit(const check* x) { std::abort(); }
};

bool match(const expr& p, const expr& e, std::map<symbol_id, expr>& matches) {
  matcher m(e, &matches);
  p.accept(&m);
  return m.match;
}

bool match(const expr& a, const expr& b) {
  matcher m(a);
  b.accept(&m);
  return m.match;
}

class substitutor : public node_mutator {
  const std::map<symbol_id, expr>& replacements;

public:
  substitutor(const std::map<symbol_id, expr>& replacements) : replacements(replacements) {}

  void visit(const variable* v) override {
    auto i = replacements.find(v->name);
    if (i != replacements.end()) {
      e = i->second;
    } else {
      e = v;
    }
  }

  void visit(const wildcard* v) override {
    auto i = replacements.find(v->name);
    if (i != replacements.end()) {
      e = i->second;
    } else {
      e = v;
    }
  }
};

expr substitute(const expr& e, const std::map<symbol_id, expr>& replacements) {
  return substitutor(replacements).mutate(e);
}

stmt substitute(const stmt& s, const std::map<symbol_id, expr>& replacements) {
  return substitutor(replacements).mutate(s);
}

}  // namespace slinky
