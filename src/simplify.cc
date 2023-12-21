#include "simplify.h"

#include <cassert>

#include "substitute.h"
#include "node_mutator.h"
#include "evaluate.h"

namespace slinky {

namespace {

expr x = variable::make(0);
expr y = variable::make(1);
expr z = variable::make(2);

struct rule {
  expr pattern;
  expr replacement;
  expr predicate;
};

expr buffer_min(expr buf, expr dim) {
  return load_buffer_meta::make(std::move(buf), buffer_meta::min, std::move(dim));
}
expr buffer_max(expr buf, expr dim) {
  return load_buffer_meta::make(std::move(buf), buffer_meta::max, std::move(dim));
}
expr buffer_extent(expr buf, expr dim) {
  return load_buffer_meta::make(std::move(buf), buffer_meta::extent, std::move(dim));
}

class simplifier : public node_mutator {
public:
  simplifier() {}

  expr apply_rules(const std::vector<rule>& rules, expr x) {
    for (const rule& r : rules) {
      std::map<symbol_id, expr> matches;
      if (match(r.pattern, x, matches)) {
        if (!r.predicate.defined() || can_prove(substitute(r.predicate, matches))) {
          x = substitute(r.replacement, matches);
          x = mutate(x);
          return x;
        }
      }
    }
    return x;
  }

  void visit(const class min* op) {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    auto ca = is_constant(a);
    auto cb = is_constant(b);
    if (ca && cb) {
      e = std::min(*ca, *cb);
      return;
    }
    if (a.same_as(op->a) && b.same_as(op->b)) {
      e = op;
    } else {
      e = min(a, b);
    }

    static std::vector<rule> rules = {
      {min(x, x), x},
      {min(x / z, y / z), min(x, y) / z, z > 0},
      {min(buffer_min(x, y), buffer_max(x, y)), buffer_min(x, y)},
    };
    e = apply_rules(rules, e);
  }

  void visit(const class max* op) {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    auto ca = is_constant(a);
    auto cb = is_constant(b);
    if (ca && cb) {
      e = std::max(*ca, *cb);
      return;
    }
    if (a.same_as(op->a) && b.same_as(op->b)) {
      e = op;
    } else {
      e = max(a, b);
    }

    static std::vector<rule> rules = {
      {max(x, x), x},
      {max(x / z, y / z), max(x, y) / z, z > 0},
      {max(buffer_min(x, y), buffer_max(x, y)), buffer_max(x, y)},
    };
    e = apply_rules(rules, e);
  }
  
  void visit(const add* op) {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    auto ca = is_constant(a);
    auto cb = is_constant(b);
    if (ca && cb) {
      e = *ca + *cb;
      return;
    }
    if (a.same_as(op->a) && b.same_as(op->b)) {
      e = op;
    } else {
      e = a + b;
    }

    static std::vector<rule> rules = {
      {(buffer_max(x, y) - buffer_min(x, y)) + 1, buffer_extent(x, y)},
    };
    e = apply_rules(rules, e);
  }

  void visit(const sub* op) {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    auto ca = is_constant(a);
    auto cb = is_constant(b);
    if (ca && cb) {
      e = *ca - *cb;
      return;
    }
    if (a.same_as(op->a) && b.same_as(op->b)) {
      e = op;
    } else {
      e = a - b;
    }

    static std::vector<rule> rules = {
      {x - x, 0},
      {(buffer_min(x, y) + buffer_extent(x, y)) - 1, buffer_max(x, y)},
    };
    e = apply_rules(rules, e);
  }

  void visit(const less* op) {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    auto ca = is_constant(a);
    auto cb = is_constant(b);
    if (ca && cb) {
      e = *ca < *cb;
      return;
    }
    if (a.same_as(op->a) && b.same_as(op->b)) {
      e = op;
    } else {
      e = a < b;
    }
  }
};

}  // namespace

expr simplify(const expr& e) { return simplifier().mutate(e); }
stmt simplify(const stmt& s) { return simplifier().mutate(s); }

bool can_prove(const expr& e) {
  expr simplified = simplify(e);
  if (const index_t* c = is_constant(simplified)) {
    return *c != 0;
  }
  return false;
}

}  // namespace slinky
