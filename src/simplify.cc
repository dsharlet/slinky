#include "simplify.h"

#include <cassert>
#include <limits>

#include "substitute.h"
#include "node_mutator.h"
#include "evaluate.h"

namespace slinky {

namespace {

expr x = variable::make(0);
expr y = variable::make(1);
expr z = variable::make(2);

expr c0 = wildcard::make(10, as_constant);
expr c1 = wildcard::make(11, as_constant);

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
  symbol_map<int> references;

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

  void visit(const variable* op) {
    auto& ref_count = references[op->name];
    if (!ref_count) {
      ref_count = 1;
    } else {
      *ref_count += 1;
    }
    e = op;
  }

  void visit(const class min* op) {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);
    if (ca && cb) {
      e = std::min(*ca, *cb);
      return;
    }
    if (ca && !cb) {
      std::swap(a, b);
    }
    if (a.same_as(op->a) && b.same_as(op->b)) {
      e = op;
    } else {
      e = min(a, b);
    }

    static std::vector<rule> rules = {
      {min(x, std::numeric_limits<index_t>::max()), x},
      {min(x, std::numeric_limits<index_t>::min()), std::numeric_limits<index_t>::min()},
      {min(x, x), x},
      {min(x / z, y / z), min(x, y) / z, z > 0},
      {min(buffer_min(x, y), buffer_max(x, y)), buffer_min(x, y)},
    };
    e = apply_rules(rules, e);
  }

  void visit(const class max* op) {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);
    if (ca && cb) {
      e = std::max(*ca, *cb);
      return;
    }
    if (ca && !cb) {
      std::swap(a, b);
    }
    if (a.same_as(op->a) && b.same_as(op->b)) {
      e = op;
    } else {
      e = max(a, b);
    }

    static std::vector<rule> rules = {
      {max(x, std::numeric_limits<index_t>::min()), x},
      {max(x, std::numeric_limits<index_t>::max()), std::numeric_limits<index_t>::max()},
      {max(x, x), x},
      {max(x / z, y / z), max(x, y) / z, z > 0},
      {max(buffer_min(x, y), buffer_max(x, y)), buffer_max(x, y)},
    };
    e = apply_rules(rules, e);
  }
  
  void visit(const add* op) {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);
    if (ca && cb) {
      e = *ca + *cb;
      return;
    }
    if (ca && !cb) {
      std::swap(a, b);
    }
    if (a.same_as(op->a) && b.same_as(op->b)) {
      e = op;
    } else {
      e = a + b;
    }

    static std::vector<rule> rules = {
      {x + 0, x},
      {(x + c0) + c1, x + (c0 + c1)},
      {(x + c0) + (y + c1), (x + y) + (c0 + c1)},
      {buffer_min(x, y) + buffer_extent(x, y), buffer_max(x, y) + 1},
    };
    e = apply_rules(rules, e);
  }

  void visit(const sub* op) {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);
    if (ca && cb) {
      e = *ca - *cb;
      return;
    } else if (cb) {
      // Canonicalize to addition with constants.
      e = mutate(a + -*cb);
      return;
    }
    if (a.same_as(op->a) && b.same_as(op->b)) {
      e = op;
    } else {
      e = a - b;
    }

    static std::vector<rule> rules = {
      {x - x, 0},
      {x - 0, x},
      {(x + c0) - (y + c1), (x - y) + (c0 - c1)},
      {buffer_max(x, y) - buffer_min(x, y), buffer_extent(x, y) - 1},
    };
    e = apply_rules(rules, e);
  }

  void visit(const mul* op) {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);
    if (ca && cb) {
      e = *ca * *cb;
      return;
    }
    if (ca && !cb) {
      std::swap(a, b);
    }
    if (a.same_as(op->a) && b.same_as(op->b)) {
      e = op;
    } else {
      e = a * b;
    }

    static std::vector<rule> rules = {
      {x*0, 0},
      {x*1, x},
    };
    e = apply_rules(rules, e);
  }

  void visit(const div* op) {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);
    if (ca && cb) {
      e = euclidean_div(*ca, *cb);
      return;
    }
    if (a.same_as(op->a) && b.same_as(op->b)) {
      e = op;
    } else {
      e = a / b;
    }

    static std::vector<rule> rules = {
      {x/1, x},
    };
    e = apply_rules(rules, e);
  }

  void visit(const less* op) {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);
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

  template <typename T>
  auto visit_let(const T* op) {
    expr value = mutate(op->value);

    scoped_value<int> ref_count(references, op->name, 0);
    auto body = mutate(op->body);

    int refs = *references[op->name];
    if (refs == 0) {
      // This let is dead
      return body;
    } else if (refs == 1 || value.as<constant>() || value.as<variable>() || value.as<load_buffer_meta>()) {
      return mutate(substitute(body, { { op->name, value } }));
    } else if (value.same_as(op->value) && body.same_as(op->body)) {
      return decltype(body){op};
    } else {
      return T::make(op->name, std::move(value), std::move(body));
    }
  }

  void visit(const let* op) { e = visit_let(op); }
  void visit(const let_stmt* op) { s = visit_let(op); }
};

}  // namespace

expr simplify(const expr& e) { return simplifier().mutate(e); }
stmt simplify(const stmt& s) { return simplifier().mutate(s); }

bool can_prove(const expr& e) {
  expr simplified = simplify(e);
  if (const index_t* c = as_constant(simplified)) {
    return *c != 0;
  }
  return false;
}

}  // namespace slinky
