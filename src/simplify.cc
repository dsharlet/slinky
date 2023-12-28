#include "simplify.h"

#include "print.h"
#include <cassert>
#include <iostream>
#include <limits>

#include "evaluate.h"
#include "node_mutator.h"
#include "substitute.h"

namespace slinky {

namespace {

expr x = variable::make(0);
expr y = variable::make(1);
expr z = variable::make(2);

expr c0 = wildcard::make(10, as_constant);
expr c1 = wildcard::make(11, as_constant);

// Check if a and b are out of (canonical) order.
bool should_commute(const expr& a, const expr& b) {
  auto order = [](node_type t) {
    switch (t) {
    case node_type::call: return 200;
    case node_type::constant: return 100;
    case node_type::wildcard: return 99;
    case node_type::variable: return 0;
    default: return 1;
    }
  };
  int ra = order(a.e->type);
  int rb = order(b.e->type);
  if (ra > rb) return true;

  const call* ca = a.as<call>();
  const call* cb = b.as<call>();
  if (ca && cb) {
    if (ca->intrinsic > cb->intrinsic) return true;
  }

  return false;
}

// Rules that are not in canonical order are unnecessary or may cause infinite recursion.
// This visitor checks that all commutable operations are in canonical order.
class assert_canonical : public recursive_node_visitor {
public:
  template <typename T>
  void check(const T* x) {
    if (should_commute(x->a, x->b)) {
      std::cerr << "Non-canonical operands: " << expr(x) << std::endl;
      std::abort();
    }
    x->a.accept(this);
    x->b.accept(this);
  }

  void visit(const add* op) { check(op); };
  void visit(const mul* op) { check(op); };
  void visit(const class min* op) { check(op); };
  void visit(const class max* op) { check(op); };
  void visit(const equal* op) { check(op); };
  void visit(const not_equal* op) { check(op); };
  void visit(const logical_and* op) { check(op); };
  void visit(const logical_or* op) { check(op); };
  void visit(const bitwise_and* op) { check(op); };
  void visit(const bitwise_or* op) { check(op); };
};

struct rule {
  expr pattern;
  expr replacement;
  expr predicate;

  rule(expr p, expr r, expr pr = expr()) : pattern(std::move(p)), replacement(std::move(r)), predicate(std::move(pr)) {
    assert_canonical v;
    pattern.accept(&v);
    replacement.accept(&v);
  }
};

expr buffer_min(expr buf, expr dim) { return load_buffer_meta::make(std::move(buf), buffer_meta::min, std::move(dim)); }
expr buffer_max(expr buf, expr dim) { return load_buffer_meta::make(std::move(buf), buffer_meta::max, std::move(dim)); }
expr buffer_extent(expr buf, expr dim) {
  return load_buffer_meta::make(std::move(buf), buffer_meta::extent, std::move(dim));
}

class simplifier : public node_mutator {
  symbol_map<int> references;
  symbol_map<box> buffer_bounds;
  symbol_map<interval_expr> expr_bounds;

public:
  simplifier() {}

  std::optional<bool> can_prove(expr c) {
    c = mutate(c);

    interval_expr bounds = bounds_of(c, expr_bounds);
    if (is_true(mutate(bounds.min))) {
      return true;
    } else if (is_false(mutate(bounds.max))) {
      return false;
    } else {
      return std::optional<bool>();
    }
  }

  bool can_prove_true(const expr& c) { 
    std::optional<bool> p = can_prove(c);
    return p && *p;
  }

  expr apply_rules(const std::vector<rule>& rules, expr x) {
    for (const rule& r : rules) {
      std::map<symbol_id, expr> matches;
      if (match(r.pattern, x, matches)) {
        if (!r.predicate.defined() || can_prove_true(substitute(r.predicate, matches))) {
          //std::cout << x << " " << r.pattern << " -> " << r.replacement << std::endl;
          x = substitute(r.replacement, matches);
          x = mutate(x);
          return x;
        }
      }
    }
    return x;
  }

  void visit(const variable* op) override {
    auto& ref_count = references[op->name];
    if (!ref_count) {
      ref_count = 1;
    } else {
      *ref_count += 1;
    }
    e = op;
  }

  void visit(const class min* op) override {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    if (should_commute(a, b)) {
      std::swap(a, b);
    }
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);
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
        // Constant simplifications
        {min(x, indeterminate()), indeterminate()},
        {min(x, std::numeric_limits<index_t>::max()), x},
        {min(x, positive_infinity()), x},
        {min(x, std::numeric_limits<index_t>::min()), std::numeric_limits<index_t>::min()},
        {min(x, negative_infinity()), negative_infinity()},
        {min(min(x, c0), c1), min(x, min(c0, c1))},
        {min(x, x + c0), x, c0 > 0},
        {min(x, x + c0), x + c0, c0 < 0},
        {min(x + c0, y + c1), min(x, y + c1 - c0) + c0},
        {min(x + c0, c1), min(x, c1 - c0) + c0},

        // Algebraic simplifications
        {min(x, x), x},
        {min(x, min(x, y)), min(x, y)},
        {min(x / z, y / z), min(x, y) / z, z > 0},
        {min(x + z, y + z), z + min(x, y)},
        {min(x + z, z + y), z + min(x, y)},
        {min(z + x, z + y), z + min(x, y)},
        {min(z + x, y + z), z + min(x, y)},
        {min(x - z, y - z), min(x, y) - z},
        {min(z - x, z - y), z - max(x, y)},

        // Buffer meta simplifications
        {min(buffer_min(x, y), buffer_max(x, y)), buffer_min(x, y)},
        {min(buffer_max(x, y), buffer_min(x, y)), buffer_min(x, y)},
        {min(buffer_max(x, y) + c0, buffer_min(x, y)), buffer_min(x, y), c0 > 0},
        {min(buffer_min(x, y) , buffer_max(x, y) + c0), buffer_min(x, y), c0 > 0},
        {min(buffer_min(x, y) + c0, buffer_max(x, y)), buffer_min(x, y) + c0, c0 < 0},
        {min(buffer_max(x, y), buffer_min(x, y) + c0), buffer_min(x, y) + c0, c0 < 0},
    };
    e = apply_rules(rules, e);
  }

  void visit(const class max* op) override {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    if (should_commute(a, b)) {
      std::swap(a, b);
    }
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);
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
        // Constant simplifications
        {max(x, indeterminate()), indeterminate()},
        {max(x, std::numeric_limits<index_t>::min()), x},
        {max(x, negative_infinity()), x},
        {max(x, std::numeric_limits<index_t>::max()), std::numeric_limits<index_t>::max()},
        {max(x, positive_infinity()), positive_infinity()},
        {max(max(x, c0), c1), max(x, max(c0, c1))},
        {max(x, x + c0), x + c0, c0 > 0},
        {max(x, x + c0), x, c0 < 0},
        {max(x + c0, y + c1), max(x, y + c1 - c0) + c0},
        {max(x + c0, c1), max(x, c1 - c0) + c0},

        // Algebraic simplifications
        {max(x, x), x},
        {max(x, max(x, y)), max(x, y)},
        {max(x / z, y / z), max(x, y) / z, z > 0},
        {max(x + z, y + z), z + max(x, y)},
        {max(x + z, z + y), z + max(x, y)},
        {max(z + x, z + y), z + max(x, y)},
        {max(z + x, y + z), z + max(x, y)},
        {max(x - z, y - z), max(x, y) - z},
        {max(z - x, z - y), z - min(x, y)},

        // Buffer meta simplifications
        {max(buffer_min(x, y), buffer_max(x, y)), buffer_max(x, y)},
        {max(buffer_max(x, y), buffer_min(x, y)), buffer_max(x, y)},
        {max(buffer_max(x, y) + c0, buffer_min(x, y)), buffer_max(x, y) + c0, c0 > 0},
        {max(buffer_min(x, y), buffer_max(x, y) + c0), buffer_max(x, y) + c0, c0 > 0},
        {max(buffer_min(x, y) + c0, buffer_max(x, y)), buffer_max(x, y), c0 < 0},
        {max(buffer_max(x, y), buffer_min(x, y) + c0), buffer_max(x, y), c0 < 0},
    };
    e = apply_rules(rules, e);
  }

  void visit(const add* op) override {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    if (should_commute(a, b)) {
      std::swap(a, b);
    }
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);
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
        {x + indeterminate(), indeterminate()},
        {positive_infinity() + indeterminate(), indeterminate()},
        {negative_infinity() + positive_infinity(), indeterminate()},
        {c0 + positive_infinity(), positive_infinity()},
        {c0 + negative_infinity(), negative_infinity()},
        {x + 0, x},
        {x + x, x * 2},
        {(x + c0) + c1, x + (c0 + c1)},
        {(c0 - x) + c1, (c0 + c1) - x},
        {x + (c0 - y), (x - y) + c0},
        {x + (y + c0), (x + y) + c0},
        {(x + c0) - y, (x - y) + c0},
        {(x + c0) + c1, x + (c0 + c1)},
        {(x + c0) + (y + c1), (x + y) + (c0 + c1)},
        {buffer_min(x, y) + buffer_extent(x, y), buffer_max(x, y) + 1},
        {buffer_extent(x, y) + buffer_min(x, y), buffer_max(x, y) + 1},
        {(z - buffer_max(x, y)) + buffer_min(x, y), (z - buffer_extent(x, y)) + 1},
        {buffer_min(x, y) + (z - buffer_max(x, y)), (z - buffer_extent(x, y)) + 1},
        {(z - buffer_min(x, y)) + buffer_max(x, y), (z + buffer_extent(x, y)) - 1},
        {buffer_max(x, y) + (z - buffer_min(x, y)), (z + buffer_extent(x, y)) - 1},
    };
    e = apply_rules(rules, e);
  }

  void visit(const sub* op) override {
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
        {x - indeterminate(), indeterminate()},
        {indeterminate() - x, indeterminate()},
        {positive_infinity() - positive_infinity(), indeterminate()},
        {positive_infinity() - negative_infinity(), positive_infinity()},
        {negative_infinity() - negative_infinity(), indeterminate()},
        {negative_infinity() - positive_infinity(), negative_infinity()},
        {c0 - positive_infinity(), negative_infinity()},
        {c0 - negative_infinity(), positive_infinity()},
        {x - x, 0},
        {x - 0, x},
        {x - (c0 - y), (x + y) - c0},
        {c0 - (x - y), (y - x) + c0},
        {x - (y + c0), (x - y) - c0},
        {(c0 - x) - y, c0 - (x + y)},
        {(x + c0) - y, (x - y) + c0},
        {(x + y) - x, y},
        {(y + x) - x, y},
        {x - (x + y), -y},
        {x - (y + x), -y},
        {(c0 - x) - (y - z), ((z - x) - y) + c0},
        {(x + c0) - (y + c1), (x - y) + (c0 - c1)},
        {buffer_max(x, y) - buffer_min(x, y), buffer_extent(x, y) - 1},
        {buffer_max(x, y) - (z + buffer_min(x, y)), (buffer_extent(x, y) - z) - 1},
        {(z + buffer_max(x, y)) - buffer_min(x, y), (z + buffer_extent(x, y)) - 1},
    };
    e = apply_rules(rules, e);
  }

  void visit(const mul* op) override {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    if (should_commute(a, b)) {
      std::swap(a, b);
    }
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);
    if (ca && cb) {
      e = *ca * *cb;
      return;
    }
    if (a.same_as(op->a) && b.same_as(op->b)) {
      e = op;
    } else {
      e = a * b;
    }

    static std::vector<rule> rules = {
        {x * indeterminate(), indeterminate()},
        {positive_infinity() * positive_infinity(), positive_infinity()},
        {negative_infinity() * positive_infinity(), negative_infinity()},
        {negative_infinity() * negative_infinity(), positive_infinity()},
        {c0 * positive_infinity(), positive_infinity(), c0 > 0},
        {c0 * negative_infinity(), negative_infinity(), c0 > 0},
        {c0 * positive_infinity(), negative_infinity(), c0 < 0},
        {c0 * negative_infinity(), positive_infinity(), c0 < 0},
        {x * 0, 0},
        {(x * c0) * c1, x * (c0 * c1)},
        {(x + c0) * c1, x * c1 + c0 * c1},
        {(c0 - x) * c1, c0 * c1 - x * c1},
        {x * 1, x},
    };
    e = apply_rules(rules, e);
  }

  void visit(const div* op) override {
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
        {x / indeterminate(), indeterminate()},
        {indeterminate() / x, indeterminate()},
        {positive_infinity() / positive_infinity(), indeterminate()},
        {positive_infinity() / negative_infinity(), indeterminate()},
        {negative_infinity() / positive_infinity(), indeterminate()},
        {negative_infinity() / negative_infinity(), indeterminate()},
        {c0 / positive_infinity(), 0},
        {c0 / negative_infinity(), 0},
        {positive_infinity() / c0, positive_infinity(), c0 > 0},
        {negative_infinity() / c0, negative_infinity(), c0 > 0},
        {positive_infinity() / c0, negative_infinity(), c0 < 0},
        {negative_infinity() / c0, positive_infinity(), c0 < 0},
        {x / 1, x},
    };
    e = apply_rules(rules, e);
  }

  void visit(const mod* op) override {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);
    if (ca && cb) {
      e = euclidean_mod(*ca, *cb);
      return;
    }
    if (a.same_as(op->a) && b.same_as(op->b)) {
      e = op;
    } else {
      e = a % b;
    }

    static std::vector<rule> rules = {
        {x % 1, 0}, {x % x, 0},  // We define x % 0 to be 0.
    };
    e = apply_rules(rules, e);
  }

  void visit(const less* op) override {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);
    if (ca && cb) {
      e = *ca < *cb;
      return;
    }

    if (cb) {
      if (a.same_as(op->a) && b.same_as(op->b)) {
        e = op;
      } else {
        e = a < b;
      }
    } else if (ca) {
      e = mutate(-b < -*ca);
    } else {
      e = mutate(a - b < 0);
    }

    static std::vector<rule> rules = {
        {positive_infinity() < c0, false},
        {negative_infinity() < c0, true},
        {x + c0 < c1, x < c1 - c0},
        {c0 - x < c1, -x < c1 - c0, c0 != 0},
        {buffer_extent(x, y) < c0, false, c0 < 0},
        {-buffer_extent(x, y) < c0, true, -c0 < 0},
    };
    e = apply_rules(rules, e);
  }

  void visit(const less_equal* op) override {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);
    if (ca && cb) {
      e = *ca <= *cb;
      return;
    }

    if (cb) {
      if (a.same_as(op->a) && b.same_as(op->b)) {
        e = op;
      } else {
        e = a <= b;
      }
    } else if (ca) {
      e = mutate(-b <= -*ca);
    } else {
      e = mutate(a - b <= 0);
    }

    static std::vector<rule> rules = {
        {positive_infinity() <= c0, false},
        {negative_infinity() <= c0, true},
        {x + c0 <= c1, x <= c1 - c0},
        {c0 - x <= c1, -x <= c1 - c0, c0 != 0},
        {buffer_extent(x, y) <= c0, false, c0 <= 0},
        {-buffer_extent(x, y) <= c0, true, -c0 <= 0},
    };
    e = apply_rules(rules, e);
  }

  void visit(const equal* op) override {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    if (should_commute(a, b)) {
      std::swap(a, b);
    }
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);
    if (ca && cb) {
      e = *ca == *cb;
      return;
    }
    // Canonicalize to other == constant
    if (cb) {
      if (a.same_as(op->a) && b.same_as(op->b)) {
        e = op;
      } else {
        e = a == b;
      }
    } else {
      e = mutate(a - b == 0);
    }

    static std::vector<rule> rules = {
        {x + c0 == c1, x == c1 - c0},
        {c0 - x == c1, -x == c1 - c0, c0 != 0},
    };
    e = apply_rules(rules, e);
  }

  void visit(const not_equal* op) override {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    if (should_commute(a, b)) {
      std::swap(a, b);
    }
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);
    if (ca && cb) {
      e = *ca != *cb;
      return;
    }
    // Canonicalize to other == constant
    if (cb) {
      if (a.same_as(op->a) && b.same_as(op->b)) {
        e = op;
      } else {
        e = a != b;
      }
    } else {
      e = mutate(a - b != 0);
    }

    static std::vector<rule> rules = {
        {x + c0 != c1, x != c1 - c0},
        {c0 - x != c1, -x != c1 - c0, c0 != 0},
    };
    e = apply_rules(rules, e);
  }

  void visit(const class select* op) override {
    expr c = mutate(op->condition);
    std::optional<bool> const_c = can_prove(c);
    if (const_c) {
      if (*const_c) {
        e = mutate(op->true_value);
      } else {
        e = mutate(op->false_value);
      }
      return;
    }

    expr t = mutate(op->true_value);
    expr f = mutate(op->false_value);
    if (match(t, f)) {
      e = t;
    } else if (c.same_as(op->condition) && t.same_as(op->true_value) && f.same_as(op->false_value)) {
      e = op;
    } else {
      e = select::make(std::move(c), std::move(t), std::move(f));
    }
  }

  void visit(const logical_and* op) override {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    if (should_commute(a, b)) {
      std::swap(a, b);
    }
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);

    if (ca && cb) {
      e = *ca != 0 && *cb != 0;
      return;
    } else if (cb && *cb == 0) {
      e = b;
      return;
    }

    if (a.same_as(op->a) && b.same_as(op->b)) {
      e = op;
    } else {
      e = a && b;
    }

    static std::vector<rule> rules = {
        {x && x, x},
    };
    e = apply_rules(rules, e);
  }

  void visit(const logical_or* op) override {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    if (should_commute(a, b)) {
      std::swap(a, b);
    }
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);

    if (ca && cb) {
      e = *ca != 0 || *cb != 0;
      return;
    } else if (cb && *cb == 1) {
      e = b;
      return;
    }

    if (a.same_as(op->a) && b.same_as(op->b)) {
      e = op;
    } else {
      e = a || b;
    }

    static std::vector<rule> rules = {
        {x || x, x},
    };
    e = apply_rules(rules, e);
  }

  static bool can_evaluate(intrinsic fn) {
    switch (fn) {
    case intrinsic::abs: return true;
    default: return false;
    }
  }

  void visit(const call* op) override {
    std::vector<expr> args;
    args.reserve(op->args.size());
    bool changed = false;
    bool constant = true;
    for (const expr& i : op->args) {
      expr new_i = mutate(i);
      constant = constant && as_constant(new_i);
      changed = changed || !new_i.same_as(i);
      args.push_back(new_i);
    }

    if (changed) {
      e = call::make(op->intrinsic, std::move(args));
    } else {
      e = op;
    }

    if (can_evaluate(op->intrinsic) && constant) {
      e = evaluate(e);
      return;
    }

    static std::vector<rule> rules = { 
        {abs(negative_infinity()), positive_infinity()},
        {abs(-x), abs(x)},
        {abs(abs(x)), abs(x)},
    };
    e = apply_rules(rules, e);
  }

  template <typename T>
  auto visit_let(const T* op) {
    expr value = mutate(op->value);
    auto set_bounds = set_value_in_scope(expr_bounds, op->name, bounds_of(value, expr_bounds));

    auto ref_count = set_value_in_scope(references, op->name, 0);
    auto body = mutate(op->body);

    int refs = *references[op->name];
    if (refs == 0) {
      // This let is dead
      return body;
    } else if (refs == 1 || value.as<constant>() || value.as<variable>()) {
      return mutate(substitute(body, {{op->name, value}}));
    } else if (value.same_as(op->value) && body.same_as(op->body)) {
      return decltype(body){op};
    } else {
      return T::make(op->name, std::move(value), std::move(body));
    }
  }

  void visit(const let* op) override { e = visit_let(op); }
  void visit(const let_stmt* op) override { s = visit_let(op); }

  void visit(const loop* op) override {
    expr begin = mutate(op->begin);
    expr end = mutate(op->end);

    // TODO(https://github.com/dsharlet/slinky/issues/9): We can't actually simplify anything using this yet.
    auto set_bounds = set_value_in_scope(expr_bounds, op->name, range(begin, end));
    stmt body = mutate(op->body);

    if (begin.same_as(op->begin) && end.same_as(op->end) && body.same_as(op->body)) {
      s = op;
    } else {
      s = loop::make(op->name, std::move(begin), std::move(end), std::move(body));
    }
  }

  void visit(const if_then_else* op) override {
    expr c = mutate(op->condition);
    std::optional<bool> const_c = can_prove(c);
    if (const_c) {
      if (*const_c) {
        s = mutate(op->true_body);
      } else {
        s = op->false_body.defined() ? mutate(op->false_body) : stmt();
      }
      return;
    }

    stmt t = mutate(op->true_body);
    stmt f = mutate(op->false_body);
    if (f.defined() && match(t, f)) {
      s = t;
    } else if (c.same_as(op->condition) && t.same_as(op->true_body) && f.same_as(op->false_body)) {
      s = op;
    } else {
      s = if_then_else::make(std::move(c), std::move(t), std::move(f));
    }
  }

  void visit(const block* op) override {
    stmt a = mutate(op->a);
    stmt b = mutate(op->b);

    const if_then_else* a_if = a.as<if_then_else>();
    const if_then_else* b_if = b.as<if_then_else>();

    if (a_if && b_if && match(a_if->condition, b_if->condition)) {
      stmt true_body = mutate(block::make({a_if->true_body, b_if->true_body}));
      stmt false_body = mutate(block::make({a_if->false_body, b_if->false_body}));
      s = if_then_else::make(a_if->condition, true_body, false_body);
    } else if (!a.defined() && !b.defined()) {
      s = stmt();
    } else if (!a.defined()) {
      s = b;
    } else if (!b.defined()) {
      s = a;
    } else if (a.same_as(op->a) && b.same_as(op->b)) {
      s = op;
    } else {
      s = block::make(std::move(a), std::move(b));
    }
  }

  void visit(const allocate* op) override {
    std::vector<dim_expr> dims;
    box bounds;
    dims.reserve(op->dims.size());
    for (const dim_expr& i : op->dims) {
      interval_expr bounds_i = {mutate(i.bounds.min), mutate(i.bounds.max)};
      dims.emplace_back(bounds_i, mutate(i.stride_bytes), mutate(i.fold_factor));
      bounds.push_back(bounds_i);
    }
    auto set_bounds = set_value_in_scope(buffer_bounds, op->name, std::move(bounds));
    stmt body = mutate(op->body);
    s = allocate::make(op->type, op->name, op->elem_size, std::move(dims), std::move(body));
  }

  void visit(const make_buffer* op) override {
    expr base = mutate(op->base);
    std::vector<dim_expr> dims;
    box bounds;
    dims.reserve(op->dims.size());
    for (const dim_expr& i : op->dims) {
      interval_expr bounds_i = {mutate(i.bounds.min), mutate(i.bounds.max)};
      dims.emplace_back(bounds_i, mutate(i.stride_bytes), mutate(i.fold_factor));
      bounds.push_back(bounds_i);
    }
    auto set_bounds = set_value_in_scope(buffer_bounds, op->name, std::move(bounds));
    stmt body = mutate(op->body);
    s = make_buffer::make(op->name, std::move(base), op->elem_size, std::move(dims), std::move(body));
  }

  void visit(const crop_buffer* op) override {
    // This is the bounds of the buffer as we understand them, for simplifying the inner scope.
    box bounds(op->bounds.size());
    // This is the new bounds of the crop operation. Crops that are no-ops become undefined here.
    box new_bounds(op->bounds.size());

    // If possible, rewrite crop_buffer of one dimension to crop_dim.
    std::optional<box> prev_bounds = buffer_bounds[op->name];
    int dims_count = 0;
    bool changed = false;
    for (int i = 0; i < static_cast<int>(op->bounds.size()); ++i) {
      expr min = mutate(op->bounds[i].min);
      expr max = mutate(op->bounds[i].max);
      bounds[i] = {min, max};
      changed = changed || !bounds[i].same_as(op->bounds[i]);

      // If the new bounds are the same as the existing bounds, set the crop in this dimension to
      // be undefined.
      if (prev_bounds && i < prev_bounds->size()) {
        if (can_prove_true(min == (*prev_bounds)[i].min) && can_prove_true(max == (*prev_bounds)[i].max)) {
          min = expr();
          max = expr();
        }
      }
      new_bounds[i] = {min, max};
      dims_count += min.defined() && max.defined() ? 1 : 0;
    }

    auto set_bounds = set_value_in_scope(buffer_bounds, op->name, bounds);
    stmt body = mutate(op->body);

    // Remove trailing undefined bounds.
    while (new_bounds.size() > 0 && !new_bounds.back().min.defined() && !new_bounds.back().max.defined()) {
      new_bounds.pop_back();
    }
    if (new_bounds.empty()) {
      // This crop was a no-op.
      s = std::move(body);
    } else if (dims_count == 1) {
      // This crop is of one dimension, replace it with crop_dim.
      // We removed undefined trailing bounds, so this must be the dim we want.
      int d = new_bounds.size() - 1;
      interval_expr& bounds_d = new_bounds[d];
      s = crop_dim::make(op->name, d, bounds_d.min, mutate(bounds_d.extent()), std::move(body));
    } else if (changed || !body.same_as(op->body)) {
      s = crop_buffer::make(op->name, std::move(new_bounds), std::move(body));
    } else {
      s = op;
    }
  }

  void visit(const crop_dim* op) override {
    expr min = mutate(op->min);
    expr extent = mutate(op->extent);

    std::optional<box> bounds = buffer_bounds[op->name];
    if (bounds && op->dim < bounds->size()) {
      interval_expr& dim = (*bounds)[op->dim];
      expr max = simplify(min + extent - 1);
      if (match(min, dim.min) && match(max, dim.max)) {
        // This crop is a no-op.
        s = mutate(op->body);
        return;
      }
      bounds->at(op->dim) = {min, max};
    }

    auto set_bounds = set_value_in_scope(buffer_bounds, op->name, bounds);
    stmt body = mutate(op->body);
    if (min.same_as(op->min) && extent.same_as(op->extent) && body.same_as(op->body)) {
      s = op;
    } else {
      s = crop_dim::make(op->name, op->dim, std::move(min), std::move(extent), std::move(body));
    }
  }

  void visit(const check* op) override {
    expr c = mutate(op->condition);
    std::optional<bool> const_c = can_prove(c);
    if (const_c) {
      if (*const_c) {
        s = stmt();
      } else {
        std::cerr << op->condition << " is statically false." << std::endl;
        std::abort();
      }
    } else if (c.same_as(op->condition)) {
      s = op;
    } else {
      s = check::make(std::move(c));
    }
  }
};

class find_bounds : public node_visitor {
  symbol_map<interval_expr> bounds;

public:
  find_bounds(const symbol_map<interval_expr>& bounds) : bounds(bounds) {}

  interval_expr result;

  template <typename T>
  void visit_variable(const T* x) {
    if (bounds.contains(x->name)) {
      result = *bounds.lookup(x->name);
    } else {
      result = {x, x};
    }
  }

  void visit(const variable* x) override { visit_variable(x); }
  void visit(const wildcard* x) override { visit_variable(x); }
  void visit(const constant* x) override { result = point(x); }

  void visit(const let* x) override {
    x->value.accept(this);
    auto s = set_value_in_scope(bounds, x->name, result);
    x->body.accept(this);
  }

  struct binary_result {
    interval_expr a;
    interval_expr b;
  };
  template <typename T>
  binary_result binary_bounds(const T* x) {
    x->a.accept(this);
    interval_expr ba = result;
    x->b.accept(this);
    return {ba, result};
  }

  template <typename T>
  static expr make(const T* x, const expr& a, const expr& b) {
    if (a.same_as(x->a) && b.same_as(x->b)) {
      return x;
    } else {
      return T::make(a, b);
    }
  }

  template <typename T>
  void visit_linear(const T* x) {
    binary_result r = binary_bounds(x);
    result = {make(x, r.a.min, r.b.min), make(x, r.a.max, r.b.max)};
  }

  void visit(const add* x) override { visit_linear(x); }
  void visit(const sub* x) override {
    binary_result r = binary_bounds(x);
    result = {make(x, r.a.min, r.b.max), make(x, r.a.max, r.b.min)};
  }

  void visit(const mul* x) override {
    binary_result r = binary_bounds(x);

    // TODO: I'm pretty sure there are cases missing here that would produce simpler bounds than the fallback cases.
    if (is_non_negative(r.a.min) && is_non_negative(r.b.min)) {
      // Both are >= 0, neither intervals flip.
      result = {make(x, r.a.min, r.b.min), make(x, r.a.max, r.b.max)};
    } else if (is_non_positive(r.a.max) && is_non_positive(r.b.max)) {
      // Both are <= 0, both intervals flip.
      result = {make(x, r.a.max, r.b.max), make(x, r.a.min, r.b.min)};
    } else if (r.b.is_single_point()) {
      if (is_non_negative(r.b.min)) {
        result = {make(x, r.a.min, r.b.min), make(x, r.a.max, r.b.min)};
      } else if (is_non_positive(r.b.min)) {
        result = {make(x, r.a.max, r.b.min), make(x, r.a.min, r.b.min)};
      } else {
        expr corners[] = {
            make(x, r.a.min, r.b.min),
            make(x, r.a.max, r.b.min),
        };
        result = {min(corners), max(corners)};
      }
    } else if (r.a.is_single_point()) {
      if (is_non_negative(r.a.min)) {
        result = {make(x, r.a.min, r.b.min), make(x, r.a.min, r.b.max)};
      } else if (is_non_positive(r.a.min)) {
        result = {make(x, r.a.min, r.b.max), make(x, r.a.min, r.b.min)};
      } else {
        expr corners[] = {
            make(x, r.a.min, r.b.min),
            make(x, r.a.min, r.b.max),
        };
        result = {min(corners), max(corners)};
      }
    } else {
      // We don't know anything. The results is the union of all 4 possible intervals.
      expr corners[] = {
          make(x, r.a.min, r.b.min),
          make(x, r.a.min, r.b.max),
          make(x, r.a.max, r.b.min),
          make(x, r.a.max, r.b.max),
      };
      result = {min(corners), max(corners)};
    }
  }
  void visit(const div* x) override {
    binary_result r = binary_bounds(x);
    // TODO: Tighten these bounds.
    result = r.a | -r.a;
  }
  void visit(const mod* x) override {
    binary_result r = binary_bounds(x);
    result = {0, max(abs(r.b.min), abs(r.b.max))};
    result &= r.a;
  }

  void visit(const class min* x) override { visit_linear(x); }
  void visit(const class max* x) override { visit_linear(x); }
  template <typename T>
  void visit_less(const T* x) {
    binary_result r = binary_bounds(x);
    // This bit of genius comes from
    // https://github.com/halide/Halide/blob/61b8d384b2b799cd47634e4a3b67aa7c7f580a46/src/Bounds.cpp#L829
    result = {make(x, r.a.max, r.b.min), make(x, r.a.min, r.b.max)};
  }
  void visit(const less* x) override { visit_less(x); }
  void visit(const less_equal* x) override { visit_less(x); }

  void visit(const equal* x) override {
    binary_result r = binary_bounds(x);
    result.min = 0;
    result.max = r.a.min <= r.b.max && r.b.min <= r.a.max;
  }
  void visit(const not_equal* x) override {
    binary_result r = binary_bounds(x);
    result.min = r.a.max < r.b.min || r.b.max < r.a.min;
    result.max = 1;
  }
  void visit(const logical_and* x) override { visit_linear(x); }
  void visit(const logical_or* x) override { visit_linear(x); }

  void visit(const bitwise_and* x) override { result = interval_expr::all(); }
  void visit(const bitwise_or* x) override { result = interval_expr::all(); }
  void visit(const bitwise_xor* x) override { result = interval_expr::all(); }
  void visit(const shift_left* x) override { result = interval_expr::all(); }
  void visit(const shift_right* x) override { result = interval_expr::all(); }

  void visit(const class select* x) override {
    x->condition.accept(this);
    interval_expr cb = result;
    x->true_value.accept(this);
    interval_expr tb = result;
    x->false_value.accept(this);
    interval_expr fb = result;

    if (is_true(cb.min)) {
      result = tb;
    } else if (is_false(cb.max)) {
      result = fb;
    } else {
      result = fb | tb;
    }
  }

  void visit(const load_buffer_meta* x) override { result = {x, x}; }

  void visit(const call* x) override {
    switch (x->intrinsic) {
    case intrinsic::abs: result = {0, x}; return;

    case intrinsic::positive_infinity:
    case intrinsic::negative_infinity:
    case intrinsic::indeterminate: 
      result = {x, x}; 
      return;
    }
  }

  void visit(const let_stmt* x) override { std::abort(); }
  void visit(const block* x) override { std::abort(); }
  void visit(const loop* x) override { std::abort(); }
  void visit(const if_then_else* x) override { std::abort(); }
  void visit(const call_func* x) override { std::abort(); }
  void visit(const allocate* x) override { std::abort(); }
  void visit(const make_buffer* x) override { std::abort(); }
  void visit(const crop_buffer* x) override { std::abort(); }
  void visit(const crop_dim* x) override { std::abort(); }
  void visit(const check* x) override { std::abort(); }
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

interval_expr bounds_of(const expr& e, const symbol_map<interval_expr>& bounds) {
  find_bounds fb(bounds);
  e.accept(&fb);
  return fb.result;
}

}  // namespace slinky
