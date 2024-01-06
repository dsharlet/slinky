#include "simplify.h"

#include <cassert>
#include <iostream>
#include <limits>

#include "evaluate.h"
#include "node_mutator.h"
#include "print.h"
#include "substitute.h"

namespace slinky {

namespace {

expr x = variable::make(0);
expr y = variable::make(1);
expr z = variable::make(2);
expr w = variable::make(3);

expr c0 = wildcard::make(10, as_constant);
expr c1 = wildcard::make(11, as_constant);
expr c2 = wildcard::make(12, as_constant);

expr finite_x = wildcard::make(20, is_finite);

// Check if a and b are out of (canonical) order.
bool should_commute(const expr& a, const expr& b) {
  auto order = [](node_type t) {
    switch (t) {
    case node_type::constant: return 100;
    case node_type::wildcard: return 99;
    case node_type::variable: return 0;
    case node_type::call: return -1;
    default: return 1;
    }
  };
  int ra = order(a.type());
  int rb = order(b.type());
  if (ra > rb) return true;

  const call* ca = a.as<call>();
  const call* cb = b.as<call>();
  if (ca && cb) {
    if (ca->intrinsic > cb->intrinsic) return true;
  }

  return false;
}

// Rules that are not in canonical order are unnecessary or may cause infinite loops.
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
};

// We need to generate a lot of rules that are equivalent except for commutation.
// To avoid repetitive error-prone code, we can generate all the valid commutative
// equivalents of an expression.
class commute_variants : public node_visitor {
public:
  std::vector<expr> results;

  void visit(const variable* x) override { results = {x}; }
  void visit(const wildcard* x) override { results = {x}; }
  void visit(const constant* x) override { results = {x}; }
  void visit(const let* x) override { std::abort(); }

  template <typename T>
  void visit_binary(bool commutative, const T* x) {
    // TODO: I think some of these patterns are redundant, but finding them is tricky.

    x->a.accept(this);
    std::vector<expr> a = std::move(results);
    x->b.accept(this);
    std::vector<expr> b = std::move(results);

    results.clear();
    results.reserve(a.size() * b.size());
    for (const expr& i : a) {
      for (const expr& j : b) {
        if (match(i, j)) {
          results.push_back(T::make(i, j));
        } else {
          results.push_back(T::make(i, j));
          if (commutative) {
            results.push_back(T::make(j, i));
          }
        }
      }
    }
  }

  void visit(const add* x) override { visit_binary(true, x); }
  void visit(const sub* x) override { visit_binary(false, x); }
  void visit(const mul* x) override { visit_binary(true, x); }
  void visit(const div* x) override { visit_binary(false, x); }
  void visit(const mod* x) override { visit_binary(false, x); }
  void visit(const class min* x) override { visit_binary(true, x); }
  void visit(const class max* x) override { visit_binary(true, x); }
  void visit(const equal* x) override { visit_binary(true, x); }
  void visit(const not_equal* x) override { visit_binary(true, x); }
  void visit(const less* x) override { visit_binary(false, x); }
  void visit(const less_equal* x) override { visit_binary(false, x); }
  void visit(const logical_and* x) override { visit_binary(true, x); }
  void visit(const logical_or* x) override { visit_binary(true, x); }
  void visit(const logical_not* x) override {
    x->x.accept(this);
    for (expr& i : results) {
      i = !i;
    }
  }
  void visit(const class select* x) override {
    x->condition.accept(this);
    std::vector<expr> c = std::move(results);
    x->true_value.accept(this);
    std::vector<expr> t = std::move(results);
    x->false_value.accept(this);
    std::vector<expr> f = std::move(results);

    results.clear();
    results.reserve(c.size() * t.size() * f.size());
    for (const expr& i : c) {
      for (const expr& j : t) {
        for (const expr& k : f) {
          results.push_back(select::make(i, j, k));
        }
      }
    }
  }

  void visit(const call* x) override {
    if (x->args.size() == 1) {
      x->args.front().accept(this);
      for (expr& i : results) {
        i = call::make(x->intrinsic, {i});
      }
    } else {
      results = {x};
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
  void visit(const slice_buffer* x) override { std::abort(); }
  void visit(const slice_dim* x) override { std::abort(); }
  void visit(const truncate_rank* x) override { std::abort(); }
  void visit(const check* x) override { std::abort(); }
};

class rule_set {
public:
  struct rule {
    expr pattern;
    expr replacement;
    expr predicate;

    rule(expr p, expr r, expr pr = expr())
        : pattern(std::move(p)), replacement(std::move(r)), predicate(std::move(pr)) {
      assert_canonical v;
      replacement.accept(&v);
    }
  };

private:
  std::vector<rule> rules_;

public:
  rule_set(std::initializer_list<rule> rules) {
    for (const rule& i : rules) {
      commute_variants v;
      i.pattern.accept(&v);

      for (expr& p : v.results) {
        rules_.emplace_back(std::move(p), i.replacement, i.predicate);
      }
    }
  }

  expr apply(expr x) {
    // std::cerr << "apply_rules: " << x << std::endl;
    symbol_map<expr> matches;
    for (const rule& r : rules_) {
      matches.clear();
      // std::cerr << "  Considering " << r.pattern << std::endl;
      if (match(r.pattern, x, matches)) {
        // std::cerr << "  Matched:" << std::endl;
        // for (const auto& i : matches) {
        //   std::cerr << "    " << i.first << ": " << i.second << std::endl;
        // }

        if (!r.predicate.defined() || prove_true(substitute(r.predicate, matches))) {
          // std::cerr << "  Applied " << r.pattern << " -> " << r.replacement << std::endl;
          x = substitute(r.replacement, matches);
          // std::cerr << "  Result: " << x << std::endl;
          return x;
        } else {
          // std::cerr << "  Failed predicate: " << r.predicate << std::endl;
        }
      }
    }
    // std::cerr << "  Failed" << std::endl;
    return x;
  }
};

}  // namespace

expr simplify(const class min* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return std::min(*ca, *cb);
  }
  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = min::make(std::move(a), std::move(b));
  }

  static rule_set rules = {
      // Constant simplifications
      {min(x, indeterminate()), indeterminate()},
      {min(x, std::numeric_limits<index_t>::max()), x},
      {min(x, positive_infinity()), x},
      {min(x, std::numeric_limits<index_t>::min()), std::numeric_limits<index_t>::min()},
      {min(x, negative_infinity()), negative_infinity()},
      {min(min(x, c0), c1), min(x, min(c0, c1))},
      {min(x, x + c0), x, c0 > 0},
      {min(x, x + c0), x + c0, c0 < 0},
      {min(x + c0, y + c1), min(x, y + (c1 - c0)) + c0},
      {min(x + c0, c1), min(x, c1 - c0) + c0},
      {min(c0 - x, c0 - y), c0 - max(x, y)},

      // Algebraic simplifications
      {min(x, x), x},
      {min(x, max(x, y)), x},
      {min(x, min(x, y)), min(x, y)},
      {min(max(x, y), min(x, z)), min(x, z)},
      {min(min(x, y), min(x, z)), min(x, min(y, z))},
      {min(max(x, y), max(x, z)), max(x, min(y, z))},
      {min(x, min(y, x + z)), min(y, min(x, x + z))},
      {min(x, min(y, x - z)), min(y, min(x, x - z))},
      {min(x / z, y / z), min(x, y) / z, z > 0},
      {min(x / z, y / z), max(x, y) / z, z < 0},
      {min(x * z, y * z), z * min(x, y), z > 0},
      {min(x * z, y * z), z * max(x, y), z < 0},
      {min(x + z, y + z), z + min(x, y)},
      {min(x - z, y - z), min(x, y) - z},
      {min(z - x, z - y), z - max(x, y)},

      // Buffer meta simplifications
      // TODO: These rules are sketchy, they assume buffer_max(x, y) > buffer_min(x, y), which
      // is true if we disallow empty buffers...
      {min(buffer_min(x, y), buffer_max(x, y)), buffer_min(x, y)},
      {min(buffer_min(x, y), buffer_max(x, y) + c0), buffer_min(x, y), c0 > 0},
      {min(buffer_max(x, y), buffer_min(x, y) + c0), buffer_min(x, y) + c0, c0 < 0},
  };
  return rules.apply(e);
}

expr simplify(const class max* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return std::max(*ca, *cb);
  }
  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = max::make(std::move(a), std::move(b));
  }

  static rule_set rules = {
      // Constant simplifications
      {max(x, indeterminate()), indeterminate()},
      {max(x, std::numeric_limits<index_t>::min()), x},
      {max(x, negative_infinity()), x},
      {max(x, std::numeric_limits<index_t>::max()), std::numeric_limits<index_t>::max()},
      {max(x, positive_infinity()), positive_infinity()},
      {max(max(x, c0), c1), max(x, max(c0, c1))},
      {max(x, x + c0), x + c0, c0 > 0},
      {max(x, x + c0), x, c0 < 0},
      {max(x + c0, y + c1), max(x, y + (c1 - c0)) + c0},
      {max(x + c0, c1), max(x, c1 - c0) + c0},
      {max(c0 - x, c0 - y), c0 - min(x, y)},

      // Algebraic simplifications
      {max(x, x), x},
      {max(x, min(x, y)), x},
      {max(x, max(x, y)), max(x, y)},
      {max(min(x, y), max(x, z)), max(x, z)},
      {max(max(x, y), max(x, z)), max(x, max(y, z))},
      {max(min(x, y), min(x, z)), min(x, max(y, z))},
      {max(x, max(y, x + z)), max(y, max(x, x + z))},
      {max(x, max(y, x - z)), max(y, max(x, x - z))},
      {max(x / z, y / z), max(x, y) / z, z > 0},
      {max(x / z, y / z), min(x, y) / z, z < 0},
      {max(x * z, y * z), z * max(x, y), z > 0},
      {max(x * z, y * z), z * min(x, y), z < 0},
      {max(x + z, y + z), z + max(x, y)},
      {max(x - z, y - z), max(x, y) - z},
      {max(z - x, z - y), z - min(x, y)},

      // Buffer meta simplifications
      {max(buffer_min(x, y), buffer_max(x, y)), buffer_max(x, y)},
      {max(buffer_min(x, y), buffer_max(x, y) + c0), buffer_max(x, y) + c0, c0 > 0},
      {max(buffer_max(x, y), buffer_min(x, y) + c0), buffer_max(x, y), c0 < 0},
  };
  return rules.apply(e);
}

expr simplify(const add* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return *ca + *cb;
  }
  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = add::make(std::move(a), std::move(b));
  }

  static rule_set rules = {
      {x + indeterminate(), indeterminate()},
      {positive_infinity() + indeterminate(), indeterminate()},
      {negative_infinity() + positive_infinity(), indeterminate()},
      {finite_x + positive_infinity(), positive_infinity()},
      {finite_x + negative_infinity(), negative_infinity()},
      {x + 0, x},
      {x + x, x * 2},
      {x + (x + y), y + x * 2},
      {x + (x - y), x * 2 - y},
      {x + (y - x), y},
      //{x + x * y, x * (y + 1)},  // Needs x to be non-constant or it loops with c0 * (x + c1) -> c0 * x + c0 * c1... how?
      {x * y + x * z, x * (y + z)},

      {(x + c0) + c1, x + (c0 + c1)},
      {(c0 - x) + c1, (c0 + c1) - x},
      {x + (c0 - y), (x - y) + c0},
      {x + (y + c0), (x + y) + c0},
      {(x + c0) - y, (x - y) + c0},
      {(x + c0) + (y + c1), (x + y) + (c0 + c1)},

      {min(x + c0, y + c1) + c2, min(x + (c0 + c2), y + (c1 + c2))},
      {max(x + c0, y + c1) + c2, max(x + (c0 + c2), y + (c1 + c2))},
      {min(c0 - x, y + c1) + c2, min((c0 + c2) - x, y + (c1 + c2))},
      {max(c0 - x, y + c1) + c2, max((c0 + c2) - x, y + (c1 + c2))},
      {min(c0 - x, c1 - y) + c2, min((c0 + c2) - x, (c1 + c2) - y)},
      {max(c0 - x, c1 - y) + c2, max((c0 + c2) - x, (c1 + c2) - y)},

      {select(x, c0, c1) + c2, select(x, c0 + c2, c1 + c2)},
      {select(x, y + c0, c1) + c2, select(x, y + (c0 + c2), c1 + c2)},
      {select(x, c0 - y, c1) + c2, select(x, (c0 + c2) - y, c1 + c2)},
      {select(x, c0, y + c1) + c2, select(x, c0 + c2, y + (c1 + c2))},
      {select(x, c0, c1 - y) + c2, select(x, c0 + c2, (c1 + c2) - y)},
      {select(x, y + c0, z + c1) + c2, select(x, y + (c0 + c2), z + (c1 + c2))},
      {select(x, c0 - y, z + c1) + c2, select(x, (c0 + c2) - y, z + (c1 + c2))},
      {select(x, y + c0, c1 - z) + c2, select(x, y + (c0 + c2), (c1 + c2) - z)},
      {select(x, c0 - y, c1 - z) + c2, select(x, (c0 + c2) - y, (c1 + c2) - z)},

      {buffer_min(x, y) + buffer_extent(x, y), buffer_max(x, y) + 1},
      {buffer_min(x, y) + (z - buffer_max(x, y)), (z - buffer_extent(x, y)) + 1},
      {buffer_max(x, y) + (z - buffer_min(x, y)), (buffer_extent(x, y) + z) + -1},
  };
  return rules.apply(e);
}

expr simplify(const sub* op, expr a, expr b) {
  assert(a.defined());
  assert(b.defined());
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return *ca - *cb;
  } else if (cb) {
    // Canonicalize to addition with constants.
    return simplify(static_cast<add*>(nullptr), a, -*cb);
  }

  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = sub::make(std::move(a), std::move(b));
  }

  static rule_set rules = {
      {x - indeterminate(), indeterminate()},
      {indeterminate() - x, indeterminate()},
      {positive_infinity() - positive_infinity(), indeterminate()},
      {positive_infinity() - negative_infinity(), positive_infinity()},
      {negative_infinity() - negative_infinity(), indeterminate()},
      {negative_infinity() - positive_infinity(), negative_infinity()},
      {finite_x - positive_infinity(), negative_infinity()},
      {finite_x - negative_infinity(), positive_infinity()},
      {x - x, 0},
      {x - 0, x},
      {x - (c0 - y), (x + y) - c0},
      {c0 - (x - y), (y - x) + c0},
      {x - (y + c0), (x - y) - c0},
      {(c0 - x) - y, c0 - (x + y)},
      {(x + c0) - y, (x - y) + c0},
      {(x + y) - x, y},
      {x - (x + y), -y},
      {(c0 - x) - (y - z), ((z - x) - y) + c0},
      {(x + c0) - (y + c1), (x - y) + (c0 - c1)},

      {c2 - select(x, c0, c1), select(x, c2 - c0, c2 - c1)},
      {c2 - select(x, y + c0, c1), select(x, (c2 - c0) - y, c2 - c1)},
      {c2 - select(x, c0 - y, c1), select(x, y + (c2 - c0), c2 - c1)},
      {c2 - select(x, c0, y + c1), select(x, c2 - c0, (c2 - c1) - y)},
      {c2 - select(x, c0, c1 - y), select(x, c2 - c0, y + (c2 - c1))},
      {c2 - select(x, y + c0, z + c1), select(x, (c2 - c0) - y, (c2 - c1) - z)},
      {c2 - select(x, c0 - y, z + c1), select(x, y + (c2 - c0), (c2 - c1) - z)},
      {c2 - select(x, y + c0, c1 - z), select(x, (c2 - c0) - y, z + (c2 - c1))},
      {c2 - select(x, c0 - y, c1 - z), select(x, y + (c2 - c0), z + (c2 - c1))},

      {buffer_max(x, y) - buffer_min(x, y), buffer_extent(x, y) + -1},
      {buffer_max(x, y) - (buffer_min(x, y) + z), (buffer_extent(x, y) - z) + -1},
      {(buffer_max(x, y) + z) - buffer_min(x, y), (buffer_extent(x, y) + z) + -1},
  };
  return rules.apply(e);
}

expr simplify(const mul* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return *ca * *cb;
  }
  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = mul::make(std::move(a), std::move(b));
  }

  static rule_set rules = {
      {x * indeterminate(), indeterminate()},
      {positive_infinity() * positive_infinity(), positive_infinity()},
      {negative_infinity() * positive_infinity(), negative_infinity()},
      {negative_infinity() * negative_infinity(), positive_infinity()},
      {c0 * positive_infinity(), positive_infinity(), c0 > 0},
      {c0 * negative_infinity(), negative_infinity(), c0 > 0},
      {c0 * positive_infinity(), negative_infinity(), c0 < 0},
      {c0 * negative_infinity(), positive_infinity(), c0 < 0},
      {x * 0, 0},
      {x * 1, x},
      {(x * c0) * c1, x * (c0 * c1)},
      {(x + c0) * c1, x * c1 + c0 * c1},
      {(0 - x) * c1, x * (-c1)},
      {(c0 - x) * c1, c0 * c1 - x * c1},
  };
  return rules.apply(e);
}

expr simplify(const div* op, expr a, expr b) {
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return euclidean_div(*ca, *cb);
  }
  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = div::make(std::move(a), std::move(b));
  }

  static rule_set rules = {
      {x / indeterminate(), indeterminate()},
      {indeterminate() / x, indeterminate()},
      {positive_infinity() / positive_infinity(), indeterminate()},
      {positive_infinity() / negative_infinity(), indeterminate()},
      {negative_infinity() / positive_infinity(), indeterminate()},
      {negative_infinity() / negative_infinity(), indeterminate()},
      {finite_x / positive_infinity(), 0},
      {finite_x / negative_infinity(), 0},
      {positive_infinity() / c0, positive_infinity(), c0 > 0},
      {negative_infinity() / c0, negative_infinity(), c0 > 0},
      {positive_infinity() / c0, negative_infinity(), c0 < 0},
      {negative_infinity() / c0, positive_infinity(), c0 < 0},
      {x / 0, 0},
      {0 / x, 0},
      {x / 1, x},
      {x / x, x != 0},
  };
  return rules.apply(e);
}

expr simplify(const mod* op, expr a, expr b) {
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return euclidean_mod(*ca, *cb);
  }
  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = mod::make(std::move(a), std::move(b));
  }

  static rule_set rules = {
      {x % 1, 0},
      {x % 0, 0},
      {x % x, 0},
  };
  return rules.apply(e);
}

expr simplify(const less* op, expr a, expr b) {
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return *ca < *cb;
  }

  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = less::make(std::move(a), std::move(b));
  }

  static rule_set rules = {
      {positive_infinity() < finite_x, false},
      {negative_infinity() < finite_x, true},
      {finite_x < positive_infinity(), true},
      {finite_x < negative_infinity(), false},
      {x < x, false},
      {x + c0 < c1, x < c1 - c0},
      {x < x + y, 0 < y},
      {x + y < x, y < 0},
      {x - y < x, 0 < y},
      {0 - x < c0, -c0 < x},
      {c0 - x < c1, c0 - c1 < x},
      {c0 < c1 - x, x < c1 - c0},

      {x < x + y, 0 < y},
      {x + y < x, y < 0},
      {x < x - y, y < 0},
      {x - y < x, 0 < y},
      {x + y < x + z, y < z},
      {x - y < x - z, z < y},
      {x - y < z - y, x < z},

      {min(x, y) < x, y < x},
      {max(x, y) < x, false},
      {x < max(x, y), x < y},
      {x < min(x, y), false},
      {min(x, y) < max(x, y), x != y},
      {max(x, y) < min(x, y), false},
      {min(x, y) < min(x, z), y < min(x, z)},

      {buffer_extent(x, y) < c0, false, c0 < 0},
      {c0 < buffer_extent(x, y), true, c0 < 0},
  };
  return rules.apply(e);
}

expr simplify(const less_equal* op, expr a, expr b) {
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return *ca <= *cb;
  }

  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = less_equal::make(std::move(a), std::move(b));
  }

  static rule_set rules = {
      {positive_infinity() <= finite_x, false},
      {negative_infinity() <= finite_x, true},
      {finite_x <= positive_infinity(), true},
      {finite_x <= negative_infinity(), false},
      {x <= x, true},
      {x <= x + y, 0 <= y},
      {x + y <= x, y <= 0},
      {x - y <= x, 0 <= y},
      {0 - x <= c0, -c0 <= x},
      {c0 - x <= y, c0 <= y + x},
      {x <= c1 - y, x + y <= c1},
      {x + c0 <= y + c1, x - y <= c1 - c0},

      {x <= x + y, 0 <= y},
      {x + y <= x, y <= 0},
      {x <= x - y, y <= 0},
      {x - y <= x, 0 <= y},
      {x + y <= x + z, y <= z},
      {x - y <= x - z, z <= y},
      {x - y <= z - y, x <= z},

      {min(x, y) <= x, true},
      {max(x, y) <= x, y <= x},
      {x <= max(x, y), true},
      {x <= min(x, y), x <= y},
      {min(x, y) <= max(x, y), true},
      {max(x, y) <= min(x, y), x == y},

      {buffer_extent(x, y) <= c0, false, c0 <= 0},
      {c0 <= buffer_extent(x, y), true, c0 <= 0},
  };
  return rules.apply(e);
}

expr simplify(const equal* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return *ca == *cb;
  }

  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = equal::make(std::move(a), std::move(b));
  }

  static rule_set rules = {
      {x == x, true},
      {x + c0 == c1, x == c1 - c0},
      {c0 - x == c1, -x == c1 - c0, c0 != 0},
  };
  return rules.apply(e);
}

expr simplify(const not_equal* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);
  if (ca && cb) {
    return *ca != *cb;
  }

  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = not_equal::make(std::move(a), std::move(b));
  }

  static rule_set rules = {
      {x != x, false},
      {x + c0 != c1, x != c1 - c0},
      {c0 - x != c1, -x != c1 - c0, c0 != 0},
  };
  return rules.apply(e);
}

expr simplify(const logical_and* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);

  if (ca && cb) {
    return *ca != 0 && *cb != 0;
  } else if (cb) {
    return *cb ? a : b;
  }

  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = logical_and::make(std::move(a), std::move(b));
  }

  static rule_set rules = {
      {x && x, x},
      {x && !x, false},
      {!x && !y, !(x || y)},
      {x && (x && y), x && y},
      {x && (x || y), x},
  };
  return rules.apply(e);
}

expr simplify(const logical_or* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }
  const index_t* ca = as_constant(a);
  const index_t* cb = as_constant(b);

  if (ca && cb) {
    return *ca != 0 || *cb != 0;
  } else if (cb) {
    return *cb ? b : a;
  }

  expr e;
  if (op && a.same_as(op->a) && b.same_as(op->b)) {
    e = op;
  } else {
    e = logical_or::make(std::move(a), std::move(b));
  }

  static rule_set rules = {
      {x || x, x},
      {x || !x, true},
      {!x || !y, !(x && y)},
      {x || (x && y), x},
      {x || (x || y), x || y},
  };
  return rules.apply(e);
}

expr simplify(const logical_not* op, expr value) {
  const index_t* cv = as_constant(value);
  if (cv) {
    return *cv == 0;
  }

  expr e;
  if (op && value.same_as(op->x)) {
    e = op;
  } else {
    e = logical_not::make(std::move(value));
  }

  static rule_set rules = {
      {!!x, x},
      {!(x == y), x != y},
      {!(x != y), x == y},
      {!(x < y), y <= x},
      {!(x <= y), y < x},
  };
  return rules.apply(e);
}

expr simplify(const class select* op, expr c, expr t, expr f) {
  std::optional<bool> const_c = attempt_to_prove(c);
  if (const_c) {
    if (*const_c) {
      return op->true_value;
    } else {
      return op->false_value;
    }
  }

  expr e;
  if (match(t, f)) {
    return t;
  } else if (op && c.same_as(op->condition) && t.same_as(op->true_value) && f.same_as(op->false_value)) {
    e = op;
  } else {
    e = select::make(std::move(c), std::move(t), std::move(f));
  }
  static rule_set rules = {
      {select(!x, y, z), select(x, z, y)},

      // Pull common expressions out
      {select(x, y, y + z), y + select(x, 0, z)},
      {select(x, y + z, y), y + select(x, z, 0)},
      {select(x, y + z, y + w), y + select(x, z, w)},
      {select(x, z - y, w - y), select(x, z, w) - y},
  };
  return rules.apply(e);
}

expr simplify(const call* op, std::vector<expr> args) {
  bool constant = true;
  bool changed = false;
  assert(op->args.size() == args.size());
  for (std::size_t i = 0; i < args.size(); ++i) {
    constant = constant && as_constant(args[i]);
    changed = changed || !args[i].same_as(op->args[i]);
  }

  if (op->intrinsic == intrinsic::buffer_at) {
    // Trailing undefined indices can be removed.
    for (index_t d = 1; d < static_cast<index_t>(args.size()); ++d) {
      // buffer_at(b, buffer_min(b, 0)) is equivalent to buffer_base(b)
      if (args[d].defined() && match(args[d], buffer_min(args[0], d - 1))) {
        args[d] = expr();
        changed = true;
      }
    }
    // Trailing undefined args have no effect.
    while (args.size() > 1 && !args.back().defined()) {
      args.pop_back();
      changed = true;
    }

    if (args.size() == 1) {
      return call::make(intrinsic::buffer_base, std::move(args));
    }
  }

  expr e;
  if (op && !changed) {
    e = op;
  } else {
    e = call::make(op->intrinsic, std::move(args));
  }

  if (can_evaluate(op->intrinsic) && constant) {
    return evaluate(e);
  }

  static rule_set rules = {
      {abs(negative_infinity()), positive_infinity()},
      {abs(-x), abs(x)},
      {abs(abs(x)), abs(x)},
  };
  return rules.apply(e);
}

namespace {

// This is based on the simplifier in Halide: https://github.com/halide/Halide/blob/main/src/Simplify_Internal.h
class simplifier : public node_mutator {
  symbol_map<int> references;
  symbol_map<box_expr> buffer_bounds;
  bounds_map expr_bounds;

  interval_expr result_bounds;

  void set_result(expr e, interval_expr bounds) {
    assert(!result_bounds.min.defined() && !result_bounds.max.defined());
    result_bounds = std::move(bounds);
    node_mutator::set_result(std::move(e));
  }
  void set_result(stmt s) {
    assert(!result_bounds.min.defined() && !result_bounds.max.defined());
    result_bounds = interval_expr();
    node_mutator::set_result(std::move(s));
  }

public:
  simplifier(const bounds_map& expr_bounds) : expr_bounds(expr_bounds) {}

  expr mutate(const expr& e, interval_expr* bounds) {
    expr result = node_mutator::mutate(e);
    if (bounds) {
      if (bounds != &result_bounds) {
        *bounds = std::move(result_bounds);
      }
    } else {
      result_bounds = {expr(), expr()};
    }
    return result;
  }
  expr mutate(const expr& e) override { return mutate(e, nullptr); }
  stmt mutate(const stmt& s) override { return node_mutator::mutate(s); }

  void mutate_and_set_result(const expr& e) {
    assert(!result_bounds.min.defined() && !result_bounds.max.defined());
    node_mutator::set_result(mutate(e, &result_bounds));
  }

  interval_expr mutate(
      const interval_expr& x, interval_expr* min_bounds = nullptr, interval_expr* max_bounds = nullptr) {
    interval_expr result = {mutate(x.min, min_bounds), mutate(x.max, max_bounds)};
    if (!result.min.same_as(result.max) && match(result.min, result.max)) {
      // If the bounds are the same, make sure same_as returns true.
      result.max = result.min;
    }
    return result;
  }

  std::optional<bool> attempt_to_prove(const expr& e) {
    interval_expr bounds;
    // Visits to variables mutate this state, we don't want to do that while trying to prove some other expression.
    symbol_map<int> refs;
    std::swap(references, refs);
    mutate(e, &bounds);
    std::swap(references, refs);
    if (is_true(bounds.min)) {
      return true;
    } else if (is_false(bounds.max)) {
      return false;
    } else {
      return {};
    }
  }

  bool prove_true(const expr& e) {
    std::optional<bool> result = attempt_to_prove(e);
    return result && *result;
  }

  bool prove_false(const expr& e) {
    std::optional<bool> result = attempt_to_prove(e);
    return result && !*result;
  }

  void visit(const variable* op) override {
    auto& ref_count = references[op->sym];
    if (!ref_count) {
      ref_count = 1;
    } else {
      *ref_count += 1;
    }
    std::optional<interval_expr> bounds = expr_bounds[op->sym];
    if (bounds) {
      set_result(op, std::move(*bounds));
    } else {
      set_result(op, {op, op});
    }
  }

  void visit(const constant* op) override { set_result(op, {op, op}); }

  template <typename T>
  void visit_binary(const T* op) {
    interval_expr a_bounds;
    expr a = mutate(op->a, &a_bounds);
    interval_expr b_bounds;
    expr b = mutate(op->b, &b_bounds);

    expr result = simplify(op, std::move(a), std::move(b));
    if (!result.same_as(op)) {
      mutate_and_set_result(result);
    } else {
      set_result(result, bounds_of(op, std::move(a_bounds), std::move(b_bounds)));
    }
  }

  void visit(const class min* op) override {
    interval_expr a_bounds;
    expr a = mutate(op->a, &a_bounds);
    interval_expr b_bounds;
    expr b = mutate(op->b, &b_bounds);

    std::optional<bool> lt = attempt_to_prove(a < b);
    if (lt && *lt) {
      set_result(std::move(a), std::move(a_bounds));
      return;
    } else if (lt && !*lt) {
      set_result(std::move(b), std::move(b_bounds));
      return;
    }

    expr result = simplify(op, std::move(a), std::move(b));
    if (!result.same_as(op)) {
      mutate_and_set_result(result);
    } else {
      set_result(result, bounds_of(op, std::move(a_bounds), std::move(b_bounds)));
    }
  }
  void visit(const class max* op) override {
    interval_expr a_bounds;
    expr a = mutate(op->a, &a_bounds);
    interval_expr b_bounds;
    expr b = mutate(op->b, &b_bounds);

    std::optional<bool> gt = attempt_to_prove(a > b);
    if (gt && *gt) {
      set_result(std::move(a), std::move(a_bounds));
      return;
    } else if (gt && !*gt) {
      set_result(std::move(b), std::move(b_bounds));
      return;
    }

    expr result = simplify(op, std::move(a), std::move(b));
    if (!result.same_as(op)) {
      mutate_and_set_result(result);
    } else {
      set_result(result, bounds_of(op, std::move(a_bounds), std::move(b_bounds)));
    }
  }
  void visit(const add* op) override { visit_binary(op); }

  void visit(const sub* op) override {
    interval_expr a_bounds;
    expr a = mutate(op->a, &a_bounds);
    interval_expr b_bounds;
    expr b = mutate(op->b, &b_bounds);
    const index_t* cb = as_constant(b);

    if (cb && *cb < 0) {
      // Canonicalize to addition with constants.
      mutate_and_set_result(a + -*cb);
    } else {
      expr result = simplify(op, std::move(a), std::move(b));
      if (result.same_as(op)) {
        set_result(std::move(result), bounds_of(op, std::move(a_bounds), std::move(b_bounds)));
      } else {
        mutate_and_set_result(result);
      }
    }
  }

  void visit(const mul* op) override { visit_binary(op); }
  void visit(const div* op) override { visit_binary(op); }
  void visit(const mod* op) override { visit_binary(op); }
  void visit(const less* op) override { visit_binary(op); }
  void visit(const less_equal* op) override { visit_binary(op); }
  void visit(const equal* op) override { visit_binary(op); }
  void visit(const not_equal* op) override { visit_binary(op); }
  void visit(const logical_and* op) override { visit_binary(op); }
  void visit(const logical_or* op) override { visit_binary(op); }
  void visit(const logical_not* op) override {
    interval_expr bounds;
    expr a = mutate(op->x, &bounds);

    if (is_true(bounds.min)) {
      set_result(false, {0, 0});
    } else if (is_false(bounds.max)) {
      set_result(true, {1, 1});
    } else {
      expr result = simplify(op, std::move(a));
      if (result.same_as(op)) {
        set_result(result, bounds_of(op, bounds));
      } else {
        mutate_and_set_result(result);
      }
    }
  }

  void visit(const class select* op) override {
    interval_expr c_bounds;
    expr c = mutate(op->condition, &c_bounds);
    if (is_true(c_bounds.min)) {
      mutate_and_set_result(op->true_value);
      return;
    } else if (is_false(c_bounds.max)) {
      mutate_and_set_result(op->false_value);
      return;
    }

    interval_expr t_bounds;
    expr t = mutate(op->true_value, &t_bounds);
    interval_expr f_bounds;
    expr f = mutate(op->false_value, &f_bounds);

    expr e = simplify(op, std::move(c), std::move(t), std::move(f));
    if (e.same_as(op)) {
      set_result(e, bounds_of(op, std::move(c_bounds), std::move(t_bounds), std::move(f_bounds)));
    } else {
      mutate_and_set_result(e);
    }
  }

  void visit(const call* op) override {
    std::vector<expr> args;
    std::vector<interval_expr> args_bounds;
    args.reserve(op->args.size());
    args_bounds.reserve(op->args.size());
    for (const expr& i : op->args) {
      interval_expr i_bounds;
      args.push_back(mutate(i, &i_bounds));
      args_bounds.push_back(std::move(i_bounds));
    }

    expr e = simplify(op, std::move(args));
    if (e.same_as(op)) {
      set_result(e, bounds_of(op, std::move(args_bounds)));
    } else {
      mutate_and_set_result(e);
    }
  }

  void visit(const let* op) override {
    interval_expr value_bounds;
    expr value = mutate(op->value, &value_bounds);

    auto set_bounds = set_value_in_scope(expr_bounds, op->sym, value_bounds);
    auto ref_count = set_value_in_scope(references, op->sym, 0);
    interval_expr body_bounds;
    expr body = mutate(op->body, &body_bounds);

    int refs = *references[op->sym];
    if (refs == 0) {
      // This let is dead
      set_result(body, std::move(body_bounds));
    } else if (refs == 1 || value.as<constant>() || value.as<variable>()) {
      mutate_and_set_result(substitute(body, op->sym, value));
    } else if (value.same_as(op->value) && body.same_as(op->body)) {
      set_result(op, std::move(body_bounds));
    } else {
      set_result(let::make(op->sym, std::move(value), std::move(body)), std::move(body_bounds));
    }
  }

  void visit(const let_stmt* op) override {
    interval_expr value_bounds;
    expr value = mutate(op->value, &value_bounds);

    auto set_bounds = set_value_in_scope(expr_bounds, op->sym, value_bounds);
    auto ref_count = set_value_in_scope(references, op->sym, 0);
    stmt body = mutate(op->body);

    int refs = *references[op->sym];
    if (refs == 0) {
      // This let is dead
      set_result(body);
    } else if (refs == 1 || value.as<constant>() || value.as<variable>()) {
      set_result(mutate(substitute(body, op->sym, value)));
    } else if (value.same_as(op->value) && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(let_stmt::make(op->sym, std::move(value), std::move(body)));
    }
  }

  void visit(const loop* op) override {
    interval_expr min_bounds;
    interval_expr max_bounds;
    interval_expr bounds = mutate(op->bounds, &min_bounds, &max_bounds);
    expr step = mutate(op->step);

    if (!step.defined()) {
      step = 1;
    }

    if (prove_true(min_bounds.min > max_bounds.max)) {
      // This loop is dead.
      set_result(stmt());
      return;
    } else if (prove_true(bounds.min + step > bounds.max)) {
      // The loop only runs once.
      set_result(mutate(let_stmt::make(op->sym, bounds.min, op->body)));
      return;
    }

    auto set_bounds = set_value_in_scope(expr_bounds, op->sym, bounds);
    stmt body = mutate(op->body);

    if (is_constant(step, 1)) {
      step = expr();
    }

    if (bounds.same_as(op->bounds) && step.same_as(op->step) && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(loop::make(op->sym, std::move(bounds), std::move(step), std::move(body)));
    }
  }

  void visit(const if_then_else* op) override {
    interval_expr c_bounds;
    expr c = mutate(op->condition, &c_bounds);
    if (prove_true(c_bounds.min)) {
      set_result(mutate(op->true_body));
      return;
    } else if (prove_false(c_bounds.max)) {
      set_result(mutate(op->false_body));
      return;
    }

    stmt t = mutate(op->true_body);
    stmt f = mutate(op->false_body);

    if (const logical_not* n = c.as<logical_not>()) {
      c = n->x;
      std::swap(t, f);
    }

    if (!t.defined() && !f.defined()) {
      set_result(t);
    } else if (t.defined() && f.defined() && match(t, f)) {
      set_result(t);
    } else if (c.same_as(op->condition) && t.same_as(op->true_body) && f.same_as(op->false_body)) {
      set_result(op);
    } else {
      set_result(if_then_else::make(std::move(c), std::move(t), std::move(f)));
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
      set_result(if_then_else::make(a_if->condition, true_body, false_body));
    } else if (!a.defined() && !b.defined()) {
      set_result(stmt());
    } else if (!a.defined()) {
      set_result(b);
    } else if (!b.defined()) {
      set_result(a);
    } else if (a.same_as(op->a) && b.same_as(op->b)) {
      set_result(op);
    } else {
      set_result(block::make(std::move(a), std::move(b)));
    }
  }

  void visit(const allocate* op) override {
    std::vector<dim_expr> dims;
    box_expr bounds;
    dims.reserve(op->dims.size());
    for (const dim_expr& i : op->dims) {
      interval_expr bounds_i = mutate(i.bounds);
      dims.emplace_back(bounds_i, mutate(i.stride), mutate(i.fold_factor));
      bounds.push_back(bounds_i);
    }
    auto set_bounds = set_value_in_scope(buffer_bounds, op->sym, std::move(bounds));
    stmt body = mutate(op->body);
    set_result(allocate::make(op->storage, op->sym, op->elem_size, std::move(dims), std::move(body)));
  }

  void visit(const make_buffer* op) override {
    expr base = mutate(op->base);
    expr elem_size = mutate(op->elem_size);
    std::vector<dim_expr> dims;
    box_expr bounds;
    dims.reserve(op->dims.size());
    bounds.reserve(op->dims.size());
    bool changed = false;
    for (const dim_expr& d : op->dims) {
      interval_expr new_bounds = mutate(d.bounds);
      dim_expr new_dim = {new_bounds, mutate(d.stride), mutate(d.fold_factor)};
      changed = changed || !new_dim.same_as(d);
      dims.push_back(std::move(new_dim));
      bounds.push_back(std::move(new_bounds));
    }

    auto set_bounds = set_value_in_scope(buffer_bounds, op->sym, std::move(bounds));
    stmt body = mutate(op->body);

    // Check if this make_buffer is equivalent to truncate_rank
    var buf(op->sym);
    if (match(base, buffer_base(buf)) && match(elem_size, buffer_elem_size(buf))) {
      bool is_truncate = true;
      for (index_t d = 0; d < static_cast<index_t>(dims.size()); ++d) {
        is_truncate = is_truncate && match(dims[d], buffer_dim(buf, d));
      }
      if (is_truncate) {
        set_result(truncate_rank::make(op->sym, dims.size(), std::move(body)));
        return;
      }
    }

    // Check if this make_buffer is equivalent to slice_buffer or crop_buffer
    if (const call* bc = base.as<call>()) {
      if (bc->intrinsic == intrinsic::buffer_at && match(bc->args[0], buf) && match(elem_size, buffer_elem_size(buf))) {
        // To be a slice, we need every dimension that is present in the buffer_at call to be skipped, and the rest of
        // the dimensions to be identity.
        index_t dim = 0;
        bool is_slice = true;
        for (index_t d = 0; d < static_cast<index_t>(dims.size()); ++d) {
          if (d + 1 < static_cast<index_t>(bc->args.size()) && bc->args[d + 1].defined()) {
            // Skip this dimension.
            ++dim;
          } else {
            // This arg is undefined. We need to find the next dimension here to be a slice.
            is_slice = is_slice && match(dims[dim], buffer_dim(buf, d));
          }
        }
        if (is_slice) {
          std::vector<expr> at(bc->args.begin() + 1, bc->args.end());
          set_result(slice_buffer::make(op->sym, std::move(at), std::move(body)));
          return;
        }

        // To be a crop, we need dimensions to either be identity, or the buffer_at argument is the same as the min.
        bool is_crop = true;
        box_expr crop_bounds(dims.size());
        for (index_t d = 0; d < static_cast<index_t>(dims.size()); ++d) {
          if (!match(dims[d].stride, buffer_stride(buf, d)) ||
              !match(dims[d].fold_factor, buffer_fold_factor(buf, d))) {
            is_crop = false;
            break;
          }

          // If the argument is defined, we need the min to be the same as the argument.
          // If it is not defined, it must be buffer_min(buf, d).
          bool has_at_d = d + 1 < static_cast<index_t>(bc->args.size()) && bc->args[d + 1].defined();
          expr crop_min = has_at_d ? bc->args[d + 1] : buffer_min(buf, d);
          if (match(dims[d].bounds.min, crop_min)) {
            crop_bounds[d] = dims[d].bounds;
          } else {
            is_crop = false;
            break;
          }
        }
        if (is_crop) {
          set_result(mutate(crop_buffer::make(op->sym, std::move(crop_bounds), std::move(body))));
          return;
        }
      }
    }

    if (changed || !base.same_as(op->base) || !elem_size.same_as(op->elem_size) || !body.same_as(op->body)) {
      set_result(make_buffer::make(op->sym, std::move(base), std::move(elem_size), std::move(dims), std::move(body)));
    } else {
      set_result(op);
    }
  }

  void visit(const crop_buffer* op) override {
    // This is the bounds of the buffer as we understand them, for simplifying the inner scope.
    box_expr bounds(op->bounds.size());
    // This is the new bounds of the crop operation. Crops that are no-ops become undefined here.
    box_expr new_bounds(op->bounds.size());

    // If possible, rewrite crop_buffer of one dimension to crop_dim.
    std::optional<box_expr> prev_bounds = buffer_bounds[op->sym];
    index_t dims_count = 0;
    bool changed = false;
    for (index_t i = 0; i < static_cast<index_t>(op->bounds.size()); ++i) {
      interval_expr bounds_i = mutate(op->bounds[i]);
      changed = changed || !bounds_i.same_as(op->bounds[i]);

      bounds[i] = bounds_i;

      // If the new bounds are the same as the existing bounds, set the crop in this dimension to
      // be undefined.
      if (prev_bounds && i < static_cast<index_t>(prev_bounds->size())) {
        if (prove_true(bounds_i.min == (*prev_bounds)[i].min) && prove_true(bounds_i.max == (*prev_bounds)[i].max)) {
          bounds_i.min = expr();
          bounds_i.max = expr();
        }
      }
      new_bounds[i] = bounds_i;
      dims_count += bounds_i.min.defined() && bounds_i.max.defined() ? 1 : 0;
    }

    auto set_bounds = set_value_in_scope(buffer_bounds, op->sym, bounds);
    stmt body = op->body;
    for (index_t d = 0; d < static_cast<index_t>(new_bounds.size()); ++d) {
      if (new_bounds[d].min.defined() && new_bounds[d].max.defined()) {
        body = substitute_bounds(body, op->sym, d, new_bounds[d]);
      }
    }
    body = mutate(body);

    // Remove trailing undefined bounds.
    while (new_bounds.size() > 0 && !new_bounds.back().min.defined() && !new_bounds.back().max.defined()) {
      new_bounds.pop_back();
    }
    if (new_bounds.empty()) {
      // This crop was a no-op.
      set_result(std::move(body));
    } else if (dims_count == 1) {
      // This crop is of one dimension, replace it with crop_dim.
      // We removed undefined trailing bounds, so this must be the dim we want.
      int d = static_cast<int>(new_bounds.size()) - 1;
      set_result(crop_dim::make(op->sym, d, std::move(new_bounds[d]), std::move(body)));
    } else if (changed || !body.same_as(op->body)) {
      set_result(crop_buffer::make(op->sym, std::move(new_bounds), std::move(body)));
    } else {
      set_result(op);
    }
  }

  void visit(const crop_dim* op) override {
    interval_expr bounds = mutate(op->bounds);
    if (!bounds.min.defined() && !bounds.max.defined()) {
      set_result(mutate(op->body));
      return;
    }

    std::optional<box_expr> buf_bounds = buffer_bounds[op->sym];
    if (buf_bounds && op->dim < static_cast<index_t>(buf_bounds->size())) {
      interval_expr& dim = (*buf_bounds)[op->dim];
      if (prove_true(bounds.min == dim.min) && prove_true(bounds.max == dim.max)) {
        // This crop is a no-op.
        set_result(mutate(op->body));
        return;
      }
      (*buf_bounds)[op->dim] = bounds;
    }

    auto set_bounds = set_value_in_scope(buffer_bounds, op->sym, buf_bounds);
    stmt body = substitute_bounds(op->body, op->sym, op->dim, bounds);
    body = mutate(body);
    if (bounds.same_as(op->bounds) && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(crop_dim::make(op->sym, op->dim, std::move(bounds), std::move(body)));
    }
  }

  void visit(const slice_buffer* op) override {
    // Update the bounds for the slice. Sliced dimensions are removed from the bounds.
    std::optional<box_expr> bounds = buffer_bounds[op->sym];
    std::vector<expr> at(op->at.size());
    std::size_t dims_count = 0;
    bool changed = false;
    for (index_t i = 0; i < static_cast<index_t>(op->at.size()); ++i) {
      if (op->at[i].defined()) {
        at[i] = mutate(op->at[i]);
        changed = changed || !at[i].same_as(op->at[i]);

        // We sliced this dimension. Remove it from the bounds.
        if (bounds && i < static_cast<index_t>(bounds->size())) {
          bounds->erase(bounds->begin() + i);
        }
        ++dims_count;
      }
    }

    auto set_bounds = set_value_in_scope(buffer_bounds, op->sym, bounds);
    stmt body = mutate(op->body);

    // Remove trailing undefined bounds.
    while (at.size() > 0 && !at.back().defined()) {
      at.pop_back();
    }
    changed = changed || at.size() != op->at.size();
    if (at.empty()) {
      // This slice was a no-op.
      set_result(std::move(body));
    } else if (dims_count == 1) {
      // This slice is of one dimension, replace it with slice_dim.
      // We removed undefined trailing bounds, so this must be the dim we want.
      int d = static_cast<int>(at.size()) - 1;
      set_result(slice_dim::make(op->sym, d, std::move(at[d]), std::move(body)));
    } else if (changed || !body.same_as(op->body)) {
      set_result(slice_buffer::make(op->sym, std::move(at), std::move(body)));
    } else {
      set_result(op);
    }
  }

  void visit(const slice_dim* op) override {
    expr at = mutate(op->at);

    std::optional<box_expr> bounds = buffer_bounds[op->sym];
    if (bounds && op->dim < static_cast<index_t>(bounds->size())) {
      bounds->erase(bounds->begin() + op->dim);
    }

    auto set_bounds = set_value_in_scope(buffer_bounds, op->sym, bounds);
    stmt body = mutate(op->body);
    if (at.same_as(op->at) && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(slice_dim::make(op->sym, op->dim, std::move(at), std::move(body)));
    }
  }

  void visit(const truncate_rank* op) override {
    std::optional<box_expr> bounds = buffer_bounds[op->sym];
    if (bounds && static_cast<int>(bounds->size()) > op->rank) {
      bounds->resize(op->rank);
    }

    auto set_bounds = set_value_in_scope(buffer_bounds, op->sym, bounds);
    stmt body = mutate(op->body);
    if (body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(truncate_rank::make(op->sym, op->rank, std::move(body)));
    }
  }

  void visit(const check* op) override {
    if (!op->condition.defined()) {
      set_result(op);
      return;
    }

    interval_expr c_bounds;
    expr c = mutate(op->condition, &c_bounds);
    if (is_true(c_bounds.min)) {
      set_result(stmt());
    } else if (is_false(c_bounds.max)) {
      std::cerr << op->condition << " is statically false." << std::endl;
      std::abort();
    } else if (c.same_as(op->condition)) {
      set_result(op);
    } else {
      set_result(check::make(std::move(c)));
    }
  }
};

}  // namespace

expr simplify(const expr& e, const bounds_map& bounds) { return simplifier(bounds).mutate(e, nullptr); }
stmt simplify(const stmt& s, const bounds_map& bounds) { return simplifier(bounds).mutate(s); }
interval_expr simplify(const interval_expr& e, const bounds_map& bounds) {
  simplifier s(bounds);
  return {s.mutate(e.min, nullptr), s.mutate(e.max, nullptr)};
}

namespace {

template <typename T>
interval_expr bounds_of_linear(const T* x, interval_expr a, interval_expr b) {
  return {simplify(x, std::move(a.min), std::move(b.min)), simplify(x, std::move(a.max), std::move(b.max))};
}

template <typename T>
interval_expr bounds_of_less(const T* x, interval_expr a, interval_expr b) {
  // This bit of genius comes from
  // https://github.com/halide/Halide/blob/61b8d384b2b799cd47634e4a3b67aa7c7f580a46/src/Bounds.cpp#L829
  return {simplify(x, std::move(a.max), std::move(b.min)), simplify(x, std::move(a.min), std::move(b.max))};
}

}  // namespace

interval_expr bounds_of(const add* x, interval_expr a, interval_expr b) {
  return bounds_of_linear(x, std::move(a), std::move(b));
}
interval_expr bounds_of(const sub* x, interval_expr a, interval_expr b) {
  return {simplify(x, std::move(a.min), std::move(b.max)), simplify(x, std::move(a.max), std::move(b.min))};
}
interval_expr bounds_of(const mul* x, interval_expr a, interval_expr b) {
  // TODO: I'm pretty sure there are cases missing here that would produce simpler bounds than the fallback cases.
  if (is_non_negative(a.min) && is_non_negative(b.min)) {
    // Both are >= 0, neither intervals flip.
    return {simplify(x, a.min, b.min), simplify(x, a.max, b.max)};
  } else if (is_non_positive(a.max) && is_non_positive(b.max)) {
    // Both are <= 0, both intervals flip.
    return {simplify(x, a.max, b.max), simplify(x, a.min, b.min)};
  } else if (b.is_single_point()) {
    if (is_non_negative(b.min)) {
      return {simplify(x, a.min, b.min), simplify(x, a.max, b.min)};
    } else if (is_non_positive(b.min)) {
      return {simplify(x, a.max, b.min), simplify(x, a.min, b.min)};
    } else {
      expr corners[] = {
          simplify(x, a.min, b.min),
          simplify(x, a.max, b.min),
      };
      return {min(corners), max(corners)};
    }
  } else if (a.is_single_point()) {
    if (is_non_negative(a.min)) {
      return {simplify(x, a.min, b.min), simplify(x, a.min, b.max)};
    } else if (is_non_positive(a.min)) {
      return {simplify(x, a.min, b.max), simplify(x, a.min, b.min)};
    } else {
      expr corners[] = {
          simplify(x, a.min, b.min),
          simplify(x, a.min, b.max),
      };
      return {min(corners), max(corners)};
    }
  } else {
    // We don't know anything. The results is the union of all 4 possible intervals.
    expr corners[] = {
        simplify(x, a.min, b.min),
        simplify(x, a.min, b.max),
        simplify(x, a.max, b.min),
        simplify(x, a.max, b.max),
    };
    return {min(corners), max(corners)};
  }
}
interval_expr bounds_of(const div* x, interval_expr a, interval_expr b) {
  // Because b is an integer, the bounds of a will only be shrunk
  // (we define division by 0 to be 0). The absolute value of the
  // bounds are maximized when b is 1 or -1.
  if (b.is_single_point() && is_zero(b.min)) {
    return {0, 0};
  } else if (is_positive(b.min)) {
    // b > 0 => the biggest result in absolute value occurs at the min of b.
    return (a | -a) / b.min;
  } else if (is_negative(b.max)) {
    // b < 0 => the biggest result in absolute value occurs at the max of b.
    return (a | -a) / b.max;
  } else {
    return a | -a;
  }
}
interval_expr bounds_of(const mod* x, interval_expr a, interval_expr b) { return {0, max(abs(b.min), abs(b.max))}; }

interval_expr bounds_of(const class min* x, interval_expr a, interval_expr b) {
  return bounds_of_linear(x, std::move(a), std::move(b));
}
interval_expr bounds_of(const class max* x, interval_expr a, interval_expr b) {
  return bounds_of_linear(x, std::move(a), std::move(b));
}

interval_expr bounds_of(const less* x, interval_expr a, interval_expr b) {
  return bounds_of_less(x, std::move(a), std::move(b));
}
interval_expr bounds_of(const less_equal* x, interval_expr a, interval_expr b) {
  return bounds_of_less(x, std::move(a), std::move(b));
}
interval_expr bounds_of(const equal* x, interval_expr a, interval_expr b) {
  return {0, a.min <= b.max && b.min <= a.max};
}
interval_expr bounds_of(const not_equal* x, interval_expr a, interval_expr b) {
  return {a.max < b.min || b.max < a.min, 1};
}

interval_expr bounds_of(const logical_and* x, interval_expr a, interval_expr b) {
  return bounds_of_linear(x, std::move(a), std::move(b));
}
interval_expr bounds_of(const logical_or* x, interval_expr a, interval_expr b) {
  return bounds_of_linear(x, std::move(a), std::move(b));
}
interval_expr bounds_of(const logical_not* x, interval_expr a) {
  return {simplify(x, std::move(a.max)), simplify(x, std::move(a.min))};
}

interval_expr bounds_of(const class select* x, interval_expr c, interval_expr t, interval_expr f) {
  if (is_true(c.min)) {
    return t;
  } else if (is_false(c.max)) {
    return f;
  } else {
    return f | t;
  }
}

interval_expr bounds_of(const call* x, std::vector<interval_expr> args) {
  switch (x->intrinsic) {
  case intrinsic::abs:
    assert(args.size() == 1);
    if (is_positive(args[0].min)) {
      return {args[0].min, args[0].max};
    } else {
      expr abs_min = simplify(x, {args[0].min});
      expr abs_max = simplify(x, {args[0].max});
      return {0, simplify(static_cast<const class max*>(nullptr), std::move(abs_min), std::move(abs_max))};
    }
  default: return {x, x};
  }
}

interval_expr bounds_of(const expr& e, const bounds_map& expr_bounds) {
  simplifier s(expr_bounds);
  interval_expr bounds;
  s.mutate(e, &bounds);
  return bounds;
}

std::optional<bool> attempt_to_prove(const expr& condition, const bounds_map& expr_bounds) {
  simplifier s(expr_bounds);
  return s.attempt_to_prove(condition);
}

bool prove_true(const expr& condition, const bounds_map& expr_bounds) {
  simplifier s(expr_bounds);
  return s.prove_true(condition);
}

bool prove_false(const expr& condition, const bounds_map& expr_bounds) {
  simplifier s(expr_bounds);
  return s.prove_false(condition);
}

interval_expr where_true(const expr& condition, symbol_id var) {
  // TODO: This needs a proper implementation. For now, a ridiculous hack: trial and error.
  // We use the leaves of the expression as guesses around which to search.
  // We could use every node in the expression...
  class initial_guesses : public recursive_node_visitor {
  public:
    std::vector<expr> leaves;

    void visit(const variable* x) { leaves.push_back(x); }
    void visit(const constant* x) { leaves.push_back(x); }
    void visit(const call* x) { if (is_buffer_intrinsic(x->intrinsic)) leaves.push_back(x); }
  };

  initial_guesses v;
  condition.accept(&v);

  std::vector<expr> offsets;
  offsets.push_back(negative_infinity());
  for (index_t i = -10; i <= 10; ++i) {
    offsets.push_back(i);
  }
  offsets.push_back(positive_infinity());

  interval_expr result = interval_expr::none();
  for (const expr& i : v.leaves) {
    interval_expr result_i;
    for (const expr& j : offsets) {
      if (!result_i.min.defined()) {
        // Find the first offset where the expression is true.
        if (prove_true(substitute(condition, var, i + j))) {
          result_i.min = i + j;
          result_i.max = result_i.min;
        }
      } else if (prove_true(substitute(condition, var, i + j))) {
        // Find the last offset where the expression is true.
        result_i.max = i + j;
      }
    }
    if (result_i.min.defined()) {
      result.min = simplify(min(result.min, result_i.min));
    }
    if (result_i.max.defined()) {
      result.max = simplify(max(result.max, result_i.max));
    }
  }
  return result;
}

namespace {

class derivative : public node_mutator {
  symbol_id dx;

public:
  derivative(symbol_id dx) : dx(dx) {}

  template <typename T>
  void visit_variable(const T* x) {
    if (x->sym == dx) {
      set_result(1);
    } else {
      set_result(expr(0));
    }
  }

  void visit(const variable* x) override { visit_variable(x); }
  void visit(const wildcard* x) override { visit_variable(x); }
  void visit(const constant*) override { set_result(expr(0)); }

  void visit(const mul* x) override {
    if (depends_on(x->a, dx) && depends_on(x->b, dx)) {
      expr da = mutate(x->a);
      expr db = mutate(x->b);
      set_result(simplify(x, x->a, db) + simplify(x, da, x->b));
    } else if (depends_on(x->a, dx)) {
      set_result(simplify(x, mutate(x->a), x->b));
    } else if (depends_on(x->b, dx)) {
      set_result(simplify(x, x->a, mutate(x->b)));
    } else {
      set_result(expr(0));
    }
  }
  void visit(const div* x) override {
    if (depends_on(x->a, dx) && depends_on(x->b, dx)) {
      expr da = mutate(x->a);
      expr db = mutate(x->b);
      set_result((da * x->b - x->a * db) / (x->b * x->b));
    } else if (depends_on(x->a, dx)) {
      set_result(mutate(x->a) / x->b);
    } else if (depends_on(x->b, dx)) {
      expr db = mutate(x->b);
      set_result(-x->a / (x->b * x->b));
    } else {
      set_result(expr(0));
    }
  }

  virtual void visit(const mod* x) override { set_result(indeterminate()); }
  virtual void visit(const class min* x) override { set_result(select(x->a < x->b, mutate(x->a), mutate(x->b))); }
  virtual void visit(const class max* x) override { set_result(select(x->b < x->a, mutate(x->a), mutate(x->b))); }

  template <typename T>
  void visit_compare(const T* x) {
    if (depends_on(x->a, dx) || depends_on(x->b, dx)) {
      set_result(indeterminate());
    } else {
      set_result(expr(0));
    }
  }

  virtual void visit(const equal* x) override { visit_compare(x); }
  virtual void visit(const not_equal* x) override { visit_compare(x); }
  virtual void visit(const less* x) override { visit_compare(x); }
  virtual void visit(const less_equal* x) override { visit_compare(x); }
  virtual void visit(const logical_and* x) override {}
  virtual void visit(const logical_or* x) override {}
  virtual void visit(const logical_not* x) override { set_result(-mutate(x->x)); }

  virtual void visit(const class select* x) override {
    set_result(select(x->condition, mutate(x->true_value), mutate(x->false_value)));
  }

  virtual void visit(const call* x) override { std::abort(); }
};

}  // namespace

expr differentiate(const expr& f, symbol_id x) { return derivative(x).mutate(f); }

}  // namespace slinky
