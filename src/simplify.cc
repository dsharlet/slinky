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
      {min(x, min(x, y)), min(x, y)},
      {min(max(x, y), min(x, z)), min(x, z)},
      {min(min(x, y), min(x, z)), min(x, min(y, z))},
      {min(max(x, y), max(x, z)), max(x, min(y, z))},
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
      {max(x, max(x, y)), max(x, y)},
      {max(min(x, y), max(x, z)), max(x, z)},
      {max(max(x, y), max(x, z)), max(x, max(y, z))},
      {max(min(x, y), min(x, z)), min(x, max(y, z))},
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
      {c0 + positive_infinity(), positive_infinity()},
      {c0 + negative_infinity(), negative_infinity()},
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
      {buffer_max(x, y) + (z - buffer_min(x, y)), (buffer_extent(x, y) + z) - 1},
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

      {buffer_max(x, y) - buffer_min(x, y), buffer_extent(x, y) - 1},
      {buffer_max(x, y) - (buffer_min(x, y) + z), (buffer_extent(x, y) - z) - 1},
      {(buffer_max(x, y) + z) - buffer_min(x, y), (buffer_extent(x, y) + z) - 1},
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
      {c0 / positive_infinity(), 0},
      {c0 / negative_infinity(), 0},
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
      {positive_infinity() < c0, false},
      {negative_infinity() < c0, true},
      {x < x, false},
      {x + c0 < c1, x < c1 - c0},
      {x < x + y, 0 < y},
      {x + y < x, y < 0},
      {x - y < x, 0 < y},
      {0 - x < c0, -c0 < x},
      {c0 - x < c1, c0 - c1 < x},
      {c0 < c1 - x, x < c1 - c0},

      {x + y < x + z, y < z},

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
      {positive_infinity() <= c0, false},
      {negative_infinity() <= c0, true},
      {c0 <= positive_infinity(), true},
      {c0 <= negative_infinity(), false},
      {x <= x, true},
      {x <= x + y, 0 <= y},
      {x + y <= x, y <= 0},
      {x - y <= x, 0 <= y},
      {0 - x <= c0, -c0 <= x},
      {c0 - x <= y, c0 <= y + x},
      {x <= c1 - y, x + y <= c1},
      {x + c0 <= y + c1, x - y <= c1 - c0},

      {x + y <= x + z, y <= z},

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

class simplifier : public node_mutator {
  symbol_map<int> references;
  symbol_map<box_expr> buffer_bounds;
  bounds_map expr_bounds;

public:
  simplifier(const bounds_map& expr_bounds) : expr_bounds(expr_bounds) {}

  interval_expr mutate(const interval_expr& x) {
    interval_expr result = {mutate(x.min), mutate(x.max)};
    if (!result.min.same_as(result.max) && match(result.min, result.max)) {
      // If the bounds are the same, make sure same_as returns true.
      result.max = result.min;
    }
    return result;
  }
  using node_mutator::mutate;

  void visit(const variable* op) override {
    auto& ref_count = references[op->sym];
    if (!ref_count) {
      ref_count = 1;
    } else {
      *ref_count += 1;
    }
    set_result(op);
  }

  template <typename T>
  void visit_binary(const T* op) {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    expr e = simplify(op, std::move(a), std::move(b));
    if (!e.same_as(op)) {
      e = mutate(e);
    }
    set_result(e);
  }

  void visit(const class min* op) override { visit_binary(op); }
  void visit(const class max* op) override { visit_binary(op); }
  void visit(const add* op) override { visit_binary(op); }

  void visit(const sub* op) override {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    const index_t* cb = as_constant(b);
    if (cb && *cb < 0) {
      // Canonicalize to addition with constants.
      set_result(mutate(a + -*cb));
    } else {
      expr e = simplify(op, std::move(a), std::move(b));
      if (!e.same_as(op)) {
        e = mutate(e);
      }
      set_result(e);
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
    expr x = mutate(op->x);
    expr e = simplify(op, std::move(x));
    if (!e.same_as(op)) {
      e = mutate(e);
    }
    set_result(e);
  }

  void visit(const class select* op) override {
    expr c = mutate(op->condition);
    std::optional<bool> const_c = attempt_to_prove(c, expr_bounds);
    if (const_c) {
      if (*const_c) {
        set_result(mutate(op->true_value));
      } else {
        set_result(mutate(op->false_value));
      }
      return;
    }

    expr t = mutate(op->true_value);
    expr f = mutate(op->false_value);

    expr e = simplify(op, std::move(c), std::move(t), std::move(f));
    if (!e.same_as(op)) {
      e = mutate(e);
    }
    set_result(e);
  }

  void visit(const call* op) override {
    std::vector<expr> args;
    args.reserve(op->args.size());
    for (const expr& i : op->args) {
      args.push_back(mutate(i));
    }

    expr e = simplify(op, std::move(args));
    if (!e.same_as(op)) {
      e = mutate(e);
    }
    set_result(e);
  }

  template <typename T>
  auto visit_let(const T* op) {
    expr value = mutate(op->value);
    auto set_bounds = set_value_in_scope(expr_bounds, op->sym, bounds_of(value, expr_bounds));

    auto ref_count = set_value_in_scope(references, op->sym, 0);
    auto body = mutate(op->body);

    int refs = *references[op->sym];
    if (refs == 0) {
      // This let is dead
      return body;
    } else if (refs == 1 || value.as<constant>() || value.as<variable>()) {
      return mutate(substitute(body, op->sym, value));
    } else if (value.same_as(op->value) && body.same_as(op->body)) {
      return decltype(body){op};
    } else {
      return T::make(op->sym, std::move(value), std::move(body));
    }
  }

  void visit(const let* op) override { set_result(visit_let(op)); }
  void visit(const let_stmt* op) override { set_result(visit_let(op)); }

  void visit(const loop* op) override {
    interval_expr bounds = mutate(op->bounds);
    expr step = mutate(op->step);

    if (!step.defined()) {
      step = 1;
    }

    if (prove_true(bounds.min + step > bounds.max)) {
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
    expr c = mutate(op->condition);
    std::optional<bool> const_c = attempt_to_prove(c, expr_bounds);
    if (const_c) {
      if (*const_c) {
        set_result(mutate(op->true_body));
      } else {
        set_result(mutate(op->false_body));
      }
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
    stmt body = mutate(op->body);

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
    stmt body = mutate(op->body);
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

    expr c = mutate(op->condition);
    std::optional<bool> const_c = attempt_to_prove(c, expr_bounds);
    if (const_c) {
      if (*const_c) {
        set_result(stmt());
      } else {
        std::cerr << op->condition << " is statically false." << std::endl;
        std::abort();
      }
    } else if (c.same_as(op->condition)) {
      set_result(op);
    } else {
      set_result(check::make(std::move(c)));
    }
  }
};

}  // namespace

expr simplify(const expr& e, const bounds_map& bounds) { return simplifier(bounds).mutate(e); }
stmt simplify(const stmt& s, const bounds_map& bounds) { return simplifier(bounds).mutate(s); }

namespace {

class find_bounds : public node_visitor {
  bounds_map bounds;

public:
  find_bounds(const bounds_map& bounds) : bounds(bounds) {}

  interval_expr result;

  template <typename T>
  void visit_variable(const T* x) {
    if (bounds.contains(x->sym)) {
      result = *bounds.lookup(x->sym);
    } else {
      result = {x, x};
    }
  }

  void visit(const variable* x) override { visit_variable(x); }
  void visit(const wildcard* x) override { visit_variable(x); }
  void visit(const constant* x) override { result = point(x); }

  void visit(const let* x) override {
    x->value.accept(this);
    auto s = set_value_in_scope(bounds, x->sym, result);
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
  void visit_linear(const T* x) {
    binary_result r = binary_bounds(x);
    result = {simplify(x, r.a.min, r.b.min), simplify(x, r.a.max, r.b.max)};
  }

  void visit(const add* x) override { visit_linear(x); }
  void visit(const sub* x) override {
    binary_result r = binary_bounds(x);
    result = {simplify(x, r.a.min, r.b.max), simplify(x, r.a.max, r.b.min)};
  }

  void visit(const mul* x) override {
    binary_result r = binary_bounds(x);

    // TODO: I'm pretty sure there are cases missing here that would produce simpler bounds than the fallback cases.
    if (is_non_negative(r.a.min) && is_non_negative(r.b.min)) {
      // Both are >= 0, neither intervals flip.
      result = {simplify(x, r.a.min, r.b.min), simplify(x, r.a.max, r.b.max)};
    } else if (is_non_positive(r.a.max) && is_non_positive(r.b.max)) {
      // Both are <= 0, both intervals flip.
      result = {simplify(x, r.a.max, r.b.max), simplify(x, r.a.min, r.b.min)};
    } else if (r.b.is_single_point()) {
      if (is_non_negative(r.b.min)) {
        result = {simplify(x, r.a.min, r.b.min), simplify(x, r.a.max, r.b.min)};
      } else if (is_non_positive(r.b.min)) {
        result = {simplify(x, r.a.max, r.b.min), simplify(x, r.a.min, r.b.min)};
      } else {
        expr corners[] = {
            simplify(x, r.a.min, r.b.min),
            simplify(x, r.a.max, r.b.min),
        };
        result = {min(corners), max(corners)};
      }
    } else if (r.a.is_single_point()) {
      if (is_non_negative(r.a.min)) {
        result = {simplify(x, r.a.min, r.b.min), simplify(x, r.a.min, r.b.max)};
      } else if (is_non_positive(r.a.min)) {
        result = {simplify(x, r.a.min, r.b.max), simplify(x, r.a.min, r.b.min)};
      } else {
        expr corners[] = {
            simplify(x, r.a.min, r.b.min),
            simplify(x, r.a.min, r.b.max),
        };
        result = {min(corners), max(corners)};
      }
    } else {
      // We don't know anything. The results is the union of all 4 possible intervals.
      expr corners[] = {
          simplify(x, r.a.min, r.b.min),
          simplify(x, r.a.min, r.b.max),
          simplify(x, r.a.max, r.b.min),
          simplify(x, r.a.max, r.b.max),
      };
      result = {min(corners), max(corners)};
    }
  }
  void visit(const div* x) override {
    binary_result r = binary_bounds(x);
    // Because b is an integer, the bounds of a will only be shrunk
    // (we define division by 0 to be 0). The absolute value of the
    // bounds are maximized when b is 1 or -1.
    if (r.b.is_single_point() && is_zero(r.b.min)) {
      result = {0, 0};
    } else if (is_positive(r.b.min)) {
      // b > 0 => the biggest result in absolute value occurs at the min of b.
      result = (r.a | -r.a) / r.b.min;
    } else if (is_negative(r.b.max)) {
      // b < 0 => the biggest result in absolute value occurs at the max of b.
      result = (r.a | -r.a) / r.b.max;
    } else {
      result = r.a | -r.a;
    }
  }
  void visit(const mod* x) override {
    binary_result r = binary_bounds(x);
    result = {0, max(abs(r.b.min), abs(r.b.max))};
  }

  void visit(const class min* x) override { visit_linear(x); }
  void visit(const class max* x) override { visit_linear(x); }
  template <typename T>
  void visit_less(const T* x) {
    binary_result r = binary_bounds(x);
    // This bit of genius comes from
    // https://github.com/halide/Halide/blob/61b8d384b2b799cd47634e4a3b67aa7c7f580a46/src/Bounds.cpp#L829
    result = {simplify(x, r.a.max, r.b.min), simplify(x, r.a.min, r.b.max)};
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
  void visit(const logical_not* x) override {
    x->x.accept(this);
    result = {simplify(x, result.max), simplify(x, result.min)};
  }

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

  void visit(const call* x) override {
    switch (x->intrinsic) {
    case intrinsic::abs: result = {0, x}; return;
    default: result = {x, x}; return;
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

}  // namespace

interval_expr bounds_of(const expr& e, const bounds_map& bounds) {
  find_bounds fb(bounds);
  e.accept(&fb);
  return fb.result;
}

std::optional<bool> attempt_to_prove(const expr& condition, const bounds_map& expr_bounds) {
  simplifier s(expr_bounds);

  expr c = s.mutate(condition);

  interval_expr bounds = bounds_of(c, expr_bounds);
  if (is_true(s.mutate(bounds.min))) {
    return true;
  } else if (is_false(s.mutate(bounds.max))) {
    return false;
  } else {
    return {};
  }
}

bool prove_true(const expr& condition, const bounds_map& bounds) {
  std::optional<bool> r = attempt_to_prove(condition, bounds);
  return r && *r;
}

bool prove_false(const expr& condition, const bounds_map& bounds) {
  std::optional<bool> r = attempt_to_prove(condition, bounds);
  return r && !*r;
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
