#include "builder/simplify.h"

#include <algorithm>
#include <cassert>
#include <iostream>

#include "builder/substitute.h"
#include "runtime/evaluate.h"
#include "runtime/print.h"

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
  void check(const T* op) {
    if (should_commute(op->a, op->b)) {
      std::cerr << "Non-canonical operands: " << expr(op) << std::endl;
      std::abort();
    }
    op->a.accept(this);
    op->b.accept(this);
  }

  void visit(const add* op) override { check(op); };
  void visit(const mul* op) override { check(op); };
  void visit(const class min* op) override { check(op); };
  void visit(const class max* op) override { check(op); };
  void visit(const equal* op) override { check(op); };
  void visit(const not_equal* op) override { check(op); };
  void visit(const logical_and* op) override { check(op); };
  void visit(const logical_or* op) override { check(op); };
};

// We need to generate a lot of rules that are equivalent except for commutation.
// To avoid repetitive error-prone code, we can generate all the valid commutative
// equivalents of an expression.
class commute_variants : public node_visitor {
public:
  std::vector<expr> results;

  void visit(const variable* op) override { results = {op}; }
  void visit(const wildcard* op) override { results = {op}; }
  void visit(const constant* op) override { results = {op}; }
  void visit(const let* op) override { std::abort(); }

  template <typename T>
  void visit_binary(bool commutative, const T* op) {
    // TODO: I think some of these patterns are redundant, but finding them is tricky.

    op->a.accept(this);
    std::vector<expr> a = std::move(results);
    op->b.accept(this);
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

  void visit(const add* op) override { visit_binary(true, op); }
  void visit(const sub* op) override { visit_binary(false, op); }
  void visit(const mul* op) override { visit_binary(true, op); }
  void visit(const div* op) override { visit_binary(false, op); }
  void visit(const mod* op) override { visit_binary(false, op); }
  void visit(const class min* op) override { visit_binary(true, op); }
  void visit(const class max* op) override { visit_binary(true, op); }
  void visit(const equal* op) override { visit_binary(true, op); }
  void visit(const not_equal* op) override { visit_binary(true, op); }
  void visit(const less* op) override { visit_binary(false, op); }
  void visit(const less_equal* op) override { visit_binary(false, op); }
  void visit(const logical_and* op) override { visit_binary(true, op); }
  void visit(const logical_or* op) override { visit_binary(true, op); }
  void visit(const logical_not* op) override {
    op->a.accept(this);
    for (expr& i : results) {
      i = !i;
    }
  }
  void visit(const class select* op) override {
    op->condition.accept(this);
    std::vector<expr> c = std::move(results);
    op->true_value.accept(this);
    std::vector<expr> t = std::move(results);
    op->false_value.accept(this);
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

  void visit(const call* op) override {
    if (op->args.size() == 1) {
      op->args.front().accept(this);
      for (expr& i : results) {
        i = call::make(op->intrinsic, {i});
      }
    } else {
      results = {op};
    }
  }

  void visit(const let_stmt* op) override { std::abort(); }
  void visit(const block* op) override { std::abort(); }
  void visit(const loop* op) override { std::abort(); }
  void visit(const call_stmt* op) override { std::abort(); }
  void visit(const copy_stmt* op) override { std::abort(); }
  void visit(const allocate* op) override { std::abort(); }
  void visit(const make_buffer* op) override { std::abort(); }
  void visit(const clone_buffer* op) override { std::abort(); }
  void visit(const crop_buffer* op) override { std::abort(); }
  void visit(const crop_dim* op) override { std::abort(); }
  void visit(const slice_buffer* op) override { std::abort(); }
  void visit(const slice_dim* op) override { std::abort(); }
  void visit(const truncate_rank* op) override { std::abort(); }
  void visit(const check* op) override { std::abort(); }
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

  expr apply(expr op) {
    // std::cerr << "apply_rules: " << op << std::endl;
    symbol_map<expr> matches;
    for (const rule& r : rules_) {
      matches.clear();
      // std::cerr << "  Considering " << r.pattern << std::endl;
      if (match(r.pattern, op, matches)) {
        // std::cerr << "  Matched:" << std::endl;
        // for (const auto& i : matches) {
        //   std::cerr << "    " << i.first << ": " << i.second << std::endl;
        // }

        if (!r.predicate.defined() || prove_true(substitute(r.predicate, matches))) {
          // std::cerr << "  Applied " << r.pattern << " -> " << r.replacement << std::endl;
          op = substitute(r.replacement, matches);
          // std::cerr << "  Result: " << op << std::endl;
          return op;
        } else {
          // std::cerr << "  Failed predicate: " << r.predicate << std::endl;
        }
      }
    }
    // std::cerr << "  Failed" << std::endl;
    return op;
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
      {min(x + c0, c1), min(x, c1 - c0) + c0},
      {min(c0 - x, c0 - y), c0 - max(x, y)},
      {min(x, -x), -abs(x)},
      {min(x + c0, c0 - x), c0 - abs(x)},

      // Algebraic simplifications
      {min(x, x), x},
      {min(x, max(x, y)), x},
      {min(x, min(x, y)), min(x, y)},
      {min(min(x, y), y + c0), min(x, min(y, y + c0))},
      {min(min(x, y + c0), y), min(x, min(y, y + c0))},
      {min(max(x, y), min(x, z)), min(x, z)},
      {min(min(x, y), min(x, z)), min(x, min(y, z))},
      {min(max(x, y), max(x, z)), max(x, min(y, z))},
      {min(x, min(y, x + z)), min(y, min(x, x + z))},
      {min(x, min(y, x - z)), min(y, min(x, x - z))},
      {min(min(x, (y + z)), (y + w)), min(x, min(y + z, y + w))},
      {min(x / z, y / z), min(x, y) / z, z > 0},
      {min(x / z, y / z), max(x, y) / z, z < 0},
      {min(x * z, y * z), z * min(x, y), z > 0},
      {min(x * z, y * z), z * max(x, y), z < 0},
      {min(x + z, y + z), z + min(x, y)},
      {min(x - z, y - z), min(x, y) - z},
      {min(z - x, z - y), z - max(x, y)},
      {min(x + z, z - y), z + min(x, -y)},

      // Buffer meta simplifications
      // TODO: These rules are sketchy, they assume buffer_max(x, y) > buffer_min(x, y), which
      // is true if we disallow empty buffers...
      {min(buffer_min(x, y), buffer_max(x, y)), buffer_min(x, y)},
      {min(buffer_min(x, y), buffer_max(x, y) + c0), buffer_min(x, y), c0 > 0},
      {min(buffer_max(x, y), buffer_min(x, y) + c0), buffer_min(x, y) + c0, c0 < 0},
      {min(buffer_max(x, y) + c0, buffer_min(x, y) + c1), buffer_min(x, y) + c1, c0 > c1},
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
      {max(x + c0, c1), max(x, c1 - c0) + c0},
      {max(c0 - x, c0 - y), c0 - min(x, y)},
      {max(x, -x), abs(x)},
      {max(x + c0, c0 - x), abs(x) + c0},

      // Algebraic simplifications
      {max(x, x), x},
      {max(x, min(x, y)), x},
      {max(x, max(x, y)), max(x, y)},
      {max(max(x, y), y + c0), max(x, max(y, y + c0))},
      {max(max(x, y + c0), y), max(x, max(y, y + c0))},
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
      {max(buffer_max(x, y) + c0, buffer_min(x, y) + c1), buffer_max(x, y) + c0, c0 > c1},
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
      //{x + x * y, x * (y + 1)},  // Needs x to be non-constant or it loops with c0 * (x + c1) -> c0 * x + c0 * c1...
      // how?
      {x * y + x * z, x * (y + z)},
      {(x + y) + (x + z), x * 2 + (y + z)},
      {(x - y) + (x + z), x * 2 + (z - y)},
      {(y - x) + (x + z), y + z},
      {(x + y) + (x - z), x * 2 + (y - z)},
      {(x + y) + (z - x), y + z},
      {(x - y) + (x - z), x * 2 - (y + z)},
      {(y - x) + (x - z), y - z},
      {(x - y) + (z - x), z - y},
      {(y - x) + (z - x), (y + z) + x * -2},

      {(x + c0) + c1, x + (c0 + c1)},
      {(c0 - x) + c1, (c0 + c1) - x},
      {x + (c0 - y), (x - y) + c0},
      {x + (y + c0), (x + y) + c0},
      {(x + c0) + (y + c1), (x + y) + (c0 + c1)},

      {min(x, y - z) + z, min(y, x + z)},
      {max(x, y - z) + z, max(y, x + z)},

      {min(x + c0, y + c1) + c2, min(x + (c0 + c2), y + (c1 + c2))},
      {max(x + c0, y + c1) + c2, max(x + (c0 + c2), y + (c1 + c2))},
      {min(c0 - x, y + c1) + c2, min((c0 + c2) - x, y + (c1 + c2))},
      {max(c0 - x, y + c1) + c2, max((c0 + c2) - x, y + (c1 + c2))},
      {min(c0 - x, c1 - y) + c2, min((c0 + c2) - x, (c1 + c2) - y)},
      {max(c0 - x, c1 - y) + c2, max((c0 + c2) - x, (c1 + c2) - y)},
      {min(x, y + c0) + c1, min(x + c1, y + (c0 + c1))},
      {max(x, y + c0) + c1, max(x + c1, y + (c0 + c1))},

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
      {x - c0 * y, x + y * (-c0)},
      {x - (c0 - y), (x + y) - c0},
      {c0 - (x - y), (y - x) + c0},
      {x - (y + c0), (x - y) - c0},
      {(c0 - x) - y, c0 - (x + y)},
      {(x + c0) - y, (x - y) + c0},
      {(x + y) - x, y},
      {(x - y) - x, -y},
      {x - (x + y), -y},
      {x - (x - y), y},
      {(x + y) - (x + z), y - z},
      {(x - y) - (z - y), x - z},
      {(x - y) - (x - z), z - y},
      {(c0 - x) - (y - z), ((z - x) - y) + c0},
      {(x + c0) - (y + c1), (x - y) + (c0 - c1)},

      {(x + y) / c0 - x / c0, (y + (x % c0)) / c0, c0 > 0},

      {min(x, y + z) - z, min(y, x - z)},
      {max(x, y + z) - z, max(y, x - z)},

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
      {x / -1, -x},
      {x / x, x != 0},

      {(x / c0) / c1, x / (c0 * c1), c0 > 0 && c1 > 0},
      {(x / c0 + c1) / c2, (x + (c1 * c0)) / (c0 * c2), c0 > 0 && c2 > 0},
      {(x * c0) / c1, x * (c0 / c1), c0 % c1 == 0 && c1 > 0},

      {(x + c0) / c1, x / c1 + c0 / c1, c0 % c1 == 0},
      {(c0 - x) / c1, c0 / c1 + (-x / c1), c0 != 0 && c0 % c1 == 0},
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
      {min(x, min(y, z)) < y, min(x, z) < y},
      {max(x, y) < x, false},
      {x < max(x, y), x < y},
      {x < min(x, y), false},
      {min(x, y) < max(x, y), x != y},
      {max(x, y) < min(x, y), false},
      {min(x, y) < min(x, z), y < min(x, z)},

      {c0 < max(x, c1), c0 < x || c0 < c1},
      {c0 < min(x, c1), c0 < x && c0 < c1},
      {max(x, c0) < c1, x < c1 && c0 < c1},
      {min(x, c0) < c1, x < c1 || c0 < c1},

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

      {(x + c0) / c1 <= x / c1, c0 <= 0},
      {x / c1 <= (x + c0) / c1, 0 <= c0},

      {x <= x + y, 0 <= y},
      {x + y <= x, y <= 0},
      {x <= x - y, y <= 0},
      {x - y <= x, 0 <= y},
      {x + y <= x + z, y <= z},
      {x - y <= x - z, z <= y},
      {x - y <= z - y, x <= z},

      {min(x, y) <= x, true},
      {min(x, min(y, z)) <= y, true},
      {max(x, y) <= x, y <= x},
      {x <= max(x, y), true},
      {x <= min(x, y), x <= y},
      {min(x, y) <= max(x, y), true},
      {max(x, y) <= min(x, y), x == y},

      {c0 <= max(x, c1), c0 <= x || c0 <= c1},
      {c0 <= min(x, c1), c0 <= x && c0 <= c1},
      {max(x, c0) <= c1, x <= c1 && c0 <= c1},
      {min(x, c0) <= c1, x <= c1 || c0 <= c1},

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

expr simplify(const logical_not* op, expr a) {
  const index_t* cv = as_constant(a);
  if (cv) {
    return *cv == 0;
  }

  expr e;
  if (op && a.same_as(op->a)) {
    e = op;
  } else {
    e = logical_not::make(std::move(a));
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

}  // namespace slinky