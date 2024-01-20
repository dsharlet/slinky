#include "builder/simplify.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "builder/node_mutator.h"
#include "builder/substitute.h"
#include "runtime/depends_on.h"
#include "runtime/evaluate.h"
#include "runtime/expr.h"
#include "runtime/print.h"
#include "runtime/util.h"

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
  void visit(const if_then_else* op) override { std::abort(); }
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

// Check if the buffer metadata for `sym` is mutated in `s`.
bool is_buffer_mutated(symbol_id sym, const stmt& s) {
  class visitor : public recursive_node_visitor {
  public:
    symbol_id sym;
    bool result = false;

    visitor(symbol_id sym) : sym(sym) {}

    void visit(const crop_buffer* op) override {
      result = result || op->sym == sym;
      recursive_node_visitor::visit(op);
    }
    void visit(const crop_dim* op) override {
      result = result || op->sym == sym;
      recursive_node_visitor::visit(op);
    }
    void visit(const slice_buffer* op) override {
      result = result || op->sym == sym;
      recursive_node_visitor::visit(op);
    }
    void visit(const slice_dim* op) override {
      result = result || op->sym == sym;
      recursive_node_visitor::visit(op);
    }
    void visit(const truncate_rank* op) override {
      result = result || op->sym == sym;
      recursive_node_visitor::visit(op);
    }

    // If these nodes shadow the symbol we are looking for, ignore them.
    void visit(const allocate* op) override {
      if (op->sym == sym) return;
      recursive_node_visitor::visit(op);
    }
    void visit(const make_buffer* op) override {
      if (op->sym == sym) return;
      recursive_node_visitor::visit(op);
    }
    void visit(const clone_buffer* op) override {
      if (op->sym == sym) return;
      recursive_node_visitor::visit(op);
    }
    void visit(const let_stmt* op) override {
      if (op->sym == sym) return;
      recursive_node_visitor::visit(op);
    }

    using recursive_node_visitor::visit;
  };
  visitor v(sym);
  s.accept(&v);
  return v.result;
}

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
      //{x + x * y, x * (y + 1)},  // Needs x to be non-constant or it loops with c0 * (x + c1) -> c0 * x + c0 * c1... how?
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

namespace {

// This is based on the simplifier in Halide: https://github.com/halide/Halide/blob/main/src/Simplify_Internal.h
class simplifier : public node_mutator {
  symbol_map<int> references;
  symbol_map<box_expr> buffer_bounds;
  symbol_map<bool> bounds_used;
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
    if (!result.is_point() && match(result.min, result.max)) {
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

  void visit_symbol(symbol_id sym, bool bounds_used = true) {
    auto& ref_count = references[sym];
    if (!ref_count) {
      ref_count = 1;
    } else {
      *ref_count += 1;
    }
    if (bounds_used) {
      this->bounds_used[sym] = true;
    }
  }

  void visit(const variable* op) override {
    visit_symbol(op->sym);
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
    expr a = mutate(op->a, &bounds);

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
    if (!body.defined()) {
      set_result(stmt());
      return;
    }

    int refs = *references[op->sym];
    if (refs == 0) {
      // This let is dead
      set_result(body);
    } else if (is_variable(value, op->sym)) {
      set_result(body);
      // TODO: We could try substituting lets used once, or for simple lets, but we need to be careful because we can't
      // substitute values passed to call_stmt.
    } else if (value.same_as(op->value) && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(let_stmt::make(op->sym, std::move(value), std::move(body)));
    }
  }

  void visit(const loop* op) override {
    interval_expr bounds = mutate(op->bounds);
    expr step = mutate(op->step);

    if (prove_true(bounds.min > bounds.max)) {
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
    if (!body.defined()) {
      set_result(stmt());
      return;
    }

    if (op->mode == loop_mode::serial) {
      // Due to either scheduling or other simplifications, we can end up with a loop that runs a single call or copy on
      // contiguous crops of a buffer. In these cases, we can drop the loop in favor of just calling the body on the
      // union of the bounds covered by the loop.
      stmt result = body;
      std::vector<std::tuple<symbol_id, int, interval_expr>> new_crops;
      bool drop_loop = true;
      while (true) {
        // For now, we only handle crop_dim. I don't think crop_buffer can ever yield this simplification?
        if (const crop_dim* crop = result.as<crop_dim>()) {
          // Find the bounds of the crop on the next iteration.
          interval_expr next_iter = {
              substitute(crop->bounds.min, op->sym, var(op->sym) + op->step),
              substitute(crop->bounds.max, op->sym, var(op->sym) + op->step),
          };
          if (prove_true(crop->bounds.max + 1 >= next_iter.min || next_iter.max + 1 >= crop->bounds.min)) {
            result = crop->body;
            interval_expr new_crop = {
                substitute(crop->bounds.min, op->sym, op->bounds.min),
                substitute(crop->bounds.max, op->sym, op->bounds.max),
            };
            new_crops.emplace_back(crop->sym, crop->dim, new_crop);
          } else {
            // This crop was not contiguous, we can't drop the loop.
            drop_loop = false;
            break;
          }
        } else if (result.as<call_stmt>() || result.as<copy_stmt>()) {
          // We've found the actual body of the loop.
          break;
        } else {
          // TODO: We might be able to handle other cases too, like blocks of copies all to the same buffer (a
          // concatenate?).
          drop_loop = false;
          break;
        }
      }
      if (drop_loop) {
        // Rewrite the crops to cover the whole loop, and drop the loop.
        for (auto i = new_crops.rbegin(); i != new_crops.rend(); ++i) {
          result = crop_dim::make(std::get<0>(*i), std::get<1>(*i), std::get<2>(*i), result);
        }
        set_result(std::move(result));
        return;
      }
    }

    if (bounds.same_as(op->bounds) && step.same_as(op->step) && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(loop::make(op->sym, op->mode, std::move(bounds), std::move(step), std::move(body)));
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
      c = n->a;
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

  void visit(const call_stmt* op) override {
    for (symbol_id i : op->inputs) {
      visit_symbol(i, /*bounds_used=*/false);
    }
    for (symbol_id o : op->outputs) {
      visit_symbol(o);
    }
    node_mutator::visit(op);
  }

  void visit(const copy_stmt* op) override {
    visit_symbol(op->src);
    visit_symbol(op->dst);
    node_mutator::visit(op);
  }

  void visit(const allocate* op) override {
    std::vector<dim_expr> dims;
    box_expr bounds;
    dims.reserve(op->dims.size());
    stmt body = op->body;
    bool changed = false;
    for (std::size_t d = 0; d < op->dims.size(); ++d) {
      interval_expr bounds_d = mutate(op->dims[d].bounds);
      body = substitute_bounds(body, op->sym, d, bounds_d);
      dim_expr new_dim = {bounds_d, mutate(op->dims[d].stride), mutate(op->dims[d].fold_factor)};
      if (is_one(new_dim.fold_factor) || prove_true(new_dim.bounds.extent() == 1)) {
        new_dim.stride = 0;
        new_dim.fold_factor = expr();
      }
      changed = changed || !new_dim.same_as(op->dims[d]);
      dims.push_back(std::move(new_dim));
      bounds.push_back(std::move(bounds_d));
    }
    auto set_bounds = set_value_in_scope(buffer_bounds, op->sym, std::move(bounds));
    body = mutate(body);
    if (!body.defined()) {
      set_result(stmt());
    } else if (changed || !body.same_as(op->body)) {
      set_result(allocate::make(op->sym, op->storage, op->elem_size, std::move(dims), std::move(body)));
    } else {
      set_result(op);
    }
  }

  void visit(const make_buffer* op) override {
    expr base = mutate(op->base);
    expr elem_size = mutate(op->elem_size);
    std::vector<dim_expr> dims;
    box_expr bounds;
    dims.reserve(op->dims.size());
    bounds.reserve(op->dims.size());
    bool changed = false;
    stmt body = op->body;
    for (std::size_t d = 0; d < op->dims.size(); ++d) {
      interval_expr new_bounds = mutate(op->dims[d].bounds);
      body = substitute_bounds(body, op->sym, d, new_bounds);
      dim_expr new_dim = {new_bounds, mutate(op->dims[d].stride), mutate(op->dims[d].fold_factor)};
      if (is_one(new_dim.fold_factor) || prove_true(new_dim.bounds.extent() == 1)) {
        new_dim.stride = 0;
        new_dim.fold_factor = expr();
      }
      changed = changed || !new_dim.same_as(op->dims[d]);
      dims.push_back(std::move(new_dim));
      bounds.push_back(std::move(new_bounds));
    }

    auto set_bounds = set_value_in_scope(buffer_bounds, op->sym, std::move(bounds));
    body = mutate(body);
    if (!body.defined()) {
      set_result(stmt());
      return;
    }

    if (const call* bc = base.as<call>()) {
      if (bc->intrinsic == intrinsic::buffer_base) {
        // Check if this make_buffer is truncate_rank, or a clone.
        const symbol_id* src_buf = as_variable(bc->args[0]);
        if (src_buf) {
          var buf(*src_buf);
          if (match(elem_size, buffer_elem_size(buf))) {
            bool is_clone = true;
            for (index_t d = 0; d < static_cast<index_t>(dims.size()); ++d) {
              is_clone = is_clone && match(dims[d], buffer_dim(buf, d));
            }
            if (is_clone) {
              if (*src_buf == op->sym) {
                set_result(mutate(truncate_rank::make(op->sym, dims.size(), std::move(body))));
                return;
              }
              const std::optional<box_expr>& src_bounds = buffer_bounds[*src_buf];
              if (src_bounds && src_bounds->size() == dims.size()) {
                if (!is_buffer_mutated(op->sym, body) && !is_buffer_mutated(*src_buf, body)) {
                  // This is a clone of src_buf, and we never mutate either buffer, we can just re-use it.
                  set_result(let_stmt::make(op->sym, buf, std::move(body)));
                } else {
                  // This is a clone of src_buf, but we've mutated one of them. Use clone_buffer instead.
                  set_result(clone_buffer::make(op->sym, *src_buf, std::move(body)));
                }
                return;
              }
            }
          }
        }
      }

      // Check if this make_buffer is equivalent to slice_buffer or crop_buffer
      var buf(op->sym);
      if (bc->intrinsic == intrinsic::buffer_at && match(bc->args[0], buf) && match(elem_size, buffer_elem_size(buf))) {
        // To be a slice, we need every dimension that is present in the buffer_at call to be skipped, and the rest of
        // the dimensions to be identity.
        std::size_t dim = 0;
        std::size_t slice_rank = 0;
        bool is_slice = true;
        for (index_t d = 0; d < static_cast<index_t>(dims.size()); ++d) {
          if (d + 1 < static_cast<index_t>(bc->args.size()) && bc->args[d + 1].defined()) {
            // Skip this dimension.
            ++dim;
          } else {
            // This arg is undefined. We need to find the next dimension here to be a slice.
            ++slice_rank;
            is_slice = is_slice && match(dims[dim], buffer_dim(buf, d));
          }
        }
        if (is_slice && slice_rank == dims.size()) {
          std::vector<expr> at(bc->args.begin() + 1, bc->args.end());
          set_result(slice_buffer::make(op->sym, std::move(at), std::move(body)));
          return;
        }

        // To be a crop, we need dimensions to either be identity, or the buffer_at argument is the same as the min.
        bool is_crop = bc->args.size() <= dims.size() + 1;
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

  // Crop bounds like min(buffer_max(x, d), y) can be rewritten to just y because the crop will clamp anyways.
  static expr simplify_crop_bound(expr x, symbol_id sym, int dim) {
    if (const class max* m = x.as<class max>()) {
      if (is_buffer_min(m->a, sym, dim)) return simplify_crop_bound(m->b, sym, dim);
      if (is_buffer_min(m->b, sym, dim)) return simplify_crop_bound(m->a, sym, dim);
    } else if (const class min* m = x.as<class min>()) {
      if (is_buffer_max(m->a, sym, dim)) return simplify_crop_bound(m->b, sym, dim);
      if (is_buffer_max(m->b, sym, dim)) return simplify_crop_bound(m->a, sym, dim);
    }
    return x;
  }

  static interval_expr simplify_crop_bounds(interval_expr i, symbol_id sym, int dim) {
    return {simplify_crop_bound(i.min, sym, dim), simplify_crop_bound(i.max, sym, dim)};
  }

  void visit(const crop_buffer* op) override {
    // This is the bounds of the buffer as we understand them, for simplifying the inner scope.
    box_expr bounds(op->bounds.size());
    // This is the new bounds of the crop operation. Crops that are no-ops become undefined here.
    box_expr new_bounds(op->bounds.size());

    // If possible, rewrite crop_buffer of one dimension to crop_dim.
    expr sym_var = variable::make(op->sym);
    std::optional<box_expr> prev_bounds = buffer_bounds[op->sym];
    index_t dims_count = 0;
    bool changed = false;
    for (index_t i = 0; i < static_cast<index_t>(op->bounds.size()); ++i) {
      interval_expr bounds_i = simplify_crop_bounds(mutate(op->bounds[i]), op->sym, i);
      if (prove_true(bounds_i.min <= buffer_min(sym_var, i))) bounds_i.min = expr();
      if (prove_true(bounds_i.max >= buffer_max(sym_var, i))) bounds_i.max = expr();
      changed = changed || !bounds_i.same_as(op->bounds[i]);

      bounds[i] = bounds_i;

      // If the new bounds are the same as the existing bounds, set the crop in this dimension to
      // be undefined.
      if (prev_bounds && i < static_cast<index_t>(prev_bounds->size())) {
        if (prove_true(bounds_i.min <= (*prev_bounds)[i].min)) bounds_i.min = expr();
        if (prove_true(bounds_i.max >= (*prev_bounds)[i].max)) bounds_i.max = expr();
      }
      if (bounds_i.min.defined()) new_bounds[i].min = bounds_i.min;
      if (bounds_i.max.defined()) new_bounds[i].max = bounds_i.max;
      dims_count += bounds_i.min.defined() || bounds_i.max.defined() ? 1 : 0;
    }

    stmt body = op->body;
    for (index_t d = 0; d < static_cast<index_t>(new_bounds.size()); ++d) {
      if (new_bounds[d].min.defined() || new_bounds[d].max.defined()) {
        body = substitute_bounds(body, op->sym, d, new_bounds[d]);
      }
    }
    {
      auto set_bounds_used = set_value_in_scope(bounds_used, op->sym, false);
      auto set_bounds = set_value_in_scope(buffer_bounds, op->sym, bounds);
      body = mutate(body);
      if (!body.defined() || !*bounds_used[op->sym]) {
        set_result(body);
        return;
      }
    }

    // Remove trailing undefined bounds.
    while (!new_bounds.empty() && !new_bounds.back().min.defined() && !new_bounds.back().max.defined()) {
      new_bounds.pop_back();
    }
    if (new_bounds.empty()) {
      // This crop was a no-op.
      set_result(std::move(body));
    } else if (const block* b = body.as<block>()) {
      set_result(block::make(
          mutate(crop_buffer::make(op->sym, new_bounds, b->a)), mutate(crop_buffer::make(op->sym, new_bounds, b->b))));
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
    interval_expr bounds = simplify_crop_bounds(mutate(op->bounds), op->sym, op->dim);
    expr sym_var = variable::make(op->sym);
    if (prove_true(bounds.min <= buffer_min(sym_var, op->dim))) bounds.min = expr();
    if (prove_true(bounds.max >= buffer_max(sym_var, op->dim))) bounds.max = expr();
    if (!bounds.min.defined() && !bounds.max.defined()) {
      set_result(mutate(op->body));
      return;
    }

    std::optional<box_expr> buf_bounds = buffer_bounds[op->sym];
    if (buf_bounds && op->dim < static_cast<index_t>(buf_bounds->size())) {
      interval_expr& dim = (*buf_bounds)[op->dim];
      if (prove_true(bounds.min <= dim.min)) bounds.min = expr();
      if (prove_true(bounds.max >= dim.max)) bounds.max = expr();

      if (!bounds.min.defined() && !bounds.max.defined()) {
        // This crop is a no-op.
        set_result(mutate(op->body));
        return;
      }
      if (bounds.min.defined()) (*buf_bounds)[op->dim].min = bounds.min;
      if (bounds.max.defined()) (*buf_bounds)[op->dim].max = bounds.max;
    }

    stmt body = substitute_bounds(op->body, op->sym, op->dim, bounds);
    {
      auto set_bounds_used = set_value_in_scope(bounds_used, op->sym, false);
      auto set_bounds = set_value_in_scope(buffer_bounds, op->sym, buf_bounds);
      body = mutate(body);
      if (!body.defined() || !*bounds_used[op->sym]) {
        set_result(body);
        return;
      }
    }


    if (const slice_dim* slice = body.as<slice_dim>()) {
      if (slice->sym == op->sym && slice->dim == op->dim) {
        // This is a slice of the same dimension of the buffer we just cropped.
        // Don't drop the clamp that crop performs.
        expr at = clamp(slice->at, bounds.min, bounds.max);
        set_result(mutate(slice_dim::make(op->sym, op->dim, at, slice->body)));
        return;
      }
    } else if (const crop_dim* crop = body.as<crop_dim>()) {
      if (crop->sym == op->sym) {
        if (crop->dim == op->dim) {
          // Two nested crops of the same dimension, do one crop of the intersection instead.
          set_result(mutate(crop_dim::make(op->sym, op->dim, bounds & crop->bounds, crop->body)));
          return;
        } else {
          // TODO: This is a nested crop of the same buffer, use crop_buffer instead.
        }
      }
    }

    if (const block* b = body.as<block>()) {
      set_result(block::make(mutate(crop_dim::make(op->sym, op->dim, bounds, b->a)),
          mutate(crop_dim::make(op->sym, op->dim, bounds, b->b))));
    } else if (bounds.same_as(op->bounds) && body.same_as(op->body)) {
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
    stmt body = op->body;
    if (bounds) {
      body = substitute_bounds(body, op->sym, *bounds);
    }

    {
      auto set_bounds = set_value_in_scope(buffer_bounds, op->sym, bounds);
      body = mutate(body);
    }
    if (!body.defined()) {
      set_result(stmt());
      return;
    }

    // Remove trailing undefined bounds.
    while (!at.empty() && !at.back().defined()) {
      at.pop_back();
    }
    changed = changed || at.size() != op->at.size();
    if (at.empty()) {
      // This slice was a no-op.
      set_result(std::move(body));
    } else if (const block* b = body.as<block>()) {
      set_result(
          block::make(mutate(slice_buffer::make(op->sym, at, b->a)), mutate(slice_buffer::make(op->sym, at, b->b))));
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
    stmt body = op->body;
    if (bounds) {
      if (op->dim < static_cast<index_t>(bounds->size())) {
        bounds->erase(bounds->begin() + op->dim);
      }
      body = substitute_bounds(body, op->sym, *bounds);
    }

    {
      auto set_bounds = set_value_in_scope(buffer_bounds, op->sym, bounds);
      body = mutate(body);
    }
    if (!body.defined()) {
      set_result(stmt());
    } else if (const block* b = body.as<block>()) {
      set_result(block::make(
          mutate(slice_dim::make(op->sym, op->dim, at, b->a)), mutate(slice_dim::make(op->sym, op->dim, at, b->b))));
    } else if (at.same_as(op->at) && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(slice_dim::make(op->sym, op->dim, std::move(at), std::move(body)));
    }
  }

  void visit(const truncate_rank* op) override {
    std::optional<box_expr> bounds = buffer_bounds[op->sym];
    if (bounds) {
      if (static_cast<int>(bounds->size()) > op->rank) {
        bounds->resize(op->rank);
      } else {
        // truncate_rank can't add dimensions.
        assert(static_cast<int>(bounds->size()) > op->rank);
        // This truncate is a no-op.
        set_result(mutate(op->body));
        return;
      }
    }

    stmt body;
    {
      auto set_bounds = set_value_in_scope(buffer_bounds, op->sym, bounds);
      body = mutate(op->body);
    }
    if (!body.defined()) {
      set_result(stmt());
    } else if (const block* b = body.as<block>()) {
      set_result(block::make(
          mutate(truncate_rank::make(op->sym, op->rank, b->a)), mutate(truncate_rank::make(op->sym, op->rank, b->b))));
    } else if (body.same_as(op->body)) {
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
interval_expr bounds_of_linear(const T* op, interval_expr a, interval_expr b) {
  return {simplify(op, std::move(a.min), std::move(b.min)), simplify(op, std::move(a.max), std::move(b.max))};
}

template <typename T>
interval_expr bounds_of_less(const T* op, interval_expr a, interval_expr b) {
  // This bit of genius comes from
  // https://github.com/halide/Halide/blob/61b8d384b2b799cd47634e4a3b67aa7c7f580a46/src/Bounds.cpp#L829
  return {simplify(op, std::move(a.max), std::move(b.min)), simplify(op, std::move(a.min), std::move(b.max))};
}

}  // namespace

interval_expr bounds_of(const add* op, interval_expr a, interval_expr b) {
  return bounds_of_linear(op, std::move(a), std::move(b));
}
interval_expr bounds_of(const sub* op, interval_expr a, interval_expr b) {
  return {simplify(op, std::move(a.min), std::move(b.max)), simplify(op, std::move(a.max), std::move(b.min))};
}
interval_expr bounds_of(const mul* op, interval_expr a, interval_expr b) {
  // TODO: I'm pretty sure there are cases missing here that would produce simpler bounds than the fallback cases.
  if (is_non_negative(a.min) && is_non_negative(b.min)) {
    // Both are >= 0, neither intervals flip.
    return {simplify(op, a.min, b.min), simplify(op, a.max, b.max)};
  } else if (is_non_positive(a.max) && is_non_positive(b.max)) {
    // Both are <= 0, both intervals flip.
    return {simplify(op, a.max, b.max), simplify(op, a.min, b.min)};
  } else if (b.is_point()) {
    if (is_non_negative(b.min)) {
      return {simplify(op, a.min, b.min), simplify(op, a.max, b.min)};
    } else if (is_non_positive(b.min)) {
      return {simplify(op, a.max, b.min), simplify(op, a.min, b.min)};
    } else {
      expr corners[] = {
          simplify(op, a.min, b.min),
          simplify(op, a.max, b.min),
      };
      return {
          simplify(static_cast<const class min*>(nullptr), corners[0], corners[1]),
          simplify(static_cast<const class max*>(nullptr), corners[0], corners[1]),
      };
    }
  } else if (a.is_point()) {
    if (is_non_negative(a.min)) {
      return {simplify(op, a.min, b.min), simplify(op, a.min, b.max)};
    } else if (is_non_positive(a.min)) {
      return {simplify(op, a.min, b.max), simplify(op, a.min, b.min)};
    } else {
      expr corners[] = {
          simplify(op, a.min, b.min),
          simplify(op, a.min, b.max),
      };
      return {
          simplify(static_cast<const class min*>(nullptr), corners[0], corners[1]),
          simplify(static_cast<const class max*>(nullptr), corners[0], corners[1]),
      };
    }
  } else {
    // We don't know anything. The results is the union of all 4 possible intervals.
    expr corners[] = {
        simplify(op, a.min, b.min),
        simplify(op, a.min, b.max),
        simplify(op, a.max, b.min),
        simplify(op, a.max, b.max),
    };
    return {
        simplify(static_cast<const class min*>(nullptr),
            simplify(static_cast<const class min*>(nullptr), corners[0], corners[1]),
            simplify(static_cast<const class min*>(nullptr), corners[2], corners[3])),
        simplify(static_cast<const class max*>(nullptr),
            simplify(static_cast<const class max*>(nullptr), corners[0], corners[1]),
            simplify(static_cast<const class max*>(nullptr), corners[2], corners[3])),
    };
  }
}
interval_expr bounds_of(const div* op, interval_expr a, interval_expr b) {
  // Because b is an integer, the bounds of a will only be shrunk
  // (we define division by 0 to be 0). The absolute value of the
  // bounds are maximized when b is 1 or -1.
  if (b.is_point() && is_zero(b.min)) {
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
interval_expr bounds_of(const mod* op, interval_expr a, interval_expr b) { return {0, max(abs(b.min), abs(b.max))}; }

interval_expr bounds_of(const class min* op, interval_expr a, interval_expr b) {
  return bounds_of_linear(op, std::move(a), std::move(b));
}
interval_expr bounds_of(const class max* op, interval_expr a, interval_expr b) {
  return bounds_of_linear(op, std::move(a), std::move(b));
}

interval_expr bounds_of(const less* op, interval_expr a, interval_expr b) {
  return bounds_of_less(op, std::move(a), std::move(b));
}
interval_expr bounds_of(const less_equal* op, interval_expr a, interval_expr b) {
  return bounds_of_less(op, std::move(a), std::move(b));
}
interval_expr bounds_of(const equal* op, interval_expr a, interval_expr b) {
  return {0, simplify(static_cast<const logical_and*>(nullptr),
                 simplify(static_cast<const less_equal*>(nullptr), a.min, b.max),
                 simplify(static_cast<const less_equal*>(nullptr), b.min, a.max))};
}
interval_expr bounds_of(const not_equal* op, interval_expr a, interval_expr b) {
  return {simplify(static_cast<const logical_or*>(nullptr), simplify(static_cast<const less*>(nullptr), a.max, b.min),
              simplify(static_cast<const less*>(nullptr), b.max, a.min)),
      1};
}

interval_expr bounds_of(const logical_and* op, interval_expr a, interval_expr b) {
  return bounds_of_linear(op, std::move(a), std::move(b));
}
interval_expr bounds_of(const logical_or* op, interval_expr a, interval_expr b) {
  return bounds_of_linear(op, std::move(a), std::move(b));
}
interval_expr bounds_of(const logical_not* op, interval_expr a) {
  return {simplify(op, std::move(a.max)), simplify(op, std::move(a.min))};
}

interval_expr bounds_of(const class select* op, interval_expr c, interval_expr t, interval_expr f) {
  if (is_true(c.min)) {
    return t;
  } else if (is_false(c.max)) {
    return f;
  } else {
    return f | t;
  }
}

interval_expr bounds_of(const call* op, std::vector<interval_expr> args) {
  switch (op->intrinsic) {
  case intrinsic::abs:
    assert(args.size() == 1);
    if (is_positive(args[0].min)) {
      return {args[0].min, args[0].max};
    } else {
      expr abs_min = simplify(op, {args[0].min});
      expr abs_max = simplify(op, {args[0].max});
      return {0, simplify(static_cast<const class max*>(nullptr), std::move(abs_min), std::move(abs_max))};
    }
  default: return {op, op};
  }
}

interval_expr bounds_of(const expr& x, const bounds_map& expr_bounds) {
  simplifier s(expr_bounds);
  interval_expr bounds;
  s.mutate(x, &bounds);
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

    void visit(const variable* op) { leaves.push_back(op); }
    void visit(const constant* op) { leaves.push_back(op); }
    void visit(const call* op) { if (is_buffer_intrinsic(op->intrinsic)) leaves.push_back(op); }
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
  void visit_variable(const T* op) {
    if (op->sym == dx) {
      set_result(1);
    } else {
      set_result(expr(0));
    }
  }

  void visit(const variable* op) override { visit_variable(op); }
  void visit(const wildcard* op) override { visit_variable(op); }
  void visit(const constant*) override { set_result(expr(0)); }

  void visit(const mul* op) override {
    if (depends_on(op->a, dx) && depends_on(op->b, dx)) {
      expr da = mutate(op->a);
      expr db = mutate(op->b);
      set_result(simplify(op, op->a, db) + simplify(op, da, op->b));
    } else if (depends_on(op->a, dx)) {
      set_result(simplify(op, mutate(op->a), op->b));
    } else if (depends_on(op->b, dx)) {
      set_result(simplify(op, op->a, mutate(op->b)));
    } else {
      set_result(expr(0));
    }
  }
  void visit(const div* op) override {
    if (depends_on(op->a, dx) && depends_on(op->b, dx)) {
      expr da = mutate(op->a);
      expr db = mutate(op->b);
      set_result((da * op->b - op->a * db) / (op->b * op->b));
    } else if (depends_on(op->a, dx)) {
      set_result(mutate(op->a) / op->b);
    } else if (depends_on(op->b, dx)) {
      expr db = mutate(op->b);
      set_result(-op->a / (op->b * op->b));
    } else {
      set_result(expr(0));
    }
  }

  virtual void visit(const mod* op) override { set_result(indeterminate()); }
  virtual void visit(const class min* op) override { set_result(select(op->a < op->b, mutate(op->a), mutate(op->b))); }
  virtual void visit(const class max* op) override { set_result(select(op->b < op->a, mutate(op->a), mutate(op->b))); }

  template <typename T>
  void visit_compare(const T* op) {
    if (depends_on(op->a, dx) || depends_on(op->b, dx)) {
      set_result(indeterminate());
    } else {
      set_result(expr(0));
    }
  }

  virtual void visit(const equal* op) override { visit_compare(op); }
  virtual void visit(const not_equal* op) override { visit_compare(op); }
  virtual void visit(const less* op) override { visit_compare(op); }
  virtual void visit(const less_equal* op) override { visit_compare(op); }
  virtual void visit(const logical_and* op) override {}
  virtual void visit(const logical_or* op) override {}
  virtual void visit(const logical_not* op) override { set_result(-mutate(op->a)); }

  virtual void visit(const class select* op) override {
    set_result(select(op->condition, mutate(op->true_value), mutate(op->false_value)));
  }

  virtual void visit(const call* op) override { std::abort(); }
};

}  // namespace

expr differentiate(const expr& f, symbol_id x) { return derivative(x).mutate(f); }

}  // namespace slinky
