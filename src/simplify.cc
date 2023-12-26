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

struct rule {
  expr pattern;
  expr replacement;
  expr predicate;
};

expr buffer_min(expr buf, expr dim) { return load_buffer_meta::make(std::move(buf), buffer_meta::min, std::move(dim)); }
expr buffer_max(expr buf, expr dim) { return load_buffer_meta::make(std::move(buf), buffer_meta::max, std::move(dim)); }
expr buffer_extent(expr buf, expr dim) {
  return load_buffer_meta::make(std::move(buf), buffer_meta::extent, std::move(dim));
}

class simplifier : public node_mutator {
  symbol_map<int> references;
  symbol_map<box> buffer_bounds;
  symbol_map<interval> expr_bounds;

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
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);
    if (ca && cb) {
      e = std::min(*ca, *cb);
      return;
    }
    if (ca && !cb) { std::swap(a, b); }
    if (a.same_as(op->a) && b.same_as(op->b)) {
      e = op;
    } else {
      e = min(a, b);
    }

    static std::vector<rule> rules = {
        {min(x, std::numeric_limits<index_t>::max()), x},
        {min(x, std::numeric_limits<index_t>::min()), std::numeric_limits<index_t>::min()},
        {min(c0, min(x, c1)), min(x, min(c0, c1))},
        {min(min(x, c0), c1), min(x, min(c0, c1))},
        {min(x, x + c0), x, c0 > 0},
        {min(x, x + c0), x + c0, c0 < 0},
        {min(x + c0, y + c1), min(x, y + c1 - c0) + c0},
        {min(x + c0, c1), min(x, c1 - c0) + c0},
        {min(x + c0, y), min(y, x + c0)},
        {min(x, x), x},
        {min(x / z, y / z), min(x, y) / z, z > 0},
        {min(x + z, y + z), min(x, y) + z},
        {min(z + x, z + y), z + min(x, y)},
        {min(x - z, y - z), min(x, y) - z},
        {min(z - x, z - y), z - max(x, y)},
        {min(buffer_min(x, y), buffer_max(x, y)), buffer_min(x, y)},
        {min(buffer_min(x, y), buffer_max(x, y) + c0), buffer_min(x, y), c0 > 0},
        {min(buffer_max(x, y), buffer_min(x, y)), buffer_min(x, y)},
        {min(buffer_max(x, y), buffer_min(x, y) + c0), buffer_min(x, y) + c0, c0 < 0},
    };
    e = apply_rules(rules, e);
  }

  void visit(const class max* op) override {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);
    if (ca && cb) {
      e = std::max(*ca, *cb);
      return;
    }
    if (ca && !cb) { std::swap(a, b); }
    if (a.same_as(op->a) && b.same_as(op->b)) {
      e = op;
    } else {
      e = max(a, b);
    }

    static std::vector<rule> rules = {
        {max(x, std::numeric_limits<index_t>::min()), x},
        {max(x, std::numeric_limits<index_t>::max()), std::numeric_limits<index_t>::max()},
        {max(c0, max(x, c1)), max(x, max(c0, c1))},
        {max(max(x, c0), c1), max(x, max(c0, c1))},
        {max(x, x + c0), x + c0, c0 > 0},
        {max(x, x + c0), x, c0 < 0},
        {max(x + c0, y + c1), max(x, y + c1 - c0) + c0},
        {max(x + c0, c1), max(x, c1 - c0) + c0},
        {max(x + c0, y), max(y, x + c0)},
        {max(x, x), x},
        {max(x / z, y / z), max(x, y) / z, z > 0},
        {max(x + z, y + z), max(x, y) + z},
        {max(z + x, z + y), z + max(x, y)},
        {max(x - z, y - z), max(x, y) - z},
        {max(z - x, z - y), z - min(x, y)},
        {max(buffer_min(x, y), buffer_max(x, y)), buffer_max(x, y)},
        {max(buffer_min(x, y), buffer_max(x, y) + c0), buffer_max(x, y) + c0, c0 > 0},
        {max(buffer_max(x, y), buffer_min(x, y)), buffer_max(x, y)},
        {max(buffer_max(x, y), buffer_min(x, y) + c0), buffer_max(x, y), c0 < 0},
    };
    e = apply_rules(rules, e);
  }

  void visit(const add* op) override {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);
    if (ca && cb) {
      e = *ca + *cb;
      return;
    }
    if (ca && !cb) { std::swap(a, b); }
    if (a.same_as(op->a) && b.same_as(op->b)) {
      e = op;
    } else {
      e = a + b;
    }

    static std::vector<rule> rules = {
        {x + 0, x},
        {x + x, x * 2},
        {x + (0 - y), x - y},
        {(0 - x) + y, y - x},
        {(x + c0) + c1, x + (c0 + c1)},
        {(x + c0) + (y + c1), (x + y) + (c0 + c1)},
        {buffer_min(x, y) + buffer_extent(x, y), buffer_max(x, y) + 1},
        {buffer_extent(x, y) + buffer_min(x, y), buffer_max(x, y) + 1},
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
        {x - x, 0},
        {x - 0, x},
        {x - (0 - y), x + y},
        {(x + y) - x, y},
        {(x + c0) - (y + c1), (x - y) + (c0 - c1)},
        {buffer_max(x, y) - buffer_min(x, y), buffer_extent(x, y) - 1},
    };
    e = apply_rules(rules, e);
  }

  void visit(const mul* op) override {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);
    if (ca && cb) {
      e = *ca * *cb;
      return;
    }
    if (ca && !cb) { std::swap(a, b); }
    if (a.same_as(op->a) && b.same_as(op->b)) {
      e = op;
    } else {
      e = a * b;
    }

    static std::vector<rule> rules = {
        {x * 0, 0},
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
        {x % 1, 0},
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
    if (ca) {
      if (a.same_as(op->a) && b.same_as(op->b)) {
        e = op;
      } else {
        e = a < b;
      }
    } else if (cb) {
      e = mutate(-*cb < -a);
    } else {
      e = mutate(0 < b - a);
    }

    static std::vector<rule> rules = {
        {c0 < c1 + x, c0 - c1 < x},
        {c0 < c1 - x, c0 - c1 < -x, c1 != 0},
        {c0 < x + c1, c0 - c1 < x},
        {c0 < buffer_extent(x, y), true, c0 < 0},
        {c0 < -buffer_extent(x, y), false, 0 < c0},
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
    // Canonicalize to constant <= other
    if (ca) {
      if (a.same_as(op->a) && b.same_as(op->b)) {
        e = op;
      } else {
        e = a <= b;
      }
    } else if (cb) {
      e = mutate(-*cb <= -a);
    } else {
      e = mutate(0 <= b - a);
    }

    static std::vector<rule> rules = {
        {c0 <= c1 + x, c0 - c1 <= x},
        {c0 <= c1 - x, c0 - c1 <= -x, c1 != 0},
        {c0 <= x + c1, c0 - c1 <= x},
        {c0 <= buffer_extent(x, y), true, c0 <= 0},
        {c0 <= -buffer_extent(x, y), false, 0 <= c0},
    };
    e = apply_rules(rules, e);
  }

  void visit(const equal* op) override {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);
    if (ca && cb) {
      e = *ca == *cb;
      return;
    }
    // Canonicalize to constant == other
    if (ca) {
      if (a.same_as(op->a) && b.same_as(op->b)) {
        e = op;
      } else {
        e = a == b;
      }
    } else if (cb) {
      e = b == a;
    } else {
      e = mutate(0 == b - a);
    }

    static std::vector<rule> rules = {
        {c0 == c1 + x, c0 - c1 == x},
        {c0 == c1 - x, c0 - c1 == -x, c1 != 0},
        {c0 == x + c1, c0 - c1 == x},
    };
    e = apply_rules(rules, e);
  }

  void visit(const not_equal* op) override {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);
    if (ca && cb) {
      e = *ca != *cb;
      return;
    }
    // Canonicalize to constant == other
    if (ca) {
      if (a.same_as(op->a) && b.same_as(op->b)) {
        e = op;
      } else {
        e = a != b;
      }
    } else if (cb) {
      e = b != a;
    } else {
      e = mutate(0 != b - a);
    }

    static std::vector<rule> rules = {
        {c0 != c1 + x, c0 - c1 != x},
        {c0 != c1 - x, c0 - c1 != -x, c1 != 0},
        {c0 != x + c1, c0 - c1 != x},
    };
    e = apply_rules(rules, e);
  }

  void visit(const select* op) override {
    expr c = mutate(op->condition);
    if (is_true(c)) {
      e = mutate(op->true_value);
      return;
    } else if (is_false(c)) {
      e = mutate(op->false_value);
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
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);

    // Canonicalize to constant && other
    if (cb && !ca) {
      std::swap(a, b);
      std::swap(ca, cb);
    }

    if (ca && cb) {
      e = *ca != 0 && *cb != 0;
      return;
    } else if (ca) {
      if (*ca) {
        e = b;
        return;
      } else {
        e = std::move(a);
        return;
      }
    }

    if (a.same_as(op->a) && b.same_as(op->b)) {
      e = op;
    } else {
      e = a && b;
    }
  }

  void visit(const logical_or* op) override {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);

    // Canonicalize to constant && other
    if (cb && !ca) { 
      std::swap(a, b);
      std::swap(ca, cb);
    }

    if (ca && cb) {
      e = *ca != 0 || *cb != 0;
      return;
    } else if (ca) {
      if (*ca) {
        e = std::move(a);
        return;
      } else {
        e = std::move(b);
        return;
      }
    }

    if (a.same_as(op->a) && b.same_as(op->b)) {
      e = op;
    } else {
      e = a || b;
    }
  }

  template <typename T>
  auto visit_let(const T* op) {
    expr value = mutate(op->value);

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

    // TODO: We can't actually simplify anything using this yet.
    auto set_bounds = set_value_in_scope(expr_bounds, op->name, interval(begin, end - 1));
    stmt body = mutate(op->body);

    if (begin.same_as(op->begin) && end.same_as(op->end) && body.same_as(op->body)) {
      s = op;
    } else {
      s = loop::make(op->name, std::move(begin), std::move(end), std::move(body));
    }
  }

  void visit(const if_then_else* op) override { 
    expr c = mutate(op->condition); 
    if (is_true(c)) {
      s = mutate(op->true_body);
      return;
    } else if (is_false(c)) {
      s = op->false_body.defined() ? mutate(op->false_body) : stmt();
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

  void visit(const allocate* op) override {
    std::vector<dim_expr> dims;
    box bounds;
    dims.reserve(op->dims.size());
    for (const dim_expr& i : op->dims) {
      expr min = mutate(i.min);
      expr extent = mutate(i.extent);
      dims.emplace_back(min, extent, mutate(i.stride_bytes), mutate(i.fold_factor));
      bounds.emplace_back(min, mutate(min + extent - 1));
    }
    auto set_bounds = set_value_in_scope(buffer_bounds, op->name, std::move(bounds));
    stmt body = mutate(op->body);
    s = allocate::make(op->type, op->name, op->elem_size, std::move(dims), std::move(body));
  }

  virtual void visit(const make_buffer* op) override {
    expr base = mutate(op->base);
    std::vector<dim_expr> dims;
    box bounds;
    dims.reserve(op->dims.size());
    for (const dim_expr& i : op->dims) {
      expr min = mutate(i.min);
      expr extent = mutate(i.extent);
      dims.emplace_back(min, extent, mutate(i.stride_bytes), mutate(i.fold_factor));
      bounds.emplace_back(min, mutate(min + extent - 1));
    }
    auto set_bounds = set_value_in_scope(buffer_bounds, op->name, std::move(bounds));
    stmt body = mutate(op->body);
    s = make_buffer::make(op->name, std::move(base), op->elem_size, std::move(dims), std::move(body));
  }

  void visit(const crop_buffer* op) override {
    box bounds(op->bounds.size());

    // If possible, rewrite crop_buffer of one dimension to crop_dim.
    int one_dim = -1;
    bool any_defined = false;
    std::optional<box> prev_bounds = buffer_bounds[op->name];
    for (int i = 0; i < static_cast<int>(op->bounds.size()); ++i) {
      expr min = mutate(op->bounds[i].min);
      expr max = mutate(op->bounds[i].max);
      if (prev_bounds && i < prev_bounds->size()) {
        // TODO: If we switch to min/max everywhere (instead of min/extent in some places),
        // then we could check and default these individually.
        if (match(min, (*prev_bounds)[i].min) && match(max, (*prev_bounds)[i].max)) {
          min = expr();
          max = expr();
        }
      }
      if (min.defined() || max.defined()) {
        if (!any_defined) {
          one_dim = i;
        } else {
          one_dim = -1;
        }
        any_defined = true;
      }
      bounds[i] = {min, max};
    }

    auto set_bounds = set_value_in_scope(buffer_bounds, op->name, bounds);
    stmt body = mutate(op->body);
    if (one_dim >= 0) {
      interval& dim = bounds[one_dim];
      s = crop_dim::make(op->name, one_dim, dim.min, dim.extent(), std::move(body));
    } else {
      // Remove trailing undefined bounds.
      while (bounds.size() > 0 && !bounds.back().min.defined() && !bounds.back().max.defined()) {
        bounds.pop_back();
      }
      s = crop_buffer::make(op->name, std::move(bounds), std::move(body));
    }
  }

  void visit(const crop_dim* op) override {
    expr min = mutate(op->min);
    expr extent = mutate(op->extent);

    std::optional<box> bounds = buffer_bounds[op->name];
    if (bounds && bounds->size() > op->dim) {
      interval& dim = (*bounds)[op->dim];
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
};

}  // namespace

expr simplify(const expr& e) { return simplifier().mutate(e); }
stmt simplify(const stmt& s) { return simplifier().mutate(s); }

bool can_prove(const expr& e) {
  expr simplified = simplify(e);
  if (const index_t* c = as_constant(simplified)) { return *c != 0; }
  return false;
}

}  // namespace slinky
