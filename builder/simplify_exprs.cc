#include "builder/simplify.h"

#include <algorithm>
#include <cassert>

#include "builder/rewrite.h"
#include "builder/simplify_rules.h"
#include "runtime/evaluate.h"

namespace slinky {

using namespace rewrite;

expr simplify(const class min* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }

  auto ca = as_constant(a);
  auto cb = as_constant(b);
  if (ca && cb) {
    return std::min(*ca, *cb);
  }

  if (is_indeterminate(a)) return a;
  if (is_indeterminate(b)) return b;

  auto r = make_rewriter(min(pattern_expr{a}, pattern_expr{b}));
  if (apply_min_rules(r)) {
    return r.result;
  } else if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return expr(op);
  } else {
    return min::make(std::move(a), std::move(b));
  }
}

expr simplify(const class max* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }

  auto ca = as_constant(a);
  auto cb = as_constant(b);
  if (ca && cb) {
    return std::max(*ca, *cb);
  }

  if (is_indeterminate(a)) return a;
  if (is_indeterminate(b)) return b;

  auto r = make_rewriter(max(pattern_expr{a}, pattern_expr{b}));
  if (apply_max_rules(r)) {
    return r.result;
  } else if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return expr(op);
  } else {
    return max::make(std::move(a), std::move(b));
  }
}

expr simplify(const add* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }

  auto ca = as_constant(a);
  auto cb = as_constant(b);
  if (ca && cb && !add_overflows(*ca, *cb)) {
    return *ca + *cb;
  }

  if (is_indeterminate(a)) return a;
  if (is_indeterminate(b)) return b;
  int inf_a = is_infinity(a);
  int inf_b = is_infinity(b);
  if (inf_a && inf_b) return inf_a == inf_b ? a : slinky::indeterminate();

  auto r = make_rewriter(pattern_expr{a} + pattern_expr{b});
  if (apply_add_rules(r)) {
    return r.result;
  } else if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return expr(op);
  } else {
    return add::make(std::move(a), std::move(b));
  }
}

expr simplify(const sub* op, expr a, expr b) {
  auto ca = as_constant(a);
  auto cb = as_constant(b);
  if (ca && cb && !sub_overflows(*ca, *cb)) {
    return *ca - *cb;
  } else if (cb && !sub_overflows<index_t>(0, *cb)) {
    // Canonicalize to addition with constants.
    return simplify(static_cast<add*>(nullptr), a, -*cb);
  }

  if (is_indeterminate(a)) return a;
  if (is_indeterminate(b)) return b;
  int inf_a = is_infinity(a);
  int inf_b = is_infinity(b);
  if (inf_a && inf_b) return inf_a == inf_b ? slinky::indeterminate() : a;

  auto r = make_rewriter(pattern_expr{a} - pattern_expr{b});
  if (apply_sub_rules(r)) {
    return r.result;
  } else if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return expr(op);
  } else {
    return sub::make(std::move(a), std::move(b));
  }
}

expr simplify(const mul* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }

  auto ca = as_constant(a);
  auto cb = as_constant(b);
  if (ca && cb && !mul_overflows(*ca, *cb)) {
    return *ca * *cb;
  }

  if (is_indeterminate(a)) return a;
  if (is_indeterminate(b)) return b;
  int inf_a = is_infinity(a);
  int inf_b = is_infinity(b);
  if (inf_a && inf_b) return infinity(inf_a * inf_b);

  auto r = make_rewriter(pattern_expr{a} * pattern_expr{b});
  if (apply_mul_rules(r)) {
    return r.result;
  } else if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return expr(op);
  } else {
    return mul::make(std::move(a), std::move(b));
  }
}

expr simplify(const div* op, expr a, expr b) {
  auto ca = as_constant(a);
  auto cb = as_constant(b);
  if (ca && cb) {
    return euclidean_div(*ca, *cb);
  }

  if (is_indeterminate(a)) return a;
  if (is_indeterminate(b)) return b;
  if (is_infinity(a) && is_infinity(b)) return slinky::indeterminate();

  auto r = make_rewriter(pattern_expr{a} / pattern_expr{b});
  if (apply_div_rules(r)) {
    return r.result;
  } else if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return expr(op);
  } else {
    return div::make(std::move(a), std::move(b));
  }
}

expr simplify(const mod* op, expr a, expr b) {
  auto ca = as_constant(a);
  auto cb = as_constant(b);
  if (ca && cb) {
    return euclidean_mod(*ca, *cb);
  }

  auto r = make_rewriter(pattern_expr{a} % pattern_expr{b});
  if (apply_mod_rules(r)) {
    return r.result;
  } else if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return expr(op);
  } else {
    return mod::make(std::move(a), std::move(b));
  }
}

expr simplify(const less* op, expr a, expr b) {
  auto ca = as_constant(a);
  auto cb = as_constant(b);
  if (ca && cb) {
    return *ca < *cb;
  }

  auto r = make_rewriter(pattern_expr{a} < pattern_expr{b});
  if (apply_less_rules(r)) {
    return r.result;
  } else if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return expr(op);
  } else {
    return less::make(std::move(a), std::move(b));
  }
}

expr simplify(const less_equal* op, expr a, expr b) {
  // Rewrite to !(b < a) and simplify that instead.
  expr result = simplify(static_cast<const logical_not*>(nullptr),
      simplify(static_cast<const less*>(nullptr), std::move(b), std::move(a)));
  if (op) {
    if (const less_equal* le = result.as<less_equal>()) {
      if (le->a.same_as(op->a) && le->b.same_as(op->b)) {
        return expr(op);
      }
    }
  }
  return result;
}

expr simplify(const equal* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }

  auto ca = as_constant(a);
  auto cb = as_constant(b);
  if (ca && cb) {
    return *ca == *cb;
  }

  auto r = make_rewriter(pattern_expr{a} == pattern_expr{b});
  if (apply_equal_rules(r)) {
    return r.result;
  } else if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return expr(op);
  } else {
    return equal::make(std::move(a), std::move(b));
  }
}

expr simplify(const not_equal* op, expr a, expr b) {
  // Rewrite to !(a == b) and simplify that instead.
  expr result = simplify(static_cast<const logical_not*>(nullptr),
      simplify(static_cast<const equal*>(nullptr), std::move(a), std::move(b)));
  if (op) {
    if (const not_equal* ne = result.as<not_equal>()) {
      if (ne->a.same_as(op->a) && ne->b.same_as(op->b)) {
        return expr(op);
      }
    }
  }
  return result;
}

expr simplify(const logical_and* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }
  auto ca = as_constant(a);
  auto cb = as_constant(b);

  if (ca && cb) {
    return *ca != 0 && *cb != 0;
  } else if (cb) {
    return *cb ? boolean(a) : false;
  }

  auto r = make_rewriter(pattern_expr{a} && pattern_expr{b});
  if (apply_logical_and_rules(r)) {
    return r.result;
  } else if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return expr(op);
  } else {
    return logical_and::make(std::move(a), std::move(b));
  }
}

expr simplify(const logical_or* op, expr a, expr b) {
  if (should_commute(a, b)) {
    std::swap(a, b);
  }
  auto ca = as_constant(a);
  auto cb = as_constant(b);

  if (ca && cb) {
    return *ca != 0 || *cb != 0;
  } else if (cb) {
    return *cb ? true : boolean(a);
  }

  auto r = make_rewriter(pattern_expr{a} || pattern_expr{b});
  if (apply_logical_or_rules(r)) {
    return r.result;
  } else if (op && a.same_as(op->a) && b.same_as(op->b)) {
    return expr(op);
  } else {
    return logical_or::make(std::move(a), std::move(b));
  }
}

expr simplify(const class logical_not* op, expr a) {
  auto cv = as_constant(a);
  if (cv) {
    return *cv == 0;
  }

  auto r = make_rewriter(!pattern_expr{a});
  if (apply_logical_not_rules(r)) {
    return r.result;
  } else if (op && a.same_as(op->a)) {
    return expr(op);
  } else {
    return logical_not::make(std::move(a));
  }
}

expr simplify(const class select* op, expr c, expr t, expr f) {
  if (is_true(c)) {
    return t;
  } else if (is_false(c)) {
    return f;
  }

  auto r = make_rewriter(select(pattern_expr{c}, pattern_expr{t}, pattern_expr{f}));
  if (apply_select_rules(r)) {
    return r.result;
  } else if (op && c.same_as(op->condition) && t.same_as(op->true_value) && f.same_as(op->false_value)) {
    return expr(op);
  } else {
    return select::make(std::move(c), std::move(t), std::move(f));
  }
}

expr simplify(const call* op, intrinsic fn, std::vector<expr> args) {
  bool constant = true;
  bool changed = op == nullptr;
  for (std::size_t i = 0; i < args.size(); ++i) {
    constant = constant && as_constant(args[i]);
    changed = changed || !args[i].same_as(op->args[i]);
  }

  if (fn == intrinsic::buffer_at) {
    for (index_t d = 1; d < static_cast<index_t>(args.size()); ++d) {
      auto buf = as_variable(args[0]);
      assert(buf);
      // buffer_at(b, buffer_min(b, 0)) is equivalent to buffer_at(b, <>)
      if (args[d].defined() && is_variable(args[d], *buf, buffer_field::min, d - 1)) {
        args[d] = expr();
        changed = true;
      }
    }
    // Trailing undefined args have no effect.
    while (args.size() > 1 && !args.back().defined()) {
      args.pop_back();
      changed = true;
    }
  } else if (fn == intrinsic::abs) {
    assert(args.size() == 1);
    if (is_non_negative(args[0])) {
      return args[0];
    } else if (is_non_positive(args[0])) {
      return simplify(static_cast<const sub*>(nullptr), 0, std::move(args[0]));
    }
  }

  expr e;
  if (!changed) {
    assert(op);
    e = expr(op);
  } else {
    e = call::make(fn, std::move(args));
  }

  if (constant && can_evaluate(fn)) {
    return evaluate(e);
  }

  rewriter r(e);
  if (apply_call_rules(r)) {
    return r.result;
  } else {
    return e;
  }
}

interval_expr simplify_intersection(interval_expr a, interval_expr b) {
  if (a.min.defined() && b.min.defined()) {
    a.min = simplify(static_cast<const class max*>(nullptr), std::move(a.min), std::move(b.min));
  } else if (!a.min.defined()) {
    a.min = std::move(b.min);
  }
  if (a.max.defined() && b.max.defined()) {
    a.max = simplify(static_cast<const class min*>(nullptr), std::move(a.max), std::move(b.max));
  } else if (!a.max.defined()) {
    a.max = std::move(b.max);
  }
  if (match(a.min, a.max)) {
    // Ensure that is_point is true if the result is a point.
    a.max = a.min;
  }
  return a;
}

}  // namespace slinky