#include "builder/simplify.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "base/arithmetic.h"
#include "base/chrome_trace.h"
#include "builder/node_mutator.h"
#include "builder/rewrite.h"
#include "builder/substitute.h"
#include "runtime/depends_on.h"
#include "runtime/evaluate.h"
#include "runtime/expr.h"
#include "runtime/print.h"

namespace slinky {

namespace {

expr strip_boolean(expr x) {
  if (const not_equal* ne = x.as<not_equal>()) {
    if (is_zero(ne->b)) {
      return strip_boolean(ne->a);
    }
    // This should be canonicalized to the RHS.
    assert(!is_zero(ne->a));
  }
  return x;
}

const expr& eval_buffer_intrinsic(intrinsic fn, const dim_expr& d) {
  switch (fn) {
  case intrinsic::buffer_min: return d.bounds.min;
  case intrinsic::buffer_max: return d.bounds.max;
  case intrinsic::buffer_stride: return d.stride;
  case intrinsic::buffer_fold_factor: return d.fold_factor;
  default: std::abort();
  }
}

bool deep_is_point(const interval_expr& x) { return x.is_point() || match(x.min, x.max); }

// Ensure that an interval that is a point in a deep equality sense is also a point in a shallow equality sense.
void ensure_is_point(interval_expr& x) {
  if (!x.is_point() && match(x.min, x.max)) {
    x.max = x.min;
  }
}

// Rewrite `make_decl(block::make(stmts))` to be `block::make(make_decl(i) for i in stmts if i depends on sym else i)`.
template <class Fn>
stmt lift_decl_invariants(std::vector<stmt> stmts, var sym, Fn&& make_decl) {
  for (stmt& i : stmts) {
    if (depends_on(i, sym).any()) {
      i = make_decl(i);
    }
  }
  return block::make(std::move(stmts));
}

// Given an expression that produces a base pointer for a buffer, find the buffer the pointer is from.
var find_buffer(const expr& e) {
  class v : public recursive_node_visitor {
  public:
    var result;

    void visit(const call* op) override {
      if (op->intrinsic == intrinsic::buffer_at) {
        assert(!result.defined());
        result = *as_variable(op->args[0]);
      }
    }
  };
  v finder;
  if (e.defined()) e.accept(&finder);
  return finder.result;
}

template <typename T>
bool match(const std::vector<T>& a, const std::vector<T>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!match(a[i], b[i])) return false;
  }
  return true;
}

// Like the above, except `a` is represented by a single non-default value at index `idx`.
template <typename T>
bool match(int idx, const T& a, const std::vector<T>& b) {
  if (idx >= static_cast<int>(b.size())) return false;
  for (int i = 0; i < static_cast<int>(b.size()); ++i) {
    if (i == idx) {
      if (!match(a, b[idx])) return false;
    } else {
      if (!match(T(), b[i])) return false;
    }
  }
  return true;
}

// Find the buffers accessed in a stmt. This is trickier than it seems, we need to track the lineage of buffers and
// report the buffer as it is visible to the caller. So something like:
//
//   x = crop_dim(y, ...) {
//     call(f, {x}, {})
//   }
//
// Will report that y is consumed in the stmt, not x.
class find_buffers_accessed : public recursive_node_visitor {
  bool consumed;
  symbol_map<var> aliases;

public:
  std::set<var> result;

  find_buffers_accessed(bool consumed) : consumed(consumed) {}

  std::optional<var> lookup_alias(var x) {
    if (aliases.contains(x)) {
      return aliases.lookup(x);
    } else {
      return x;
    }
  }

  void visit_buffer(var i) {
    if (aliases.contains(i)) {
      if (aliases.lookup(i)->defined()) {
        result.insert(*aliases.lookup(i));
      }
    } else {
      result.insert(i);
    }
  }

  void visit(const call_stmt* op) override {
    if (consumed) {
      for (const var& i : op->inputs) {
        visit_buffer(i);
      }
    } else {
      for (const var& i : op->outputs) {
        visit_buffer(i);
      }
    }
  }
  void visit(const copy_stmt* op) override {
    if (consumed) {
      visit_buffer(op->src);
    } else {
      visit_buffer(op->dst);
    }
  }

  void visit(const allocate* op) override {
    if (!op->body.defined()) return;
    auto s = set_value_in_scope(aliases, op->sym, var());
    op->body.accept(this);
  }
  void visit(const make_buffer* op) override {
    if (!op->body.defined()) return;
    auto s = set_value_in_scope(aliases, op->sym, lookup_alias(find_buffer(op->base)));
    op->body.accept(this);
  }

  template <typename T>
  void visit_buffer_alias(const T* op) {
    if (!op->body.defined()) return;
    auto s = set_value_in_scope(aliases, op->sym, lookup_alias(op->src));
    op->body.accept(this);
  }
  void visit(const crop_buffer* op) override { visit_buffer_alias(op); }
  void visit(const crop_dim* op) override { visit_buffer_alias(op); }
  void visit(const slice_buffer* op) override { visit_buffer_alias(op); }
  void visit(const slice_dim* op) override { visit_buffer_alias(op); }
  void visit(const transpose* op) override { visit_buffer_alias(op); }
  void visit(const clone_buffer* op) override { visit_buffer_alias(op); }
};

std::set<var> buffers_accessed(const stmt& s, bool consumed) {
  find_buffers_accessed accessed(consumed);
  if (s.defined()) s.accept(&accessed);
  return accessed.result;
}

// The algorithm at https://en.cppreference.com/w/cpp/algorithm/set_intersection, but detects any intersection.
template <typename It>
bool empty_intersection(It a_begin, It a_end, It b_begin, It b_end) {
  It a = a_begin;
  It b = b_begin;
  while (a != a_end && b != b_end) {
    if (*a == *b) {
      return false;
    } else if (*a < *b) {
      ++a;
    } else {
      ++b;
    }
  }
  return true;
}

template <typename T>
bool empty_intersection(const std::set<T>& a, const std::set<T>& b) {
  return empty_intersection(a.begin(), a.end(), b.begin(), b.end());
}

// If a constant can be added to an expression without creating new nodes, this helper produces that expression,
// otherwise expr().
class constant_adder : public node_mutator {
public:
  index_t c;

  constant_adder(index_t c) : c(c) {}

  void visit(const constant* op) override { set_result(op->value + c); };
  void visit(const variable* op) override { set_result(expr()); }

  void visit(const let* op) override {
    expr body = mutate(op->body);
    if (body.defined()) {
      set_result(let::make(op->lets, std::move(body)));
    } else {
      set_result(expr());
    }
  }

  template <typename T>
  void visit_add_sub(const T* op, int sign_b) {
    expr a = mutate(op->a);
    if (a.defined()) {
      set_result(T::make(std::move(a), op->b));
      return;
    }
    c *= sign_b;
    expr b = mutate(op->b);
    c *= sign_b;
    if (b.defined()) {
      set_result(T::make(op->a, std::move(b)));
      return;
    }
    set_result(expr());
  }

  void visit(const add* op) override { visit_add_sub(op, /*sign_b=*/1); }
  void visit(const sub* op) override { visit_add_sub(op, /*sign_b=*/-1); }

  template <typename T>
  void visit_min_max(const T* op) {
    expr a = mutate(op->a);
    if (a.defined()) {
      expr b = mutate(op->b);
      if (b.defined()) {
        set_result(T::make(std::move(a), std::move(b)));
        return;
      }
    }
    set_result(expr());
  }
  void visit(const class min* op) override { visit_min_max(op); }
  void visit(const class max* op) override { visit_min_max(op); }
  void visit(const class select* op) override {
    expr t = mutate(op->true_value);
    if (t.defined()) {
      expr f = mutate(op->false_value);
      if (f.defined()) {
        set_result(select::make(op->condition, std::move(t), std::move(f)));
        return;
      }
    }
    set_result(expr());
  }

  void visit(const mul* op) override {
    if (auto b = as_constant(op->b)) {
      // a*b + c == (a + c/b)*b if c%b == 0
      if (*b != 0 && euclidean_mod(c, *b) == 0) {
        c = euclidean_div(c, *b);
        expr a = mutate(op->a);
        c *= *b;
        if (a.defined()) {
          set_result(mul::make(std::move(a), op->b));
          return;
        }
      }
    }
    set_result(expr());
  }
  void visit(const div* op) override { set_result(expr()); }
  void visit(const mod* op) override { set_result(expr()); }
  void visit(const equal*) override { set_result(expr()); }
  void visit(const not_equal*) override { set_result(expr()); }
  void visit(const less*) override { set_result(expr()); }
  void visit(const less_equal*) override { set_result(expr()); }
  void visit(const logical_and*) override { set_result(expr()); }
  void visit(const logical_or*) override { set_result(expr()); }
  void visit(const logical_not*) override { set_result(expr()); }
  void visit(const call*) override { set_result(expr()); }
};

expr add_constant(const expr& a, index_t b) { return constant_adder(b).mutate(a); }

// This is based on the simplifier in Halide: https://github.com/halide/Halide/blob/main/src/Simplify_Internal.h
class simplifier : public node_mutator {
public:
  struct buffer_info {
    expr elem_size;

    // The dimension metadata for this buffer.
    std::vector<dim_expr> dims;

    // If true, we know that all the dimensions in the buffer are in the `dims` vector above. If not, there may be more
    // dimensions we don't know about.
    bool all_dims_known = false;

    // The op that defined this buffer.
    stmt decl;

    // Identifies the buffer this buffer is a descendent of, if any.
    var src;
  };

  struct expr_info {
    interval_expr bounds;
    alignment_type alignment;

    void trim_bounds_using_alignment() {
      if (alignment.modulus == 0) {
        bounds = point(alignment.remainder);
      } else if (alignment.modulus > 1) {
        auto bounds_min = as_constant(bounds.min);
        if (bounds_min) {
          index_t adjustment;
          bool no_overflow =
              !sub_with_overflow(alignment.remainder, euclidean_mod(*bounds_min, alignment.modulus), adjustment);
          adjustment = euclidean_mod(adjustment, alignment.modulus);
          index_t new_min;
          no_overflow &= !add_with_overflow(*bounds_min, adjustment, new_min);
          if (no_overflow) {
            bounds.min = new_min;
          }
        }
        auto bounds_max = as_constant(bounds.max);
        if (bounds_max) {
          index_t adjustment;
          bool no_overflow =
              !sub_with_overflow(euclidean_mod(*bounds_max, alignment.modulus), alignment.remainder, adjustment);
          adjustment = euclidean_mod(adjustment, alignment.modulus);
          index_t new_max;
          no_overflow &= !sub_with_overflow(*bounds_max, adjustment, new_max);
          if (no_overflow) {
            bounds.max = new_max;
          }
        }
      }

      if (bounds.is_point()) {
        auto c = as_constant(bounds.min);
        if (c) {
          alignment.modulus = 0;
          alignment.remainder = *c;
        }
      }
    }
  };

private:
  symbol_map<buffer_info> buffers;
  symbol_map<expr_info> info_map;

  expr_info result_info;

  void set_result(expr e, expr_info info) {
    assert(!result_info.bounds.min.defined() && !result_info.bounds.max.defined());
    result_info = std::move(info);
    node_mutator::set_result(std::move(e));
  }
  void set_result(const base_expr_node* e, expr_info info) { set_result(expr(e), std::move(info)); }
  void set_result(stmt s) { node_mutator::set_result(std::move(s)); }
  void set_result(const base_stmt_node* s) { set_result(stmt(s)); }
  // Dummy for template code.
  void set_result(stmt s, const expr_info&) { set_result(std::move(s)); }
  void set_result(const base_stmt_node* s, const expr_info&) { set_result(stmt(s)); }

public:
  simplifier() {}
  simplifier(const bounds_map& bounds, const alignment_map& alignment) {
    for (size_t ix = 0; ix < bounds.size(); ix++) {
      var id = var(ix);
      if (!bounds[id]) continue;
      info_map[id] = {*bounds[id], alignment_type()};
    }

    for (size_t ix = 0; ix < alignment.size(); ix++) {
      var id = var(ix);
      if (!alignment[id]) continue;
      if (info_map[id]) {
        info_map[id]->alignment = *alignment[id];
      } else {
        info_map[id] = {interval_expr(), *alignment[id]};
      }
    }
  }

  expr mutate(const expr& e, expr_info* info) {
    expr result = node_mutator::mutate(e);
    if (info) {
      ensure_is_point(result_info.bounds);
      if (info != &result_info) {
        *info = std::move(result_info);
      }
    } else {
      result_info = {{expr(), expr()}, alignment_type()};
    }
    return result;
  }
  // Dummy for template code.
  stmt mutate(const stmt& s, expr_info* info) { return node_mutator::mutate(s); }
  expr mutate(const expr& e) override { return mutate(e, nullptr); }
  stmt mutate(const stmt& s) override { return mutate(s, nullptr); }

  // When mutating a value x interpreted as boolean, we need to effectively mutate x != 0, but we can't do that directly
  // because it risks breaking the simplifiers ability to check if an expression has not changed. This helper emulates
  // this.
  expr mutate_boolean(const expr& e, expr_info* info) {
    expr result = strip_boolean(mutate(e, info));
    if (info && !is_boolean(result)) {
      info->bounds = bounds_of(static_cast<const not_equal*>(nullptr), std::move(info->bounds), point(0));
    }
    return result;
  }

  void mutate_and_set_result(const expr& e) {
    assert(!result_info.bounds.min.defined() && !result_info.bounds.max.defined());
    node_mutator::set_result(mutate(e, &result_info));
  }

  interval_expr mutate(const interval_expr& x, expr_info* min_info, expr_info* max_info) {
    if (deep_is_point(x)) {
      expr result = mutate(x.min, min_info);
      if (min_info && max_info) {
        *max_info = *min_info;
      }
      return point(std::move(result));
    } else {
      interval_expr result = {mutate(x.min, min_info), mutate(x.max, max_info)};
      // If the interval is of the form [select(b < a, b + 1, a), b], i.e. checking if the interval is empty, we can
      // just rewrite it to [a, b], because all empty intervals are equivalent.
      rewrite::pattern_wildcard<0> a;
      rewrite::pattern_wildcard<1> b;
      rewrite::match_context ctx;
      if (match(ctx, select(b < a, b + 1, a), result.min) && match(ctx.matched(b), result.max)) {
        result = {ctx.matched(a), ctx.matched(b)};
      }
      ensure_is_point(result);

      return result;
    }
  }
  interval_expr mutate(const interval_expr& x) override { return mutate(x, nullptr, nullptr); }

  // This class manages information learned from conditions that can be used to improve bounds.
  class knowledge {
    // This could be made a variant if we need to be able to update more than one kind of symbol_map.
    std::vector<scoped_value_in_symbol_map<expr_info>> k;

    symbol_map<expr_info>& info_map;

    alignment_type get_alignment(const var& v) { return info_map[v] ? info_map[v]->alignment : alignment_type(); }

  public:
    knowledge(symbol_map<expr_info>& info_map) : info_map(info_map) {}
    knowledge(const knowledge&) = delete;
    knowledge(knowledge&&) = default;
    ~knowledge() {
      // Destroy our knowledge in reverse order.
      while (!k.empty()) {
        k.pop_back();
      }
    }

    void learn_from_equal(const expr& a, const expr& b) {
      if (const variable* v = a.as<variable>()) {
        // bounds of a are [b, b].
        k.push_back(set_value_in_scope(info_map, v->sym, {point(b), get_alignment(v->sym)}));
      }
      if (const variable* v = b.as<variable>()) {
        // bounds of b are [a, a].
        k.push_back(set_value_in_scope(info_map, v->sym, {point(a), get_alignment(v->sym)}));
      }
    }

    void learn_from_less(const expr& a, const expr& b) {
      if (const variable* v = a.as<variable>()) {
        // a has an upper bound of b - 1
        const std::optional<expr_info>& old_info = info_map[v->sym];
        const expr& lb = old_info ? old_info->bounds.min : expr();
        k.push_back(set_value_in_scope(info_map, v->sym, {{lb, b - 1}, get_alignment(v->sym)}));
      }
      if (const variable* v = b.as<variable>()) {
        // b has a lower bound of a + 1
        const std::optional<expr_info>& old_info = info_map[v->sym];
        const expr& ub = old_info ? old_info->bounds.max : expr();
        k.push_back(set_value_in_scope(info_map, v->sym, {{a + 1, ub}, get_alignment(v->sym)}));
      }
    }
    void learn_from_less_equal(const expr& a, const expr& b) {
      if (const variable* v = a.as<variable>()) {
        // a has an upper bound of b
        const std::optional<expr_info>& old_info = info_map[v->sym];
        const expr& lb = old_info ? old_info->bounds.min : expr();
        k.push_back(set_value_in_scope(info_map, v->sym, {{lb, b}, get_alignment(v->sym)}));
      }
      if (const variable* v = b.as<variable>()) {
        // b has a lower bound of a
        const std::optional<expr_info>& old_info = info_map[v->sym];
        const expr& ub = old_info ? old_info->bounds.max : expr();
        k.push_back(set_value_in_scope(info_map, v->sym, {{a, ub}, get_alignment(v->sym)}));
      }
    }

    void learn_from_true(const expr& c) {
      if (const logical_and* a = c.as<logical_and>()) {
        learn_from_true(a->a);
        learn_from_true(a->b);
      } else if (const logical_not* n = c.as<logical_not>()) {
        learn_from_false(n->a);
      } else if (const equal* eq = c.as<equal>()) {
        learn_from_equal(eq->a, eq->b);
      } else if (const less* lt = c.as<less>()) {
        learn_from_less(lt->a, lt->b);
      } else if (const less_equal* lt = c.as<less_equal>()) {
        learn_from_less_equal(lt->a, lt->b);
      }
    }
    void learn_from_false(const expr& c) {
      if (const logical_or* a = c.as<logical_or>()) {
        learn_from_false(a->a);
        learn_from_false(a->b);
      } else if (const logical_not* n = c.as<logical_not>()) {
        learn_from_true(n->a);
      } else if (const not_equal* ne = c.as<not_equal>()) {
        learn_from_equal(ne->a, ne->b);
      } else if (const less* lt = c.as<less>()) {
        learn_from_less_equal(lt->b, lt->a);
      } else if (const less_equal* lt = c.as<less_equal>()) {
        learn_from_less(lt->b, lt->a);
      }
    }
  };

  knowledge learn_from_true(const expr& c) {
    knowledge result(info_map);
    result.learn_from_true(c);
    return result;
  }
  knowledge learn_from_false(const expr& c) {
    knowledge result(info_map);
    result.learn_from_false(c);
    return result;
  }

  // When we attempt to prove things about bounds, we sometimes get constant expressions, but we can't recursively
  // simplify without a high risk of infinite recursion. We can evaluate these as constants instead.
  static bool prove_constant_true(const expr& e) {
    if (!e.defined()) return false;

    std::optional<index_t> ec = evaluate_constant(e);
    if (ec) return *ec != 0;

    // e is constant true if we know it has bounds that don't include zero.
    std::optional<index_t> a = evaluate_constant(constant_lower_bound(e));
    if (a && *a > 0) return true;
    std::optional<index_t> b = evaluate_constant(constant_upper_bound(e));
    return b && *b < 0;
  }

  static bool prove_constant_false(const expr& e) {
    if (!e.defined()) return false;

    std::optional<index_t> ec = evaluate_constant(e);
    if (ec) return *ec == 0;

    // e is constant false if we know its bounds are [0, 0].
    std::optional<index_t> a = evaluate_constant(constant_lower_bound(e));
    if (!a) return false;
    std::optional<index_t> b = evaluate_constant(constant_upper_bound(e));
    if (!b) return false;
    return *a == 0 && *b == 0;
  }

  std::optional<bool> attempt_to_prove(const expr& e) {
    scoped_trace trace("attempt_to_prove");
    expr_info info;
    mutate_boolean(e, &info);
    if (prove_constant_true(info.bounds.min)) {
      return true;
    } else if (prove_constant_false(info.bounds.max)) {
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
    std::optional<expr_info> info = info_map[op->sym];
    if (info) {
      if (!info->bounds.min.defined()) info->bounds.min = expr(op);
      if (!info->bounds.max.defined()) info->bounds.max = expr(op);
      set_result(op, std::move(*info));
    } else {
      set_result(op, {point(expr(op)), alignment_type()});
    }
  }

  void visit(const constant* op) override { set_result(op, {point(expr(op)), {0, op->value}}); }

  template <typename T>
  void visit_min_max(const T* op) {
    expr_info a_info;
    expr a = mutate(op->a, &a_info);
    expr_info b_info;
    expr b = mutate(op->b, &b_info);

    if (!a.defined() || !b.defined()) {
      set_result(expr(), expr_info());
      return;
    }

    // We need to check between the bounds and a/b themselves to avoid the possibility of something like:
    // min(x, y + 1) not simplifying if we know the bounds of x are [0, y] and the bounds of y are [z, w],
    // because we end up looking at min(y, z + 1) instead of min(y, y + 1).
    // TODO: This is quite expensive, we should try to find a better way.
    if (prove_constant_false(simplify(static_cast<const less*>(nullptr), b_info.bounds.min, a)) ||
        prove_constant_false(simplify(static_cast<const less*>(nullptr), b, a_info.bounds.max)) ||
        prove_constant_false(simplify(static_cast<const less*>(nullptr), b_info.bounds.min, a_info.bounds.max))) {
      if (T::static_type == expr_node_type::min) {
        set_result(std::move(a), std::move(a_info));
      } else {
        set_result(std::move(b), std::move(b_info));
      }
      return;
    } else if (prove_constant_false(simplify(static_cast<const less*>(nullptr), a_info.bounds.min, b)) ||
               prove_constant_false(simplify(static_cast<const less*>(nullptr), a, b_info.bounds.max)) ||
               prove_constant_false(
                   simplify(static_cast<const less*>(nullptr), a_info.bounds.min, b_info.bounds.max))) {
      if (T::static_type == expr_node_type::min) {
        set_result(std::move(b), std::move(b_info));
      } else {
        set_result(std::move(a), std::move(a_info));
      }
      return;
    }

    expr result = simplify(op, std::move(a), std::move(b));
    if (!result.same_as(op)) {
      mutate_and_set_result(result);
    } else {
      set_result(std::move(result),
          {bounds_of(op, std::move(a_info.bounds), std::move(b_info.bounds)), a_info.alignment | b_info.alignment});
    }
  }

  void visit(const class min* op) override { visit_min_max(op); }
  void visit(const class max* op) override { visit_min_max(op); }

  alignment_type modulus_of(const add* op, const alignment_type& a, const alignment_type& b) { return a + b; }
  alignment_type modulus_of(const sub* op, const alignment_type& a, const alignment_type& b) { return a - b; }
  alignment_type modulus_of(const mul* op, const alignment_type& a, const alignment_type& b) { return a * b; }
  alignment_type modulus_of(const div* op, const alignment_type& a, const alignment_type& b) { return a / b; }
  alignment_type modulus_of(const mod* op, const alignment_type& a, const alignment_type& b) { return a % b; }

  template <typename T>
  void visit_binary(const T* op) {
    expr_info a_info;
    expr a = mutate(op->a, &a_info);
    expr_info b_info;
    expr b = mutate(op->b, &b_info);

    if (!a.defined() || !b.defined()) {
      set_result(expr(), expr_info());
      return;
    }

    auto a_mod = a_info.alignment.modulus;
    auto a_rem = a_info.alignment.remainder;

    rewrite::pattern_wildcard<0> x;

    rewrite::pattern_constant<0> c0;
    rewrite::pattern_constant<1> c1;

    // It's really ugly to have rules here instead of simplify_rules, but plumbing bounds and alignment seems difficult.
    if (T::static_type == expr_node_type::div) {
      auto r = rewrite::make_rewriter(rewrite::pattern_expr{a} / rewrite::pattern_expr{b});
      // Taken from https://github.com/halide/Halide/blob/main/src/Simplify_Div.cpp#L125-L167.
      // clang-format off
      if (r((x + c0) / c1, x / c1 + eval(a_rem / c1 - (a_rem - c0) / c1), eval(a_mod % c1 == 0)) ||
          r((c0 - x) / c1, eval(a_rem / c1 + (c0 - a_rem) / c1) - x / c1, eval(a_mod % c1 == 0)) ||
          false) {
        mutate_and_set_result(r.result);
        return;
      }
      // clang-format on
    }

    expr result = simplify(op, std::move(a), std::move(b));
    if (!result.same_as(op)) {
      mutate_and_set_result(result);
    } else {
      expr_info info = {bounds_of(op, std::move(a_info.bounds), std::move(b_info.bounds)),
          modulus_of(op, a_info.alignment, b_info.alignment)};
      info.trim_bounds_using_alignment();
      set_result(std::move(result), std::move(info));
    }
  }
  void visit(const add* op) override {
    if (auto bc = as_constant(op->b)) {
      // We have a lot of rules that pull constants out of expressions. Sometimes we end up with complicated expressions
      // that add a constant, e.g. select(x, max(2, y + 3), 4) - 1 and we could put that constant back inside. However,
      // writing rules for all of these rewrites would be very tedious, so we handle it here instead.
      expr result = add_constant(op->a, *bc);
      if (result.defined()) {
        mutate_and_set_result(result);
        return;
      }
    }
    visit_binary(op);
  }
  void visit(const sub* op) override { visit_binary(op); }
  void visit(const mul* op) override { visit_binary(op); }
  void visit(const div* op) override { visit_binary(op); }
  void visit(const mod* op) override { visit_binary(op); }

  template <typename T>
  void visit_logical(const T* op, bool coerce_boolean = false) {
    expr_info a_info;
    expr a = coerce_boolean ? mutate_boolean(op->a, &a_info) : mutate(op->a, &a_info);
    expr_info b_info;
    expr b = coerce_boolean ? mutate_boolean(op->b, &b_info) : mutate(op->b, &b_info);

    if (!a.defined() || !b.defined()) {
      set_result(expr(), expr_info());
      return;
    }

    expr result = simplify(op, std::move(a), std::move(b));
    if (!result.same_as(op)) {
      mutate_and_set_result(result);
    } else {
      interval_expr result_info = bounds_of(op, std::move(a_info.bounds), std::move(b_info.bounds));
      if (prove_constant_true(result_info.min)) {
        set_result(expr(true), {point(true), alignment_type()});
      } else if (prove_constant_false(result_info.max)) {
        set_result(expr(false), {point(false), alignment_type()});
      } else {
        set_result(std::move(result), {std::move(result_info), alignment_type()});
      }
    }
  }
  void visit(const less* op) override { visit_logical(op); }
  void visit(const less_equal* op) override { visit_logical(op); }
  void visit(const equal* op) override { visit_logical(op); }
  void visit(const not_equal* op) override { visit_logical(op); }
  void visit(const logical_and* op) override { visit_logical(op, /*coerce_boolean=*/true); }
  void visit(const logical_or* op) override { visit_logical(op, /*coerce_boolean=*/true); }

  void visit(const logical_not* op) override {
    expr_info info;
    expr a = mutate_boolean(op->a, &info);

    if (!a.defined()) {
      set_result(expr(), expr_info());
    } else if (prove_constant_true(info.bounds.min)) {
      set_result(expr(false), {point(false), alignment_type()});
    } else if (prove_constant_false(info.bounds.max)) {
      set_result(expr(true), {point(true), alignment_type()});
    } else {
      expr result = simplify(op, std::move(a));
      if (result.same_as(op)) {
        set_result(std::move(result), {bounds_of(op, std::move(info.bounds)), alignment_type()});
      } else {
        mutate_and_set_result(result);
      }
    }
  }

  // substitute c = true into x.
  static expr substitute_true(expr x, const expr& c) {
    if (const logical_and* l = c.as<logical_and>()) {
      // If we assume a && b is true, then a and b both must be true.
      x = substitute_true(x, l->a);
      x = substitute_true(x, l->b);
    } else if (const logical_not* l = c.as<logical_not>()) {
      x = substitute_false(x, l->a);
    } else if (is_boolean(c) && !as_constant(c)) {
      x = substitute(x, c, true);
    }
    // Do this separately because we might be able to substitute c and one side of an equals too.
    if (const equal* e = c.as<equal>()) {
      if (e->b.as<constant>()) {
        x = substitute(x, e->a, e->b);
      } else if (e->a.as<constant>()) {
        x = substitute(x, e->b, e->a);
      }
    }
    return x;
  }

  // substitute c = false into x.
  static expr substitute_false(expr x, const expr& c) {
    if (const logical_or* l = c.as<logical_or>()) {
      // If we assume a || b is false, then a and b both must be false.
      x = substitute_false(x, l->a);
      x = substitute_false(x, l->b);
    } else if (const logical_not* l = c.as<logical_not>()) {
      x = substitute_true(x, l->a);
    } else if (is_boolean(c) && !as_constant(c)) {
      x = substitute(x, c, false);
    }
    // Do this separately because we might be able to substitute c and one side of an equals too.
    if (const not_equal* e = c.as<not_equal>()) {
      if (e->b.as<constant>()) {
        x = substitute(x, e->a, e->b);
      } else if (e->a.as<constant>()) {
        x = substitute(x, e->b, e->a);
      }
    }
    return x;
  }

  void visit(const class select* op) override {
    expr_info c_info;
    // When simplifying expressions treated as bools, we need to force them to have the result 0 or 1.
    expr c = mutate_boolean(op->condition, &c_info);
    if (!c.defined()) {
      set_result(expr(), expr_info());
      return;
    } else if (prove_constant_true(c_info.bounds.min)) {
      mutate_and_set_result(op->true_value);
      return;
    } else if (prove_constant_false(c_info.bounds.max)) {
      mutate_and_set_result(op->false_value);
      return;
    }

    expr t = op->true_value;
    expr f = op->false_value;

    t = substitute_true(t, c);
    f = substitute_false(f, c);

    expr t_when_c_false = substitute_false(t, c);
    expr f_when_c_true = substitute_true(f, c);
    if (!t_when_c_false.same_as(t) && prove_true(t_when_c_false == f)) {
      mutate_and_set_result(t);
      return;
    } else if (!f_when_c_true.same_as(f) && prove_true(f_when_c_true == t)) {
      mutate_and_set_result(f);
      return;
    }

    expr_info t_info;
    {
      auto knowledge = learn_from_true(c);
      t = mutate(t, &t_info);
    }
    expr_info f_info;
    {
      auto knowledge = learn_from_false(c);
      f = mutate(f, &f_info);
    }

    if (!t.defined() && !f.defined()) {
      set_result(expr(), expr_info());
      return;
    }

    expr e = simplify(op, std::move(c), std::move(t), std::move(f));
    if (e.same_as(op)) {
      expr_info info = {bounds_of(op, std::move(c_info.bounds), std::move(t_info.bounds), std::move(f_info.bounds)),
          t_info.alignment | f_info.alignment};
      info.trim_bounds_using_alignment();
      set_result(std::move(e), std::move(info));
    } else {
      mutate_and_set_result(e);
    }
  }

  static bool should_substitute(const expr& e) { return e.as<constant>() || e.as<variable>(); }

  void visit(const call* op) override {
    std::vector<expr> args;
    std::vector<interval_expr> args_bounds;
    args.reserve(op->args.size());
    args_bounds.reserve(op->args.size());
    for (const expr& i : op->args) {
      expr_info i_info;
      args.push_back(mutate(i, &i_info));
      args_bounds.push_back(std::move(i_info.bounds));
    }

    if (is_buffer_intrinsic(op->intrinsic)) {
      assert(args.size() >= 1);
      if (!args[0].defined()) {
        set_result(expr(), expr_info());
        return;
      }
      auto buf = as_variable(op->args[0]);
      assert(buf);
      const std::optional<buffer_info>& info = buffers[*buf];
      if (info) {
        // TODO: We substitute here because we can't prove things like buffer_elem_size(x) == buffer_elem_size(y) where
        // x is a crop of y. If we can fix that, we don't need to substitute here, which seems better.
        if (op->intrinsic == intrinsic::buffer_elem_size) {
          const expr& value = info->elem_size;
          if (should_substitute(value) || value.as<call>()) {
            set_result(value, {point(value), alignment_type()});
          } else {
            set_result(op, {point(value), alignment_type()});
          }
          return;
        } else if (is_buffer_dim_intrinsic(op->intrinsic)) {
          auto dim = as_constant(op->args[1]);
          assert(dim);
          if (*dim < static_cast<index_t>(info->dims.size())) {
            const expr& value = eval_buffer_intrinsic(op->intrinsic, info->dims[*dim]);
            if (should_substitute(value) || value.as<call>()) {
              set_result(value, {point(value), alignment_type()});
            } else {
              set_result(op, {point(value), alignment_type()});
            }
            return;
          }
        } else if (op->intrinsic == intrinsic::buffer_at) {
          for (int d = 0; d < static_cast<int>(std::min(info->dims.size(), args.size() - 1)); ++d) {
            if (!info->dims[d].fold_factor.defined() && prove_true(args[d + 1] == info->dims[d].bounds.min)) {
              // This argument is equal to the default value, and we know it is in bounds.
              args[d + 1] = expr();
            } else if (info->dims[d].fold_factor.defined() && prove_true(args[d + 1] == 0)) {
              // This argument is equal to the default value, and we know it is in bounds.
              args[d + 1] = expr();
            }
          }
        }
      }
    } else if (op->intrinsic == intrinsic::abs) {
      assert(args.size() == 1);
      assert(args_bounds.size() == 1);
      if (prove_constant_true(args_bounds[0].min >= 0)) {
        set_result(std::move(args[0]), {std::move(args_bounds[0]), alignment_type()});
        return;
      } else if (prove_constant_true(args_bounds[0].max <= 0)) {
        mutate_and_set_result(-args[0]);
        return;
      }
    }

    expr e = simplify(op, op->intrinsic, std::move(args));
    if (e.same_as(op)) {
      set_result(std::move(e), {bounds_of(op, std::move(args_bounds)), alignment_type()});
    } else {
      mutate_and_set_result(e);
    }
  }

  template <typename T>
  void visit_let(const T* op) {
    std::vector<std::pair<var, expr>> lets;
    lets.reserve(op->lets.size());

    using sv_type = scoped_value_in_symbol_map<expr_info>;
    std::vector<sv_type> scoped_values;
    scoped_values.reserve(op->lets.size());

    bool values_changed = false;
    for (const auto& s : op->lets) {
      expr_info value_info;
      lets.emplace_back(s.first, mutate(s.second, &value_info));
      values_changed = values_changed || !lets.back().second.same_as(s.second);

      assert(!info_map.contains(s.first));
      scoped_values.push_back(set_value_in_scope(info_map, s.first, std::move(value_info)));
    }

    expr_info body_info;
    auto body = mutate(op->body, &body_info);

    bool substituted = false;
    for (auto it = lets.rbegin(); it != lets.rend();) {
      scoped_values.pop_back();
      auto deps = depends_on(body, it->first);
      // Find any deps on this variable in the inner let values.
      for (auto inner = lets.rbegin(); inner != it; ++inner) {
        depends_on(inner->second, it->first, deps);
      }

      if (!deps.any()) {
        // Prune dead lets
        it = std::make_reverse_iterator(lets.erase(std::next(it).base()));
        values_changed = true;
      } else if (should_substitute(it->second)) {
        body = substitute(body, it->first, it->second);
        for (auto inner = lets.rbegin(); inner != it; ++inner) {
          inner->second = substitute(inner->second, it->first, it->second);
        }
        it = std::make_reverse_iterator(lets.erase(std::next(it).base()));
        values_changed = true;
        substituted = true;
      } else {
        ++it;
      }
    }
    if (substituted) {
      body = mutate(body, &body_info);
    }

    if (lets.empty()) {
      // All lets were removed.
      set_result(std::move(body), std::move(body_info));
    } else if (!values_changed && body.same_as(op->body)) {
      set_result(op, std::move(body_info));
    } else {
      set_result(T::make(std::move(lets), std::move(body)), std::move(body_info));
    }
  }

  void visit(const let* op) override { visit_let(op); }
  void visit(const let_stmt* op) override { visit_let(op); }

  stmt mutate_with_buffer(const base_stmt_node* decl, stmt body, var buf, var src, std::optional<buffer_info> buffer) {
    if (buffer) {
      buffer->decl = stmt(decl);
      buffer->src = src;
    }
    auto set_buffer = set_value_in_scope(buffers, buf, std::move(buffer));
    return mutate(body);
  }
  stmt mutate_with_buffer(const base_stmt_node* decl, stmt body, var buf, std::optional<buffer_info> buffer) {
    if (buffer) buffer->decl = stmt(decl);
    auto set_buffer = set_value_in_scope(buffers, buf, std::move(buffer));
    return mutate(body);
  }

  stmt mutate_with_bounds(stmt body, var v, interval_expr bounds) {
    assert(!info_map.contains(v));
    auto set_bounds = set_value_in_scope(info_map, v, {std::move(bounds), alignment_type()});
    return mutate(body);
  }

  // Find all buffers accessed in `s`, adding them, and all the aliases of them, to `bufs`.
  void buffers_accessed_via_aliases(const stmt& s, bool consumed, std::set<var>& bufs) {
    std::set<var> raw = buffers_accessed(s, consumed);
    for (var i : raw) {
      while (i.defined()) {
        bufs.insert(i);
        const std::optional<buffer_info>& info = buffers.lookup(i);
        if (info && i != info->src) {
          i = info->src;
        } else {
          break;
        }
      }
    }
  }

  // Substitute expr() in for loop_var only in crop bounds within s, and only if those crop bounds do not crop a folded
  // dimension.
  stmt remove_loop_var_in_crop_bounds(const stmt& s, var loop_var) {
    class m : public node_mutator {
      symbol_map<buffer_info>& buffers;
      var loop_var;

    public:
      m(symbol_map<buffer_info>& buffers, var loop_var) : buffers(buffers), loop_var(loop_var) {}

      interval_expr mutate_crop_bounds(var src, int dim, const interval_expr& bounds) {
        std::optional<buffer_info>& src_info = buffers[src];
        if (!src_info || dim >= static_cast<int>(src_info->dims.size())) {
          // We don't know about this buffer or dimension, it might be folded.
          return bounds;
        }
        if (src_info->dims[dim].fold_factor.defined() && !is_constant(src_info->dims[dim].fold_factor, dim::unfolded)) {
          // This dimension is folded, don't drop crops.
          return bounds;
        }
        return substitute(bounds, loop_var, expr());
      }

      void visit(const crop_dim* op) override {
        interval_expr bounds = mutate_crop_bounds(op->src, op->dim, op->bounds);
        stmt body = mutate(op->body);
        if (!bounds.same_as(op->bounds) || !body.same_as(op->body)) {
          set_result(crop_dim::make(op->sym, op->src, op->dim, std::move(bounds), std::move(body)));
        } else {
          set_result(op);
        }
      }
      void visit(const crop_buffer* op) override {
        box_expr bounds(op->bounds.size());
        bool changed = false;
        for (std::size_t d = 0; d < op->bounds.size(); ++d) {
          bounds[d] = mutate_crop_bounds(op->src, d, op->bounds[d]);
          changed = changed || !bounds[d].same_as(op->bounds[d]);
        }
        stmt body = mutate(op->body);
        if (changed || !body.same_as(op->body)) {
          set_result(crop_buffer::make(op->sym, op->src, std::move(bounds), std::move(body)));
        } else {
          set_result(op);
        }
      }
    };
    return m(buffers, loop_var).mutate(s);
  }

  void visit(const loop* op) override {
    interval_expr bounds = mutate(op->bounds);
    expr step = mutate(op->step);

    // TODO: Try not to assume that step > 0.
    auto knowledge = learn_from_true(step > 0);

    if (prove_true(bounds.min > bounds.max)) {
      // This loop is dead.
      set_result(stmt());
      return;
    } else if (prove_true(bounds.min + step > bounds.max)) {
      // The loop only runs at most once. It's safe to run the body even if the loop is empty, because we assume we can
      // move loops freely in and out of calls, even if the buffers are empty.
      set_result(mutate(substitute(op->body, op->sym, bounds.min)));
      return;
    }

    stmt body = mutate_with_bounds(op->body, op->sym, bounds);
    scoped_trace trace("visit(const loop*)");
    if (!body.defined()) {
      set_result(stmt());
      return;
    } else if (!depends_on(body, op->sym).any()) {
      // The body does not depend on the loop, drop the loop.
      set_result(std::move(body));
      return;
    } else if (const block* b = body.as<block>()) {
      scoped_trace trace("licm");
      // This LICM is more aggressive than the typical compiler. Because we can freely add or remove loops around calls,
      // if we find a loop invariant stmt that consumes a loop variant stmt, we can try to force the loop varying stmt
      // to be loop invariant, and remove both from the loop.
      // These accumulate results in reverse order.
      // Here, we keep the original stmt, in case we need to put it back in the loop.
      std::vector<std::pair<stmt, stmt>> lifted;
      std::vector<stmt> loop_body;
      loop_body.reserve(b->stmts.size());

      // Find out if i produces something consumed by a loop invariant.
      for (auto ri = b->stmts.rbegin(); ri != b->stmts.rend(); ++ri) {
        stmt i = *ri;

        if (!depends_on(i, op->sym).var) {
          // i is loop invariant. Add it to the lifted result.
          lifted.push_back({i, i});
        } else {
          // i depends on the loop. We are effectively reordering result ahead of i, we need to make sure we can do
          // that.
          std::set<var> produced_by_i;
          buffers_accessed_via_aliases(i, /*consumed=*/false, produced_by_i);

          // The loop invariants might depend on i. If so, we need to figure out what to do.
          for (auto j = lifted.begin(); j != lifted.end();) {
            std::set<var> consumed_by_j;
            buffers_accessed_via_aliases(j->first, /*consumed=*/true, consumed_by_j);

            if (empty_intersection(produced_by_i, consumed_by_j)) {
              // This loop invariant is independent of i.
              ++j;
              continue;
            }

            // j depends on i, so j is not loop invariant. Can we make i loop invariant? We can if we can delete the
            // references to the loop variable.
            stmt i_lifted = remove_loop_var_in_crop_bounds(i, op->sym);
            if (!depends_on(i_lifted, op->sym).var) {
              // We made i loop invariant, add it to the loop invariant result before it is used.
              ++j;
              lifted.insert(j, {mutate(i_lifted), i});
              i = stmt();
              break;
            } else {
              // We can't delete the references to the loop variable here, so j is not loop invariant. Put the original
              // stmt back in the loop.
              loop_body.push_back(j->second);
              j = lifted.erase(j);
            }
          }

          if (i.defined()) {
            // We didn't lift i, put it in the loop.
            loop_body.push_back(std::move(i));
          }
        }
      }
      if (!lifted.empty()) {
        // We found something to lift out of the loop.
        std::vector<stmt> result;
        result.reserve(lifted.size() + 1);
        for (auto i = lifted.rbegin(); i != lifted.rend(); ++i) {
          result.push_back(i->first);
        }
        std::reverse(loop_body.begin(), loop_body.end());
        result.push_back(mutate(loop::make(op->sym, op->max_workers, bounds, step, block::make(std::move(loop_body)))));
        set_result(block::make(std::move(result)));
        return;
      } else {
        // We didn't find anything to lift out of the loop, proceed on to other possible simplifications.
      }
    }

    if (op->is_serial()) {
      scoped_trace trace("drop_loop");
      // Due to either scheduling or other simplifications, we can end up with a loop that runs a single call or copy on
      // contiguous crops of a buffer. In these cases, we can drop the loop in favor of just calling the body on the
      // union of the bounds covered by the loop.
      stmt result = body;
      std::vector<std::tuple<var, var, int, interval_expr>> new_crops;
      bool drop_loop = true;
      while (true) {
        // For now, we only handle crop_dim. I don't think crop_buffer can ever yield this simplification?
        if (const crop_dim* crop = result.as<crop_dim>()) {
          // If we can prove that the union of either the current and next iteration, or previous and current iteration,
          // is the whole iteration domain, then we can drop the loop. It's helpful to check both, because usually
          // clamps that make this proof hard only exist in one direction.
          interval_expr next_iter = substitute(crop->bounds, op->sym, expr(op->sym) + op->step);
          interval_expr prev_iter = substitute(crop->bounds, op->sym, expr(op->sym) - op->step);
          auto set_bounds_of_sym = set_value_in_scope(info_map, op->sym, {bounds, alignment_type()});
          // TODO: Currently we only support crops that monotonically increase the crop bounds as the loop progresses.
          if (prove_true((next_iter.min > crop->bounds.min && crop->bounds.max + 1 >= next_iter.min) || 
                         (crop->bounds.min > prev_iter.min && prev_iter.max + 1 >= crop->bounds.min))) {
            result = crop->body;
            expr_info info_of_min, info_of_max;
            mutate(crop->bounds, &info_of_min, &info_of_max);
            interval_expr crop_bounds = info_of_min.bounds | info_of_max.bounds;
            // If the original loop was empty, we need to hack the crop bounds to produce an empty buffer.
            crop_bounds.min = select(bounds.max < bounds.min, crop_bounds.max + 1, crop_bounds.min);
            new_crops.emplace_back(crop->sym, crop->src, crop->dim, std::move(crop_bounds));
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
          result = crop_dim::make(std::get<0>(*i), std::get<1>(*i), std::get<2>(*i), std::get<3>(*i), result);
        }
        set_result(mutate(result));
        return;
      }
    }

    if (bounds.same_as(op->bounds) && step.same_as(op->step) && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(loop::make(op->sym, op->max_workers, std::move(bounds), std::move(step), std::move(body)));
    }
  }

  void visit(const block* op) override {
    std::vector<stmt> stmts;
    stmts.reserve(op->stmts.size());
    bool changed = false;
    for (const stmt& s : op->stmts) {
      stmts.push_back(mutate(s));
      changed = changed || !stmts.back().same_as(s);
    }

    if (!changed) {
      set_result(op);
    } else {
      set_result(block::make(std::move(stmts)));
    }
  }

  dim_expr mutate(const dim_expr& d) {
    dim_expr result = {mutate(d.bounds), mutate(d.stride), mutate(d.fold_factor)};
    if (is_constant(result.fold_factor, dim::unfolded)) result.fold_factor = expr();
    if (is_constant(result.stride, dim::auto_stride)) result.stride = expr();
    return result;
  }

  template <typename T>
  buffer_info mutate_buffer(const T* op) {
    scoped_trace trace("mutate_buffer");
    buffer_info info;
    info.elem_size = mutate(op->elem_size);
    info.dims.reserve(op->dims.size());
    for (std::size_t d = 0; d < op->dims.size(); ++d) {
      info.dims.push_back(mutate(op->dims[d]));
    }
    info.all_dims_known = true;
    info.decl = stmt(op);
    return info;
  }

  template <typename T>
  static bool buffer_changed(const T* op, const buffer_info& info) {
    if (op->dims.size() != info.dims.size()) return true;
    if (!info.elem_size.same_as(op->elem_size)) return true;
    for (std::size_t d = 0; d < op->dims.size(); ++d) {
      if (!info.dims[d].same_as(op->dims[d])) return true;
    }
    return false;
  }

  void visit(const allocate* op) override {
    buffer_info info = mutate_buffer(op);
    stmt body = mutate_with_buffer(op, op->body, op->sym, info);
    scoped_trace trace("visit(const allocate*)");
    auto deps = depends_on(body, op->sym);
    if (!deps.any()) {
      set_result(std::move(body));
      return;
    } else if (!deps.buffer_data()) {
      // We only needed the buffer meta, not the allocation itself.
      set_result(mutate(substitute_buffer(body, op->sym, info.elem_size, info.dims)));
      return;
    }

    stmt before, after;
    if (const block* b = body.as<block>()) {
      // Split the body into 3 parts: the part that depends on the allocation, and anything before or after that.
      const auto depends_on_alloc = [=](const stmt& s) { return depends_on(s, op->sym).any(); };
      auto end_before = std::find_if(b->stmts.begin(), b->stmts.end(), depends_on_alloc);
      if (end_before != b->stmts.end()) {
        before = block::make({b->stmts.begin(), end_before});
        auto end_body = std::find_if(b->stmts.rbegin(), b->stmts.rend(), depends_on_alloc).base();
        after = block::make({end_body, b->stmts.end()});
        body = block::make({end_before, end_body});
      } else {
        set_result(block::make({b->stmts.begin(), end_before}));
        return;
      }
    }

    if (buffer_changed(op, info) || !body.same_as(op->body)) {
      set_result(block::make({std::move(before),
          allocate::make(op->sym, op->storage, std::move(info.elem_size), std::move(info.dims), std::move(body)),
          std::move(after)}));
    } else {
      set_result(op);
    }
  }

  // If d is equal to buffer_dim(sym, x), return x, otherwise return -1.
  int is_buffer_dim(const dim_expr& d, var sym) {
    const call* min = match_call(d.bounds.min, intrinsic::buffer_min, sym);
    if (min) {
      assert(min->args.size() == 2);
      auto dim = as_constant(min->args[1]);
      assert(dim);
      if (match_call(d.bounds.max, intrinsic::buffer_max, sym, *dim) &&
          match_call(d.stride, intrinsic::buffer_stride, sym, *dim) &&
          match_call(d.fold_factor, intrinsic::buffer_fold_factor, sym, *dim)) {
        return *dim;
      }
    }
    return -1;
  }

  bool is_buffer_meta(const expr& x, const expr& value, intrinsic fn, var sym, int dim) {
    return (!x.defined() && !value.defined()) || match_call(x, fn, sym, dim) || prove_true(x == value);
  }

  // Returns true if d can be represented as buffer_dim(sym, dim)
  bool is_buffer_dim(const dim_expr& d, const dim_expr& src, var sym, int dim) {
    return is_buffer_meta(d.bounds.min, src.bounds.min, intrinsic::buffer_min, sym, dim) &&
           is_buffer_meta(d.bounds.max, src.bounds.max, intrinsic::buffer_max, sym, dim) &&
           is_buffer_meta(d.stride, src.stride, intrinsic::buffer_stride, sym, dim) &&
           is_buffer_meta(d.fold_factor, src.fold_factor, intrinsic::buffer_fold_factor, sym, dim);
  }

  // If we know that buffer metadata has some values, rewrite references to that dim to use buffer intrinsics
  // when those references use the same values.
  void canonicalize_buffer_meta(expr& x, const expr& value, intrinsic fn, var sym) {
    if (!match_call(x, fn, sym) && prove_true(x == value)) x = call::make(fn, {sym});
  }
  void canonicalize_buffer(buffer_info& buf, const buffer_info& src, var sym) {
    scoped_trace trace("canonicalize_buffer");
    canonicalize_buffer_meta(buf.elem_size, src.elem_size, intrinsic::buffer_elem_size, sym);
    for (dim_expr& d : buf.dims) {
      for (int src_d = 0; src_d < static_cast<int>(src.dims.size()); ++src_d) {
        if (is_buffer_dim(d, src.dims[src_d], sym, src_d)) {
          d = buffer_dim(sym, src_d);
          break;
        }
      }
    }
  }

  void visit(const make_buffer* op) override {
    expr base = mutate(op->base);
    buffer_info info = mutate_buffer(op);

    // To avoid redundant nested simplifications, try to substitute the buffer both before and after mutating the body.
    // TODO: It may be impossible for depends_on_result::buffer_data() to change due to simplification, so the second
    // check below could be unnecessary.
    if (can_substitute_buffer(depends_on(op->body, op->sym))) {
      // We only needed the buffer meta, not the buffer itself.
      set_result(mutate(substitute_buffer(op->body, op->sym, info.elem_size, info.dims)));
      return;
    }
    stmt body = mutate_with_buffer(op, op->body, op->sym, find_buffer(base), info);
    scoped_trace trace("visit(const make_buffer*)");
    auto deps = depends_on(body, op->sym);
    if (!deps.any()) {
      // This make_buffer is unused.
      set_result(std::move(body));
      return;
    } else if (can_substitute_buffer(depends_on(op->body, op->sym))) {
      // We only needed the buffer meta, not the buffer itself.
      set_result(mutate(substitute_buffer(body, op->sym, info.elem_size, info.dims)));
      return;
    }

    if (const call* bc = as_intrinsic(base, intrinsic::buffer_at)) {
      // Check if this make_buffer is equivalent to transpose, slice_buffer or crop_buffer
      auto src_buf = as_variable(bc->args[0]);
      assert(src_buf);

      const std::optional<buffer_info>& src_info = buffers[*src_buf];
      if (src_info) {
        // Before trying to do anything, try to normalize the dimensions to be in terms of src_buf metadata.
        canonicalize_buffer(info, *src_info, *src_buf);
      }

      if (match(info.elem_size, buffer_elem_size(*src_buf))) {
        // To be a slice, we need every dimension that is present in the buffer_at call to be skipped, and the rest of
        // the dimensions to be identity.
        int dim = 0;
        std::size_t slice_rank = 0;
        std::size_t at_rank =
            std::count_if(bc->args.begin() + 1, bc->args.end(), [](const expr& i) { return i.defined(); });
        bool is_slice = true;
        for (int d = 0; d < static_cast<int>(info.dims.size() + at_rank); ++d) {
          if (d + 1 < static_cast<index_t>(bc->args.size()) && bc->args[d + 1].defined()) {
            // Skip this dimension.
            ++dim;
          } else if (slice_rank < info.dims.size()) {
            // This arg is undefined. We need to find the next dimension here to be a slice.
            is_slice = is_slice && is_buffer_dim(info.dims[slice_rank++], *src_buf) == dim++;
          } else {
            is_slice = false;
            break;
          }
        }
        if (is_slice && slice_rank == info.dims.size()) {
          std::vector<expr> at(bc->args.begin() + 1, bc->args.end());
          stmt result = slice_buffer::make(op->sym, op->sym, std::move(at), std::move(body));
          // make_buffer drops trailing dims, do the same here.
          result = transpose::make_truncate(op->sym, *src_buf, info.dims.size() + at_rank, std::move(result));
          set_result(mutate(result));
          return;
        }

        // To be a crop, we need dimensions to either be identity, or the buffer_at argument is the same as the min.
        bool is_crop = bc->args.size() <= info.dims.size() + 1;
        box_expr crop_bounds(info.dims.size());
        for (index_t d = 0; d < static_cast<index_t>(info.dims.size()); ++d) {
          if (!match_call(info.dims[d].stride, intrinsic::buffer_stride, *src_buf, d) ||
              !match_call(info.dims[d].fold_factor, intrinsic::buffer_fold_factor, *src_buf, d)) {
            is_crop = false;
            break;
          }

          // If the argument to buffer_at is defined, we need the min to be the same as the argument.
          // If it is not defined, it must be buffer_min(buf, d).
          bool has_at_d = d + 1 < static_cast<index_t>(bc->args.size()) && bc->args[d + 1].defined();
          expr crop_min = has_at_d ? bc->args[d + 1] : buffer_min(*src_buf, d);
          if (match(info.dims[d].bounds.min, crop_min)) {
            // We rewrite src -> sym in the truncate below.
            crop_bounds[d] = substitute(info.dims[d].bounds, *src_buf, op->sym);
          } else {
            is_crop = false;
            break;
          }
        }
        if (is_crop) {
          stmt result = crop_buffer::make(op->sym, op->sym, std::move(crop_bounds), std::move(body));
          // make_buffer drops trailing dims, do the same here.
          result = transpose::make_truncate(op->sym, *src_buf, info.dims.size(), std::move(result));
          set_result(mutate(result));
          return;
        }

        // To be a transpose, we need buffer_at to be the base of src_buf, and each dimension to be a dimension of the
        // original buffer.
        // TODO: This could probably be built into the slice check above.
        bool is_transpose = bc->args.size() == 1;
        std::vector<int> permutation;
        permutation.reserve(info.dims.size());
        for (std::size_t d = 0; d < info.dims.size(); ++d) {
          int dim = is_buffer_dim(info.dims[d], *src_buf);
          if (dim >= 0) {
            permutation.push_back(dim);
          } else {
            is_transpose = false;
            break;
          }
        }
        if (is_transpose) {
          set_result(mutate(transpose::make(op->sym, *src_buf, std::move(permutation), std::move(body))));
          return;
        }
      }
    }

    if (const block* b = body.as<block>()) {
      set_result(lift_decl_invariants(b->stmts, op->sym,
          [&](stmt body) { return make_buffer::make(op->sym, base, info.elem_size, info.dims, std::move(body)); }));
    } else if (buffer_changed(op, info) || !base.same_as(op->base) || !body.same_as(op->body)) {
      set_result(make_buffer::make(
          op->sym, std::move(base), std::move(info.elem_size), std::move(info.dims), std::move(body)));
    } else {
      set_result(op);
    }
  }

  std::optional<buffer_info> get_buffer_info(var buf, int rank) {
    std::optional<buffer_info> info = buffers[buf];
    if (!info) {
      info = buffer_info();
    }
    info->dims.resize(std::max(info->dims.size(), static_cast<std::size_t>(rank)));
    if (!info->elem_size.defined()) info->elem_size = buffer_elem_size(buf);
    for (int d = 0; d < static_cast<int>(info->dims.size()); ++d) {
      if (!info->dims[d].bounds.min.defined()) info->dims[d].bounds.min = buffer_min(buf, d);
      if (!info->dims[d].bounds.max.defined()) info->dims[d].bounds.max = buffer_max(buf, d);
      if (!info->dims[d].stride.defined()) info->dims[d].stride = buffer_stride(buf, d);
      if (!info->dims[d].fold_factor.defined()) info->dims[d].fold_factor = buffer_fold_factor(buf, d);
    }
    return info;
  }

  // Crop bounds like min(buffer_max(x, d), y) can be rewritten to just y because the crop will clamp anyways.
  static expr simplify_crop_bound(expr x, var sym, int dim) {
    if (const class max* m = x.as<class max>()) {
      if (match_call(m->a, intrinsic::buffer_min, sym, dim)) return simplify_crop_bound(m->b, sym, dim);
      if (match_call(m->b, intrinsic::buffer_min, sym, dim)) return simplify_crop_bound(m->a, sym, dim);
    } else if (const class min* m = x.as<class min>()) {
      if (match_call(m->a, intrinsic::buffer_max, sym, dim)) return simplify_crop_bound(m->b, sym, dim);
      if (match_call(m->b, intrinsic::buffer_max, sym, dim)) return simplify_crop_bound(m->a, sym, dim);
    }
    return x;
  }

  template <typename T>
  static void enumerate_bounds(expr x, std::set<expr, node_less>& bounds) {
    bounds.insert(x);
    if (const T* t = x.as<T>()) {
      enumerate_bounds<T>(t->a, bounds);
      enumerate_bounds<T>(t->b, bounds);
    }
  }

  template <typename T>
  expr remove_redundant_bounds(expr x, const std::set<expr, node_less>& bounds) {
    if (bounds.count(x)) return expr();
    if (const T* t = x.as<T>()) {
      bool a_is_bound = bounds.count(t->a);
      bool b_is_bound = bounds.count(t->b);
      if (a_is_bound && b_is_bound) {
        return expr();
      } else if (a_is_bound) {
        return remove_redundant_bounds<T>(t->b, bounds);
      } else if (b_is_bound) {
        return remove_redundant_bounds<T>(t->a, bounds);
      }
    } else if (const add* xa = x.as<add>()) {
      if (as_constant(xa->b)) {
        // We have T(x + y, b). We can rewrite to T(x, b - y) + y, and if we can eliminate the bound, the whole
        // bound is redundant.
        for (const expr& i : bounds) {
          expr removed = remove_redundant_bounds<T>(xa->a, {mutate(i - xa->b)});
          if (!removed.same_as(xa->a)) {
            return std::move(removed) + xa->b;
          }
        }
      }
    } else if (const class select* xs = x.as<class select>()) {
      for (const expr& i : bounds) {
        if (const class select* bs = i.as<class select>()) {
          if (match(xs->condition, bs->condition)) {
            // We have T(select(c, xt, xf), select(c, bt, bf)), rewrite to select(c, T(xt, bt), T(xf, bf)) and attempt
            // to eliminate bounds.
            expr t = remove_redundant_bounds<T>(xs->true_value, {bs->true_value});
            expr f = remove_redundant_bounds<T>(xs->false_value, {bs->false_value});
            if (!t.same_as(xs->true_value) || !f.same_as(xs->false_value)) {
              return select(xs->condition, std::move(t), std::move(f));
            }
          }
        }
      }
      // Also try select(x, T(xt, b), T(xf, b))
      expr t = remove_redundant_bounds<T>(xs->true_value, bounds);
      expr f = remove_redundant_bounds<T>(xs->false_value, bounds);
      if (!t.same_as(xs->true_value) || !f.same_as(xs->false_value)) {
        return select(xs->condition, std::move(t), std::move(f));
      }
    }
    return x;
  }

  interval_expr mutate_crop_bounds(const interval_expr& crop, var buf, int dim, interval_expr& buffer) {
    if (!crop.min.defined() && !crop.max.defined()) return crop;
    scoped_trace trace("mutate_crop_bounds");

    interval_expr result = mutate(crop);

    // Find and remove redundant clamps in the crop bounds.
    // TODO: This seems like a hack. We should be able to do this with the simplifier itself. We basically want to do
    // something like:
    //
    //   result = simplify(result & x, {{x, buffer}})
    //
    // and then remove the clamps of x. But this is pretty tricky.
    std::set<expr, node_less> mins = {buffer_min(buf, dim)};
    std::set<expr, node_less> maxs = {buffer_max(buf, dim)};
    enumerate_bounds<class max>(buffer.min, mins);
    enumerate_bounds<class min>(buffer.max, maxs);
    interval_expr deduped = {
        remove_redundant_bounds<class max>(result.min, mins),
        remove_redundant_bounds<class min>(result.max, maxs),
    };
    if (!deduped.same_as(result)) {
      result = mutate(deduped);
    }

    // TODO: We should not need to compare to both buffer_bounds(buf, dim) and buffer.
    if (prove_true(result.min <= buffer.min || result.min <= buffer_min(buf, dim))) result.min = expr();
    if (prove_true(result.max >= buffer.max || result.max >= buffer_max(buf, dim))) result.max = expr();

    // We already proved above that this min/max is necessary (otherwise result would be undefined here).
    if (result.min.defined()) buffer.min = max(buffer.min, result.min);
    if (result.max.defined()) buffer.max = min(buffer.max, result.max);

    // We might have written a select into an interval that tries to preserve the empty-ness of the interval.
    // But this might be unnecessary. Try to remove unnecessary selects here.
    rewrite::pattern_wildcard<0> b;
    rewrite::pattern_wildcard<1> d;
    rewrite::pattern_wildcard<2> empty_min;
    rewrite::pattern_wildcard<3> new_min;
    rewrite::match_context ctx;
    if (match(ctx, select(buffer_max(b, d) < buffer_min(b, d), empty_min, new_min), result.min)) {
      if (is_variable(ctx.matched(b), buf) && is_constant(ctx.matched(d), dim)) {
        // This select is a check that the dimension we are cropping is empty.
        // If the buffer is empty, it doesn't matter what we do, the resulting crop will still be empty, so we can
        // just take the new min.
        result.min = ctx.matched(new_min);
      }
    }

    return result;
  }

  static bool crop_needed(const depends_on_result& deps) {
    // We don't need a crop if the buffer is only used as an input to a call. But we do need the crop if it is used as
    // an input to a copy, which uses the bounds of the input for padding.
    return deps.buffer_output || deps.buffer_src || deps.buffer_dst || deps.buffer_bounds;
  }

  void visit_crop(const base_stmt_node* op, var op_sym, var op_src, const box_expr& op_bounds, stmt op_body) {
    std::optional<buffer_info> info = get_buffer_info(op_src, op_bounds.size());
    box_expr bounds(op_bounds.size());

    // If possible, rewrite crop_buffer of one dimension to crop_dim.
    bool changed = false;
    for (index_t i = 0; i < static_cast<index_t>(op_bounds.size()); ++i) {
      bounds[i] = mutate_crop_bounds(op_bounds[i], op_src, i, info->dims[i].bounds);
      changed = changed || !bounds[i].same_as(op_bounds[i]);
    }
    stmt body = mutate_with_buffer(op, op_body, op_sym, op_src, std::move(info));
    scoped_trace trace("visit_crop");
    auto deps = depends_on(body, op_sym);
    if (!deps.any()) {
      set_result(std::move(body));
      return;
    }

    // Remove trailing undefined bounds.
    while (!bounds.empty() && !bounds.back().min.defined() && !bounds.back().max.defined()) {
      bounds.pop_back();
    }

    if (!crop_needed(deps)) {
      // Add clamps for the implicit bounds like crop would have done.
      for (index_t d = 0; d < static_cast<index_t>(bounds.size()); ++d) {
        bounds[d] &= slinky::buffer_bounds(op_src, d);
      }
      body = substitute_bounds(body, op_sym, bounds);
      body = substitute(body, op_sym, op_src);
      set_result(mutate(body));
      return;
    }

    // Rewrite nested crops to be one crop where possible.
    var sym = op_sym;
    while (true) {
      if (const crop_buffer* c = body.as<crop_buffer>()) {
        // The inner crop might use the outer buffer's bounds, substitute them.
        box_expr c_bounds;
        c_bounds.reserve(c->bounds.size());
        for (const interval_expr& i : c->bounds) {
          c_bounds.push_back({
              substitute_bounds(i.min, op_sym, bounds),
              substitute_bounds(i.max, op_sym, bounds),
          });
        }
        if (op_sym == c->src && !depends_on(c->body, op_sym).any()) {
          // Nested crops of the same buffer, and the crop isn't used.
          bounds.resize(std::max(bounds.size(), c_bounds.size()));
          bounds = bounds & c_bounds;
          for (interval_expr& i : bounds) {
            i = mutate(i);
          }
          sym = c->sym;
          body = c->body;
          continue;
        } else if (op_src == c->src && match(c->bounds, bounds)) {
          // Two crops producing the same buffer, we can just use one of them and discard the other.
          body = substitute(c->body, c->sym, sym);
          continue;
        }
      } else if (const crop_dim* c = body.as<crop_dim>()) {
        // The inner crop might use the outer buffer's bounds, substitute them.
        interval_expr c_bounds = {
            substitute_bounds(c->bounds.min, op_sym, bounds),
            substitute_bounds(c->bounds.max, op_sym, bounds),
        };
        if (op_sym == c->src && !depends_on(c->body, op_sym).any()) {
          // Nested crops of the same buffer, and the crop isn't used.
          if (c->dim < static_cast<int>(bounds.size())) {
            bounds[c->dim] = mutate(bounds[c->dim] & c_bounds);
          } else {
            bounds.resize(c->dim + 1);
            bounds[c->dim] = mutate(c_bounds);
          }
          sym = c->sym;
          body = c->body;
          continue;
        } else if (c->src == op_src && match(c->dim, c->bounds, bounds)) {
          // Two crops producing the same buffer, we can just use one of them and discard the other.
          body = substitute(c->body, c->sym, sym);
          continue;
        }
      }
      break;
    }

    // If this was a crop_buffer, and we only have one dim, we're going to change it to a crop_dim.
    const int dims_count = std::count_if(
        bounds.begin(), bounds.end(), [](const interval_expr& i) { return i.min.defined() || i.max.defined(); });
    changed = changed || (dims_count == 1 && op->type != crop_dim::static_type) || !body.same_as(op_body);

    auto make_crop = [&](const stmt& body) -> stmt {
      if (!changed && body.same_as(op_body)) {
        return stmt(op);
      } else if (dims_count == 1) {
        // This crop is of one dimension, replace it with crop_dim.
        // We removed undefined trailing bounds, so this must be the dim we want.
        int d = static_cast<int>(bounds.size()) - 1;
        return crop_dim::make(sym, op_src, d, bounds[d], body);
      } else {
        return crop_buffer::make(sym, op_src, bounds, body);
      }
    };

    if (bounds.empty()) {
      // This crop was a no-op.
      set_result(substitute(body, sym, op_src));
    } else if (const block* b = body.as<block>()) {
      set_result(lift_decl_invariants(b->stmts, sym, make_crop));
    } else {
      set_result(make_crop(body));
    }
  }

  void visit(const crop_buffer* op) override { visit_crop(op, op->sym, op->src, op->bounds, op->body); }

  void visit(const crop_dim* op) override {
    box_expr bounds(op->dim + 1);
    bounds[op->dim] = op->bounds;
    visit_crop(op, op->sym, op->src, bounds, op->body);
  }

  static void update_sliced_buffer_metadata(symbol_map<expr_info>& info_map, var sym, span<const int> sliced) {
    for (std::optional<expr_info>& i : info_map) {
      if (!i) continue;
      i->bounds = slinky::update_sliced_buffer_metadata(i->bounds, sym, sliced);
    }
  }
  static void update_sliced_buffer_metadata(symbol_map<buffer_info>& buffers, var sym, span<const int> sliced) {
    for (std::optional<buffer_info>& i : buffers) {
      if (!i) continue;
      for (dim_expr& j : i->dims) {
        j.bounds = slinky::update_sliced_buffer_metadata(j.bounds, sym, sliced);
        j.stride = slinky::update_sliced_buffer_metadata(j.stride, sym, sliced);
        j.fold_factor = slinky::update_sliced_buffer_metadata(j.fold_factor, sym, sliced);
      }
    }
  }

  void visit_slice(const base_stmt_node* op, var op_sym, var op_src, const std::vector<expr>& op_at, stmt op_body) {
    std::vector<expr> at(op_at.size());
    std::vector<int> sliced_dims;
    bool changed = false;
    for (index_t i = 0; i < static_cast<index_t>(op_at.size()); ++i) {
      if (op_at[i].defined()) {
        at[i] = mutate(op_at[i]);
        changed = changed || !at[i].same_as(op_at[i]);
        sliced_dims.push_back(i);
      }
    }

    symbol_map<buffer_info> old_buffers = buffers;
    symbol_map<expr_info> old_info_map = info_map;
    std::optional<buffer_info> info = buffers[op_src];

    if (info) {
      info->decl = stmt(op);
      buffers[op_sym] = std::move(info);
    } else {
      buffers[op_sym] = std::nullopt;
    }
    update_sliced_buffer_metadata(buffers, op_sym, sliced_dims);
    update_sliced_buffer_metadata(info_map, op_sym, sliced_dims);
    stmt body = mutate(op_body);
    buffers = std::move(old_buffers);
    info_map = std::move(old_info_map);

    if (!depends_on(body, op_sym).any()) {
      set_result(std::move(body));
      return;
    }

    // Remove trailing undefined bounds.
    while (!at.empty() && !at.back().defined()) {
      at.pop_back();
    }

    while (true) {
      if (const slice_buffer* s = body.as<slice_buffer>()) {
        if (s->src == op_src && match(s->at, at)) {
          // Two slices producing the same buffer, we can just use one of them and discard the other.
          body = substitute(s->body, s->sym, op_sym);
          continue;
        }
      } else if (const slice_dim* s = body.as<slice_dim>()) {
        if (s->src == op_src && match(s->dim, s->at, at)) {
          // Two slices producing the same buffer, we can just use one of them and discard the other.
          body = substitute(s->body, s->sym, op_sym);
          continue;
        }
      }
      break;
    }

    changed = changed || at.size() != op_at.size() || !body.same_as(op_body);

    // If this was a slice_buffer, and we only have one dimension, we're going to change it to a slice_dim.
    const int at_count = std::count_if(at.begin(), at.end(), [](const expr& i) { return i.defined(); });
    changed = changed || (at_count == 1 && op->type != slice_dim::static_type);

    auto make_slice = [&](const stmt& body) -> stmt {
      if (!changed && body.same_as(op)) {
        return stmt(op);
      } else if (at_count == 1) {
        // This slice is of one dimension, replace it with slice_dim.
        // We removed undefined trailing bounds, so this must be the dim we want.
        int d = static_cast<int>(at.size()) - 1;
        return slice_dim::make(op_sym, op_src, d, at[d], body);
      } else {
        return slice_buffer::make(op_sym, op_src, at, body);
      }
    };

    if (at.empty()) {
      // This slice was a no-op.
      set_result(substitute(body, op_sym, op_src));
    } else if (const block* b = body.as<block>()) {
      set_result(lift_decl_invariants(b->stmts, op_sym, make_slice));
    } else {
      set_result(make_slice(body));
    }
  }

  void visit(const slice_buffer* op) override { visit_slice(op, op->sym, op->src, op->at, op->body); }

  void visit(const slice_dim* op) override {
    std::vector<expr> at(op->dim + 1);
    at[op->dim] = op->at;
    visit_slice(op, op->sym, op->src, at, op->body);
  }

  void visit(const transpose* op) override {
    const std::optional<buffer_info>* src_info = &buffers[op->src];

    var src = op->src;
    std::vector<int> dims = op->dims;
    while (src_info && *src_info) {
      if (const transpose* t = (*src_info)->decl.as<transpose>()) {
        if (t->sym == src) {
          // This is a transpose of another transpose. Rewrite this to directly transpose the parent.
          dims = permute(dims, t->dims);
          src = t->src;
          src_info = &buffers[src];
          continue;
        }
      }
      break;
    }

    buffer_info sym_info;
    if (src_info && *src_info) {
      if (transpose::is_truncate(dims) && (*src_info)->all_dims_known && (*src_info)->dims.size() <= dims.size()) {
        // transpose can't add dimensions.
        assert((*src_info)->dims.size() == dims.size());
        // This truncate is a no-op.
        set_result(mutate(substitute(op->body, op->sym, src)));
        return;
      }

      sym_info.elem_size = (*src_info)->elem_size;
      // This is like `permute`, but we can't guarantee that we know all the dimensions of src_info (it could be a
      // buffer external to the pipeline).
      sym_info.dims.resize(op->dims.size());
      sym_info.all_dims_known = true;
      for (size_t i = 0; i < op->dims.size(); ++i) {
        if (op->dims[i] < static_cast<int>((*src_info)->dims.size())) {
          sym_info.dims[i] = (*src_info)->dims[op->dims[i]];
        } else {
          sym_info.dims[i] = buffer_dim(op->src, op->dims[i]);
        }
      }
    }
    sym_info.decl = stmt(op);

    stmt body = mutate_with_buffer(op, op->body, op->sym, op->src, std::move(sym_info));

    if (const block* b = body.as<block>()) {
      set_result(lift_decl_invariants(
          b->stmts, op->sym, [&](stmt body) { return mutate(transpose::make(op->sym, src, dims, std::move(body))); }));
      return;
    }

    auto deps = depends_on(body, op->sym);
    if (!deps.any()) {
      set_result(std::move(body));
    } else if (!deps.buffer_dims) {
      set_result(mutate(substitute(body, op->sym, op->src)));
    } else if (body.same_as(op->body) && src == op->src && dims == op->dims) {
      set_result(op);
    } else {
      set_result(transpose::make(op->sym, src, dims, std::move(body)));
    }
  }

  void visit(const clone_buffer* op) override {
    // Because we disallow shadowing (i.e. mutating buffers in place), clone_buffer can always be removed :)
    // Essentially, every operation is also a clone here.
    set_result(mutate(substitute(op->body, op->sym, op->src)));
  }

  void visit(const check* op) override {
    expr_info c_info;
    expr c = mutate_boolean(op->condition, &c_info);

    if (!c.defined()) {
      set_result(stmt());
    } else if (prove_constant_true(c_info.bounds.min)) {
      set_result(stmt());
    } else if (prove_constant_false(c_info.bounds.max)) {
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

expr simplify(const expr& e, const bounds_map& bounds, const alignment_map& alignment) {
  return simplifier(bounds, alignment).mutate(e, nullptr);
}

stmt simplify(const stmt& s, const bounds_map& bounds, const alignment_map& alignment) {
  scoped_trace trace("simplify");
  return simplifier(bounds, alignment).mutate(s);
}

interval_expr simplify(const interval_expr& e, const bounds_map& bounds, const alignment_map& alignment) {
  simplifier s(bounds, alignment);
  return s.mutate(e);
}

interval_expr bounds_of(const expr& x, const bounds_map& expr_bounds, const alignment_map& alignment) {
  scoped_trace trace("bounds_of");
  simplifier s(expr_bounds, alignment);
  simplifier::expr_info result;
  s.mutate(x, &result);
  return result.bounds;
}

interval_expr bounds_of(const interval_expr& x, const bounds_map& expr_bounds, const alignment_map& alignment) {
  if (deep_is_point(x)) {
    return bounds_of(x.min, expr_bounds);
  } else {
    scoped_trace trace("bounds_of");
    simplifier s(expr_bounds, alignment);
    simplifier::expr_info info_of_min, info_of_max;
    s.mutate(x, &info_of_min, &info_of_max);
    return {
        simplify(static_cast<const class min*>(nullptr), info_of_min.bounds.min, info_of_max.bounds.min),
        simplify(static_cast<const class max*>(nullptr), info_of_min.bounds.max, info_of_max.bounds.max),
    };
  }
}

namespace {

class constant_bound : public node_mutator {
  // > 0 -> we are looking for an upper bound
  // < 0 -> we are looking for a lower bound
  int sign;

public:
  constant_bound(int sign) : sign(sign) {}

  template <typename T>
  void visit_min_max(const T* op, bool take_constant) {
    // We can only learn about upper bounds from min and lower bounds from max. Furthermore, it is an error to
    // attempt to recursively mutate into a max while finding upper bounds or vice versa, because we might find
    // incorrect conservative bounds in the wrong direction.
    expr a = take_constant ? mutate(op->a) : op->a;
    expr b = take_constant ? mutate(op->b) : op->b;
    auto ca = as_constant(a);
    auto cb = as_constant(b);
    if (ca && cb) {
      set_result(make_binary<T>(*ca, *cb));
    } else if (take_constant && ca) {
      set_result(std::move(a));
    } else if (take_constant && cb) {
      set_result(std::move(b));
    } else if (a.same_as(op->a) && b.same_as(op->b)) {
      set_result(op);
    } else {
      set_result(T::make(std::move(a), std::move(b)));
    }
  }
  void visit(const class min* op) override { visit_min_max(op, /*take_constant=*/sign > 0); }
  void visit(const class max* op) override { visit_min_max(op, /*take_constant=*/sign < 0); }

  template <typename T>
  void visit_add_sub(const T* op, int rhs_sign) {
    expr a = mutate(op->a);
    // When we multiply by a negative number, we need to flip whether we are looking for an upper or lower bound.
    sign *= rhs_sign;
    expr b = mutate(op->b);
    sign *= rhs_sign;
    auto ca = as_constant(a);
    auto cb = as_constant(b);
    if (ca && cb) {
      set_result(make_or_eval_binary<T>(*ca, *cb));
    } else if (a.same_as(op->a) && b.same_as(op->b)) {
      set_result(op);
    } else {
      set_result(T::make(std::move(a), std::move(b)));
    }
  }

  void visit(const add* op) override { visit_add_sub(op, 1); }
  void visit(const sub* op) override { visit_add_sub(op, -1); }

  static int sign_of(const expr& x) {
    if (is_positive(x)) return 1;
    if (is_negative(x)) return -1;
    return 0;
  }

  template <typename T>
  void visit_mul_div(const T* op) {
    // When we multiply by a negative number, we need to flip whether we are looking for an upper or lower bound.
    int sign_a = sign_of(op->a);
    int sign_b = sign_of(op->b);
    // TODO: We should be able to handle the numerator of div too, it's just tricky.
    if (std::is_same<T, mul>::value && sign_a != 0) {
      int old_sign = sign;
      sign *= sign_a;
      expr b = mutate(op->b);
      sign = old_sign;
      if (b.same_as(op->b)) {
        set_result(op);
      } else {
        set_result(T::make(op->a, std::move(b)));
      }
    } else if (sign_b != 0) {
      int old_sign = sign;
      sign *= sign_b;
      expr a = mutate(op->a);
      sign = old_sign;
      if (a.same_as(op->a)) {
        set_result(op);
      } else {
        set_result(T::make(std::move(a), op->b));
      }
    } else {
      set_result(op);
    }
  }

  void visit(const mul* op) override { visit_mul_div(op); }
  void visit(const div* op) override { visit_mul_div(op); }

  void visit(const mod* op) override { set_result(op); }

  void visit_equal() {
    // Can we tighten this? I'm not sure. We need both upper and lower bounds to say anything here.
    if (sign < 0) {
      set_result(expr(0));
    } else {
      set_result(expr(1));
    }
  }
  void visit(const equal* op) override { visit_equal(); }
  void visit(const not_equal* op) override { visit_equal(); }

  template <typename T>
  void visit_less(const T* op) {
    // This is a constant version of that found in bounds_of_less:
    // - For a lower bound, we want to know if this can ever be false, so we want the upper bound of the lhs and the
    // lower bound of the rhs.
    // - For an upper bound, we want to know if this can ever be true, so we want the lower bound of the lhs and the
    // upper bound of the rhs.
    sign = -sign;
    expr a = mutate(op->a);
    sign = -sign;
    expr b = mutate(op->b);

    auto ca = as_constant(a);
    auto cb = as_constant(b);
    if (ca && cb) {
      set_result(make_binary<T>(*ca, *cb));
    } else if (sign < 0) {
      set_result(expr(0));
    } else {
      set_result(expr(1));
    }
  }
  void visit(const less* op) override { visit_less(op); }
  void visit(const less_equal* op) override { visit_less(op); }

  template <typename T>
  void visit_logical_and_or(const T* op, bool recurse) {
    // We can recursively mutate if:
    // - We're looking for the upper bound of &&, because if either operand is definitely false, the result is false.
    // - We're looking for the lower bound of ||, because if either operand is definitely true, the result is true.
    expr a = recurse ? mutate(op->a) : op->a;
    expr b = recurse ? mutate(op->b) : op->b;

    auto ca = as_constant(a);
    auto cb = as_constant(b);

    if (ca && cb) {
      set_result(make_binary<T>(*ca != 0, *cb != 0));
    } else if (sign < 0) {
      set_result(expr(0));
    } else {
      set_result(expr(1));
    }
  }
  void visit(const logical_and* op) override { visit_logical_and_or(op, /*recurse=*/sign > 0); }
  void visit(const logical_or* op) override { visit_logical_and_or(op, /*recurse=*/sign < 0); }

  void visit(const logical_not* op) override {
    sign = -sign;
    expr a = mutate(op->a);
    sign = -sign;
    auto ca = as_constant(a);
    if (ca) {
      set_result(*ca != 0 ? 0 : 1);
    } else if (sign < 0) {
      set_result(expr(0));
    } else {
      set_result(expr(1));
    }
  }
  void visit(const class select* op) override {
    expr t = mutate(op->true_value);
    expr f = mutate(op->false_value);
    auto ct = as_constant(t);
    auto cf = as_constant(f);
    if (sign < 0 && ct && cf) {
      set_result(expr(std::min(*ct, *cf)));
    } else if (sign > 0 && ct && cf) {
      set_result(expr(std::max(*ct, *cf)));
    } else if (t.same_as(op->true_value) && f.same_as(op->false_value)) {
      set_result(op);
    } else {
      set_result(select::make(op->condition, std::move(t), std::move(f)));
    }
  }
  void visit(const call* op) override {
    switch (op->intrinsic) {
    case intrinsic::abs:
      if (sign < 0) {
        expr a = mutate(op->args[0]);
        if (auto ca = as_constant(a)) {
          set_result(std::max<index_t>(0, *ca));
        } else {
          set_result(expr(0));
        }
        return;
      }
      break;
    default: break;
    }
    set_result(op);
  }
};

}  // namespace

expr constant_lower_bound(const expr& x) { return constant_bound(/*sign=*/-1).mutate(x); }
expr constant_upper_bound(const expr& x) { return constant_bound(/*sign=*/1).mutate(x); }

std::optional<bool> attempt_to_prove(
    const expr& condition, const bounds_map& expr_bounds, const alignment_map& alignment) {
  simplifier s(expr_bounds, alignment);
  return s.attempt_to_prove(condition);
}

bool prove_true(const expr& condition, const bounds_map& expr_bounds, const alignment_map& alignment) {
  simplifier s(expr_bounds, alignment);
  return s.prove_true(condition);
}

bool prove_false(const expr& condition, const bounds_map& expr_bounds, const alignment_map& alignment) {
  simplifier s(expr_bounds, alignment);
  return s.prove_false(condition);
}

}  // namespace slinky
