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

bool deep_is_point(const interval_expr& x) { return x.is_point() || match(x.min, x.max); }

// Ensure that an interval that is a point in a deep equality sense is also a point in a shallow equality sense.
void ensure_is_point(interval_expr& x) {
  if (!x.is_point() && match(x.min, x.max)) {
    x.max = x.min;
  }
}

// Rewrite `make_decl(block::make(stmts))` to be `block::make(make_decl(i) for i in stmts if i depends on sym else i)`.
template <class Fn>
stmt lift_decl_invariants(stmt body, var sym, Fn&& make_decl) {
  if (const block* b = body.as<block>()) {
    std::vector<stmt> result;
    result.reserve(b->stmts.size());
    for (auto i = b->stmts.begin(); i != b->stmts.end();) {
      if (depends_on(*i, sym).any()) {
        std::vector<stmt> result_i;
        result_i.reserve(b->stmts.size());
        do {
          result_i.push_back(*i++);
        } while (i != b->stmts.end() && depends_on(*i, sym).any());
        if (result.empty() && i == b->stmts.end()) {
          // Every stmt in the body depended on the decl, we aren't changing it.
          return make_decl(std::move(body));
        } else {
          result.push_back(make_decl(block::make(std::move(result_i))));
        }
        if (i != b->stmts.end()) result.push_back(*i++);
      } else {
        result.push_back(*i++);
      }
    }
    return block::make(std::move(result));
  } else {
    return make_decl(std::move(body));
  }
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

expr add_constant(const expr& a, index_t b) { 
  if (b == 0) return a;
  return constant_adder(b).mutate(a); 
}

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

    // How many loops out is the `decl` found.
    int loop_depth = 0;

    buffer_info() = default;

    buffer_info(expr elem_size) : elem_size(elem_size) {}

    buffer_info(var sym, int rank) : dims(rank) {
      elem_size = buffer_elem_size(sym);
      for (int d = 0; d < rank; ++d) {
        dims[d] = buffer_dim(sym, d);
      }
    }

    void init_dims(var sym, int rank) {
      for (int d = 0; d < static_cast<int>(dims.size()); ++d) {
        if (!dims[d].bounds.min.defined()) dims[d].bounds.min = buffer_min(sym, d);
        if (!dims[d].bounds.max.defined()) dims[d].bounds.max = buffer_max(sym, d);
        if (!dims[d].stride.defined()) dims[d].stride = buffer_stride(sym, d);
        if (!dims[d].fold_factor.defined()) dims[d].fold_factor = buffer_fold_factor(sym, d);
      }
      dims.reserve(rank);
      for (int d = dims.size(); d < rank; ++d) {
        dims.push_back(buffer_dim(sym, d));
      }
    }
  };

  struct expr_info {
    interval_expr bounds;
    alignment_type alignment;
    expr replacement;

    static expr_info substitution(expr replacement) {
      expr_info result;
      result.replacement = replacement;
      return result;
    }

    var replacement_sym() const {
      if (!replacement.defined()) return var();
      auto result = as_variable(replacement);
      assert(result);
      return *result;
    }

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

      if (auto c = as_constant(bounds.as_point())) {
        alignment.modulus = 0;
        alignment.remainder = *c;
      }
    }
  };

private:
  symbol_map<buffer_info> buffers;
  symbol_map<expr_info> vars;
  bool proving = false;

  expr_info result_info;

  void set_result(expr e, expr_info info) {
    assert(!result_info.bounds.min.defined() && !result_info.bounds.max.defined());
    result_info = std::move(info);
    ensure_is_point(result_info.bounds);
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
      vars[id] = {*bounds[id], alignment_type()};
    }

    for (size_t ix = 0; ix < alignment.size(); ix++) {
      var id = var(ix);
      if (!alignment[id]) continue;
      if (vars[id]) {
        vars[id]->alignment = *alignment[id];
      } else {
        vars[id] = {interval_expr(), *alignment[id]};
      }
    }
  }

  expr mutate(const expr& e, expr_info* info) {
    expr result = node_mutator::mutate(e);
    if (result.defined() && info) {
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
    std::vector<scoped_value_in_symbol_map<expr_info>> vk;
    std::vector<scoped_value_in_symbol_map<buffer_info>> bk;
    std::vector<std::pair<expr, expr>> facts;

    symbol_map<expr_info>& vars;
    symbol_map<buffer_info>& buffers;

    void add_var_info(const variable* x, interval_expr bounds, alignment_type alignment = {}) {
      if (x->field == buffer_field::none) {
        std::optional<expr_info> info = vars.lookup(x->sym);
        if (!info) {
          ensure_is_point(bounds);
          info = {std::move(bounds), alignment};
        } else {
          info->bounds = simplify_intersection(std::move(info->bounds), std::move(bounds));
          info->alignment = info->alignment & alignment;
        }
        if (auto value = as_constant(info->bounds.as_point())) {
          // The bounds tell us this is a constant point.
          info->replacement = *value;
        }
        vk.push_back(set_value_in_scope(vars, x->sym, std::move(info)));
      } else {
        expr value = bounds.as_point();
        if (!value.defined() || !is_pure(value)) {
          // TODO: Try to resolve the circular dependency that can result if we relax this constraint.
          return;
        }
        var buf = x->sym;
        if (std::find_if(bk.begin(), bk.end(), [buf](const auto& i) { return i.sym() == buf; }) == bk.end()) {
          // Save the value if we haven't already, but we set the new values below.
          bk.push_back(scoped_value_in_symbol_map<buffer_info>(buffers, buf));
        }
        std::optional<buffer_info>& info = buffers[buf];
        if (x->dim >= 0) {
          if (!info) {
            info = buffer_info(buf, x->dim + 1);
          } else {
            info->init_dims(buf, x->dim + 1);
          }
          switch (x->field) {
          case buffer_field::min: info->dims[x->dim].bounds.min = std::move(value); break;
          case buffer_field::max: info->dims[x->dim].bounds.max = std::move(value); break;
          case buffer_field::stride: info->dims[x->dim].stride = std::move(value); break;
          case buffer_field::fold_factor: info->dims[x->dim].fold_factor = std::move(value); break;
          default: break;
          }
        } else if (x->field == buffer_field::elem_size) {
          if (!info) {
            info = buffer_info(std::move(value));
          } else {
            info->elem_size = std::move(value);
          }
        } else if (x->field == buffer_field::rank) {
          if (auto rank = as_constant(value)) {
            if (!info) {
              info = buffer_info(buf, *rank);
            } else {
              info->init_dims(buf, *rank);
            }
            info->all_dims_known = true;
          }
        }
      }
    }

  public:
    knowledge(symbol_map<expr_info>& vars, symbol_map<buffer_info>& buffers) : vars(vars), buffers(buffers) {}
    knowledge(const knowledge&) = delete;
    knowledge(knowledge&&) = default;
    ~knowledge() { exit_scope(); }

    void exit_scope() {
      // Destroy our knowledge in reverse order.
      while (!vk.empty()) {
        vk.pop_back();
      }
      while (!bk.empty()) {
        bk.pop_back();
      }
    }

    void learn_from_equal(const expr& a, const expr& b) {
      if (const variable* v = a.as<variable>()) {
        add_var_info(v, point(b));
      } else if (const mod* md = a.as<mod>()) {
        if (const variable* v = md->a.as<variable>()) {
          if (auto m = as_constant(md->b)) {
            if (auto r = as_constant(b)) {
              add_var_info(v, {}, {*m, *r});
            }
          }
        }
      } else {
        // If a is not a variable, then b is not a variable (because we canonicalize variables to the left hand side),
        // so if we're going to learn from this, we need to just learn it as a fact.
        facts.push_back({a == b, true});
      }
      if (const variable* v = b.as<variable>()) {
        add_var_info(v, point(a));
      }
    }

    void learn_from_less(const expr& a, const expr& b) {
      if (const class max* r = b.as<class max>()) {
        // a < max(x, y) ==> a < x && a < y
        learn_from_less(a, r->a);
        learn_from_less(a, r->b);
      } else if (const class min* l = a.as<class min>()) {
        // min(x, y) < b ==> x < b && y < b
        learn_from_less(l->a, b);
        learn_from_less(l->b, b);
      } else {
        const variable* av = a.as<variable>();
        const variable* bv = b.as<variable>();
        if (av) add_var_info(av, {expr(), simplify(static_cast<const add*>(nullptr), b, -1)});
        if (bv) add_var_info(bv, {simplify(static_cast<const add*>(nullptr), a, 1), expr()});
        if (!(av || bv)) {
          // We couldn't learn from this, just remember it as a fact.
          facts.push_back({a < b, true});
        }
      }
    }
    void learn_from_less_equal(const expr& a, const expr& b) {
      if (const class max* r = b.as<class max>()) {
        // a <= max(x, y) ==> a <= x && a <= y
        learn_from_less_equal(a, r->a);
        learn_from_less_equal(a, r->b);
      } else if (const class min* l = a.as<class min>()) {
        // min(x, y) <= b ==> x <= b && y <= b
        learn_from_less_equal(l->a, b);
        learn_from_less_equal(l->b, b);
      } else {
        const variable* av = a.as<variable>();
        const variable* bv = b.as<variable>();
        if (av) add_var_info(av, {expr(), b});
        if (bv) add_var_info(bv, {a, expr()});
        if (!(av || bv)) {
          // We couldn't learn from this, just remember it as a fact.
          facts.push_back({a <= b, true});
        }
      }
    }

    void learn_from_true(const expr& c) {
      if (const logical_and* a = c.as<logical_and>()) {
        learn_from_true(a->a);
        learn_from_true(a->b);
      } else if (const logical_not* n = c.as<logical_not>()) {
        learn_from_false(n->a);
      } else if (const less* lt = c.as<less>()) {
        learn_from_less(lt->a, lt->b);
      } else if (const less_equal* lt = c.as<less_equal>()) {
        learn_from_less_equal(lt->a, lt->b);
      } else if (const equal* eq = c.as<equal>()) {
        learn_from_equal(eq->a, eq->b);
      } else if (!as_constant(c) && !as_variable(c)) {
        // We couldn't learn anything, just add the whole expression as a fact, if it isn't a constant or variable,
        // which could rewrite the value from any x not zero to 1.
        facts.push_back({c, expr(true)});
      }
    }
    void learn_from_false(const expr& c) {
      if (const logical_or* a = c.as<logical_or>()) {
        learn_from_false(a->a);
        learn_from_false(a->b);
      } else if (const logical_not* n = c.as<logical_not>()) {
        learn_from_true(n->a);
      } else if (const less* lt = c.as<less>()) {
        learn_from_less_equal(lt->b, lt->a);
      } else if (const less_equal* lt = c.as<less_equal>()) {
        learn_from_less(lt->b, lt->a);
      } else if (const not_equal* ne = c.as<not_equal>()) {
        learn_from_equal(ne->a, ne->b);
      } else if (!as_constant(c)) {
        // We couldn't learn anything, just add the whole expression as a fact.
        facts.push_back({c, expr(false)});
      }
    }

    expr substitute(expr x) const {
      for (const auto& i : facts) {
        x = slinky::substitute(x, i.first, i.second);
      }
      return x;
    }
  };

  knowledge learn_from_true(const expr& c) {
    knowledge result(vars, buffers);
    result.learn_from_true(c);
    return result;
  }
  knowledge learn_from_false(const expr& c) {
    knowledge result(vars, buffers);
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

  // Attempt to prove that the interval only contains true or false.
  static std::optional<bool> attempt_to_prove(const interval_expr& e) {
    if (prove_constant_true(e.min)) {
      return true;
    } else if (prove_constant_false(e.max)) {
      return false;
    } else {
      return std::nullopt;
    }
  }

  // Find the interval where `x` is true.
  interval_expr where_true(const expr& x) {
    scoped_trace trace("where_true");
    expr_info info;
    bool old_proving = proving;
    proving = true;
    mutate_boolean(x, &info);
    proving = old_proving;
    return info.bounds;
  }

  std::optional<bool> attempt_to_prove(const expr& e) { return attempt_to_prove(where_true(e)); }
  bool prove_true(const expr& e) { return prove_constant_true(where_true(e).min); }
  bool prove_false(const expr& e) { return prove_constant_false(where_true(e).max); }

  void visit(const variable* op) override {
    if (op->field != buffer_field::none) {
      var new_sym = op->sym;
      if (vars.contains(op->sym)) {
        const expr_info& info = *vars[op->sym];
        if (info.replacement.defined()) {
          new_sym = info.replacement_sym();
        }
      }
      const std::optional<buffer_info>& info = buffers[new_sym];
      expr result = new_sym == op->sym ? expr(op) : variable::make(new_sym, op->field, op->dim);
      expr bounds = result;
      if (info) {
        // TODO: We substitute here because we can't prove things like buffer_elem_size(x) == buffer_elem_size(y) where
        // x is a crop of y. If we can fix that, we don't need to substitute here, which seems better.
        auto visit_buffer_meta_value = [&, this](expr x) {
          // There are many conditions in which we should substitute buffer meta:
          // - We're being asked to substitute it (decl is undefined).
          // - The value is something we should substitute (it's simple and pure).
          // - The value is another buffer meta expression.
          // - We're trying to prove something (as opposed to producing a simplified expression).
          if (!info->decl.defined() || should_substitute(x) || x.as<variable>() || (proving && x.defined())) {
            if (!match(x, op)) {
              // This is a value we should substitute, and it's different from what we started with.
              mutate_and_set_result(x);
              return true;
            }
          }
          if (x.defined()) {
            bounds = x;
          }
          return false;
        };
        switch (op->field) {
        case buffer_field::elem_size:
          if (visit_buffer_meta_value(info->elem_size)) return;
          break;
        case buffer_field::min:
        case buffer_field::max:
        case buffer_field::stride:
        case buffer_field::fold_factor:
          if (op->dim < static_cast<index_t>(info->dims.size())) {
            if (visit_buffer_meta_value(info->dims[op->dim].get_field(op->field))) return;
          }
          break;
        default: break;
        }
      }
      switch (op->field) {
      case buffer_field::rank:
      case buffer_field::elem_size: set_result(std::move(result), {{0, std::move(bounds)}, alignment_type()}); return;
      case buffer_field::fold_factor: set_result(std::move(result), {{1, std::move(bounds)}, alignment_type()}); return;
      default: set_result(std::move(result), {point(std::move(bounds)), alignment_type()}); return;
      }
    } else {
      if (vars.contains(op->sym)) {
        expr_info info = *vars[op->sym];
        if (info.replacement.defined()) {
          // TODO: This seems like it might be expensive, but it's the simplest way to get correct bounds and alignment
          // information.
          // TODO: Maybe we should intersect any information we already had with this?
          mutate_and_set_result(info.replacement);
          return;
        } else if (auto c = as_constant(info.bounds.as_point())) {
          set_result(info.bounds.min, {point(info.bounds.min), alignment_type()});
          return;
        } else {
          if (!info.bounds.min.defined()) info.bounds.min = expr(op);
          if (!info.bounds.max.defined()) info.bounds.max = expr(op);
          set_result(op, std::move(info));
          return;
        }
      }
    }
    set_result(op, {point(expr(op)), alignment_type()});
  }

  var visit_symbol(var x) {
    if (vars.contains(x)) {
      const expr_info& info = *vars[x];
      if (info.replacement.defined()) {
        return info.replacement_sym();
      }
    }
    return x;
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
    auto less_equal = [this](const expr& a, const expr& a_max, const expr& b, const expr& b_min) {
      return prove_constant_false(simplify(static_cast<const less*>(nullptr), b_min, a_max)) ||
             (!match(a, a_max) && prove_constant_false(simplify(static_cast<const less*>(nullptr), b_min, a))) ||
             (!match(b, b_min) && prove_constant_false(simplify(static_cast<const less*>(nullptr), b, a_max)));
    };
    if (less_equal(a, a_info.bounds.max, b, b_info.bounds.min)) {
      if (T::static_type == expr_node_type::min) {
        set_result(std::move(a), std::move(a_info));
      } else {
        set_result(std::move(b), std::move(b_info));
      }
      return;
    } else if (less_equal(b, b_info.bounds.max, a, a_info.bounds.min)) {
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

    if (T::static_type == expr_node_type::mul) {
      // TODO: This is really ugly, we should have a better way of expressing such simplifications.
      if (auto c1 = as_constant(b)) {
        if (const div* d = a.as<div>()) {
          if (auto c0 = as_constant(d->b)) {
            if (*c0 != 0) {
              // This is (x/c0)*c1. If we know x is aligned to c0, the result is x*(c1/c0).
              expr_info x_info;
              expr x = mutate(d->a, &x_info);
              if (x_info.alignment.modulus > 0 && x_info.alignment.modulus % *c0 == 0) {
                mutate_and_set_result((x * *c1) / *c0);
                return;
              }
            }
          }
        }
      }
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
    } else if (T::static_type == expr_node_type::mod) {
      auto r = rewrite::make_rewriter(rewrite::pattern_expr{a} % rewrite::pattern_expr{b});
      // Taken from https://github.com/halide/Halide/blob/main/src/Simplify_Div.cpp#L125-L167.
      // clang-format off
      if (r(x % c0, eval(a_rem % c0), eval(a_mod % c0 == 0)) ||
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

    expr result = simplify(op, a, b);
    if (!result.same_as(op)) {
      mutate_and_set_result(result);
    } else {
      // Similar to the way we handle min/max, we need to check the bounds against the expression itself.
      interval_expr result_bounds = bounds_of(op, a_info.bounds, b_info.bounds);
      if (auto proven = attempt_to_prove(result_bounds)) {
        set_result(expr(*proven), {point(*proven), alignment_type()});
      } else if (auto proven = attempt_to_prove(bounds_of(op, point(a), b_info.bounds))) {
        set_result(expr(*proven), {point(*proven), alignment_type()});
      } else if (auto proven = attempt_to_prove(bounds_of(op, a_info.bounds, point(b)))) {
        set_result(expr(*proven), {point(*proven), alignment_type()});
      } else {
        set_result(std::move(result), {std::move(result_bounds), alignment_type()});
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
    } else if (auto proven = attempt_to_prove(info.bounds)) {
      set_result(expr(!*proven), {point(!*proven), alignment_type()});
    } else {
      expr result = simplify(op, std::move(a));
      if (result.same_as(op)) {
        set_result(std::move(result), {bounds_of(op, std::move(info.bounds)), alignment_type()});
      } else {
        mutate_and_set_result(result);
      }
    }
  }

  void visit(const class select* op) override {
    expr_info c_info;
    // When simplifying expressions treated as bools, we need to force them to have the result 0 or 1.
    expr c = mutate_boolean(op->condition, &c_info);
    if (!c.defined()) {
      set_result(expr(), expr_info());
      return;
    } else if (auto proven = attempt_to_prove(c_info.bounds)) {
      mutate_and_set_result(*proven ? op->true_value : op->false_value);
      return;
    }

    expr t, f;
    expr_info t_info, f_info;
    {
      auto k = learn_from_true(c);
      t = mutate(op->true_value, &t_info);
      expr learned = k.substitute(t);
      if (!learned.same_as(t)) {
        t = mutate(learned, &t_info);
      }
    }
    {
      auto k = learn_from_false(c);
      f = mutate(op->false_value, &f_info);
      expr learned = k.substitute(f);
      if (!learned.same_as(f)) {
        f = mutate(learned, &f_info);
      }
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

  static bool should_substitute(const expr& e) {
    return e.as<constant>() || (e.as<variable>() && e.as<variable>()->field == buffer_field::none);
  }

  void visit(const call* op) override {
    std::vector<expr> args;
    std::vector<interval_expr> args_bounds;
    args.reserve(op->args.size());
    args_bounds.reserve(op->args.size());
    bool changed = false;
    for (const expr& i : op->args) {
      expr_info i_info;
      args.push_back(mutate(i, &i_info));
      changed = changed || !args.back().same_as(i);
      args_bounds.push_back(std::move(i_info.bounds));
    }

    if (op->intrinsic == intrinsic::buffer_at) {
      assert(args.size() >= 1);
      if (!args[0].defined()) {
        set_result(expr(), expr_info());
        return;
      }
      auto buf = as_variable(args[0]);
      assert(buf);
      const std::optional<buffer_info>& info = buffers[*buf];
      if (info) {
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
      expr value = mutate(s.second, &value_info);
      if (should_substitute(value)) {
        value_info = expr_info::substitution(std::move(value));
        values_changed = true;
      } else {
        lets.emplace_back(s.first, std::move(value));
        values_changed = values_changed || !lets.back().second.same_as(s.second);
      }

      assert(!vars.contains(s.first));
      scoped_values.push_back(set_value_in_scope(vars, s.first, std::move(value_info)));
    }

    expr_info body_info;
    auto body = mutate(op->body, &body_info);

    scoped_values.clear();
    for (auto it = lets.rbegin(); it != lets.rend();) {
      auto deps = depends_on(body, it->first);
      // Find any deps on this variable in the inner let values.
      for (auto inner = lets.rbegin(); inner != it; ++inner) {
        depends_on(inner->second, it->first, deps);
      }

      if (!deps.any()) {
        // Prune dead lets
        it = std::make_reverse_iterator(lets.erase(std::next(it).base()));
        values_changed = true;
      } else {
        ++it;
      }
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
    assert(!vars.contains(buf));
    return mutate(body);
  }
  // if `decl` is nullptr, the buffer will be substituted.
  stmt mutate_with_buffer(const base_stmt_node* decl, stmt body, var buf, std::optional<buffer_info> buffer) {
    if (buffer) buffer->decl = stmt(decl);
    auto set_buffer = set_value_in_scope(buffers, buf, std::move(buffer));
    return mutate(body);
  }

  stmt mutate_with_bounds(stmt body, var v, interval_expr bounds, alignment_type alignment = {}) {
    assert(!vars.contains(v));
    auto set_bounds = set_value_in_scope(vars, v, {std::move(bounds), alignment});
    return mutate(body);
  }

  // Find all buffers accessed in `s`, adding them, and all the aliases of them, to `bufs`.
  void buffers_accessed_via_aliases(const stmt& s, bool consumed, std::set<var>& bufs) {
    std::vector<var> raw = find_buffer_dependencies(s, consumed, !consumed);
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
      auto s = set_value_in_scope(vars, op->sym, expr_info::substitution(bounds.min));
      set_result(mutate(op->body));
      return;
    }

    for (auto& i : buffers) {
      if (i) ++i->loop_depth;
    }
    alignment_type alignment;
    if (auto cstep = as_constant(step)) alignment.modulus = *cstep;
    if (auto cmin = as_constant(bounds.min)) alignment.remainder = *cmin;
    stmt body = mutate_with_bounds(op->body, op->sym, bounds, alignment);
    for (auto& i : buffers) {
      if (i) --i->loop_depth;
    }
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
          auto set_bounds_of_sym = set_value_in_scope(vars, op->sym, {bounds, alignment_type()});
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

  void visit(const call_stmt* op) override {
    call_stmt::symbol_list inputs = op->inputs;
    call_stmt::symbol_list outputs = op->outputs;
    bool changed = false;
    auto visit_symbol_list = [&, this](call_stmt::symbol_list& list) {
      for (var& i : list) {
        var new_i = visit_symbol(i);
        if (new_i != i) {
          i = new_i;
          changed = true;
        }
      }
    };
    visit_symbol_list(inputs);
    visit_symbol_list(outputs);
    if (changed) {
      set_result(call_stmt::make(op->target, std::move(inputs), std::move(outputs), op->attrs));
    } else {
      set_result(op);
    }
  }

  void visit(const copy_stmt* op) override {
    var src = visit_symbol(op->src);
    var dst = visit_symbol(op->dst);

    std::vector<scoped_value_in_symbol_map<expr_info>> decls;
    for (var i : op->dst_x) {
      decls.push_back(set_value_in_scope(vars, i, expr_info()));
    }

    std::vector<expr> src_x;
    src_x.reserve(op->src_x.size());
    bool changed = false;
    for (const expr& i : op->src_x) {
      src_x.push_back(mutate(i));
      changed = changed || !src_x.back().same_as(i);
    }

    if (changed || src != op->src || dst != op->dst) {
      set_result(copy_stmt::make(src, std::move(src_x), dst, op->dst_x, op->padding));
    } else {
      set_result(op);
    }
  }

  void visit(const block* op) override {
    std::vector<stmt> stmts;
    stmts.reserve(op->stmts.size());
    bool changed = false;
    // Learn from checks in this scope and store the knowledge in k.
    knowledge k(vars, buffers);
    for (const stmt& s : op->stmts) {
      stmts.push_back(mutate(s));
      if (const check* c = stmts.back().as<check>()) {
        k.learn_from_true(c->condition);
      }
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
  bool mutate_buffer(const T* op, buffer_info& info) {
    scoped_trace trace("mutate_buffer");
    info = buffer_info(mutate(op->elem_size));
    bool changed = !info.elem_size.same_as(op->elem_size);
    info.dims.reserve(op->dims.size());
    for (std::size_t d = 0; d < op->dims.size(); ++d) {
      info.dims.push_back(mutate(op->dims[d]));
      changed = changed || !info.dims.back().same_as(op->dims[d]);
    }
    info.all_dims_known = true;
    info.decl = stmt(op);
    return changed;
  }

  void visit(const allocate* op) override {
    buffer_info info;
    bool changed = mutate_buffer(op, info);
    stmt body = mutate_with_buffer(op, op->body, op->sym, info);
    scoped_trace trace("visit(const allocate*)");
    auto deps = depends_on(body, op->sym);
    if (!deps.any()) {
      set_result(std::move(body));
      return;
    } else if (!deps.buffer_data()) {
      // We only needed the buffer meta, not the allocation itself.
      set_result(mutate_with_buffer(nullptr, op->body, op->sym, std::move(info)));
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

    if (changed || !body.same_as(op->body)) {
      set_result(block::make({std::move(before),
          allocate::make(op->sym, op->storage, std::move(info.elem_size), std::move(info.dims), std::move(body)),
          std::move(after)}));
    } else {
      set_result(op);
    }
  }

  // If d is equal to buffer_dim(sym, x), return x, otherwise return -1.
  int is_buffer_dim(const dim_expr& d, var sym) {
    if (is_variable(d.bounds.min, sym, buffer_field::min)) {
      int dim = d.bounds.min.as<variable>()->dim;
      if (is_variable(d.bounds.max, sym, buffer_field::max, dim) &&
          is_variable(d.stride, sym, buffer_field::stride, dim) &&
          is_variable(d.fold_factor, sym, buffer_field::fold_factor, dim)) {
        return dim;
      }
    }
    return -1;
  }

  bool is_buffer_meta(const expr& x, const expr& value, var sym, buffer_field field, int dim) {
    return (!x.defined() && !value.defined()) || is_variable(x, sym, field, dim) || prove_true(x == value);
  }

  // Returns true if d can be represented as buffer_dim(sym, dim)
  bool is_buffer_dim(const dim_expr& d, const dim_expr& src, var sym, int dim) {
    if (!is_buffer_meta(d.bounds.min, src.bounds.min, sym, buffer_field::min, dim)) return false;
    if (!is_buffer_meta(d.bounds.max, src.bounds.max, sym, buffer_field::max, dim)) return false;

    if (prove_true(src.bounds.min == src.bounds.max)) {
      // The extent is 1, the stride and fold factor don't matter.
      return true;
    } else {
      return is_buffer_meta(d.stride, src.stride, sym, buffer_field::stride, dim) &&
             is_buffer_meta(d.fold_factor, src.fold_factor, sym, buffer_field::fold_factor, dim);
    }
  }

  // If we know that buffer metadata has some values, rewrite references to that dim to use buffer intrinsics
  // when those references use the same values.
  void canonicalize_buffer_meta(expr& x, const expr& value, buffer_field field, var sym) {
    if (!is_variable(x, sym, field) && prove_true(x == value)) x = variable::make(sym, field);
  }
  void canonicalize_buffer(buffer_info& buf, const buffer_info& src, var sym) {
    scoped_trace trace("canonicalize_buffer");
    canonicalize_buffer_meta(buf.elem_size, src.elem_size, buffer_field::elem_size, sym);
    for (int buf_d = 0; buf_d < static_cast<int>(buf.dims.size()); ++buf_d) {
      dim_expr& d = buf.dims[buf_d];
      // Try buf_d first, to prefer making identical buffers.
      if (buf_d < static_cast<int>(src.dims.size()) && is_buffer_dim(d, src.dims[buf_d], sym, buf_d)) {
        d = buffer_dim(sym, buf_d);
      } else {
        for (int src_d = 0; src_d < static_cast<int>(src.dims.size()); ++src_d) {
          if (src_d != buf_d && is_buffer_dim(d, src.dims[src_d], sym, src_d)) {
            d = buffer_dim(sym, src_d);
            break;
          }
        }
      }
    }
  }

  void visit(const make_buffer* op) override {
    expr base = mutate(op->base);
    buffer_info info;
    bool changed = mutate_buffer(op, info);

    // To avoid redundant nested simplifications, try to substitute the buffer both before and after mutating the body.
    // TODO: It may be impossible for depends_on_result::buffer_data() to change due to simplification, so the second
    // check below could be unnecessary.
    if (can_substitute_buffer(depends_on(op->body, op->sym))) {
      // We only needed the buffer meta, not the buffer itself.
      set_result(mutate_with_buffer(nullptr, op->body, op->sym, find_buffer_dependency(base), std::move(info)));
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

      auto make_truncate = [&](var src, std::size_t rank, stmt body) {
        if (src_info && src_info->all_dims_known && src_info->dims.size() == rank) {
          // We know all the dims, and the rank is already what we want to truncate to.
          return body;
        }
        // In this special case, we allow shadowing.
        return transpose::make_truncate(src, src, rank, std::move(body));
      };

      if (match(info.elem_size, buffer_elem_size(*src_buf))) {
        // To be a slice, we need every dimension that is present in the buffer_at call to be skipped, and the rest of
        // the dimensions to be identity.
        auto is_slice = [&]() {
          int dim = 0;
          std::size_t slice_rank = 0;
          std::size_t at_rank =
              std::count_if(bc->args.begin() + 1, bc->args.end(), [](const expr& i) { return i.defined(); });
          for (int d = 0; d < static_cast<int>(info.dims.size() + at_rank); ++d) {
            if (d + 1 < static_cast<index_t>(bc->args.size()) && bc->args[d + 1].defined()) {
              // Skip this dimension.
              ++dim;
            } else if (slice_rank < info.dims.size()) {
              // This arg is undefined. We need to find the next dimension here to be a slice.
              if (is_buffer_dim(info.dims[slice_rank++], *src_buf) != dim++) return false;
            } else {
              return false;
            }
          }
          return slice_rank == info.dims.size();
        };
        if (is_slice()) {
          // make_buffer drops trailing dims, do the same here.
          stmt body = make_truncate(op->sym, info.dims.size(), op->body);
          std::vector<expr> at(bc->args.begin() + 1, bc->args.end());
          set_result(mutate(slice_buffer::make(op->sym, *src_buf, std::move(at), std::move(body))));
          return;
        }

        // To be a crop, we need dimensions to either be identity, or the buffer_at argument is the same as the min.
        auto is_crop = [&]() {
          if (bc->args.size() > info.dims.size() + 1) return false;
          for (index_t d = 0; d < static_cast<index_t>(info.dims.size()); ++d) {
            if (!is_variable(info.dims[d].stride, *src_buf, buffer_field::stride, d) ||
                !is_variable(info.dims[d].fold_factor, *src_buf, buffer_field::fold_factor, d)) {
              return false;
            }

            // If the argument to buffer_at is defined, we need the min to be the same as the argument.
            // If it is not defined, it must be buffer_min(buf, d).
            bool has_at_d = d + 1 < static_cast<index_t>(bc->args.size()) && bc->args[d + 1].defined();
            expr crop_min = has_at_d ? bc->args[d + 1] : buffer_min(*src_buf, d);
            if (!match(info.dims[d].bounds.min, crop_min)) {
              return false;
            }
          }
          return true;
        };
        if (is_crop()) {
          // make_buffer drops trailing dims, do the same here.
          stmt body = make_truncate(op->sym, info.dims.size(), op->body);
          set_result(mutate(crop_buffer::make(op->sym, *src_buf, dims_bounds(info.dims), std::move(body))));
          return;
        }

        // To be a transpose, we need buffer_at to be the base of src_buf, and each dimension to be a dimension of the
        // original buffer.
        // TODO: This could probably be built into the slice check above.
        std::vector<int> permutation;
        auto is_transpose = [&]() {
          if (bc->args.size() != 1) return false;
          permutation.reserve(info.dims.size());
          for (std::size_t d = 0; d < info.dims.size(); ++d) {
            int dim = is_buffer_dim(info.dims[d], *src_buf);
            if (dim >= 0) {
              permutation.push_back(dim);
            } else {
              return false;
            }
          }
          return true;
        };
        if (is_transpose()) {
          set_result(mutate(transpose::make(op->sym, *src_buf, std::move(permutation), op->body)));
          return;
        }
      }
    }

    stmt body = mutate_with_buffer(op, op->body, op->sym, find_buffer_dependency(base), info);
    scoped_trace trace("visit(const make_buffer*)");

    changed = changed || !base.same_as(op->base);
    auto make_make_buffer = [&](stmt body) {
      auto deps = depends_on(body, op->sym);
      if (!deps.any()) {
        // This make_buffer is unused.
        return body;
      } else if (can_substitute_buffer(deps)) {
        // We only needed the buffer meta, not the buffer itself.
        return mutate_with_buffer(nullptr, body, op->sym, find_buffer_dependency(base), std::move(info));
      } else if (changed || !body.same_as(op->body)) {
        return make_buffer::make(op->sym, base, info.elem_size, info.dims, std::move(body));
      } else {
        return stmt(op);
      }
    };

    set_result(lift_decl_invariants(body, op->sym, make_make_buffer));
  }

  std::optional<buffer_info> get_buffer_info(var buf, int rank) {
    std::optional<buffer_info> info = buffers[buf];
    if (!info) {
      info = buffer_info(buf, rank);
    } else {
      info->init_dims(buf, rank);
    }
    return info;
  }

  template <typename T>
  static void enumerate_bounds(expr x, std::set<expr, node_less>& bounds, index_t offset = 0) {
    if (const add* a = x.as<add>()) {
      if (auto c = as_constant(a->b)) {
        // Remove constant adds and remember the offset for later.
        x = a->a;
        offset += *c;
      }
    }
    auto add_offset = [offset](expr x) {
      return offset ? simplify(static_cast<const class add*>(nullptr), std::move(x), offset) : x;
    };
    if (const T* t = x.as<T>()) {
      enumerate_bounds<T>(t->a, bounds, offset);
      enumerate_bounds<T>(t->b, bounds, offset);
    } else if (const class select* s = x.as<class select>()) {
      // Move constants into select here.
      if (offset) {
        bounds.insert(select(s->condition, add_offset(s->true_value), add_offset(s->false_value)));
      } else {
        bounds.insert(x);
      }
    } else {
      bounds.insert(add_offset(std::move(x)));
    }
  }

  template <typename T>
  expr remove_redundant_bounds(expr x, const std::set<expr, node_less>& bounds) {
    auto is_redundant = [&](const expr& x) {
      // A bound x is redundant if the bounds already have a value i such that T(x, i) == i
      for (const expr& i : bounds) {
        if (std::is_same<T, class min>::value && prove_true(i <= x)) return true;
        if (std::is_same<T, class max>::value && prove_true(i >= x)) return true;
      }
      return false;
    };
    if (is_redundant(x)) return expr();
    if (const T* t = x.as<T>()) {
      bool a_redundant = is_redundant(t->a);
      bool b_redundant = is_redundant(t->b);
      if (a_redundant && b_redundant) {
        return expr();
      } else if (a_redundant) {
        return remove_redundant_bounds<T>(t->b, bounds);
      } else if (b_redundant) {
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
    } else if (const div* xa = x.as<div>()) {
      if (as_constant(xa->b)) {
        // We have T(x / y, b). We can rewrite to T(x, b * y) / y, and if we can eliminate the bound, the whole
        // bound is redundant.
        for (const expr& i : bounds) {
          expr removed = remove_redundant_bounds<T>(xa->a, {mutate(i * xa->b)});
          if (!removed.same_as(xa->a)) {
            return std::move(removed) / xa->b;
          }
        }
      }
    } else if (const mul* xa = x.as<mul>()) {
      if (as_constant(xa->b)) {
        // We have T(x * y, b). We can do something similar to the above, but we need to be careful because division
        // is not invertible. Since we're looking for bounds, we can use a conservative rounding. If we're looking for
        // redundant mins, we should round up, and redundant maxes should round down.
        for (const expr& i : bounds) {
          expr rounded = std::is_same<T, class min>::value ? i + (xa->b - 1) : i;
          expr removed = remove_redundant_bounds<T>(xa->a, {mutate(rounded / xa->b)});
          if (!removed.same_as(xa->a)) {
            return std::move(removed) * xa->b;
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

    if (result.min.defined() && prove_true(result.min <= buffer.min)) result.min = expr();
    if (result.max.defined() && prove_true(result.max >= buffer.max)) result.max = expr();

    // TODO: I think it might be possible to avoid generating these min/max + mutate in some cases.
    if (result.min.defined()) buffer.min = mutate(max(buffer.min, result.min));
    if (result.max.defined()) buffer.max = mutate(min(buffer.max, result.max));

    // We might have written a select into an interval that tries to preserve the empty-ness of the interval.
    // But this might be unnecessary. Try to remove unnecessary selects here.
    rewrite::pattern_wildcard<0> x;
    rewrite::pattern_wildcard<1> y;
    rewrite::pattern_wildcard<2> z;
    rewrite::pattern_wildcard<3> w;
    rewrite::match_context ctx;
    if (match(ctx, select(x < y, z, w), result.min)) {
      if (is_variable(ctx.matched(x), buf, buffer_field::max, dim) &&
          is_variable(ctx.matched(y), buf, buffer_field::min, dim)) {
        // This select is a check that the dimension we are cropping is empty.
        // If the buffer is empty, it doesn't matter what we do, the resulting crop will still be empty, so we can
        // just take the new min.
        result.min = ctx.matched(w);
      }
    } else if (match(ctx, x + min(y, z), result.min)) {
      // This is the same as above, but the select was simplified.
      if (is_variable(ctx.matched(y), buf, buffer_field::max, dim) || match(ctx.matched(y), result.max)) {
        result.min = mutate(ctx.matched(x) + ctx.matched(z));
      } else if (is_variable(ctx.matched(z), buf, buffer_field::max, dim) || match(ctx.matched(z), result.max)) {
        result.min = mutate(ctx.matched(x) + ctx.matched(y));
      }
    }

    return result;
  }

  static bool crop_needed(const depends_on_result& deps) {
    // We don't need a crop if the buffer is only used as an input to a call. But we do need the crop if it is used as
    // an input to a copy, which uses the bounds of the input for padding.
    return deps.buffer_output || deps.buffer_src || deps.buffer_dst || deps.buffer_bounds;
  }

  void visit_crop(const base_stmt_node* op, var op_sym, var op_src, box_expr op_bounds, stmt op_body) {
    std::optional<buffer_info> info = get_buffer_info(op_src, op_bounds.size());

    while (info && info->decl.defined() && info->loop_depth == 0) {
      if (const crop_buffer* c = info->decl.as<crop_buffer>()) {
        // Substitute the outer crop bounds into this crop's bounds.
        auto c_dims = make_dims_from_bounds(c->bounds);
        for (interval_expr& i : op_bounds) {
          i = substitute_buffer(i, op_src, c_dims);
        }
        // Nested crops of the same buffer, and the crop isn't used.
        op_bounds.resize(std::max(op_bounds.size(), c->bounds.size()));
        op_bounds = c->bounds & op_bounds;
        op_src = c->src;
        info = get_buffer_info(op_src, op_bounds.size());
        op = nullptr;
      } else if (const crop_dim* c = info->decl.as<crop_dim>()) {
        // Substitute the outer crop bounds into this crop's bounds.
        auto c_dims = make_dims_from_bounds(c->dim, c->bounds);
        for (interval_expr& i : op_bounds) {
          i = substitute_buffer(i, op_src, c_dims);
        }
        // Nested crops of the same buffer, and the crop isn't used.
        op_bounds.resize(std::max<int>(op_bounds.size(), c->dim + 1));
        op_bounds[c->dim] = c->bounds & op_bounds[c->dim];
        op_src = c->src;
        info = get_buffer_info(op_src, op_bounds.size());
        op = nullptr;
      } else {
        break;
      }
    }

    // If possible, rewrite crop_buffer of one dimension to crop_dim.
    bool changed = op == nullptr;
    box_expr bounds(op_bounds.size());
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
    } else if (!crop_needed(deps)) {
      set_result(substitute(body, op_sym, op_src));
      return;
    }

    // Remove trailing undefined bounds.
    while (!bounds.empty() && !bounds.back().min.defined() && !bounds.back().max.defined()) {
      bounds.pop_back();
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
        return crop_dim::make(op_sym, op_src, d, bounds[d], body);
      } else {
        return crop_buffer::make(op_sym, op_src, bounds, body);
      }
    };

    if (bounds.empty()) {
      // This crop was a no-op.
      set_result(substitute(body, op_sym, op_src));
    } else {
      set_result(lift_decl_invariants(body, op_sym, make_crop));
    }
  }

  void visit(const crop_buffer* op) override {
    var src = visit_symbol(op->src);
    visit_crop(src == op->src ? op : nullptr, op->sym, src, op->bounds, op->body);
  }

  void visit(const crop_dim* op) override {
    var src = visit_symbol(op->src);
    box_expr bounds(op->dim + 1);
    bounds[op->dim] = op->bounds;
    visit_crop(src == op->src ? op : nullptr, op->sym, src, std::move(bounds), op->body);
  }

  void visit_slice(const base_stmt_node* op, var op_sym, var op_src, const std::vector<expr>& op_at, stmt op_body) {
    std::vector<expr> at(op_at.size());
    bool changed = op == nullptr;
    std::optional<buffer_info> info = buffers[op_src];
    for (index_t i = static_cast<index_t>(op_at.size()) - 1; i >= 0; --i) {
      if (op_at[i].defined()) {
        at[i] = mutate(op_at[i]);
        changed = changed || !at[i].same_as(op_at[i]);
        if (info && static_cast<index_t>(info->dims.size()) > i) info->dims.erase(info->dims.begin() + i);
      }
    }

    stmt body = mutate_with_buffer(op, op_body, op_sym, op_src, std::move(info));
    if (!depends_on(body, op_sym).any()) {
      set_result(std::move(body));
      return;
    }

    // Remove trailing undefined bounds.
    while (!at.empty() && !at.back().defined()) {
      at.pop_back();
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
    } else {
      set_result(lift_decl_invariants(body, op_sym, make_slice));
    }
  }

  void visit(const slice_buffer* op) override {
    var src = visit_symbol(op->src);
    visit_slice(src == op->src ? op : nullptr, op->sym, src, op->at, op->body);
  }

  void visit(const slice_dim* op) override {
    var src = visit_symbol(op->src);
    std::vector<expr> at(op->dim + 1);
    at[op->dim] = op->at;
    visit_slice(src == op->src ? op : nullptr, op->sym, src, at, op->body);
  }

  void visit(const transpose* op) override {
    const std::optional<buffer_info>* src_info = &buffers[op->src];

    var src = visit_symbol(op->src);
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

    buffer_info sym_info{expr()};
    if (src_info && *src_info) {
      if (transpose::is_truncate(dims) && (*src_info)->all_dims_known && (*src_info)->dims.size() <= dims.size()) {
        // transpose can't add dimensions.
        assert((*src_info)->dims.size() == dims.size());
        // This truncate is a no-op.
        auto s = set_value_in_scope(vars, op->sym, expr_info::substitution(variable::make(src)));
        set_result(mutate(op->body));
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

    stmt body = mutate_with_buffer(op, op->body, op->sym, src, std::move(sym_info));

    auto make_transpose = [&](stmt body) {
      auto deps = depends_on(body, op->sym);
      if (!deps.any()) {
        return body;
      } else if (!deps.buffer_dims) {
        return substitute(body, op->sym, src);
      } else if (body.same_as(op->body) && src == op->src && dims == op->dims) {
        return stmt(op);
      } else {
        return transpose::make(op->sym, src, dims, std::move(body));
      }
    };

    set_result(lift_decl_invariants(body, op->sym, make_transpose));
  }

  void visit(const clone_buffer* op) override {
    // Because we disallow shadowing (i.e. mutating buffers in place), clone_buffer can always be removed :)
    // Essentially, every operation is also a clone here.
    var src = visit_symbol(op->src);
    auto s = set_value_in_scope(vars, op->sym, expr_info::substitution(variable::make(src)));
    set_result(mutate(op->body));
  }

  void visit(const check* op) override {
    expr_info c_info;
    expr c = mutate_boolean(op->condition, &c_info);

    if (!c.defined()) {
      set_result(stmt());
    } else if (auto proven = attempt_to_prove(c_info.bounds)) {
      if (*proven) {
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

  using node_mutator::visit;
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
    expr a = mutate(op->a);
    expr b = mutate(op->b);
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
    auto make = [](expr a, expr b) {
      auto ac = as_constant(a);
      auto bc = as_constant(b);
      if (ac && bc) {
        return make_or_eval_binary<T>(*ac, *bc);
      } else {
        return T::make(std::move(a), std::move(b));
      }
    };
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
        set_result(make(op->a, std::move(b)));
      }
    } else if (sign_b != 0) {
      int old_sign = sign;
      sign *= sign_b;
      expr a = mutate(op->a);
      sign = old_sign;
      if (a.same_as(op->a)) {
        set_result(op);
      } else {
        set_result(make(std::move(a), op->b));
      }
    } else {
      set_result(op);
    }
  }

  void visit(const mul* op) override { visit_mul_div(op); }
  void visit(const div* op) override { visit_mul_div(op); }

  void visit(const mod* op) override {
    // We know that 0 <= a % b < upper_bound(abs(b)). We might be able to do better if a is constant, but even that is
    // not easy, because an upper bound of a is not necessarily an upper bound of a % b.
    if (sign < 0) {
      set_result(expr(0));
      return;
    }
    expr equiv = max(0, max(-op->b, op->b) - 1);
    expr result = mutate(equiv);
    if (!equiv.same_as(result)) {
      set_result(std::move(result));
    } else {
      set_result(op);
    }
  }

  void visit_logical() {
    // If we don't know anything about a logical op, the result is either 0 or 1.
    set_result(expr(sign < 0 ? 0 : 1));
  }
  void visit(const equal* op) override { visit_logical(); }
  void visit(const not_equal* op) override { visit_logical(); }

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
    } else {
      visit_logical();
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
    } else {
      visit_logical();
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
    } else {
      visit_logical();
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
    if (op->intrinsic == intrinsic::abs) {
      expr equiv = max(0, max(op->args[0], -op->args[0]));
      expr result = mutate(equiv);
      if (!equiv.same_as(result)) {
        set_result(std::move(result));
        return;
      }
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
