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

#include "base/chrome_trace.h"
#include "builder/node_mutator.h"
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
    } else if (is_zero(ne->a)) {
      return strip_boolean(ne->b);
    }
  }
  return x;
}

expr eval_buffer_intrinsic(intrinsic fn, const dim_expr& d) {
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
interval_expr ensure_is_point(const interval_expr& x) {
  if (deep_is_point(x)) {
    return point(x.min);
  } else {
    return x;
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

// This is based on the simplifier in Halide: https://github.com/halide/Halide/blob/main/src/Simplify_Internal.h
class simplifier : public node_mutator {
  struct buffer_info {
    expr elem_size;

    // The dimension metadata for this buffer.
    std::vector<dim_expr> dims;

    // The op that defined this buffer.
    stmt decl;
  };
  symbol_map<buffer_info> buffers;
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
  // Dummy for template code.
  void set_result(stmt s, interval_expr) { set_result(std::move(s)); }

public:
  simplifier() {}
  simplifier(const bounds_map& expr_bounds) : expr_bounds(expr_bounds) {}

  expr mutate(const expr& e, interval_expr* bounds) {
    expr result = node_mutator::mutate(e);
    if (bounds) {
      result_bounds = ensure_is_point(result_bounds);
      if (bounds != &result_bounds) {
        *bounds = std::move(result_bounds);
      }
    } else {
      result_bounds = {expr(), expr()};
    }
    return result;
  }
  // Dummy for template code.
  stmt mutate(const stmt& s, interval_expr* bounds) { return node_mutator::mutate(s); }
  expr mutate(const expr& e) override { return mutate(e, nullptr); }
  stmt mutate(const stmt& s) override { return mutate(s, nullptr); }

  void mutate_and_set_result(const expr& e) {
    assert(!result_bounds.min.defined() && !result_bounds.max.defined());
    node_mutator::set_result(mutate(e, &result_bounds));
  }

  interval_expr mutate(
      const interval_expr& x, interval_expr* min_bounds = nullptr, interval_expr* max_bounds = nullptr) {
    if (deep_is_point(x)) {
      expr result = mutate(x.min, min_bounds);
      if (min_bounds && max_bounds) {
        *max_bounds = *min_bounds;
      }
      return point(result);
    } else {
      interval_expr result = {mutate(x.min, min_bounds), mutate(x.max, max_bounds)};
      result = ensure_is_point(result);
      return result;
    }
  }

  // When we attempt to prove things about bounds, we sometimes get constant expressions, but we can't recursively
  // simplify without a high risk of infinite recursion. We can evaluate these as constants instead.
  static bool prove_constant_true(const expr& e) {
    if (!e.defined()) return false;

    std::optional<index_t> ec = evaluate_constant(e);
    if (ec) return *ec != 0;

    // This might have a constant bound we can use.
    expr predicate = constant_lower_bound(e) > 0 || constant_upper_bound(e) < 0;
    std::optional<index_t> result = evaluate_constant(predicate);
    return result && *result != 0;
  }

  static bool prove_constant_false(const expr& e) {
    if (!e.defined()) return false;

    std::optional<index_t> ec = evaluate_constant(e);
    if (ec) return *ec == 0;

    // This might have a constant bound we can use.
    expr predicate = constant_lower_bound(e) == 0 && constant_upper_bound(e) == 0;
    std::optional<index_t> result = evaluate_constant(predicate);
    return result && *result != 0;
  }

  std::optional<bool> attempt_to_prove(const expr& e) {
    scoped_trace trace("attempt_to_prove");
    interval_expr bounds;
    mutate(boolean(e), &bounds);
    if (prove_constant_true(bounds.min)) {
      return true;
    } else if (prove_constant_false(bounds.max)) {
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
    std::optional<interval_expr> bounds = expr_bounds[op->sym];
    if (bounds) {
      if (!bounds->min.defined()) bounds->min = op;
      if (!bounds->max.defined()) bounds->max = op;
      set_result(op, std::move(*bounds));
    } else {
      set_result(op, {op, op});
    }
  }

  void visit(const constant* op) override { set_result(op, {op, op}); }

  void visit(const class min* op) override {
    interval_expr a_bounds;
    expr a = mutate(op->a, &a_bounds);
    interval_expr b_bounds;
    expr b = mutate(op->b, &b_bounds);

    if (!a.defined() || !b.defined()) {
      set_result(expr(), interval_expr());
      return;
    }

    expr result = simplify(op, a, b);
    if (!result.same_as(op)) {
      mutate_and_set_result(result);
    } else if (prove_constant_true(simplify(static_cast<const less_equal*>(nullptr), a_bounds.max, b_bounds.min))) {
      set_result(std::move(a), std::move(a_bounds));
    } else if (prove_constant_true(simplify(static_cast<const less_equal*>(nullptr), b_bounds.max, a_bounds.min))) {
      set_result(std::move(b), std::move(b_bounds));
    } else {
      set_result(result, bounds_of(op, std::move(a_bounds), std::move(b_bounds)));
    }
  }
  void visit(const class max* op) override {
    interval_expr a_bounds;
    expr a = mutate(op->a, &a_bounds);
    interval_expr b_bounds;
    expr b = mutate(op->b, &b_bounds);

    if (!a.defined() || !b.defined()) {
      set_result(expr(), interval_expr());
      return;
    }

    expr result = simplify(op, a, b);
    if (!result.same_as(op)) {
      mutate_and_set_result(result);
    } else if (prove_constant_true(simplify(static_cast<const less_equal*>(nullptr), a_bounds.max, b_bounds.min))) {
      set_result(std::move(b), std::move(b_bounds));
    } else if (prove_constant_true(simplify(static_cast<const less_equal*>(nullptr), b_bounds.max, a_bounds.min))) {
      set_result(std::move(a), std::move(a_bounds));
    } else {
      set_result(result, bounds_of(op, std::move(a_bounds), std::move(b_bounds)));
    }
  }

  template <typename T>
  void visit_binary(const T* op) {
    interval_expr a_bounds;
    expr a = mutate(op->a, &a_bounds);
    interval_expr b_bounds;
    expr b = mutate(op->b, &b_bounds);

    if (!a.defined() || !b.defined()) {
      set_result(expr(), interval_expr());
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
  void visit(const sub* op) override { visit_binary(op); }
  void visit(const mul* op) override { visit_binary(op); }
  void visit(const div* op) override { visit_binary(op); }
  void visit(const mod* op) override { visit_binary(op); }

  template <typename T>
  void visit_logical(const T* op, bool coerce_boolean = false) {
    interval_expr a_bounds;
    expr a = mutate(coerce_boolean ? boolean(op->a) : op->a, &a_bounds);
    interval_expr b_bounds;
    expr b = mutate(coerce_boolean ? boolean(op->b) : op->b, &b_bounds);
    if (coerce_boolean) {
      a = strip_boolean(a);
      b = strip_boolean(b);
    }

    if (!a.defined() || !b.defined()) {
      set_result(expr(), interval_expr());
      return;
    }

    expr result = simplify(op, std::move(a), std::move(b));
    if (!result.same_as(op)) {
      mutate_and_set_result(result);
    } else {
      interval_expr result_bounds = bounds_of(op, std::move(a_bounds), std::move(b_bounds));
      if (prove_constant_true(result_bounds.min)) {
        set_result(true, {1, 1});
      } else if (prove_constant_false(result_bounds.max)) {
        set_result(false, {0, 0});
      } else {
        set_result(result, std::move(result_bounds));
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
    interval_expr bounds;
    expr a = strip_boolean(mutate(boolean(op->a), &bounds));

    if (!a.defined()) {
      set_result(expr(), interval_expr());
    } else if (prove_constant_true(bounds.min)) {
      set_result(false, {0, 0});
    } else if (prove_constant_false(bounds.max)) {
      set_result(true, {1, 1});
    } else {
      expr result = simplify(op, std::move(a));
      if (result.same_as(op)) {
        set_result(result, bounds_of(op, std::move(bounds)));
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
    return x;
  }

  void visit(const class select* op) override {
    interval_expr c_bounds;
    // When simplifying expressions treated as bools, we need to force them to have the result 0 or 1.
    expr c = strip_boolean(mutate(boolean(op->condition), &c_bounds));
    if (!c.defined()) {
      set_result(expr(), interval_expr());
      return;
    } else if (prove_constant_true(c_bounds.min)) {
      mutate_and_set_result(op->true_value);
      return;
    } else if (prove_constant_false(c_bounds.max)) {
      mutate_and_set_result(op->false_value);
      return;
    }

    expr t = op->true_value;
    expr f = op->false_value;

    t = substitute_true(t, c);
    f = substitute_false(f, c);

    interval_expr t_bounds;
    t = mutate(t, &t_bounds);
    interval_expr f_bounds;
    f = mutate(f, &f_bounds);

    if (!t.defined() || !f.defined()) {
      set_result(expr(), interval_expr());
      return;
    }

    expr e = simplify(op, std::move(c), std::move(t), std::move(f));
    if (e.same_as(op)) {
      set_result(e, bounds_of(op, std::move(c_bounds), std::move(t_bounds), std::move(f_bounds)));
    } else {
      mutate_and_set_result(e);
    }
  }

  static bool should_substitute(expr& e) { return e.as<constant>() || e.as<variable>(); }

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

    if (is_buffer_intrinsic(op->intrinsic)) {
      assert(args.size() >= 1);
      if (!args[0].defined()) {
        set_result(expr(), interval_expr());
        return;
      }
      const var* buf = as_variable(op->args[0]);
      assert(buf);
      const std::optional<buffer_info>& info = buffers[*buf];
      if (info) {
        // TODO: We substitute here because we can't prove things like buffer_elem_size(x) == buffer_elem_size(y) where
        // x is a crop of y. If we can fix that, we don't need to substitute here, which seems better.
        if (op->intrinsic == intrinsic::buffer_elem_size) {
          expr value = info->elem_size;
          if (should_substitute(value) || value.as<call>()) {
            set_result(value, point(value));
          } else {
            set_result(op, point(value));
          }
          return;
        } else if (is_buffer_dim_intrinsic(op->intrinsic)) {
          const index_t* dim = as_constant(op->args[1]);
          assert(dim);
          if (*dim < static_cast<index_t>(info->dims.size())) {
            expr value = eval_buffer_intrinsic(op->intrinsic, info->dims[*dim]);
            if (should_substitute(value) || value.as<call>()) {
              set_result(value, point(value));
            } else {
              set_result(op, point(value));
            }
            return;
          }
        } else if (op->intrinsic == intrinsic::buffer_at) {
          for (int d = 0; d < static_cast<int>(std::min(info->dims.size(), args.size() - 1)); ++d) {
            if (prove_true(args[d + 1] == info->dims[d].bounds.min)) {
              // This argument is equal to the default value, and we know it is in bounds.
              args[d + 1] = expr();
            }
          }
        }
      }
    }

    expr e = simplify(op, op->intrinsic, std::move(args));
    if (e.same_as(op)) {
      set_result(e, bounds_of(op, std::move(args_bounds)));
    } else {
      mutate_and_set_result(e);
    }
  }

  template <typename T>
  void visit_let(const T* op) {
    std::vector<std::pair<var, expr>> lets;
    lets.reserve(op->lets.size());

    using sv_type = scoped_value_in_symbol_map<interval_expr>;
    std::vector<sv_type> scoped_values;
    scoped_values.reserve(op->lets.size());

    bool values_changed = false;
    for (const auto& s : op->lets) {
      interval_expr value_bounds;
      lets.emplace_back(s.first, mutate(s.second, &value_bounds));
      values_changed = values_changed || !lets.back().second.same_as(s.second);

      assert(!expr_bounds.contains(s.first));
      scoped_values.push_back(set_value_in_scope(expr_bounds, s.first, value_bounds));
    }

    interval_expr body_bounds;
    auto body = mutate(op->body, &body_bounds);

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
      body = mutate(body, &body_bounds);
    }

    if (lets.empty()) {
      // All lets were removed.
      set_result(body, std::move(body_bounds));
    } else if (!values_changed && body.same_as(op->body)) {
      set_result(op, std::move(body_bounds));
    } else {
      set_result(T::make(std::move(lets), std::move(body)), std::move(body_bounds));
    }
  }

  void visit(const let* op) override { visit_let(op); }
  void visit(const let_stmt* op) override { visit_let(op); }

  stmt mutate_with_buffer(stmt decl, stmt body, var buf, std::optional<buffer_info> buffer) {
    if (buffer) buffer->decl = decl;
    auto set_buffer = set_value_in_scope(buffers, buf, std::move(buffer));
    return mutate(body);
  }

  stmt mutate_with_bounds(stmt body, var v, interval_expr bounds) {
    assert(!expr_bounds.contains(v));
    auto set_bounds = set_value_in_scope(expr_bounds, v, std::move(bounds));
    return mutate(body);
  }

  void visit(const loop* op) override {
    interval_expr bounds = mutate(op->bounds);
    expr step = mutate(op->step);

    if (prove_true(bounds.min > bounds.max)) {
      // This loop is dead.
      set_result(stmt());
      return;
    } else if (prove_true(bounds.min <= bounds.max && bounds.min + step > bounds.max)) {
      // The loop only runs once.
      set_result(mutate(let_stmt::make(op->sym, bounds.min, op->body)));
      return;
    }

    stmt body = mutate_with_bounds(op->body, op->sym, bounds);
    if (!body.defined()) {
      set_result(stmt());
      return;
    } else if (!depends_on(body, op->sym).any()) {
      // The body does not depend on the loop, drop the loop.
      set_result(std::move(body));
      return;
    } else if (const block* b = body.as<block>()) {
      // This next bit of logic implements loop invariant code motion. It is allowed to split the loop around invariant
      // code, turning a loop into possibly multiple loops, with loop invariant code between the loops.
      std::vector<stmt> result;
      result.reserve(b->stmts.size());

      // We build loops by adding to their body, and possibly "flushing" to an actual loop if we reach the end of where
      // we want a loop to be.
      std::vector<stmt> loop_body;
      loop_body.reserve(b->stmts.size());
      auto flush_loop = [&]() {
        if (loop_body.empty()) return;
        stmt body = block::make(std::move(loop_body));
        loop_body.clear();
        result.push_back(loop::make(op->sym, op->max_workers, bounds, step, std::move(body)));
      };
      for (const stmt& i : b->stmts) {
        if (depends_on(i, op->sym).any()) {
          // This stmt should be in the loop.
          loop_body.push_back(i);
        } else {
          // This stmt should not be in the loop. If we already have some loop body, we need to make a loop now, and
          // then put this stmt after that loop.
          flush_loop();
          result.push_back(i);
        }
      }
      if (!result.empty()) {
        flush_loop();
        set_result(mutate(block::make(std::move(result))));
        return;
      } else {
        // We didn't find anything to lift out of the loop, proceed on to other possible simplifications.
      }
    }

    if (op->is_serial()) {
      // Due to either scheduling or other simplifications, we can end up with a loop that runs a single call or copy on
      // contiguous crops of a buffer. In these cases, we can drop the loop in favor of just calling the body on the
      // union of the bounds covered by the loop.
      stmt result = body;
      std::vector<std::tuple<var, var, int, interval_expr>> new_crops;
      bool drop_loop = true;
      while (true) {
        // For now, we only handle crop_dim. I don't think crop_buffer can ever yield this simplification?
        if (const crop_dim* crop = result.as<crop_dim>()) {
          // Find the bounds of the crop on the next iteration.
          interval_expr next_iter = substitute(crop->bounds, op->sym, expr(op->sym) + op->step);
          if (prove_true(crop->bounds.max + 1 >= next_iter.min || next_iter.max + 1 >= crop->bounds.min)) {
            result = crop->body;
            auto set_bounds_of_sym = set_value_in_scope(expr_bounds, op->sym, bounds);
            interval_expr bounds_of_min, bounds_of_max;
            mutate(crop->bounds, &bounds_of_min, &bounds_of_max);
            new_crops.emplace_back(crop->sym, crop->src, crop->dim, bounds_of_min | bounds_of_max);
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
    return result;
  }

  template <typename T>
  buffer_info mutate_buffer(const T* op) {
    buffer_info info;
    info.elem_size = mutate(op->elem_size);
    info.dims.reserve(op->dims.size());
    for (std::size_t d = 0; d < op->dims.size(); ++d) {
      info.dims.push_back(mutate(op->dims[d]));
    }
    info.decl = op;
    return info;
  }

  template <typename T>
  bool buffer_changed(const T* op, const buffer_info& info) {
    if (!info.elem_size.same_as(op->elem_size)) return true;
    for (std::size_t d = 0; d < op->dims.size(); ++d) {
      if (!info.dims[d].same_as(op->dims[d])) return true;
    }
    return false;
  }

  void visit(const allocate* op) override {
    buffer_info info = mutate_buffer(op);
    stmt body = mutate_with_buffer(op, op->body, op->sym, info);
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
      const index_t* dim = as_constant(min->args[1]);
      assert(dim);
      if (match_call(d.bounds.max, intrinsic::buffer_max, sym, *dim) &&
          match_call(d.stride, intrinsic::buffer_stride, sym, *dim) &&
          match_call(d.fold_factor, intrinsic::buffer_fold_factor, sym, *dim)) {
        return *dim;
      }
    }
    return -1;
  }

  // If we know that buffer metadata has some values, rewrite references to that dim to use buffer intrinsics
  // when those references use the same values.
  void canonicalize_buffer_meta(expr& x, const expr& value, intrinsic fn, var sym) {
    if (!match_call(x, fn, sym) && prove_true(x == value)) x = call::make(fn, {sym});
  }
  void canonicalize_buffer_meta(expr& x, const expr& value, intrinsic fn, var sym, int dim) {
    if (!match_call(x, fn, sym, dim) && prove_true(x == value)) x = call::make(fn, {sym, dim});
  }
  void canonicalize_dim(dim_expr& dim, const dim_expr& src, var sym, int src_d) {
    canonicalize_buffer_meta(dim.bounds.min, src.bounds.min, intrinsic::buffer_min, sym, src_d);
    canonicalize_buffer_meta(dim.bounds.max, src.bounds.max, intrinsic::buffer_max, sym, src_d);
    canonicalize_buffer_meta(dim.stride, src.stride, intrinsic::buffer_stride, sym, src_d);
    canonicalize_buffer_meta(dim.fold_factor, src.fold_factor, intrinsic::buffer_fold_factor, sym, src_d);
  }
  void canonicalize_buffer(buffer_info& buf, const buffer_info& src, var sym) {
    canonicalize_buffer_meta(buf.elem_size, src.elem_size, intrinsic::buffer_elem_size, sym);
    for (dim_expr& d : buf.dims) {
      for (int src_d = 0; src_d < static_cast<int>(src.dims.size()); ++src_d) {
        if (is_buffer_dim(d, sym) >= 0) continue;
        canonicalize_dim(d, src.dims[src_d], sym, src_d);
      }
    }
  }

  void visit(const make_buffer* op) override {
    expr base = mutate(op->base);
    buffer_info info = mutate_buffer(op);

    // To avoid redundant nested simplifications, try to substitute the buffer both before and after mutating the body.
    // TODO: It may be impossible for depends_on_result::buffer_data() to change due to simplification, so the second
    // check below could be unnecessary.
    if (!depends_on(op->body, op->sym).buffer_data()) {
      // We only needed the buffer meta, not the buffer itself.
      set_result(mutate(substitute_buffer(op->body, op->sym, info.elem_size, info.dims)));
      return;
    }
    stmt body = mutate_with_buffer(op, op->body, op->sym, info);
    auto deps = depends_on(body, op->sym);
    if (!deps.any()) {
      // This make_buffer is unused.
      set_result(std::move(body));
      return;
    } else if (!deps.buffer_data()) {
      // We only needed the buffer meta, not the buffer itself.
      set_result(mutate(substitute_buffer(body, op->sym, info.elem_size, info.dims)));
      return;
    }

    if (const call* bc = as_intrinsic(base, intrinsic::buffer_at)) {
      // Check if this make_buffer is equivalent to transpose, slice_buffer or crop_buffer
      const var* src_buf = as_variable(bc->args[0]);
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
  static expr remove_redundant_bounds(expr x, const std::set<expr, node_less>& bounds) {
    if (bounds.count(x)) return expr();
    while (const T* t = x.as<T>()) {
      bool a_is_bound = bounds.count(t->a);
      bool b_is_bound = bounds.count(t->b);
      if (a_is_bound && b_is_bound) {
        return expr();
      } else if (a_is_bound) {
        x = t->b;
      } else if (b_is_bound) {
        x = t->a;
      } else {
        break;
      }
    }
    return x;
  }

  interval_expr mutate_crop_bounds(const interval_expr& crop, var buf, int dim, interval_expr& buffer) {
    if (!crop.min.defined() && !crop.max.defined()) return crop;

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
    result.min = remove_redundant_bounds<class max>(result.min, mins);
    result.max = remove_redundant_bounds<class min>(result.max, maxs);

    // TODO: We should not need to compare to both buffer_bounds(buf, dim) and buffer.
    if (prove_true(result.min <= buffer.min || result.min <= buffer_min(buf, dim))) result.min = expr();
    if (prove_true(result.max >= buffer.max || result.max >= buffer_max(buf, dim))) result.max = expr();

    // We already proved above that this min/max is necessary (otherwise result would be undefined here.
    if (result.min.defined()) buffer.min = max(buffer.min, result.min);
    if (result.max.defined()) buffer.max = min(buffer.max, result.max);

    return result;
  }

  static bool crop_needed(const depends_on_result& deps) {
    // We don't need a crop if the buffer is only used as an input to a call. But we do need the crop if it is used as
    // an input to a copy, which uses the bounds of the input for padding.
    return deps.buffer_output || deps.buffer_src || deps.buffer_dst;
  }

  template <typename T>
  void visit_crop(const T* op, const box_expr& op_bounds) {
    std::optional<buffer_info> info = get_buffer_info(op->src, op_bounds.size());
    box_expr bounds(op_bounds.size());

    // If possible, rewrite crop_buffer of one dimension to crop_dim.
    bool changed = false;
    for (index_t i = 0; i < static_cast<index_t>(op_bounds.size()); ++i) {
      bounds[i] = mutate_crop_bounds(op_bounds[i], op->src, i, info->dims[i].bounds);
      changed = changed || !bounds[i].same_as(op_bounds[i]);
    }
    stmt body = mutate_with_buffer(op, op->body, op->sym, std::move(info));
    auto deps = depends_on(body, op->sym);
    if (!deps.any()) {
      set_result(std::move(body));
      return;
    } else if (!crop_needed(deps)) {
      // Add clamps for the implicit bounds like crop would have done.
      for (index_t d = 0; d < static_cast<index_t>(bounds.size()); ++d) {
        bounds[d] &= slinky::buffer_bounds(op->src, d);
      }
      body = substitute_bounds(body, op->sym, bounds);
      body = substitute(body, op->sym, op->src);
      set_result(mutate(body));
      return;
    }

    // Rewrite nested crops to be one crop.
    var sym = op->sym;
    while (true) {
      if (const crop_buffer* c = body.as<crop_buffer>()) {
        if (op->sym == c->src && !depends_on(c->body, op->sym).any()) {
          bounds.resize(std::max(bounds.size(), c->bounds.size()));
          bounds = bounds & c->bounds;
          sym = c->sym;
          body = c->body;
          continue;
        }
      } else if (const crop_dim* c = body.as<crop_dim>()) {
        if (op->sym == c->src && !depends_on(c->body, op->sym).any()) {
          if (c->dim < static_cast<int>(bounds.size())) {
            bounds[c->dim] &= c->bounds;
          } else {
            bounds.resize(c->dim + 1);
            bounds[c->dim] = c->bounds;
          }
          sym = c->sym;
          body = c->body;
          continue;
        }
      }
      break;
    }

    // Remove trailing undefined bounds.
    while (!bounds.empty() && !bounds.back().min.defined() && !bounds.back().max.defined()) {
      bounds.pop_back();
    }

    // If this was a crop_buffer, and we only have one dim, we're going to change it to a crop_dim.
    const int dims_count = std::count_if(
        bounds.begin(), bounds.end(), [](const interval_expr& i) { return i.min.defined() || i.max.defined(); });
    changed = changed || (dims_count == 1 && std::is_same_v<T, crop_buffer>) || !body.same_as(op->body);

    auto make_crop = [&](const stmt& body) -> stmt {
      if (!changed) {
        return op;
      } else if (dims_count == 1) {
        // This crop is of one dimension, replace it with crop_dim.
        // We removed undefined trailing bounds, so this must be the dim we want.
        int d = static_cast<int>(bounds.size()) - 1;
        return crop_dim::make(sym, op->src, d, bounds[d], body);
      } else {
        return crop_buffer::make(sym, op->src, bounds, body);
      }
    };

    if (bounds.empty()) {
      // This crop was a no-op.
      set_result(substitute(body, sym, op->src));
    } else if (const block* b = body.as<block>()) {
      set_result(lift_decl_invariants(b->stmts, sym, make_crop));
    } else {
      set_result(make_crop(body));
    }
  }

  void visit(const crop_buffer* op) override { visit_crop(op, op->bounds); }

  void visit(const crop_dim* op) override {
    box_expr bounds(op->dim + 1);
    bounds[op->dim] = op->bounds;
    visit_crop(op, bounds);
  }

  static void update_sliced_buffer_metadata(bounds_map& bounds, var sym, span<const int> sliced) {
    for (std::optional<interval_expr>& i : bounds) {
      if (!i) continue;
      *i = slinky::update_sliced_buffer_metadata(*i, sym, sliced);
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

  template <typename T>
  void visit_slice(const T* op, const std::vector<expr>& op_at) {
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
    bounds_map old_expr_bounds = expr_bounds;
    std::optional<buffer_info> info = buffers[op->src];

    if (info) {
      info->decl = op;
      buffers[op->sym] = std::move(info);
    } else {
      buffers[op->sym] = std::nullopt;
    }
    update_sliced_buffer_metadata(buffers, op->sym, sliced_dims);
    update_sliced_buffer_metadata(expr_bounds, op->sym, sliced_dims);
    stmt body = mutate(op->body);
    buffers = std::move(old_buffers);
    expr_bounds = std::move(old_expr_bounds);

    if (!depends_on(body, op->sym).any()) {
      set_result(std::move(body));
      return;
    }

    // Remove trailing undefined bounds.
    while (!at.empty() && !at.back().defined()) {
      at.pop_back();
    }

    changed = changed || at.size() != op_at.size() || !body.same_as(op->body);

    // If this was a slice_buffer, and we only have one dimension, we're going to change it to a slice_dim.
    const int at_count = std::count_if(at.begin(), at.end(), [](const expr& i) { return i.defined(); });
    changed = changed || (at_count == 1 && std::is_same_v<T, slice_buffer>);

    auto make_slice = [&](const stmt& body) -> stmt {
      if (!changed) {
        return op;
      } else if (at_count == 1) {
        // This slice is of one dimension, replace it with slice_dim.
        // We removed undefined trailing bounds, so this must be the dim we want.
        int d = static_cast<int>(at.size()) - 1;
        return slice_dim::make(op->sym, op->src, d, at[d], body);
      } else {
        return slice_buffer::make(op->sym, op->src, at, body);
      }
    };

    if (at.empty()) {
      // This slice was a no-op.
      set_result(substitute(body, op->sym, op->src));
    } else if (const block* b = body.as<block>()) {
      set_result(lift_decl_invariants(b->stmts, op->sym, make_slice));
    } else {
      set_result(make_slice(body));
    }
  }

  void visit(const slice_buffer* op) override { visit_slice(op, op->at); }

  void visit(const slice_dim* op) override {
    std::vector<expr> at(op->dim + 1);
    at[op->dim] = op->at;
    visit_slice(op, at);
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
      if (transpose::is_truncate(dims) && (*src_info)->dims.size() <= dims.size()) {
        // transpose can't add dimensions.
        assert((*src_info)->dims.size() == dims.size());
        // This truncate is a no-op.
        set_result(mutate(substitute(op->body, op->sym, src)));
        return;
      }

      sym_info.elem_size = (*src_info)->elem_size;
      sym_info.dims = permute(op->dims, (*src_info)->dims);
    }
    sym_info.decl = op;

    stmt body = mutate_with_buffer(op, op->body, op->sym, std::move(sym_info));

    if (const block* b = body.as<block>()) {
      set_result(lift_decl_invariants(
          b->stmts, op->sym, [&](stmt body) { return mutate(transpose::make(op->sym, src, dims, std::move(body))); }));
    } else if (!depends_on(body, op->sym).any()) {
      set_result(std::move(body));
    } else if (body.same_as(op->body) && src == op->src && dims == op->dims) {
      set_result(op);
    } else {
      set_result(transpose::make(op->sym, src, dims, std::move(body)));
    }
  }

  void visit(const clone_buffer* op) override {
    std::optional<buffer_info> info = buffers[op->src];
    if (info) info->decl = op;
    stmt body = mutate_with_buffer(op, op->body, op->sym, std::move(info));

    auto make_clone = [&](const stmt& body) -> stmt {
      if (!depends_on(body, op->src).any()) {
        // We didn't use the original buffer. We can just use that instead.
        return substitute(body, op->sym, op->src);
      } else if (!body.same_as(op->body)) {
        // We don't have any nested simplifications, no need to recursively mutate.
        return clone_buffer::make(op->sym, op->src, body);
      } else {
        return op;
      }
    };

    if (const block* b = body.as<block>()) {
      set_result(lift_decl_invariants(b->stmts, op->sym, make_clone));
    } else if (!depends_on(body, op->sym).any()) {
      set_result(std::move(body));
    } else {
      set_result(make_clone(body));
    }
  }

  void visit(const check* op) override {
    interval_expr c_bounds;
    expr c = strip_boolean(mutate(boolean(op->condition), &c_bounds));

    if (!c.defined()) {
      set_result(stmt());
    } else if (prove_constant_true(c_bounds.min)) {
      set_result(stmt());
    } else if (prove_constant_false(c_bounds.max)) {
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
stmt simplify(const stmt& s, const bounds_map& bounds) {
  scoped_trace trace("simplify");
  return simplifier(bounds).mutate(s);
}
interval_expr simplify(const interval_expr& e, const bounds_map& bounds) {
  simplifier s(bounds);
  return s.mutate(e);
}

interval_expr bounds_of(const expr& x, const bounds_map& expr_bounds) {
  scoped_trace trace("bounds_of");
  simplifier s(expr_bounds);
  interval_expr result;
  s.mutate(x, &result);
  return result;
}

interval_expr bounds_of(const interval_expr& x, const bounds_map& expr_bounds) {
  if (deep_is_point(x)) {
    return bounds_of(x.min, expr_bounds);
  } else {
    scoped_trace trace("bounds_of");
    simplifier s(expr_bounds);
    interval_expr bounds_of_min, bounds_of_max;
    s.mutate(x, &bounds_of_min, &bounds_of_max);
    return {
        simplify(static_cast<const class min*>(nullptr), bounds_of_min.min, bounds_of_max.min),
        simplify(static_cast<const class max*>(nullptr), bounds_of_min.max, bounds_of_max.max),
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
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);
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
    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);
    if (ca && cb) {
      set_result(make_binary<T>(*ca, *cb));
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
  void visit_mul_div(const T* op, bool is_mul) {
    // When we multiply by a negative number, we need to flip whether we are looking for an upper or lower bound.
    int sign_a = sign_of(op->a);
    int sign_b = sign_of(op->b);
    // TODO: We should be able to handle the numerator of div too, it's just tricky.
    if (is_mul && sign_a != 0) {
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

  void visit(const mul* op) override { visit_mul_div(op, /*is_mul=*/true); }
  void visit(const div* op) override { visit_mul_div(op, /*is_mul=*/false); }

  void visit(const mod* op) override { set_result(op); }

  template <typename T>
  void visit_equal(const T* op) {
    // Can we tighten this? I'm not sure. We need both upper and lower bounds to say anything here.
    if (sign < 0) {
      set_result(expr(0));
    } else {
      set_result(expr(1));
    }
  }
  void visit(const equal* op) override { visit_equal(op); }
  void visit(const not_equal* op) override { visit_equal(op); }

  template <typename T>
  void visit_less(const T* op) {
    expr a, b;
    // This is a constant version of that found in bounds_of_less:
    // - For a lower bound, we want to know if this can ever be false, so we want the upper bound of the lhs and the
    // lower bound of the rhs.
    // - For an upper bound, we want to know if this can ever be true, so we want the lower bound of the lhs and the
    // upper bound of the rhs.
    sign = -sign;
    a = mutate(op->a);
    sign = -sign;
    b = mutate(op->b);

    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);
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
    // Whenever we mutate an expression implicitly converted to bool, we need to force it to have the value 0 or 1.
    expr a = recurse ? mutate(boolean(op->a)) : op->a;
    expr b = recurse ? mutate(boolean(op->b)) : op->b;

    const index_t* ca = as_constant(a);
    const index_t* cb = as_constant(b);

    if (ca && cb) {
      set_result(make_binary<T>(*ca, *cb));
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
    // Whenever we mutate an expression implicitly converted to bool, we need to force it to have the value 0 or 1.
    expr a = mutate(boolean(op->a));
    sign = -sign;
    const index_t* ca = as_constant(a);
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
    const index_t* ct = as_constant(t);
    const index_t* cf = as_constant(f);
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
        if (const index_t* ca = as_constant(a)) {
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

}  // namespace slinky
