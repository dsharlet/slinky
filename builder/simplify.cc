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

// std::optional::operator= causes asan failures...
template <typename T>
void assign(std::optional<T>& dst, std::optional<T> src) {
  if (src) {
    dst = std::move(src);
  } else {
    dst = std::nullopt;
  }
}

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

bool is_buffer_dim_intrinsic(intrinsic fn) {
  switch (fn) {
  case intrinsic::buffer_min:
  case intrinsic::buffer_max:
  case intrinsic::buffer_stride:
  case intrinsic::buffer_fold_factor: return true;
  default: return false;
  }
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

// This is based on the simplifier in Halide: https://github.com/halide/Halide/blob/main/src/Simplify_Internal.h
class simplifier : public node_mutator {
  struct buffer_info {
    expr elem_size;

    // The dimension metadata for this buffer.
    std::vector<dim_expr> dims;
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
        set_result(result, bounds_of(op, bounds));
      } else {
        mutate_and_set_result(result);
      }
    }
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

    if (is_boolean(c) && !as_constant(c)) {
      t = substitute(t, c, true);
      f = substitute(f, c, false);
    } else {
      // We can't substitute in this case because if c is true but not 1, it will change the meaning of the values.
    }

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
        body = mutate(substitute(body, it->first, it->second), &body_bounds);
        for (auto inner = lets.rbegin(); inner != it; ++inner) {
          inner->second = substitute(inner->second, it->first, it->second);
        }
        it = std::make_reverse_iterator(lets.erase(std::next(it).base()));
        values_changed = true;
      } else {
        ++it;
      }
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

  stmt mutate_with_buffer(stmt body, var buf, std::optional<buffer_info> buffer) {
    auto set_buffer = set_value_in_scope(buffers, buf, std::move(buffer));
    return mutate(body);
  }
  stmt mutate_with_bounds(stmt body, var buf, std::optional<box_expr> bounds) {
    std::optional<buffer_info> info;
    if (bounds) {
      info = buffer_info();
      info->dims.resize(bounds->size());
      for (std::size_t d = 0; d < info->dims.size(); ++d) {
        if ((*bounds)[d].min.defined()) info->dims[d].bounds.min = (*bounds)[d].min;
        if ((*bounds)[d].max.defined()) info->dims[d].bounds.max = (*bounds)[d].max;
      }
    }
    return mutate_with_buffer(std::move(body), buf, std::move(info));
  }

  stmt mutate_with_bounds(stmt body, var v, interval_expr bounds) {
    assert(!expr_bounds.contains(v));
    auto set_bounds = set_value_in_scope(expr_bounds, v, std::move(bounds));
    return mutate(body);
  }

  void visit(const loop* op) override {
    scoped_trace trace("visit(const loop*)");
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
      scoped_trace trace("licm");
      // Lift loop invariants from the end of the loop.
      std::vector<stmt> after;
      after.reserve(b->stmts.size());
      std::vector<stmt> loop_body = b->stmts;
      while (!loop_body.empty() && !depends_on(loop_body.back(), op->sym).any()) {
        after.push_back(std::move(loop_body.back()));
        loop_body.pop_back();
      }
      // Lift loop invariants from the beginning of the loop.
      std::vector<stmt> before;
      before.reserve(b->stmts.size());
      while (!loop_body.empty() && !depends_on(loop_body.front(), op->sym).any()) {
        before.push_back(std::move(loop_body.front()));
        loop_body.erase(loop_body.begin());
      }
      if (!before.empty() || !after.empty()) {
        std::reverse(after.begin(), after.end());
        stmt result = block::make({
            block::make(std::move(before)),
            loop::make(op->sym, op->max_workers, bounds, step, block::make(std::move(loop_body))),
            block::make(std::move(after)),
        });
        set_result(mutate(result));
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
          // Find the bounds of the crop on the next iteration.
          interval_expr next_iter = substitute(crop->bounds, op->sym, expr(op->sym) + op->step);
          if (prove_true(crop->bounds.max + 1 >= next_iter.min || next_iter.max + 1 >= crop->bounds.min)) {
            result = crop->body;
            interval_expr new_crop = bounds_of(crop->bounds, {{op->sym, bounds}});
            new_crops.emplace_back(crop->sym, crop->src, crop->dim, new_crop);
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
    stmt body = mutate_with_buffer(op->body, op->sym, info);
    auto deps = depends_on(body, op->sym);
    if (!deps.any()) {
      set_result(stmt());
      return;
    } else if (!deps.buffer_data()) {
      // We only needed the buffer meta, not the allocation itself.
      body = substitute_buffer(body, op->sym, info.elem_size, info.dims);
      set_result(mutate(body));
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

  void visit(const make_buffer* op) override {
    expr base = mutate(op->base);
    buffer_info info = mutate_buffer(op);
    stmt body = mutate_with_buffer(op->body, op->sym, info);
    auto deps = depends_on(body, op->sym);
    if (!deps.any()) {
      // This make_buffer is unused.
      set_result(std::move(body));
      return;
    } else if (!deps.buffer_data()) {
      // We only needed the buffer meta, not the buffer itself.
      body = substitute_buffer(body, op->sym, info.elem_size, info.dims);
      set_result(mutate(body));
      return;
    }

    if (const call* bc = base.as<call>()) {
      // Check if this make_buffer is equivalent to transpose, slice_buffer or crop_buffer
      if (bc->intrinsic == intrinsic::buffer_at) {
        const var* src_buf = as_variable(bc->args[0]);
        assert(src_buf);

        const std::optional<buffer_info>& src_info = buffers[*src_buf];
        if (src_info) {
          // Before trying to do anything, try to normalize the dimensions to be in terms of src_buf metadata.
          if (prove_true(info.elem_size == src_info->elem_size)) info.elem_size = buffer_elem_size(*src_buf);
          for (dim_expr& d : info.dims) {
            for (int src_d = 0; src_d < static_cast<int>(src_info->dims.size()); ++src_d) {
              const dim_expr& src_dim = src_info->dims[src_d];
              if (prove_true(d.bounds.min == src_dim.bounds.min)) d.bounds.min = buffer_min(*src_buf, src_d);
              if (prove_true(d.bounds.max == src_dim.bounds.max)) d.bounds.max = buffer_max(*src_buf, src_d);
              if (prove_true(d.stride == src_dim.stride)) d.stride = buffer_stride(*src_buf, src_d);
              if (prove_true(d.fold_factor == src_dim.fold_factor)) d.fold_factor = buffer_fold_factor(*src_buf, src_d);
            }
          }
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
              is_slice = is_slice && match(info.dims[slice_rank++], buffer_dim(*src_buf, dim++));
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
            if (!match(info.dims[d].stride, buffer_stride(*src_buf, d)) ||
                !match(info.dims[d].fold_factor, buffer_fold_factor(*src_buf, d))) {
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
          // Returns the dimension of a buffer intrinsic, or -1 if not the expected intrinsic.
          auto buffer_intrinsic_dim = [=](intrinsic fn, const expr& x) -> int {
            if (const call* c = x.as<call>()) {
              if (c->intrinsic != fn) return -1;
              assert(c->args.size() == 2);

              if (*as_variable(c->args[0]) != *src_buf) return -1;
              return *as_constant(c->args[1]);
            }
            return -1;
          };
          for (std::size_t d = 0; d < info.dims.size(); ++d) {
            int min_dim = buffer_intrinsic_dim(intrinsic::buffer_min, info.dims[d].bounds.min);
            int max_dim = buffer_intrinsic_dim(intrinsic::buffer_max, info.dims[d].bounds.max);
            int stride_dim = buffer_intrinsic_dim(intrinsic::buffer_stride, info.dims[d].stride);
            int fold_factor_dim = buffer_intrinsic_dim(intrinsic::buffer_fold_factor, info.dims[d].fold_factor);
            if (min_dim >= 0 && min_dim == max_dim && min_dim == stride_dim && fold_factor_dim == min_dim) {
              permutation.push_back(min_dim);
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
    }

    if (const block* b = body.as<block>()) {
      std::vector<stmt> stmts;
      stmts.reserve(b->stmts.size());
      for (const stmt& s : b->stmts) {
        stmts.push_back(mutate(make_buffer::make(op->sym, base, info.elem_size, info.dims, s)));
      }
      set_result(block::make(std::move(stmts)));
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
      info->dims.resize(rank);
    }
    assert(rank <= static_cast<int>(info->dims.size()));
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
      if (is_buffer_min(m->a, sym, dim)) return simplify_crop_bound(m->b, sym, dim);
      if (is_buffer_min(m->b, sym, dim)) return simplify_crop_bound(m->a, sym, dim);
    } else if (const class min* m = x.as<class min>()) {
      if (is_buffer_max(m->a, sym, dim)) return simplify_crop_bound(m->b, sym, dim);
      if (is_buffer_max(m->b, sym, dim)) return simplify_crop_bound(m->a, sym, dim);
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

  void visit(const crop_buffer* op) override {
    scoped_trace trace("visit(const crop_buffer*)");
    std::optional<buffer_info> info = get_buffer_info(op->src, op->bounds.size());
    box_expr bounds(op->bounds.size());

    // If possible, rewrite crop_buffer of one dimension to crop_dim.
    index_t dims_count = 0;
    bool changed = false;
    for (index_t i = 0; i < static_cast<index_t>(op->bounds.size()); ++i) {
      bounds[i] = mutate_crop_bounds(op->bounds[i], op->src, i, info->dims[i].bounds);
      changed = changed || !bounds[i].same_as(op->bounds[i]);

      dims_count += bounds[i].min.defined() || bounds[i].max.defined() ? 1 : 0;
    }
    stmt body = mutate_with_buffer(op->body, op->sym, std::move(info));
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

    // Remove trailing undefined bounds.
    while (!bounds.empty() && !bounds.back().min.defined() && !bounds.back().max.defined()) {
      bounds.pop_back();
    }
    if (bounds.empty()) {
      // This crop was a no-op.
      body = substitute(body, op->sym, op->src);
      set_result(std::move(body));
    } else if (const block* b = body.as<block>()) {
      std::vector<stmt> stmts;
      stmts.reserve(b->stmts.size());
      for (const stmt& s : b->stmts) {
        stmts.push_back(mutate(crop_buffer::make(op->sym, op->src, bounds, s)));
      }
      set_result(block::make(std::move(stmts)));
    } else if (dims_count == 1) {
      // This crop is of one dimension, replace it with crop_dim.
      // We removed undefined trailing bounds, so this must be the dim we want.
      int d = static_cast<int>(bounds.size()) - 1;
      set_result(mutate(crop_dim::make(op->sym, op->src, d, std::move(bounds[d]), std::move(body))));
    } else if (changed || !body.same_as(op->body)) {
      set_result(crop_buffer::make(op->sym, op->src, std::move(bounds), std::move(body)));
    } else {
      set_result(op);
    }
  }

  void visit(const crop_dim* op) override {
    scoped_trace trace("visit(const crop_dim*)");
    std::optional<buffer_info> info = get_buffer_info(op->src, op->dim + 1);
    interval_expr bounds = mutate_crop_bounds(op->bounds, op->src, op->dim, info->dims[op->dim].bounds);
    if (!bounds.min.defined() && !bounds.max.defined()) {
      // This crop is a no-op.
      stmt body = substitute(op->body, op->sym, op->src);
      set_result(mutate(body));
      return;
    }

    stmt body = mutate_with_buffer(op->body, op->sym, std::move(info));
    auto deps = depends_on(body, op->sym);
    if (!deps.any()) {
      set_result(std::move(body));
      return;
    } else if (!crop_needed(deps)) {
      // Add clamps for the implicit bounds like crop would have done.
      body = substitute_bounds(body, op->sym, op->dim, bounds & buffer_bounds(op->src, op->dim));
      body = substitute(body, op->sym, op->src);
      set_result(mutate(body));
      return;
    }

    if (const slice_dim* slice = body.as<slice_dim>()) {
      if (slice->src == op->sym && slice->dim == op->dim) {
        // This is a slice of the same dimension of the buffer we just cropped.
        // Rewrite the inner slice to just slice the src of the outer crop.
        expr at = clamp(slice->at, bounds & buffer_bounds(op->src, op->dim));
        body = slice_dim::make(slice->sym, op->src, op->dim, at, slice->body);
      }
    } else if (const crop_dim* crop = body.as<crop_dim>()) {
      if (crop->src == op->sym) {
        if (crop->dim == op->dim) {
          // Two nested crops of the same dimension. Rewrite the inner crop to do both crops, the outer crop might
          // become unused.
          body = crop_dim::make(crop->sym, op->src, op->dim, bounds & crop->bounds, crop->body);
        } else {
          // TODO: This is a nested crop of the same buffer, use crop_buffer instead.
        }
      }
    }

    if (const block* b = body.as<block>()) {
      std::vector<stmt> stmts;
      stmts.reserve(b->stmts.size());
      for (const stmt& s : b->stmts) {
        stmts.push_back(mutate(crop_dim::make(op->sym, op->src, op->dim, bounds, s)));
      }
      set_result(block::make(std::move(stmts)));
    } else if (bounds.same_as(op->bounds) && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(crop_dim::make(op->sym, op->src, op->dim, std::move(bounds), std::move(body)));
    }
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

  void visit(const slice_buffer* op) override {
    scoped_trace trace("visit(const slice_buffer*)");
    std::vector<expr> at(op->at.size());
    std::vector<int> sliced_dims;
    bool changed = false;
    for (index_t i = 0; i < static_cast<index_t>(op->at.size()); ++i) {
      if (op->at[i].defined()) {
        at[i] = mutate(op->at[i]);
        changed = changed || !at[i].same_as(op->at[i]);
        sliced_dims.push_back(i);
      }
    }

    symbol_map<buffer_info> old_buffers = buffers;
    bounds_map old_expr_bounds = expr_bounds;
    assign(buffers[op->sym], buffers[op->src]);
    update_sliced_buffer_metadata(buffers, op->sym, sliced_dims);
    update_sliced_buffer_metadata(expr_bounds, op->sym, sliced_dims);
    stmt body = mutate(op->body);
    buffers = std::move(old_buffers);
    expr_bounds = std::move(old_expr_bounds);

    auto deps = depends_on(body, op->sym);
    if (!deps.any()) {
      set_result(std::move(body));
      return;
    }

    // Remove trailing undefined bounds.
    while (!at.empty() && !at.back().defined()) {
      at.pop_back();
    }
    changed = changed || at.size() != op->at.size();
    if (at.empty()) {
      // This slice was a no-op.
      body = substitute(body, op->sym, op->src);
      set_result(std::move(body));
    } else if (const block* b = body.as<block>()) {
      std::vector<stmt> stmts;
      stmts.reserve(b->stmts.size());
      for (const stmt& s : b->stmts) {
        stmts.push_back(mutate(slice_buffer::make(op->sym, op->src, at, s)));
      }
      set_result(block::make(std::move(stmts)));
    } else if (sliced_dims.size() == 1) {
      // This slice is of one dimension, replace it with slice_dim.
      // We removed undefined trailing bounds, so this must be the dim we want.
      int d = static_cast<int>(at.size()) - 1;
      set_result(slice_dim::make(op->sym, op->src, d, std::move(at[d]), std::move(body)));
    } else if (changed || !body.same_as(op->body)) {
      set_result(slice_buffer::make(op->sym, op->src, std::move(at), std::move(body)));
    } else {
      set_result(op);
    }
  }

  void visit(const slice_dim* op) override {
    scoped_trace trace("visit(const slice_dim*)");
    expr at = mutate(op->at);

    symbol_map<buffer_info> old_buffers = buffers;
    bounds_map old_expr_bounds = expr_bounds;
    int sliced_dims[] = {op->dim};
    assign(buffers[op->sym], buffers[op->src]);
    update_sliced_buffer_metadata(buffers, op->sym, sliced_dims);
    update_sliced_buffer_metadata(expr_bounds, op->sym, sliced_dims);
    stmt body = mutate(op->body);
    buffers = std::move(old_buffers);
    expr_bounds = std::move(old_expr_bounds);

    auto deps = depends_on(body, op->sym);
    if (!deps.any()) {
      set_result(std::move(body));
    } else if (const block* b = body.as<block>()) {
      std::vector<stmt> stmts;
      stmts.reserve(b->stmts.size());
      for (const stmt& s : b->stmts) {
        stmts.push_back(mutate(slice_dim::make(op->sym, op->src, op->dim, at, s)));
      }
      set_result(block::make(std::move(stmts)));
    } else if (at.same_as(op->at) && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(slice_dim::make(op->sym, op->src, op->dim, std::move(at), std::move(body)));
    }
  }

  void visit(const transpose* op) override {
    scoped_trace trace("visit(const transpose*)");
    std::optional<buffer_info> info = buffers[op->src];
    if (info) {
      if (op->is_truncate() && info->dims.size() <= op->dims.size()) {
        // transpose can't add dimensions.
        assert(info->dims.size() == op->dims.size());
        // This truncate is a no-op.
        set_result(mutate(substitute(op->body, op->sym, op->src)));
        return;
      }
      buffer_info new_info;
      new_info.elem_size = info->elem_size;
      new_info.dims = permute(op->dims, info->dims);
      info = std::move(new_info);
    }

    stmt body = mutate_with_buffer(op->body, op->sym, std::move(info));

    if (const transpose* body_t = body.as<transpose>()) {
      if (body_t->src == op->sym) {
        // Nested transposes of the same buffer, rewrite the inner transpose to directly transpose our src.
        body = transpose::make(body_t->sym, op->src, permute(body_t->dims, op->dims), body_t->body);
      }
    }

    auto deps = depends_on(body, op->sym);
    if (!deps.any()) {
      set_result(std::move(body));
    } else if (const block* b = body.as<block>()) {
      std::vector<stmt> stmts;
      stmts.reserve(b->stmts.size());
      for (const stmt& s : b->stmts) {
        stmts.push_back(mutate(transpose::make(op->sym, op->src, op->dims, s)));
      }
      set_result(block::make(std::move(stmts)));
    } else if (body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(transpose::make(op->sym, op->src, op->dims, std::move(body)));
    }
  }

  void visit(const clone_buffer* op) override {
    stmt body = mutate_with_buffer(op->body, op->sym, buffers[op->src]);

    if (!depends_on(body, op->sym).any()) {
      set_result(std::move(body));
    } else if (!depends_on(body, op->src).any()) {
      // We didn't use the original buffer. We can just use that instead.
      set_result(mutate(substitute(body, op->sym, variable::make(op->src))));
    } else if (const block* b = body.as<block>()) {
      std::vector<stmt> stmts;
      stmts.reserve(b->stmts.size());
      for (const stmt& s : b->stmts) {
        stmts.push_back(mutate(clone_buffer::make(op->sym, op->src, s)));
      }
      set_result(block::make(std::move(stmts)));
    } else if (body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(clone_buffer::make(op->sym, op->src, std::move(body)));
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
  interval_expr bounds;
  s.mutate(x, &bounds);
  return bounds;
}

interval_expr bounds_of(const interval_expr& x, const bounds_map& expr_bounds) {
  if (deep_is_point(x)) {
    return bounds_of(x.min, expr_bounds);
  } else {
    scoped_trace trace("bounds_of");
    simplifier s(expr_bounds);
    interval_expr bounds_of_min, bounds_of_max;
    s.mutate(x.min, &bounds_of_min);
    s.mutate(x.max, &bounds_of_max);
    return s.mutate(bounds_of_min | bounds_of_max);
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

interval_expr where_true(const expr& condition, var x) {
  scoped_trace trace("where_true");
  // TODO: This needs a proper implementation. For now, a ridiculous hack: trial and error.
  // We use the leaves of the expression as guesses around which to search.
  // We could use every node in the expression...
  class initial_guesses : public recursive_node_visitor {
  public:
    std::vector<expr> leaves;

    void visit(const variable* op) override { leaves.push_back(op); }
    void visit(const constant* op) override { leaves.push_back(op); }
    void visit(const call* op) override {
      if (is_buffer_intrinsic(op->intrinsic)) leaves.push_back(op);
    }
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
        if (prove_true(substitute(condition, x, i + j))) {
          result_i.min = i + j;
          result_i.max = result_i.min;
        }
      } else if (prove_true(substitute(condition, x, i + j))) {
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

}  // namespace slinky
