#include "builder/simplify.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <limits>
#include <optional>
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

// This is based on the simplifier in Halide: https://github.com/halide/Halide/blob/main/src/Simplify_Internal.h
class simplifier : public node_mutator {
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
  // Dummy for template code.
  void set_result(stmt s, interval_expr) { set_result(std::move(s)); }

public:
  simplifier() {}
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
    if (x.is_point()) {
      expr result = mutate(x.min, min_bounds);
      if (min_bounds && max_bounds) {
        *max_bounds = *min_bounds;
      }
      return point(result);
    } else {
      interval_expr result = {mutate(x.min, min_bounds), mutate(x.max, max_bounds)};
      if (!result.is_point() && match(result.min, result.max)) {
        // If the bounds are the same, make sure same_as returns true.
        result.max = result.min;
      }
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

    if (prove_constant_true(bounds.min)) {
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
    if (prove_constant_true(c_bounds.min)) {
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

    if (is_buffer_intrinsic(op->intrinsic)) {
      assert(args.size() >= 1);
      if (!args[0].defined()) {
        set_result(expr(), interval_expr());
        return;
      }
    }

    if (op->intrinsic == intrinsic::buffer_min || op->intrinsic == intrinsic::buffer_max) {
      assert(args.size() == 2);
      const var* buf = as_variable(args[0]);
      const index_t* dim = as_constant(args[1]);
      assert(buf);
      assert(dim);
      const std::optional<box_expr>& bounds = buffer_bounds[*buf];
      if (bounds && *dim < static_cast<index_t>(bounds->size())) {
        const interval_expr& dim_bounds = (*bounds)[*dim];
        if (op->intrinsic == intrinsic::buffer_min && dim_bounds.min.defined()) {
          mutate_and_set_result(dim_bounds.min);
          return;
        } else if (op->intrinsic == intrinsic::buffer_max && dim_bounds.max.defined()) {
          mutate_and_set_result(dim_bounds.max);
          return;
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

  static bool is_trivial_let_value(expr& e) { return e.as<constant>() || e.as<variable>(); }

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
      } else if (is_trivial_let_value(it->second)) {
        // Inline single-ref lets outside of a loop, along with lets that are trivial
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

  stmt mutate_with_bounds(stmt body, var buf, std::optional<box_expr> bounds) {
    assert(!buffer_bounds.contains(buf));
    auto set_bounds = set_value_in_scope(buffer_bounds, buf, std::move(bounds));
    return mutate(body);
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

  void visit(const allocate* op) override {
    expr elem_size = mutate(op->elem_size);
    std::vector<dim_expr> dims;
    box_expr bounds;
    dims.reserve(op->dims.size());
    bool changed = false;
    for (std::size_t d = 0; d < op->dims.size(); ++d) {
      interval_expr bounds_d = mutate(op->dims[d].bounds);
      dim_expr new_dim = {bounds_d, mutate(op->dims[d].stride), mutate(op->dims[d].fold_factor)};
      if (is_constant(new_dim.fold_factor, dim::unfolded)) new_dim.fold_factor = expr();
      changed = changed || !new_dim.same_as(op->dims[d]);
      dims.push_back(std::move(new_dim));
      bounds.push_back(std::move(bounds_d));
    }
    stmt body = mutate_with_bounds(op->body, op->sym, std::move(bounds));
    if (!body.defined()) {
      set_result(stmt());
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
        changed = true;
      } else {
        set_result(block::make({b->stmts.begin(), end_before}));
        return;
      }
    }

    if (changed || !elem_size.same_as(op->elem_size) || !body.same_as(op->body)) {
      set_result(block::make({std::move(before),
          allocate::make(op->sym, op->storage, std::move(elem_size), std::move(dims), std::move(body)),
          std::move(after)}));
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
    for (std::size_t d = 0; d < op->dims.size(); ++d) {
      interval_expr new_bounds = mutate(op->dims[d].bounds);
      dim_expr new_dim = {new_bounds, mutate(op->dims[d].stride), mutate(op->dims[d].fold_factor)};
      if (is_constant(new_dim.fold_factor, dim::unfolded)) new_dim.fold_factor = expr();
      changed = changed || !new_dim.same_as(op->dims[d]);
      dims.push_back(std::move(new_dim));
      bounds.push_back(std::move(new_bounds));
    }
    stmt body = mutate_with_bounds(op->body, op->sym, std::move(bounds));
    auto deps = depends_on(body, op->sym);
    if (!deps.any()) {
      // This make_buffer is unused.
      set_result(std::move(body));
      return;
    }

    if (const call* bc = base.as<call>()) {
      // Check if this make_buffer is equivalent to transpose, slice_buffer or crop_buffer
      if (bc->intrinsic == intrinsic::buffer_at) {
        const var* src_buf = as_variable(bc->args[0]);
        assert(src_buf);
        if (match(elem_size, buffer_elem_size(*src_buf))) {
          // To be a slice, we need every dimension that is present in the buffer_at call to be skipped, and the rest of
          // the dimensions to be identity.
          int dim = 0;
          std::size_t slice_rank = 0;
          std::size_t at_rank =
              std::count_if(bc->args.begin() + 1, bc->args.end(), [](const expr& i) { return i.defined(); });
          bool is_slice = true;
          for (int d = 0; d < static_cast<int>(dims.size() + at_rank); ++d) {
            if (d + 1 < static_cast<index_t>(bc->args.size()) && bc->args[d + 1].defined()) {
              // Skip this dimension.
              ++dim;
            } else if (slice_rank < dims.size()) {
              // This arg is undefined. We need to find the next dimension here to be a slice.
              is_slice = is_slice && match(dims[slice_rank++], buffer_dim(*src_buf, dim++));
            } else {
              is_slice = false;
              break;
            }
          }
          if (is_slice && slice_rank == dims.size()) {
            std::vector<expr> at(bc->args.begin() + 1, bc->args.end());
            stmt result = slice_buffer::make(op->sym, op->sym, std::move(at), std::move(body));
            // make_buffer drops trailing dims, do the same here.
            result = transpose::make_truncate(op->sym, *src_buf, dims.size() + at_rank, std::move(result));
            set_result(mutate(result));
            return;
          }

          // To be a crop, we need dimensions to either be identity, or the buffer_at argument is the same as the min.
          bool is_crop = bc->args.size() <= dims.size() + 1;
          box_expr crop_bounds(dims.size());
          for (index_t d = 0; d < static_cast<index_t>(dims.size()); ++d) {
            if (!match(dims[d].stride, buffer_stride(*src_buf, d)) ||
                !match(dims[d].fold_factor, buffer_fold_factor(*src_buf, d))) {
              is_crop = false;
              break;
            }

            // If the argument is defined, we need the min to be the same as the argument.
            // If it is not defined, it must be buffer_min(buf, d).
            bool has_at_d = d + 1 < static_cast<index_t>(bc->args.size()) && bc->args[d + 1].defined();
            expr crop_min = has_at_d ? bc->args[d + 1] : buffer_min(*src_buf, d);
            if (match(dims[d].bounds.min, crop_min)) {
              // We rewrite src -> sym in the truncate below.
              crop_bounds[d] = substitute(dims[d].bounds, *src_buf, op->sym);
            } else {
              is_crop = false;
              break;
            }
          }
          if (is_crop) {
            stmt result = crop_buffer::make(op->sym, op->sym, std::move(crop_bounds), std::move(body));
            // make_buffer drops trailing dims, do the same here.
            result = transpose::make_truncate(op->sym, *src_buf, dims.size(), std::move(result));
            set_result(mutate(result));
            return;
          }

          // To be a transpose, we need buffer_at to be the base of src_buf, and each dimension to be a dimension of the
          // original buffer.
          // TODO: This could probably be built into the slice check above.
          bool is_transpose = bc->args.size() == 1;
          std::vector<int> permutation;
          permutation.reserve(dims.size());
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
          for (std::size_t d = 0; d < dims.size(); ++d) {
            int min_dim = buffer_intrinsic_dim(intrinsic::buffer_min, dims[d].bounds.min);
            int max_dim = buffer_intrinsic_dim(intrinsic::buffer_max, dims[d].bounds.max);
            int stride_dim = buffer_intrinsic_dim(intrinsic::buffer_stride, dims[d].stride);
            int fold_factor_dim = buffer_intrinsic_dim(intrinsic::buffer_fold_factor, dims[d].fold_factor);
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
        stmts.push_back(mutate(make_buffer::make(op->sym, base, elem_size, dims, s)));
      }
      set_result(block::make(std::move(stmts)));
    } else if (changed || !base.same_as(op->base) || !elem_size.same_as(op->elem_size) || !body.same_as(op->body)) {
      set_result(make_buffer::make(op->sym, std::move(base), std::move(elem_size), std::move(dims), std::move(body)));
    } else {
      set_result(op);
    }
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

  static interval_expr simplify_crop_bounds(interval_expr i, var sym, int dim) {
    return {simplify_crop_bound(i.min, sym, dim), simplify_crop_bound(i.max, sym, dim)};
  }

  static bool crop_needed(const depends_on_result& deps) {
    // We don't need a crop if the buffer is only used as an input to a call. But we do need the crop if it is used as
    // an input to a copy, which uses the bounds of the input for padding.
    return deps.buffer_output || deps.buffer_src || deps.buffer_dst;
  }

  // Simplify bounds, assuming they will be clamped to outer_bounds.
  interval_expr simplify_redundant_bounds(interval_expr bounds, const interval_expr& outer_bounds) {
    if (bounds.min.defined() && outer_bounds.min.defined() && prove_true(bounds.min <= outer_bounds.min)) {
      bounds.min = expr();
    }
    if (bounds.max.defined() && outer_bounds.max.defined() && prove_true(bounds.max >= outer_bounds.max)) {
      bounds.max = expr();
    }
    return bounds;
  }

  void visit(const crop_buffer* op) override {
    scoped_trace trace("visit(const crop_buffer*)");
    // This is the bounds of the buffer as we understand them, for simplifying the inner scope.
    box_expr bounds(op->bounds.size());
    // This is the new bounds of the crop operation. Crops that are no-ops become undefined here.
    box_expr new_bounds(op->bounds.size());

    // If possible, rewrite crop_buffer of one dimension to crop_dim.
    const std::optional<box_expr>& prev_bounds = buffer_bounds[op->sym];
    index_t dims_count = 0;
    bool changed = false;
    for (index_t i = 0; i < static_cast<index_t>(op->bounds.size()); ++i) {
      interval_expr bounds_i = simplify_crop_bounds(mutate(op->bounds[i]), op->src, i);
      bounds_i = simplify_redundant_bounds(bounds_i, slinky::buffer_bounds(op->src, i));
      changed = changed || !bounds_i.same_as(op->bounds[i]);

      bounds[i] = bounds_i;

      // If the new bounds are the same as the existing bounds, set the crop in this dimension to
      // be undefined.
      if (prev_bounds && i < static_cast<index_t>(prev_bounds->size())) {
        bounds_i = simplify_redundant_bounds(bounds_i, (*prev_bounds)[i]);
      }
      if (bounds_i.min.defined()) new_bounds[i].min = bounds_i.min;
      if (bounds_i.max.defined()) new_bounds[i].max = bounds_i.max;
      dims_count += bounds_i.min.defined() || bounds_i.max.defined() ? 1 : 0;
    }
    stmt body = mutate_with_bounds(op->body, op->sym, std::move(bounds));
    auto deps = depends_on(body, op->sym);
    if (!deps.any()) {
      set_result(std::move(body));
      return;
    } else if (!crop_needed(deps)) {
      // Add clamps for the implicit bounds like crop would have done.
      for (index_t d = 0; d < static_cast<index_t>(new_bounds.size()); ++d) {
        new_bounds[d] &= slinky::buffer_bounds(op->src, d);
      }
      body = substitute_bounds(body, op->sym, new_bounds);
      body = substitute(body, op->sym, op->src);
      set_result(mutate(body));
      return;
    }

    // Remove trailing undefined bounds.
    while (!new_bounds.empty() && !new_bounds.back().min.defined() && !new_bounds.back().max.defined()) {
      new_bounds.pop_back();
    }
    if (new_bounds.empty()) {
      // This crop was a no-op.
      body = substitute(body, op->sym, op->src);
      set_result(std::move(body));
    } else if (const block* b = body.as<block>()) {
      std::vector<stmt> stmts;
      stmts.reserve(b->stmts.size());
      for (const stmt& s : b->stmts) {
        stmts.push_back(mutate(crop_buffer::make(op->sym, op->src, new_bounds, s)));
      }
      set_result(block::make(std::move(stmts)));
    } else if (dims_count == 1) {
      // This crop is of one dimension, replace it with crop_dim.
      // We removed undefined trailing bounds, so this must be the dim we want.
      int d = static_cast<int>(new_bounds.size()) - 1;
      set_result(mutate(crop_dim::make(op->sym, op->src, d, std::move(new_bounds[d]), std::move(body))));
    } else if (changed || !body.same_as(op->body)) {
      set_result(crop_buffer::make(op->sym, op->src, std::move(new_bounds), std::move(body)));
    } else {
      set_result(op);
    }
  }

  void visit(const crop_dim* op) override {
    scoped_trace trace("visit(const crop_dim*)");
    interval_expr bounds = simplify_crop_bounds(mutate(op->bounds), op->src, op->dim);
    bounds = simplify_redundant_bounds(bounds, slinky::buffer_bounds(op->src, op->dim));
    if (!bounds.min.defined() && !bounds.max.defined()) {
      // This crop is a no-op.
      stmt body = substitute(op->body, op->sym, op->src);
      set_result(mutate(body));
      return;
    }

    std::optional<box_expr> buf_bounds = buffer_bounds[op->sym];
    if (buf_bounds && op->dim < static_cast<index_t>(buf_bounds->size())) {
      interval_expr& buf_bounds_dim = (*buf_bounds)[op->dim];
      bounds = simplify_redundant_bounds(bounds, buf_bounds_dim);

      if (!bounds.min.defined() && !bounds.max.defined()) {
        // This crop is a no-op.
        stmt body = substitute(op->body, op->sym, op->src);
        set_result(mutate(body));
        return;
      }
      if (bounds.min.defined()) buf_bounds_dim.min = bounds.min;
      if (bounds.max.defined()) buf_bounds_dim.max = bounds.max;
    }

    stmt body = mutate_with_bounds(op->body, op->sym, std::move(buf_bounds));
    auto deps = depends_on(body, op->sym);
    if (!deps.any()) {
      set_result(std::move(body));
      return;
    } else if (!crop_needed(deps)) {
      body = substitute_bounds(body, op->sym, op->dim, bounds & slinky::buffer_bounds(op->src, op->dim));
      body = substitute(body, op->sym, op->src);
      set_result(mutate(body));
      return;
    }

    if (const slice_dim* slice = body.as<slice_dim>()) {
      if (slice->sym == op->sym && slice->dim == op->dim) {
        // This is a slice of the same dimension of the buffer we just cropped.
        // Don't drop the clamp that crop performs.
        expr at = clamp(slice->at, bounds);
        set_result(mutate(slice_dim::make(op->sym, op->src, op->dim, at, slice->body)));
        return;
      }
    } else if (const crop_dim* crop = body.as<crop_dim>()) {
      if (crop->sym == op->sym) {
        if (crop->dim == op->dim) {
          // Two nested crops of the same dimension, do one crop of the intersection instead.
          set_result(mutate(crop_dim::make(op->sym, op->src, op->dim, bounds & crop->bounds, crop->body)));
          return;
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
  static void update_sliced_buffer_metadata(symbol_map<box_expr>& bounds, var sym, span<const int> sliced) {
    for (std::optional<box_expr>& i : bounds) {
      if (!i) continue;
      for (interval_expr& j : *i) {
        j = slinky::update_sliced_buffer_metadata(j, sym, sliced);
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

    symbol_map<box_expr> old_buffer_bounds = buffer_bounds;
    bounds_map old_expr_bounds = expr_bounds;
    update_sliced_buffer_metadata(buffer_bounds, op->sym, sliced_dims);
    update_sliced_buffer_metadata(expr_bounds, op->sym, sliced_dims);
    stmt body = mutate(op->body);
    buffer_bounds = std::move(old_buffer_bounds);
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

    symbol_map<box_expr> old_buffer_bounds = buffer_bounds;
    bounds_map old_expr_bounds = expr_bounds;
    int sliced_dims[] = {op->dim};
    update_sliced_buffer_metadata(buffer_bounds, op->sym, sliced_dims);
    update_sliced_buffer_metadata(expr_bounds, op->sym, sliced_dims);
    stmt body = mutate(op->body);
    buffer_bounds = std::move(old_buffer_bounds);
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
    std::optional<box_expr> bounds = buffer_bounds[op->sym];
    if (bounds) {
      if (op->is_truncate() && bounds->size() <= op->dims.size()) {
        // transpose can't add dimensions.
        assert(bounds->size() == op->dims.size());
        // This truncate is a no-op.
        set_result(mutate(op->body));
        return;
      }
      box_expr new_bounds(op->dims.size());
      for (std::size_t i = 0; i < op->dims.size(); ++i) {
        new_bounds[i] = (*bounds)[op->dims[i]];
      }
      bounds = std::move(new_bounds);
    }

    stmt body = mutate_with_bounds(op->body, op->sym, std::move(bounds));

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
    stmt body = mutate_with_bounds(op->body, op->sym, buffer_bounds[op->src]);

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
    expr c = mutate(op->condition, &c_bounds);

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
  if (x.is_point()) {
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
