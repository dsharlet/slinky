#include "builder/simplify.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
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

    expr e = simplify(op, op->intrinsic, std::move(args));
    if (e.same_as(op)) {
      set_result(e, bounds_of(op, std::move(args_bounds)));
    } else {
      mutate_and_set_result(e);
    }
  }

  void visit(const let* op) override {
    std::vector<std::pair<symbol_id, expr>> lets;
    lets.reserve(op->lets.size());

    using sv_type = std::pair<scoped_value_in_symbol_map<interval_expr>, scoped_value_in_symbol_map<int>>;
    std::vector<sv_type> scoped_values;
    scoped_values.reserve(op->lets.size());

    bool values_changed = false;
    for (const auto& s : op->lets) {
      interval_expr value_bounds;
      lets.emplace_back(s.first, mutate(s.second, &value_bounds));
      values_changed = values_changed || !lets.back().second.same_as(s.second);

      auto vb = set_value_in_scope(expr_bounds, s.first, value_bounds);
      auto rc = set_value_in_scope(references, s.first, 0);
      scoped_values.emplace_back(std::move(vb), std::move(rc));
    }

    interval_expr body_bounds;
    expr body = mutate(op->body, &body_bounds);

    // - Prune dead lets
    // - Inline single-ref lets, along with lets that are just constants or vars
    for (auto it = lets.begin(); it != lets.end();) {
      int refs = *references[it->first];
      if (refs == 0) {
        it = lets.erase(it);
        values_changed = true;
      } else if (refs == 1 || it->second.as<constant>() || it->second.as<variable>()) {
        body = mutate(substitute(std::move(body), it->first, it->second), &body_bounds);
        it = lets.erase(it);
        values_changed = true;
      } else {
        it++;
      }
    }

    if (lets.empty()) {
      // All lets were removed.
      set_result(body, std::move(body_bounds));
    } else if (!values_changed && body.same_as(op->body)) {
      set_result(op, std::move(body_bounds));
    } else {
      set_result(let::make(std::move(lets), std::move(body)), std::move(body_bounds));
    }
  }

  void visit(const let_stmt* op) override {
    std::vector<std::pair<symbol_id, expr>> lets;
    lets.reserve(op->lets.size());

    using sv_type = std::pair<scoped_value_in_symbol_map<interval_expr>, scoped_value_in_symbol_map<int>>;
    std::vector<sv_type> scoped_values;
    scoped_values.reserve(op->lets.size());

    bool values_changed = false;
    for (const auto& s : op->lets) {
      interval_expr value_bounds;
      lets.emplace_back(s.first, mutate(s.second, &value_bounds));
      values_changed = values_changed || !lets.back().second.same_as(s.second);

      auto vb = set_value_in_scope(expr_bounds, s.first, value_bounds);
      auto rc = set_value_in_scope(references, s.first, 0);
      scoped_values.emplace_back(std::move(vb), std::move(rc));
    }

    stmt body = mutate(op->body);
    if (!body.defined()) {
      set_result(stmt());
      return;
    }

    // First, prune any dead lets, and any lets that are just vars
    for (auto it = lets.begin(); it != lets.end();) {
      int refs = *references[it->first];
      if (refs == 0 || is_variable(it->second, it->first)) {
        it = lets.erase(it);
        values_changed = true;
      } else {
        it++;
      }
    }

    if (lets.empty()) {
      // All lets were pruned.
      set_result(body);
    } else if (!values_changed && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(let_stmt::make(std::move(lets), std::move(body)));
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
            // If the crop negates the loop variable, the min could become the max. Just do both and take the union.
            interval_expr new_crop_a = {
                substitute(crop->bounds.min, op->sym, op->bounds.min),
                substitute(crop->bounds.max, op->sym, op->bounds.max),
            };
            interval_expr new_crop_b = {
                substitute(crop->bounds.min, op->sym, op->bounds.max),
                substitute(crop->bounds.max, op->sym, op->bounds.min),
            };
            new_crops.emplace_back(crop->sym, crop->dim, new_crop_a | new_crop_b);
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
        set_result(mutate(result));
        return;
      }
    }

    if (bounds.same_as(op->bounds) && step.same_as(op->step) && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(loop::make(op->sym, op->mode, std::move(bounds), std::move(step), std::move(body)));
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
    bool changed = false;
    for (std::size_t d = 0; d < op->dims.size(); ++d) {
      interval_expr bounds_d = mutate(op->dims[d].bounds);
      dim_expr new_dim = {bounds_d, mutate(op->dims[d].stride), mutate(op->dims[d].fold_factor)};
      if (prove_true(new_dim.fold_factor == 1 || new_dim.bounds.extent() == 1)) {
        new_dim.stride = 0;
      }
      changed = changed || !new_dim.same_as(op->dims[d]);
      dims.push_back(std::move(new_dim));
      bounds.push_back(std::move(bounds_d));
    }
    stmt body = substitute_bounds(op->body, op->sym, bounds);

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
    for (std::size_t d = 0; d < op->dims.size(); ++d) {
      interval_expr new_bounds = mutate(op->dims[d].bounds);
      dim_expr new_dim = {new_bounds, mutate(op->dims[d].stride), mutate(op->dims[d].fold_factor)};
      if (prove_true(new_dim.fold_factor == 1 || new_dim.bounds.extent() == 1)) {
        new_dim.stride = 0;
      }
      changed = changed || !new_dim.same_as(op->dims[d]);
      dims.push_back(std::move(new_dim));
      bounds.push_back(std::move(new_bounds));
    }
    stmt body = substitute_bounds(op->body, op->sym, bounds);

    auto set_bounds = set_value_in_scope(buffer_bounds, op->sym, std::move(bounds));
    body = mutate(body);
    if (!body.defined()) {
      set_result(stmt());
      return;
    }

    if (const call* bc = base.as<call>()) {
      if (bc->intrinsic == intrinsic::buffer_at && bc->args.size() == 1) {
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
                // This is a clone of src_buf.
                set_result(clone_buffer::make(op->sym, *src_buf, std::move(body)));
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

  static bool crop_needed(const depends_on_result& deps) {
    // We don't need a crop if the buffer is only used as an input to a call. But we do need the crop if it is used as
    // an input to a copy, which uses the bounds of the input for padding.
    return deps.buffer_output || deps.buffer_src || deps.buffer_dst;
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
    if (!crop_needed(depends_on(body, op->sym))) {
      body = substitute_bounds(body, op->sym, new_bounds);
      set_result(mutate(body));
      return;
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
      std::vector<stmt> stmts;
      stmts.reserve(b->stmts.size());
      for (const stmt& s : b->stmts) {
        stmts.push_back(mutate(crop_buffer::make(op->sym, new_bounds, s)));
      }
      set_result(block::make(std::move(stmts)));
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

    stmt body = op->body;
    if (!crop_needed(depends_on(body, op->sym))) {
      body = substitute_bounds(body, op->sym, op->dim, bounds);
      set_result(mutate(body));
      return;
    }
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
      std::vector<stmt> stmts;
      stmts.reserve(b->stmts.size());
      for (const stmt& s : b->stmts) {
        stmts.push_back(mutate(crop_dim::make(op->sym, op->dim, bounds, s)));
      }
      set_result(block::make(std::move(stmts)));
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
      std::vector<stmt> stmts;
      stmts.reserve(b->stmts.size());
      for (const stmt& s : b->stmts) {
        stmts.push_back(mutate(slice_buffer::make(op->sym, at, s)));
      }
      set_result(block::make(std::move(stmts)));
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
    }

    {
      auto set_bounds = set_value_in_scope(buffer_bounds, op->sym, bounds);
      body = mutate(body);
    }
    if (!body.defined()) {
      set_result(stmt());
    } else if (const block* b = body.as<block>()) {
      std::vector<stmt> stmts;
      stmts.reserve(b->stmts.size());
      for (const stmt& s : b->stmts) {
        stmts.push_back(mutate(slice_dim::make(op->sym, op->dim, at, s)));
      }
      set_result(block::make(std::move(stmts)));
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
      std::vector<stmt> stmts;
      stmts.reserve(b->stmts.size());
      for (const stmt& s : b->stmts) {
        stmts.push_back(mutate(truncate_rank::make(op->sym, op->rank, s)));
      }
      set_result(block::make(std::move(stmts)));
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

}  // namespace slinky
