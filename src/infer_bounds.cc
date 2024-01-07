#include "infer_bounds.h"

#include <cassert>
#include <iostream>

#include "node_mutator.h"
#include "print.h"
#include "simplify.h"
#include "substitute.h"
#include "optimizations.h"

namespace slinky {

namespace {

// Get a reference to `n`th vector element of v, resizing the vector if necessary.
template <typename T>
T& vector_at(std::vector<T>& v, std::size_t n) {
  if (n >= v.size()) {
    v.resize(n + 1);
  }
  return v[n];
}
template <typename T>
T& vector_at(std::optional<std::vector<T>>& v, std::size_t n) {
  if (!v) {
    v = std::vector<T>(n + 1);
  }
  return vector_at(*v, n);
}

void merge_crop(std::optional<box_expr>& bounds, int dim, const interval_expr& new_bounds) {
  if (new_bounds.min.defined()) {
    vector_at(bounds, dim).min = new_bounds.min;
  }
  if (new_bounds.max.defined()) {
    vector_at(bounds, dim).max = new_bounds.max;
  }
}

void merge_crop(std::optional<box_expr>& bounds, const box_expr& new_bounds) {
  for (int d = 0; d < static_cast<int>(new_bounds.size()); ++d) {
    merge_crop(bounds, d, new_bounds[d]);
  }
}

class input_crop_remover : public node_mutator {
  symbol_map<bool> used_as_output;

public:
  void visit(const call_stmt* op) {
    for (symbol_id i : op->outputs) {
      used_as_output[i] = true;
    }
    set_result(op);
  }

  void visit(const crop_buffer* op) {
    auto s = set_value_in_scope(used_as_output, op->sym, false);
    stmt body = mutate(op->body);

    if (!*used_as_output[op->sym]) {
      set_result(body);
    } else if (body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(crop_buffer::make(op->sym, op->bounds, std::move(body)));
    }
  }

  void visit(const crop_dim* op) {
    auto s = set_value_in_scope(used_as_output, op->sym, false);
    stmt body = mutate(op->body);

    if (!*used_as_output[op->sym]) {
      set_result(body);
    } else if (body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(crop_dim::make(op->sym, op->dim, op->bounds, std::move(body)));
    }
  }
};

// This pass tries to identify where call_stmt operations need to run to satisfy the requirements of their consumers (or
// the output buffers). It updates `allocate` nodes to allocate enough memory for the uses of the allocation, and crops
// producers to the required region.
class bounds_inferrer : public node_mutator {
public:
  symbol_map<box_expr> infer;
  symbol_map<box_expr> crops;

  void visit(const allocate* alloc) override {
    auto set_bounds = set_value_in_scope(infer, alloc->sym, box_expr());
    stmt body = mutate(alloc->body);

    // When we constructed the pipeline, the buffer dimensions were set to buffer_* calls.
    // (This is a little janky because the buffers they are loading from don't exist where they are used.)
    // Here, we are building a list of replacements for those expressions. This way, if the user did something
    // like buf->dim(0).extent = buf->dim(0).extent + 10 (i.e. pad the extent by 10), we'll add 10 to our
    // inferred value.
    // TODO: Is this actually a good design...?
    std::vector<std::pair<expr, expr>> substitutions;

    expr alloc_var = variable::make(alloc->sym);

    box_expr& bounds = *infer[alloc->sym];
    expr stride = static_cast<index_t>(alloc->elem_size);
    for (index_t d = 0; d < static_cast<index_t>(bounds.size()); ++d) {
      const interval_expr& bounds_d = bounds[d];

      substitutions.emplace_back(buffer_min(alloc_var, d), bounds_d.min);
      substitutions.emplace_back(buffer_max(alloc_var, d), bounds_d.max);
      substitutions.emplace_back(buffer_stride(alloc_var, d), stride);

      // We didn't initially set up the buffer with a max, but the user might have used it.
      substitutions.emplace_back(buffer_extent(alloc_var, d), bounds_d.extent());
      stride *= bounds_d.extent();
    }

    // We need to keep replacing until nothing happens :(
    std::vector<dim_expr> dims(alloc->dims);
    while (true) {
      bool changed = false;
      for (index_t d = 0; d < static_cast<index_t>(dims.size()); ++d) {
        dim_expr& dim = dims[d];
        dim_expr new_dim = dim;
        for (auto& j : substitutions) {
          new_dim.bounds.min = substitute(new_dim.bounds.min, j.first, j.second);
          new_dim.bounds.max = substitute(new_dim.bounds.max, j.first, j.second);
          new_dim.stride = substitute(new_dim.stride, j.first, j.second);
        }
        if (!new_dim.same_as(dim)) {
          changed = true;
          dim = new_dim;
        }
      }
      if (!changed) break;
    }

    // Check that the actual bounds we generated are bigger than the inferred bounds (in case the
    // user set the bounds to something too small).
    std::vector<stmt> checks;
    for (std::size_t d = 0; d < dims.size(); ++d) {
      if (d < bounds.size()) {
        checks.push_back(check::make(dims[d].min() <= bounds[d].min));
        checks.push_back(check::make(dims[d].max() >= bounds[d].max));
      }
    }

    // Substitute the allocation bounds in any remaining inferred bounds.
    for (std::optional<box_expr>& i : infer) {
      if (!i) continue;
      for (interval_expr& j : *i) {
        for (const auto& k : substitutions) {
          j.min = substitute(j.min, k.first, k.second);
          j.max = substitute(j.max, k.first, k.second);
        }
      }
    }

    stmt s = allocate::make(alloc->storage, alloc->sym, alloc->elem_size, std::move(dims), body);
    set_result(block::make(block::make(checks), s));
  }

  void visit(const call_stmt* c) override {
    // Record the bounds we currently have from the crops.
    for (symbol_id input : c->inputs) {
      if (infer.contains(input)) {
        infer[input] = crops[input];
      }
    }
    set_result(c);
  }

  void visit(const copy_stmt* c) override {
    // Record the bounds we currently have from the crops.
    if (infer.contains(c->src)) {
      infer[c->src] = crops[c->src];
    }
    set_result(c);
  }

  void visit(const crop_buffer* c) override {
    std::optional<box_expr> crop = crops[c->sym];
    merge_crop(crop, c->bounds);
    auto set_crop = set_value_in_scope(crops, c->sym, crop);
    node_mutator::visit(c);
  }

  void visit(const crop_dim* c) override {
    std::optional<box_expr> crop = crops[c->sym];
    merge_crop(crop, c->dim, c->bounds);
    auto set_crop = set_value_in_scope(crops, c->sym, crop);
    node_mutator::visit(c);
  }

  // TODO: Need to handle this?
  void visit(const slice_buffer*) override { std::abort(); }
  void visit(const slice_dim*) override { std::abort(); }
  void visit(const truncate_rank*) override { std::abort(); }

  void visit(const loop* l) override {
    stmt body = mutate(l->body);

    stmt result;
    if (body.same_as(l->body)) {
      result = l;
    } else {
      // We rewrote the loop min.
      result = loop::make(l->sym, l->bounds, l->step, std::move(body));
    }

    // We're leaving the body of l. If any of the bounds used that loop variable, we need
    // to replace those uses with the bounds of the loop.
    for (symbol_id buf = 0; buf < infer.size(); ++buf) {
      std::optional<box_expr>& inferring = infer[buf];
      if (!inferring) continue;

      for (interval_expr& j : *inferring) {
        // We need to be careful of the case where min > max, such as when a pipeline
        // flips a dimension.
        // TODO: This seems janky/possibly not right.
        if (depends_on(j.min, l->sym)) {
          j.min = simplify(static_cast<const class min*>(nullptr), substitute(j.min, l->sym, l->bounds.min),
              substitute(j.min, l->sym, l->bounds.max));
        }
        if (depends_on(j.max, l->sym)) {
          j.max = simplify(static_cast<const class max*>(nullptr), substitute(j.max, l->sym, l->bounds.min),
              substitute(j.max, l->sym, l->bounds.max));
        }
      }
      result = crop_buffer::make(buf, *inferring, result);
    }
    set_result(result);
  }

  void visit(const block* op) override {
    // Visit blocks in reverse order. TODO: Is this really sufficient?
    stmt b = mutate(op->b);
    stmt a = mutate(op->a);
    if (a.same_as(op->a) && b.same_as(op->b)) {
      set_result(op);
    } else {
      set_result(block::make(a, b));
    }
  }
};

// Try to find cases where we can do "sliding window" or "line buffering" optimizations. When there
// is a producer that is consumed by a stencil operation in a loop, the producer can incrementally produce
// only the values required by the next iteration, and re-use the rest of the values from the previous iteration.
class slider : public node_mutator {
public:
  node_context& ctx;
  symbol_map<box_expr> buffer_bounds;
  symbol_map<std::pair<int, expr>> fold_factors;
  struct loop_info {
    symbol_id sym;
    expr orig_min;
    interval_expr bounds;
    expr step;
  };
  std::vector<loop_info> loops;

  // We need an unknown to make equations of.
  var x;

  slider(node_context& ctx) : ctx(ctx), x(ctx.insert_unique("_x")) {}

  void visit(const allocate* alloc) override {
    box_expr bounds;
    bounds.reserve(alloc->dims.size());
    for (const dim_expr& d : alloc->dims) {
      bounds.push_back(d.bounds);
    }
    auto set_buffer_bounds = set_value_in_scope(buffer_bounds, alloc->sym, bounds);
    stmt body = mutate(alloc->body);

    // When we constructed the pipeline, the buffer dimensions were set to buffer_* calls.
    // (This is a little janky because the buffers they are loading from don't exist where they are used.)
    // Here, we are building a list of replacements for those expressions. This way, if the user did something
    // like buf->dim(0).extent = buf->dim(0).extent + 10 (i.e. pad the extent by 10), we'll add 10 to our
    // inferred value.
    // TODO: Is this actually a good design...?
    std::vector<dim_expr> dims(alloc->dims);
    std::optional<std::pair<int, expr>> fold_info = fold_factors[alloc->sym];
    std::vector<std::pair<expr, expr>> replacements;
    for (index_t d = 0; d < static_cast<index_t>(alloc->dims.size()); ++d) {
      expr alloc_var = variable::make(alloc->sym);
      if (fold_info && fold_info->first == d) {
        replacements.emplace_back(buffer_fold_factor(alloc_var, d), fold_info->second);
      } else {
        replacements.emplace_back(buffer_fold_factor(alloc_var, d), -1);
      }
    }

    // We need to keep replacing until nothing happens :(
    while (true) {
      bool changed = false;
      for (index_t d = 0; d < static_cast<index_t>(dims.size()); ++d) {
        dim_expr& dim = dims[d];
        dim_expr new_dim = dim;
        for (auto& j : replacements) {
          new_dim.fold_factor = substitute(new_dim.fold_factor, j.first, j.second);
        }
        if (!new_dim.same_as(dim)) {
          changed = true;
          dim = new_dim;
        }
      }
      if (!changed) break;
    }

    set_result(allocate::make(alloc->storage, alloc->sym, alloc->elem_size, std::move(dims), body));
  }

  void visit(const call_stmt* c) override {
    stmt result = c;
    for (symbol_id output : c->outputs) {
      std::optional<box_expr>& bounds = buffer_bounds[output];
      if (!bounds) continue;

      for (size_t l = 0; l < loops.size(); ++l) {
        symbol_id loop_sym = loops[l].sym;
        expr loop_var = variable::make(loop_sym);
        const expr& loop_max = loops[l].bounds.max;

        for (int d = 0; d < static_cast<int>(bounds->size()); ++d) {
          interval_expr cur_bounds_d = (*bounds)[d];
          interval_expr prev_bounds_d = {
              substitute(cur_bounds_d.min, loop_sym, loop_var - loops[l].step),
              substitute(cur_bounds_d.max, loop_sym, loop_var - loops[l].step),
          };

          // A few things here struggle to simplify when there is a min(loop_max, x) expression involved, where x is
          // some expression that is bounded by the loop bounds. This min simplifies away if we know that x <= loop_max,
          // but the simplifier can't figure that out. As a hopefully temporary workaround, we can just substitute
          // infinity for the loop max.
          auto ignore_loop_max = [=](const expr& e) { return substitute(e, loop_max, positive_infinity()); };

          expr is_monotonic_increasing = prev_bounds_d.min <= cur_bounds_d.min && prev_bounds_d.max < cur_bounds_d.max;
          expr is_monotonic_decreasing = prev_bounds_d.min > cur_bounds_d.min && prev_bounds_d.max >= cur_bounds_d.max;
          is_monotonic_increasing = ignore_loop_max(is_monotonic_increasing);
          is_monotonic_decreasing = ignore_loop_max(is_monotonic_decreasing);

          if (prove_true(is_monotonic_increasing)) {
            // The bounds for each loop iteration are monotonically increasing,
            // so we can incrementally compute only the newly required bounds.
            expr old_min = cur_bounds_d.min;
            expr new_min = simplify(simplify(prev_bounds_d.max + 1));

            expr fold_factor = simplify(bounds_of(ignore_loop_max(cur_bounds_d.extent())).max);
            if (!depends_on(fold_factor, loop_sym)) {
              fold_factors[output] = {d, fold_factor};
            } else {
              // The fold factor didn't simplify to something that doesn't depend on the loop variable.
            }

            // Now that we're only computing the newly required parts of the domain, we need
            // to move the loop min back so we compute the whole required region. We'll insert
            // ifs around the other parts of the loop to avoid expanding the bounds that those
            // run on.
            expr new_min_at_new_loop_min = substitute(new_min, loop_sym, x);
            expr old_min_at_loop_min = substitute(old_min, loop_sym, loops[l].orig_min);
            expr new_loop_min =
                where_true(ignore_loop_max(new_min_at_new_loop_min <= old_min_at_loop_min), x.sym()).max;
            if (!is_negative_infinity(new_loop_min)) {
              loops[l].bounds.min = simplify(min(loops[l].bounds.min, new_loop_min));

              (*bounds)[d].min = new_min;
            } else {
              // We couldn't find the new loop min. We need to warm up the loop on the first iteration.
              // TODO: If another loop or func adjusts the loop min, we're going to run before the original min... that
              // seems like it might be fine anyways here, but pretty janky.
              (*bounds)[d].min = select(loop_var == loops[l].orig_min, old_min, new_min);
            }
            break;
          } else if (prove_true(is_monotonic_decreasing)) {
            // TODO: We could also try to slide when the bounds are monotonically
            // decreasing, but this is an unusual case.
          }
        }
      }
    }

    // Insert ifs around these calls, in case the loop min shifts later.
    for (const auto& l : loops) {
      if (is_positive_infinity(l.bounds.min)) continue;
      result = if_then_else::make(variable::make(l.sym) >= l.bounds.min, result, stmt());
    }
    set_result(result);
  }

  void visit(const crop_buffer* c) override {
    std::optional<box_expr> bounds = buffer_bounds[c->sym];
    merge_crop(bounds, c->bounds);
    auto set_bounds = set_value_in_scope(buffer_bounds, c->sym, bounds);
    stmt body = mutate(c->body);
    box_expr new_bounds = *buffer_bounds[c->sym];

    set_result(crop_buffer::make(c->sym, std::move(new_bounds), std::move(body)));
  }

  void visit(const crop_dim* c) override {
    std::optional<box_expr> bounds = buffer_bounds[c->sym];
    merge_crop(bounds, c->dim, c->bounds);
    auto set_bounds = set_value_in_scope(buffer_bounds, c->sym, bounds);
    stmt body = mutate(c->body);
    interval_expr new_bounds = (*buffer_bounds[c->sym])[c->dim];

    if (body.same_as(c->body) && new_bounds.same_as(c->bounds)) {
      set_result(c);
    } else {
      set_result(crop_dim::make(c->sym, c->dim, std::move(new_bounds), std::move(body)));
    }
  }

  // TODO: Need to handle this?
  void visit(const slice_buffer*) override { std::abort(); }
  void visit(const slice_dim*) override { std::abort(); }
  void visit(const truncate_rank*) override { std::abort(); }

  void visit(const loop* l) override {
    var orig_min(ctx, ctx.name(l->sym) + "_min.orig");

    loops.emplace_back(l->sym, orig_min, bounds(positive_infinity(), l->bounds.max), l->step);
    stmt body = mutate(l->body);
    expr loop_min = loops.back().bounds.min;
    loops.pop_back();

    if (is_positive_infinity(loop_min) && body.same_as(l->body)) {
      set_result(l);
    } else {
      // We rewrote the loop min.
      stmt result = loop::make(l->sym, {loop_min, l->bounds.max}, l->step, std::move(body));
      set_result(let_stmt::make(orig_min.sym(), l->bounds.min, result));
    }
  }

  void visit(const block* op) override {
    // Visit blocks in reverse order. TODO: Is this really sufficient?
    stmt b = mutate(op->b);
    stmt a = mutate(op->a);
    if (a.same_as(op->a) && b.same_as(op->b)) {
      set_result(op);
    } else {
      set_result(block::make(a, b));
    }
  }
};

stmt infer_bounds(const stmt& s, const std::vector<symbol_id>& inputs) {
  // Tell the bounds inferrer that we are inferring the bounds of the inputs too.
  bounds_inferrer infer;
  for (symbol_id i : inputs) {
    infer.infer[i] = box_expr();
  }
  stmt result = infer.mutate(s);

  // Now we should know the bounds required of the inputs. Add checks that the inputs are sufficient.
  std::vector<stmt> checks;
  for (symbol_id i : inputs) {
    expr buf_var = variable::make(i);
    const box_expr& bounds = *infer.infer[i];
    for (int d = 0; d < static_cast<int>(bounds.size()); ++d) {
      checks.push_back(check::make(buffer_min(buf_var, d) <= bounds[d].min));
      checks.push_back(check::make(buffer_max(buf_var, d) >= bounds[d].max));
      expr fold_factor = buffer_fold_factor(buf_var, d);
      checks.push_back(check::make(fold_factor <= 0 || bounds[d].extent() <= fold_factor));
    }
  }
  return block::make(block::make(checks), result);
}

}  // namespace

stmt infer_bounds(const stmt& s, node_context& ctx, const std::vector<symbol_id>& inputs) {
  stmt result = s;
  
  result = infer_bounds(s, inputs);
  result = simplify(result);

  // After simplifying and inferring the bounds of producers, we can try to run the sliding window
  // optimization.
  result = slider(ctx).mutate(result);

  result = alias_buffers(result);

  result = reduce_scopes(result);
  // At this point, crops of input buffers are unnecessary.
  // TODO: This is actually necessary for correctness in the case of folded buffers, but this shouldn't
  // be the case.
  result = input_crop_remover().mutate(result);

  return result;
}

}  // namespace slinky
