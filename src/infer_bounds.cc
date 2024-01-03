#include "infer_bounds.h"

#include <cassert>
#include <iostream>

#include "node_mutator.h"
#include "pipeline.h"
#include "print.h"
#include "simplify.h"
#include "substitute.h"

namespace slinky {

namespace {

// Get a reference to `n`th vector element of v, resizing the vector if necessary.
template <typename T>
T& vector_at(std::optional<std::vector<T>>& v, std::size_t n) {
  if (!v) {
    v = std::vector<T>(n + 1);
  } else if (n >= v->size()) {
    v->resize(n + 1);
  }
  return v->at(n);
}

class bounds_inferrer : public node_mutator {
public:
  node_context& ctx;
  symbol_map<box_expr> inferring;
  symbol_map<std::pair<int, expr>> fold_factors;
  symbol_map<box_expr> crops;
  std::vector<std::pair<symbol_id, interval_expr>> loop_bounds;
  symbol_map<std::size_t> loops_since_allocate;

  bounds_inferrer(node_context& ctx) : ctx(ctx) {}

  void visit(const allocate* alloc) override {
    {
      std::optional<box_expr>& bounds = inferring[alloc->sym];
      assert(!bounds);
      bounds = box_expr();
    }

    auto set_loops = set_value_in_scope(loops_since_allocate, alloc->sym, loop_bounds.size());
    stmt body = mutate(alloc->body);

    // When we constructed the pipeline, the buffer dimensions were set to buffer_* calls.
    // (This is a little janky because the buffers they are loading from don't exist where they are used.)
    // Here, we are building a list of replacements for those expressions. This way, if the user did something
    // like buf->dim(0).extent = buf->dim(0).extent + 10 (i.e. pad the extent by 10), we'll add 10 to our
    // inferred value.
    // TODO: Is this actually a good design...?
    std::vector<std::pair<expr, expr>> replacements;

    box_expr& inferred = *inferring[alloc->sym];
    expr stride = static_cast<index_t>(alloc->elem_size);
    std::vector<std::pair<symbol_id, expr>> lets;
    auto& fold_factor = fold_factors[alloc->sym];
    for (index_t d = 0; d < static_cast<index_t>(inferred.size()); ++d) {
      interval_expr& i = inferred[d];

      i.min = simplify(i.min);
      i.max = simplify(i.max);

      expr alloc_var = variable::make(alloc->sym);
      replacements.emplace_back(buffer_min(alloc_var, d), i.min);
      replacements.emplace_back(buffer_max(alloc_var, d), i.max);
      replacements.emplace_back(buffer_stride(alloc_var, d), stride);
      if (fold_factor && fold_factor->first == d) {
        replacements.emplace_back(buffer_fold_factor(alloc_var, d), fold_factor->second);
      } else {
        replacements.emplace_back(buffer_fold_factor(alloc_var, d), -1);
      }

      // We didn't initially set up the buffer with a max, but the user might have used it.
      replacements.emplace_back(buffer_extent(alloc_var, d), i.extent());
      stride *= i.extent();
    }

    // We need to keep replacing until nothing happens :(
    std::vector<dim_expr> dims(alloc->dims);
    while (true) {
      bool changed = false;
      for (index_t d = 0; d < static_cast<index_t>(dims.size()); ++d) {
        dim_expr& dim = dims[d];
        dim_expr new_dim = dim;
        for (auto& j : replacements) {
          new_dim.bounds.min = substitute(new_dim.bounds.min, j.first, j.second);
          new_dim.bounds.max = substitute(new_dim.bounds.max, j.first, j.second);
          new_dim.stride = substitute(new_dim.stride, j.first, j.second);
          new_dim.fold_factor = substitute(new_dim.fold_factor, j.first, j.second);
        }
        if (!new_dim.same_as(dim)) {
          changed = true;
          dim = new_dim;
        }
      }
      if (!changed) break;
    }

    // Check that the actual bounds we generated are bigger than the inferred bounds.
    std::vector<stmt> checks;
    for (index_t d = 0; d < static_cast<index_t>(dims.size()); ++d) {
      checks.push_back(check::make(dims[d].min() <= inferred[d].min));
      checks.push_back(check::make(dims[d].max() >= inferred[d].max));
    }

    stmt s = allocate::make(alloc->storage, alloc->sym, alloc->elem_size, std::move(dims), body);
    s = block::make(block::make(checks), s);
    for (const auto& i : lets) {
      s = let_stmt::make(i.first, i.second, s);
    }
    set_result(s);
  }

  expr buffer_intrinsic(symbol_id buffer, intrinsic fn, index_t d) {
    std::optional<box_expr>& bounds = inferring[buffer];
    if (bounds && d < static_cast<index_t>(bounds->size())) {
      switch (fn) {
      case intrinsic::buffer_min: return (*bounds)[d].min;
      case intrinsic::buffer_max: return (*bounds)[d].max;
      case intrinsic::buffer_extent: return (*bounds)[d].extent();
      default: break;
      }
    }
    return call::make(fn, {variable::make(buffer), d});
  }

  void visit(const call_func* c) override {
    assert(c->fn);
    // Expand the bounds required of the inputs.
    for (const func::input& input : c->fn->inputs()) {
      symbol_map<expr> mins, maxs;
      for (const func::output& output : c->fn->outputs()) {
        var arg(output.sym());
        const std::optional<box_expr>& crops_i = crops[arg];
        for (index_t d = 0; d < static_cast<index_t>(output.dims.size()); ++d) {
          symbol_id dim = output.dims[d].sym();
          if (crops_i && d < static_cast<index_t>(crops_i->size()) && (*crops_i)[d].min.defined() &&
              (*crops_i)[d].max.defined()) {
            mins[dim] = (*crops_i)[d].min;
            maxs[dim] = (*crops_i)[d].max;
          } else {
            mins[dim] = buffer_intrinsic(arg.sym(), intrinsic::buffer_min, d);
            maxs[dim] = buffer_intrinsic(arg.sym(), intrinsic::buffer_max, d);
          }
        }
      }

      std::optional<box_expr>& bounds = inferring[input.buffer->sym()];
      assert(bounds);
      bounds->reserve(input.bounds.size());
      while (bounds->size() < input.bounds.size()) {
        bounds->push_back(interval_expr::union_identity());
      }
      for (std::size_t d = 0; d < input.bounds.size(); ++d) {
        expr min = substitute(input.bounds[d].min, mins);
        expr max = substitute(input.bounds[d].max, maxs);
        // We need to be careful of the case where min > max, such as when a pipeline
        // flips a dimension.
        // TODO: This seems janky/possibly not right.
        (*bounds)[d] |= slinky::bounds(min, max) | slinky::bounds(max, min);
      }
    }

    // Add any crops necessary.
    stmt result = c;
    for (const func::output& output : c->fn->outputs()) {
      std::optional<box_expr>& bounds = inferring[output.buffer->sym()];
      if (!bounds) continue;

      // Maybe a hack? Keep the original bounds for inference purposes, but compute new bounds
      // (sliding window) for the crop.
      box_expr crop_bounds = *bounds;

      std::optional<std::size_t> first_loop = loops_since_allocate.lookup(output.buffer->sym());

      for (std::size_t l = first_loop ? *first_loop : 0; l < loop_bounds.size(); ++l) {
        symbol_id loop_sym = loop_bounds[l].first;
        expr loop_var = variable::make(loop_sym);
        expr loop_min = loop_bounds[l].second.min;
        expr loop_max = loop_bounds[l].second.max;

        box_expr prev_bounds(crop_bounds.size());
        for (int d = 0; d < static_cast<int>(crop_bounds.size()); ++d) {
          prev_bounds[d].min = substitute(crop_bounds[d].min, loop_sym, loop_var - 1);
          prev_bounds[d].max = substitute(crop_bounds[d].max, loop_sym, loop_var - 1);
          if (prove_true(prev_bounds[d].min <= crop_bounds[d].min) &&
              prove_true(prev_bounds[d].max < crop_bounds[d].max)) {
            // The bounds for each loop iteration are monotonically increasing,
            // so we can incrementally compute only the newly required bounds.
            expr& old_min = crop_bounds[d].min;
            expr new_min = prev_bounds[d].max + 1;

            expr fold_factor = simplify(bounds_of(crop_bounds[d].extent()).max);
            fold_factors[output.buffer->sym()] = {d, fold_factor};

            // Now that we're only computing the newly required parts of the domain, we need
            // to move the loop min back so we compute the whole required region. We'll insert
            // ifs around the other parts of the loop to avoid expanding the bounds that those
            // run on.
            symbol_id new_loop_min_sym = ctx.insert_unique();
            expr new_loop_min_var = variable::make(new_loop_min_sym);
            expr new_min_at_new_loop_min = substitute(new_min, loop_sym, new_loop_min_var);
            expr old_min_at_loop_min = substitute(old_min, loop_sym, loop_min);
            expr new_loop_min = where_true(new_min_at_new_loop_min <= old_min_at_loop_min, new_loop_min_sym).max;
            if (new_loop_min.defined()) {
              loop_bounds[l].second.min = simplify(loop_min - (new_min - old_min));

              old_min = new_min;
            } else {
              // We couldn't find the new loop min. We need to warm up the loop on the first iteration.
              old_min = select(loop_var == loop_min, old_min, new_min);
            }
            break;
          } else if (prove_true(prev_bounds[d].min > crop_bounds[d].min) &&
                     prove_true(prev_bounds[d].max >= crop_bounds[d].max)) {
            // TODO: We could also try to slide when the bounds are monotonically
            // decreasing, but this is an unusual case.
          }
        }
      }

      result = crop_buffer::make(output.buffer->sym(), crop_bounds, result);
    }

    // Insert ifs around these calls, in case the loop min shifts later.
    // TODO: If there was already a crop_dim here, this if goes inside it, which
    // modifies the buffer meta, which the condition (probably) depends on.
    // To fix this, we hackily move the if out below, but this is a serious hack
    // that needs to be fixed.
    for (const auto& l : loop_bounds) {
      result = if_then_else::make(variable::make(l.first) >= l.second.min, result, stmt());
    }
    set_result(result);
  }

  void visit(const crop_buffer* c) override {
    auto new_crop = set_value_in_scope(crops, c->sym, c->bounds);
    node_mutator::visit(c);
  }

  void visit(const crop_dim* c) override {
    std::optional<box_expr> cropped_bounds = crops[c->sym];
    vector_at(cropped_bounds, c->dim) = c->bounds;

    auto new_crop = set_value_in_scope(crops, c->sym, *cropped_bounds);
    stmt body = mutate(c->body);
    if (const if_then_else* body_if = body.as<if_then_else>()) {
      // TODO: HORRIBLE HACK: crop_dim modifies the buffer meta, which this if we inserted
      // above assumes didn't happen. The if should be outside the crop anyways, it's just
      // not clear how to do that yet.
      // One fix for the issue mentioned below regarding ignoring ifs in loop bodies would
      // be to substitute a clamp on the loop variable for when the if is true. It should
      // simplify away later anyways, and make it easier to track bounds. This isn't easily
      // doable due to this hack.
      set_result(if_then_else::make(
          body_if->condition, crop_dim::make(c->sym, c->dim, c->bounds, body_if->true_body), stmt()));
    } else if (body.same_as(c->body)) {
      set_result(c);
    } else {
      set_result(crop_dim::make(c->sym, c->dim, c->bounds, std::move(body)));
    }
  }

  void visit(const loop* l) override {
    loop_bounds.emplace_back(l->sym, l->bounds);
    stmt body = mutate(l->body);
    expr loop_min = loop_bounds.back().second.min;
    // The loop max should not be changed.
    assert(l->bounds.max.same_as(loop_bounds.back().second.max));
    loop_bounds.pop_back();

    stmt result;
    if (loop_min.same_as(l->bounds.min) && body.same_as(l->body)) {
      result = l;
    } else {
      // We rewrote the loop min.
      result = loop::make(l->sym, {loop_min, l->bounds.max}, l->step, std::move(body));
    }

    // We're leaving the body of l. If any of the bounds used that loop variable, we need
    // to replace those uses with the bounds of the loop.
    // TODO: This ignores ifs inserted around parts of the body of this loop, which limit the
    // range of the loop. I was debugging a failure regarding this when I made an unrelated
    // change, and it magically started working. It *shouldn't* work, I expect this bug will
    // appear again. See the TODO: HORRIBLE HACK: above for more.
    // Use the original loop min. Hack?
    loop_min = l->bounds.min;
    expr loop_max = l->bounds.max;
    for (std::optional<box_expr>& i : inferring) {
      if (!i) continue;

      for (interval_expr& j : *i) {
        // We need to be careful of the case where min > max, such as when a pipeline
        // flips a dimension.
        // TODO: This seems janky/possibly not right.
        j.min = min(substitute(j.min, l->sym, loop_min), substitute(j.min, l->sym, loop_max));
        j.max = max(substitute(j.max, l->sym, loop_min), substitute(j.max, l->sym, loop_max));
      }
    }
    set_result(result);
  }

  void visit(const block* x) override {
    // Visit blocks in reverse order. TODO: Is this really sufficient?
    stmt b = mutate(x->b);
    stmt a = mutate(x->a);
    if (a.same_as(x->a) && b.same_as(x->b)) {
      set_result(x);
    } else {
      set_result(block::make(a, b));
    }
  }
};

}  // namespace

stmt infer_bounds(const stmt& s, node_context& ctx, const std::vector<symbol_id>& inputs) {
  bounds_inferrer b(ctx);

  // Tell the bounds inferrer that we are inferring the bounds of the inputs too.
  for (symbol_id i : inputs) {
    b.inferring[i] = box_expr();
  }

  // Run it.
  stmt result = b.mutate(s);

  // Now we should know the bounds required of the inputs. Add checks that the inputs are sufficient.
  std::vector<stmt> checks;
  for (symbol_id i : inputs) {
    expr buf_var = variable::make(i);
    const box_expr& bounds = *b.inferring[i];
    for (int d = 0; d < static_cast<int>(bounds.size()); ++d) {
      checks.push_back(check::make(buffer_min(buf_var, d) <= bounds[d].min));
      checks.push_back(check::make(buffer_max(buf_var, d) >= bounds[d].max));
    }
  }
  return block::make(block::make(checks), result);
}

}  // namespace slinky
