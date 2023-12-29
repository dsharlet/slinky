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
  std::vector<std::pair<symbol_id, expr>> loop_mins;
  symbol_map<std::size_t> loops_since_allocate;

  bounds_inferrer(node_context& ctx) : ctx(ctx) {}

  void visit(const allocate* alloc) override {
    {
      std::optional<box_expr>& bounds = inferring[alloc->name];
      assert(!bounds);
      bounds = box_expr();
    }

    auto set_loops = set_value_in_scope(loops_since_allocate, alloc->name, loop_mins.size());
    stmt body = mutate(alloc->body);

    // When we constructed the pipeline, the buffer dimensions were set to load_buffer_meta expressions.
    // (This is a little janky because the buffers they are loading from don't exist where they are used.)
    // Here, we are building a list of replacements for those expressions. This way, if the user did something
    // like buf->dim(0).extent = buf->dim(0).extent + 10 (i.e. pad the extent by 10), we'll add 10 to our
    // inferred value.
    // TODO: Is this actually a good design...?
    std::vector<std::pair<expr, expr>> replacements;

    box_expr& inferred = *inferring[alloc->name];
    expr stride_bytes = static_cast<index_t>(alloc->elem_size);
    std::vector<std::pair<symbol_id, expr>> lets;
    auto& fold_factor = fold_factors[alloc->name];
    for (int d = 0; d < static_cast<int>(inferred.size()); ++d) {
      interval_expr& i = inferred[d];

      i.min = simplify(i.min);
      i.max = simplify(i.max);

      expr alloc_var = variable::make(alloc->name);
      replacements.emplace_back(buffer_min(alloc_var, d), i.min);
      replacements.emplace_back(buffer_max(alloc_var, d), i.max);
      replacements.emplace_back(buffer_stride_bytes(alloc_var, d), stride_bytes);
      if (fold_factor && fold_factor->first == d) {
        replacements.emplace_back(buffer_fold_factor(alloc_var, d), fold_factor->second);
      } else {
        replacements.emplace_back(buffer_fold_factor(alloc_var, d), -1);
      }

      // We didn't initially set up the buffer with a max, but the user might have used it.
      replacements.emplace_back(buffer_extent(alloc_var, d), i.extent());
      stride_bytes *= i.extent();
    }

    // We need to keep replacing until nothing happens :(
    std::vector<dim_expr> dims(alloc->dims);
    while (true) {
      bool changed = false;
      for (int d = 0; d < dims.size(); ++d) {
        dim_expr& dim = dims[d];
        dim_expr new_dim = dim;
        for (auto& j : replacements) {
          new_dim.bounds.min = substitute(new_dim.bounds.min, j.first, j.second);
          new_dim.bounds.max = substitute(new_dim.bounds.max, j.first, j.second);
          new_dim.stride_bytes = substitute(new_dim.stride_bytes, j.first, j.second);
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
    for (int d = 0; d < dims.size(); ++d) {
      checks.push_back(check::make(dims[d].min() <= inferred[d].min));
      checks.push_back(check::make(dims[d].max() >= inferred[d].max));
    }

    s = allocate::make(alloc->type, alloc->name, alloc->elem_size, std::move(dims), body);
    s = block::make(block::make(checks), s);
    for (const auto& i : lets) {
      s = let_stmt::make(i.first, i.second, s);
    }
  }

  expr get_buffer_meta(symbol_id buffer, buffer_meta meta, index_t d) {
    std::optional<box_expr>& bounds = inferring[buffer];
    if (bounds && d < bounds->size()) {
      switch (meta) {
      case buffer_meta::min: return (*bounds)[d].min;
      case buffer_meta::max: return (*bounds)[d].max;
      case buffer_meta::extent: return (*bounds)[d].extent();
      default: break;
      }
    }
    return load_buffer_meta::make(variable::make(buffer), meta, d);
  }

  void visit(const call_func* c) override {
    assert(c->fn);
    // Expand the bounds required of the inputs.
    for (const func::input& input : c->fn->inputs()) {
      std::map<symbol_id, expr> mins, maxs;
      // TODO(https://github.com/dsharlet/slinky/issues/7): We need a better way to map
      // inputs/outputs between func and call_func. Here, we are assuming that c->buffer_args
      // is the inputs concatenated with the outputs, in that order.
      auto arg_i = c->buffer_args.begin() + c->fn->inputs().size();
      for (const func::output& output : c->fn->outputs()) {
        const std::optional<box_expr>& crops_i = crops[*arg_i];
        expr arg = variable::make(*arg_i++);
        for (index_t d = 0; d < output.dims.size(); ++d) {
          symbol_id dim = *as_variable(output.dims[d]);
          if (crops_i && d < static_cast<int>(crops_i->size()) && (*crops_i)[d].min.defined() &&
              (*crops_i)[d].max.defined()) {
            mins[dim] = (*crops_i)[d].min;
            maxs[dim] = (*crops_i)[d].max;
          } else {
            mins[dim] = get_buffer_meta(*as_variable(arg), buffer_meta::min, d);
            maxs[dim] = get_buffer_meta(*as_variable(arg), buffer_meta::max, d);
          }
        }
      }

      std::optional<box_expr>& bounds = inferring[input.buffer->name()];
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
    s = c;
    for (const func::output& output : c->fn->outputs()) {
      std::optional<box_expr>& bounds = inferring[output.buffer->name()];
      if (!bounds) continue;

      // Maybe a hack? Keep the original bounds for inference purposes, but compute new bounds
      // (sliding window) for the crop.
      box_expr crop_bounds = *bounds;

      std::optional<std::size_t> first_loop = loops_since_allocate.lookup(output.buffer->name());

      for (std::size_t l = first_loop ? *first_loop : 0; l < loop_mins.size(); ++l) {
        symbol_id loop_name = loop_mins[l].first;
        box_expr prev_bounds(crop_bounds.size());
        std::map<symbol_id, expr> prev_iter = {{loop_name, variable::make(loop_name) - 1}};
        for (int d = 0; d < static_cast<int>(crop_bounds.size()); ++d) {
          prev_bounds[d].min = simplify(substitute(crop_bounds[d].min, prev_iter));
          prev_bounds[d].max = simplify(substitute(crop_bounds[d].max, prev_iter));
          if (can_prove(prev_bounds[d].min <= crop_bounds[d].min) &&
              can_prove(prev_bounds[d].max < crop_bounds[d].max)) {
            // The bounds for each loop iteration are monotonically increasing,
            // so we can incrementally compute only the newly required bounds.
            expr& old_min = crop_bounds[d].min;
            expr new_min = prev_bounds[d].max + 1;
            loop_mins[l].second -= simplify(new_min - old_min);

            expr fold_factor = simplify(bounds_of(crop_bounds[d].extent()).max);
            fold_factors[output.buffer->name()] = {d, fold_factor};

            old_min = new_min;
            break;
          } else if (can_prove(prev_bounds[d].min > crop_bounds[d].min) &&
                     can_prove(prev_bounds[d].max >= crop_bounds[d].max)) {
            // TODO: We could also try to slide when the bounds are monotonically
            // decreasing, but this is an unusual case.
          }
        }
      }

      s = crop_buffer::make(output.buffer->name(), crop_bounds, s);
    }

    // Insert ifs around these calls, in case the loop min shifts later.
    // TODO: If there was already a crop_dim here, this if goes inside it, which
    // modifies the buffer meta, which the condition (probably) depends on.
    // To fix this, we hackily move the if out below, but this is a serious hack
    // that needs to be fixed.
    for (const auto& l : loop_mins) {
      s = if_then_else::make(variable::make(l.first) >= l.second, s, stmt());
    }
  }

  void visit(const crop_buffer* c) override {
    auto new_crop = set_value_in_scope(crops, c->name, c->bounds);
    node_mutator::visit(c);
  }

  void visit(const crop_dim* c) override {
    std::optional<box_expr> cropped_bounds = crops[c->name];
    vector_at(cropped_bounds, c->dim) = min_extent(c->min, c->extent);

    auto new_crop = set_value_in_scope(crops, c->name, *cropped_bounds);
    node_mutator::visit(c);
    c = s.as<crop_dim>();
    if (const if_then_else* body = c->body.as<if_then_else>()) {
      // TODO: HORRIBLE HACK: crop_dim modifies the buffer meta, which this if we inserted
      // above assumes didn't happen. The if should be outside the crop anyways, it's just
      // not clear how to do that yet.
      // One fix for the issue mentioned below regarding ignoring ifs in loop bodies would
      // be to substitute a clamp on the loop variable for when the if is true. It should
      // simplify away later anyways, and make it easier to track bounds. This isn't easily
      // doable due to this hack.
      s = if_then_else::make(
          body->condition, crop_dim::make(c->name, c->dim, c->min, c->extent, body->true_body), stmt());
    }
  }

  void visit(const loop* l) override {
    loop_mins.emplace_back(l->name, l->bounds.min);
    stmt body = mutate(l->body);
    expr loop_min = loop_mins.back().second;
    loop_mins.pop_back();

    if (loop_min.same_as(l->bounds.min) && body.same_as(l->body)) {
      s = l;
    } else {
      // We rewrote the loop min.
      s = loop::make(l->name, {loop_min, l->bounds.max}, std::move(body));
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
        j.min = min(substitute(j.min, l->name, loop_min), substitute(j.min, l->name, loop_max));
        j.max = max(substitute(j.max, l->name, loop_min), substitute(j.max, l->name, loop_max));
      }
    }
  }

  void visit(const block* x) override {
    // Visit blocks in reverse order. TODO: Is this really sufficient?
    stmt b = mutate(x->b);
    stmt a = mutate(x->a);
    if (a.same_as(x->a) && b.same_as(x->b)) {
      s = x;
    } else {
      s = block::make(a, b);
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
