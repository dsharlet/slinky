#include "infer_bounds.h"

#include <cassert>
#include <iostream>

#include "node_mutator.h"
#include "pipeline.h"
#include "substitute.h"
#include "print.h"
#include "simplify.h"

namespace slinky {

namespace {

class bounds_inferrer : public node_mutator {
public:
  node_context& ctx;
  symbol_map<box> inferring;
  symbol_map<box> crops;
  std::vector<std::pair<symbol_id, expr>> loop_mins;
  symbol_map<std::size_t> loops_since_allocate;

  bounds_inferrer(node_context& ctx) : ctx(ctx) {}

  void visit(const allocate* alloc) override {
    {
      std::optional<box>& bounds = inferring[alloc->name];
      assert(!bounds);
      bounds = box();
    }
    
    auto set_loops = set_value_in_scope(loops_since_allocate, alloc->name, loop_mins.size());
    stmt body = mutate(alloc->body);

    const box& inferred = *inferring[alloc->name];
    std::vector<dim_expr> dims;
    dims.reserve(inferred.size());
    expr stride_bytes = static_cast<index_t>(alloc->elem_size);
    std::vector<std::pair<symbol_id, expr>> lets;
    for (const interval& i : inferred) {
      symbol_id extent_name = ctx.insert();
      lets.emplace_back(extent_name, simplify(i.extent()));
      expr extent = variable::make(extent_name);
      dims.emplace_back(simplify(i.min), extent, stride_bytes, -1);
      stride_bytes *= extent;
    }
    s = allocate::make(alloc->type, alloc->name, alloc->elem_size, dims, body);
    for (const auto& i : lets) {
      s = let_stmt::make(i.first, i.second, s);
    }
  }

  expr get_buffer_meta(symbol_id buffer, buffer_meta meta, index_t d) { 
    std::optional<box>& bounds = inferring[buffer];
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

  void visit(const call* c) override {
    assert(c->fn);
    // Expand the bounds required of the inputs.
    for (const func::input& input : c->fn->inputs()) {
      std::map<symbol_id, expr> mins, maxs;
      // TODO(https://github.com/dsharlet/slinky/issues/7): We need a better way to map
      // inputs/outputs between func and call. Here, we are assuming that c->buffer_args
      // is the inputs concatenated with the outputs, in that order.
      auto arg_i = c->buffer_args.begin() + c->fn->inputs().size();
      for (const func::output& output : c->fn->outputs()) {
        const std::optional<box>& cropped_bounds = crops[*arg_i];
        expr arg = variable::make(*arg_i++);
        for (index_t d = 0; d < output.dims.size(); ++d) {
          symbol_id dim = *as_variable(output.dims[d]);
          if (cropped_bounds && (*cropped_bounds)[d].min.defined() && (*cropped_bounds)[d].max.defined()) {
            mins[dim] = (*cropped_bounds)[d].min;
            maxs[dim] = (*cropped_bounds)[d].max;
          } else {
            mins[dim] = get_buffer_meta(*as_variable(arg), buffer_meta::min, d);
            maxs[dim] = get_buffer_meta(*as_variable(arg), buffer_meta::max, d);
          }
        }
      }

      std::optional<box>& bounds = inferring[input.buffer->name()];
      assert(bounds);
      bounds->reserve(input.bounds.size());
      while (bounds->size() < input.bounds.size()) { 
        bounds->push_back(interval::union_identity());
      }
      for (std::size_t d = 0; d < input.bounds.size(); ++d) {
        expr min = substitute(input.bounds[d].min, mins);
        expr max = substitute(input.bounds[d].max, maxs);
        // We need to be careful of the case where min > max, such as when a pipeline
        // flips a dimension.
        // TODO: This seems janky/possibly not right.
        (*bounds)[d] |= interval(min, max) | interval(max, min);
      }
    }

    // Add any crops necessary.
    s = c;
    for (const func::output& output : c->fn->outputs()) {
      std::optional<box>& bounds = inferring[output.buffer->name()];
      if (!bounds) continue;

      // Maybe a hack? Keep the original bounds for inference purposes, but compute new bounds
      // (sliding window) for the crop.
      box crop_bounds = *bounds;

      std::optional<std::size_t> first_loop = loops_since_allocate.lookup(output.buffer->name());

      for (std::size_t l = first_loop ? *first_loop : 0; l < loop_mins.size(); ++l) {
        symbol_id loop_name = loop_mins[l].first;
        box prev_bounds(crop_bounds.size());
        std::map<symbol_id, expr> prev_iter = {{loop_name, variable::make(loop_name) - 1}};
        for (int d = 0; d < static_cast<int>(crop_bounds.size()); ++d) {
          prev_bounds[d].min = simplify(substitute(crop_bounds[d].min, prev_iter));
          prev_bounds[d].max = simplify(substitute(crop_bounds[d].max, prev_iter));
          if (can_prove(prev_bounds[d].min < crop_bounds[d].min) && can_prove(prev_bounds[d].max < crop_bounds[d].max)) {
            expr& old_min = crop_bounds[d].min;
            expr new_min = prev_bounds[d].max + 1;
            loop_mins[l].second -= simplify(new_min - old_min);
            old_min = new_min;
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
    // TODO: This is pretty messy, a better way to implement this would be nice.
    std::optional<box> cropped_bounds = crops[c->name];
    if (!cropped_bounds) {
      cropped_bounds = box(c->dim + 1);
    } else if (c->dim >= cropped_bounds->size()) {
      cropped_bounds->resize(c->dim + 1);
    }
    (*cropped_bounds)[c->dim].min = c->min;
    (*cropped_bounds)[c->dim].max = c->min + c->extent - 1;

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
    loop_mins.emplace_back(l->name, l->begin);
    stmt body = mutate(l->body);
    expr loop_min = loop_mins.back().second;
    loop_mins.pop_back();

    if (loop_min.same_as(l->begin) && body.same_as(l->body)) {
      s = l;
    } else {
      // We rewrote the loop min.
      s = loop::make(l->name, loop_min, l->end, std::move(body));
    }

    // We're leaving the body of l. If any of the bounds used that loop variable, we need
    // to replace those uses with the bounds of the loop.
    // TODO: This ignores ifs inserted around parts of the body of this loop, which limit the
    // range of the loop. I was debugging a failure regarding this when I made an unrelated
    // change, and it magically started working. It *shouldn't* work, I expect this bug will
    // appear again. See the TODO: HORRIBLE HACK: above for more.
    // Use the original loop min. Hack?
    loop_min = l->begin;
    expr loop_max = l->end - 1;
    for (std::optional<box>& i : inferring) {
      if (!i) continue;

      for (interval& j : *i) {
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

// The idea behind this pass is:
// - Find allocations
// - Track the loops between allocations and producers
// - At the producer, for each loop:
//   - Compute the bounds produced by iteration i and i + 1
//   - Subtract the bounds
class slider : public node_mutator {
public:
  node_context& ctx;
  std::vector<symbol_id> loops;
  symbol_map<std::size_t> loop_begin;

  slider(node_context& ctx) : ctx(ctx) {}

  void visit(const call* c) override {
    assert(c->fn);



    node_mutator::visit(c);
  }

  void visit(const allocate* op) override { 
    auto set_loop_begin = set_value_in_scope(loop_begin, op->name, loops.size());
    node_mutator::visit(op);
  }

  void visit(const loop* l) override { 
    loops.push_back(l->name);
    node_mutator::visit(l);
    loops.pop_back();
  }
};

}  // namespace

stmt infer_bounds(const stmt& s, node_context& ctx, const std::vector<symbol_id>& inputs) {
  bounds_inferrer b(ctx);

  // Tell the bounds inferrer that we are inferring the bounds of the inputs too.
  for (symbol_id i : inputs) {
    b.inferring[i] = box();
  }

  // Run it.
  stmt result = b.mutate(s);

  // Now we should know the bounds required of the inputs. Add checks that the inputs are sufficient.
  std::vector<stmt> checks;
  for (symbol_id i : inputs) {
    expr buf_var = variable::make(i);
    const box& bounds = *b.inferring[i];
    index_t rank = static_cast<index_t>(bounds.size());
    checks.push_back(check::make(buf_var != 0));
    checks.push_back(check::make(load_buffer_meta::make(buf_var, buffer_meta::rank) == rank));
    checks.push_back(check::make(load_buffer_meta::make(buf_var, buffer_meta::base) != 0));
    for (int d = 0; d < rank; ++d) {
      expr min = load_buffer_meta::make(buf_var, buffer_meta::min, d);
      expr max = load_buffer_meta::make(buf_var, buffer_meta::max, d);
      checks.push_back(check::make(min <= bounds[d].min));
      checks.push_back(check::make(max >= bounds[d].max));
    }
  }
  return block::make(block::make(checks), result);
}

stmt sliding_window(const stmt& s, node_context& ctx) { return slider(ctx).mutate(s); }

}  // namespace slinky
