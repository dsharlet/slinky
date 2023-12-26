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
    assert(!inferring.contains(alloc->name));

    std::optional<box>& bounds = inferring[alloc->name];
    assert(!bounds);
    bounds = box(alloc->dims.size(), interval::union_identity);
    
    auto set_loops = set_value_in_scope(loops_since_allocate, alloc->name, loop_mins.size());
    stmt body = mutate(alloc->body);

    assert(!!bounds);
    std::vector<dim_expr> dims;
    dims.reserve(bounds->size());
    expr stride_bytes = static_cast<index_t>(alloc->elem_size);
    std::vector<std::pair<symbol_id, expr>> lets;
    for (const interval& i : *bounds) {
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

  void visit(const call* c) override {
    assert(c->fn);
    for (const func::input& input : c->fn->inputs()) {
      std::optional<box>& bounds = inferring[input.buffer->name()];
      if (!bounds) continue;

      std::map<symbol_id, expr> mins, maxs;
      // TODO: We need a better way to map inputs/outputs between func and call.
      // Here, we are assuming that c->buffer_args is the inputs concatenated with the outputs,
      // in that order.
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
            mins[dim] = load_buffer_meta::make(arg, buffer_meta::min, d);
            maxs[dim] = load_buffer_meta::make(arg, buffer_meta::max, d);
          }
        }
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

    s = c;
    for (const func::output& output : c->fn->outputs()) {
      std::optional<box>& bounds = inferring[output.buffer->name()];
      if (!bounds) continue;

      std::optional<std::size_t> first_loop = loops_since_allocate.lookup(output.buffer->name());

      bool slid = false;
      for (std::size_t l = first_loop ? *first_loop : 0; !slid && l < loop_mins.size(); ++l) {
        symbol_id loop_name = loop_mins[l].first;
        box prev_bounds(bounds->size());
        std::map<symbol_id, expr> prev_iter = {{loop_name, variable::make(loop_name) - 1}};
        for (int d = 0; d < static_cast<int>(bounds->size()); ++d) {
          prev_bounds[d].min = simplify(substitute((*bounds)[d].min, prev_iter));
          prev_bounds[d].max = simplify(substitute((*bounds)[d].max, prev_iter));
          if (can_prove(prev_bounds[d].min < (*bounds)[d].min) && can_prove(prev_bounds[d].max < (*bounds)[d].max)) {
            expr& old_min = (*bounds)[d].min;
            expr new_min = prev_bounds[d].max + 1;
            old_min = select::make(variable::make(loop_name) == loop_mins[l].second, old_min, new_min);
          }
        }
      }

      s = crop_buffer::make(output.buffer->name(), *bounds, s);
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
    std::map<symbol_id, expr> mins = {{l->name, loop_min}};
    std::map<symbol_id, expr> maxs = {{l->name, l->end - 1}};
    for (std::optional<box>& i : inferring) {
      if (!i) continue;

      for (interval& j : *i) {
        // We need to be careful of the case where min > max, such as when a pipeline
        // flips a dimension.
        // TODO: This seems janky/possibly not right.
        j.min = min(substitute(j.min, mins), substitute(j.min, maxs));
        j.max = max(substitute(j.max, mins), substitute(j.max, maxs));
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

stmt infer_bounds(const stmt& s, node_context& ctx) { return bounds_inferrer(ctx).mutate(s); }

stmt sliding_window(const stmt& s, node_context& ctx) { return slider(ctx).mutate(s); }

}  // namespace slinky
