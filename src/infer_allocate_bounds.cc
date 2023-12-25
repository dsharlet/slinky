#include "infer_allocate_bounds.h"

#include <cassert>
#include <iostream>

#include "node_mutator.h"
#include "pipeline.h"
#include "substitute.h"
#include "print.h"

namespace slinky {

namespace {

class allocate_bounds_inferrer : public node_mutator {
public:
  node_context& ctx;
  symbol_map<box> inferring;
  symbol_map<box> crops;

  allocate_bounds_inferrer(node_context& ctx) : ctx(ctx) {}

  void visit(const allocate* alloc) override {
    assert(!inferring.contains(alloc->name));

    auto& bounds = inferring[alloc->name];
    assert(!bounds);
    bounds = box(alloc->dims.size(), interval::union_identity);

    stmt body = mutate(alloc->body);

    assert(!!bounds);
    std::vector<dim_expr> dims;
    dims.reserve(bounds->size());
    expr stride_bytes = static_cast<index_t>(alloc->elem_size);
    std::vector<std::pair<symbol_id, expr>> lets;
    for (const interval& i : *bounds) {
      symbol_id extent_name = ctx.insert();
      lets.emplace_back(extent_name, i.extent());
      expr extent = variable::make(extent_name);
      dims.emplace_back(i.min, extent, stride_bytes, -1);
      stride_bytes *= extent;
    }
    s = allocate::make(alloc->type, alloc->name, alloc->elem_size, dims, body);
    for (const auto& i : lets) {
      s = let_stmt::make(i.first, i.second, s);
    }
  }

  void visit(const call* c) override {
    assert(c->fn);
    for (const auto& input : c->fn->inputs()) {
      auto& maybe_bounds = inferring[input.buffer->name()];
      if (!maybe_bounds) continue;
      std::vector<interval>& bounds = *maybe_bounds;

      std::map<symbol_id, expr> mins, maxs;
      // TODO: We need a better way to map inputs/outputs between func and call.
      // Here, we are assuming that c->buffer_args is the inputs concatenated with the outputs,
      // in that order.
      auto arg_i = c->buffer_args.begin() + c->fn->inputs().size();
      for (const auto& output : c->fn->outputs()) {
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
        bounds[d] |= interval(min, max) | interval(max, min);
      }
    }
    node_mutator::visit(c);
  }

  void visit(const crop* c) override {
    // TODO: This is pretty messy, a better way to implement this would be nice.
    std::optional<box> cropped_bounds = crops[c->name];
    if (!cropped_bounds) {
      cropped_bounds = std::vector<interval>(c->dim + 1);
    } else if (c->dim >= cropped_bounds->size()) {
      cropped_bounds->resize(c->dim + 1);
    }
    (*cropped_bounds)[c->dim].min = c->min;
    (*cropped_bounds)[c->dim].max = c->min + c->extent - 1;

    scoped_value<box> new_crop(crops, c->name, *cropped_bounds);
    node_mutator::visit(c);
  }

  void visit(const loop* l) override {
    node_mutator::visit(l);

    // We're leaving the body of l. If any of the bounds used that loop variable, we need
    // to replace those uses with the bounds of the loop.
    std::map<symbol_id, expr> mins = {{l->name, l->begin}};
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
    scoped_value<std::size_t> set_loop_begin(loop_begin, op->name, loops.size());
    node_mutator::visit(op);
  }

  void visit(const loop* l) override { 
    loops.push_back(l->name);
    node_mutator::visit(l);
    loops.pop_back();
  }
};

}  // namespace

stmt infer_allocate_bounds(const stmt& s, node_context& ctx) { return allocate_bounds_inferrer(ctx).mutate(s); }

stmt sliding_window(const stmt& s, node_context& ctx) { return slider(ctx).mutate(s); }

}  // namespace slinky
