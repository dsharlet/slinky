#include "infer_allocate_bounds.h"

#include <cassert>
#include <iostream>

#include "node_mutator.h"
#include "substitute.h"
#include "pipeline.h"

namespace slinky {

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
        // TODO: Do we need to worry about the possibility of min > max here? 
        bounds[d] |= interval(min, max);
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
};

stmt infer_allocate_bounds(const stmt& s, node_context& ctx) {
  return allocate_bounds_inferrer(ctx).mutate(s);
}

}  // namespace slinky
