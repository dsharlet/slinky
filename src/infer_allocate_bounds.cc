#include "infer_allocate_bounds.h"

#include <cassert>
#include <iostream>

#include "node_mutator.h"
#include "substitute.h"
#include "pipeline.h"

namespace slinky {

class allocate_bounds_inferrer : public node_mutator {
public:
  symbol_map<std::vector<dim_expr>>& buffers;
  symbol_map<box> inferring;

  allocate_bounds_inferrer(symbol_map<std::vector<dim_expr>>& buffers) : buffers(buffers) {}

  void visit(const allocate* alloc) override {
    assert(!inferring.contains(alloc->name));
    
    auto& bounds = inferring[alloc->name];
    assert(!bounds);
    bounds = box(alloc->dims.size());

    stmt body = mutate(alloc->body);

    assert(!!bounds);
    std::vector<dim_expr> dims;
    dims.reserve(bounds->size());
    expr stride_bytes = alloc->elem_size;
    for (const interval& i : *bounds) {
      expr extent = i.extent();
      dims.emplace_back(i.min, extent, stride_bytes, -1);
      stride_bytes *= extent;
    }
    s = allocate::make(alloc->type, alloc->name, alloc->elem_size, dims, body);
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
        expr arg = variable::make(*arg_i++);
        for (index_t d = 0; d < output.dims.size(); ++d) {
          symbol_id dim = output.dims[d].as<variable>()->name;
          mins[dim] = load_buffer_meta::make(arg, buffer_meta::min, d);
          maxs[dim] = load_buffer_meta::make(arg, buffer_meta::max, d);
        }
      }

      for (std::size_t d = 0; d < input.bounds.size(); ++d) {
        expr min = substitute(input.bounds[d].min, mins);
        expr max = substitute(input.bounds[d].max, maxs);
        interval required_d(slinky::min(min, max), slinky::max(min, max));
        if (bounds[d].min.defined() && bounds[d].max.defined()) {
          bounds[d] |= required_d;
        } else {
          bounds[d] = required_d;
        }
      }
    }
    node_mutator::visit(c);
  }
};

stmt infer_allocate_bounds(const stmt& s, symbol_map<std::vector<dim_expr>>& buffers) {
  return allocate_bounds_inferrer(buffers).mutate(s);
}

}  // namespace slinky
