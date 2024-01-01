#include "optimize_buffers.h"

#include <cassert>
#include <iostream>

#include "evaluate.h"
#include "node_mutator.h"
#include "pipeline.h"
#include "print.h"
#include "simplify.h"
#include "substitute.h"

namespace slinky {

namespace {

std::vector<expr> assert_points(std::span<const interval_expr> bounds) {
  std::vector<expr> result;
  result.reserve(bounds.size());
  for (const interval_expr& i : bounds) {
    if (!i.min.same_as(i.max)) {
      std::cerr << "Bounds must be a single point." << std::endl;
      std::abort();
    }
    result.push_back(i.min);
  }
  return result;
}

std::vector<expr> substitute(std::vector<expr> x, const symbol_map<expr>& replacements) {
  for (expr& i : x) {
    i = substitute(i, replacements);
  }
  return x;
}

class copy_implementer : public node_mutator {
  node_context& ctx;

  stmt implement_copy(
      const func* fn, std::vector<expr> bounds, std::vector<var> dims, symbol_id in_arg, symbol_id out_arg) {
    // We're always going to have a call to copy at the innermost loop.
    stmt copy = call_func::make(
        [](std::span<const index_t>, std::span<raw_buffer*> buffers) -> index_t {
          assert(buffers.size() == 2);
          const raw_buffer& in = *buffers[0];
          const raw_buffer& out = *buffers[1];
          slinky::copy(in, out);
          return 0;
        },
        {}, {in_arg, out_arg}, fn);

    // Make variables for the loops.
    std::vector<expr> loop_vars(dims.size());
    symbol_map<expr> dims_to_loop_vars;
    for (std::size_t od = 0; od < dims.size(); ++od) {
      int uses_count = 0;
      for (const expr& i : bounds) {
        if (depends_on(i, dims[od].name())) {
          uses_count++;
        }
      }

      // If this dimension is copied directly, and no other dimension's bounds depend on this dimension, we can skip
      // this loop and let the call to copy handle it.
      if (od < bounds.size() && uses_count == 1) {
        if (match(bounds[od], dims[od])) {
          bounds[od] = expr();
          continue;
        } else {
          // TODO: Try to match clamps and translations and call out to copy in those cases.
        }
      }

      // TODO: If we decide we can assume that output::dims and input::bounds must use the same context as the rest of
      // the pipeline, we could use those variables here, which would make for more readable code.
      loop_vars[od] = variable::make(ctx.insert_unique());
      dims_to_loop_vars[dims[od]] = loop_vars[od];
    }

    // Slice the buffers.
    copy = slice_buffer::make(in_arg, substitute(bounds, dims_to_loop_vars), copy);
    copy = slice_buffer::make(out_arg, loop_vars, copy);

    // Make the loops.
    for (int od = 0; od < static_cast<int>(dims.size()); ++od) {
      if (loop_vars[od].defined()) {
        interval_expr bounds = {buffer_min(var(out_arg), od), buffer_max(var(out_arg), od)};
        copy = loop::make(*as_variable(loop_vars[od]), bounds, copy);
      }
    }
    return copy;
  }

public:
  copy_implementer(node_context& ctx) : ctx(ctx) {}

  void visit(const call_func* c) override {
    if (c->target) {
      // This call is not a copy.
      set_result(c);
      return;
    }

    assert(c->fn->outputs().size() == 1);
    const func::output& output = c->fn->outputs().front();

    std::vector<var> dims = output.dims;

    // We're going to implement multiple input copies by simply copying the input to the output each time, assuming
    // the padding is not replaced.
    // TODO: We could be smarter about this, and not have this limitation.
    assert(c->fn->inputs().size() == 1 || c->fn->padding().empty());

    std::vector<stmt> results;
    results.reserve(c->fn->inputs().size());

    assert(c->buffer_args.size() == c->fn->inputs().size() + 1);
    auto arg_i = c->buffer_args.begin();
    symbol_id output_arg = c->buffer_args.back();

    assert(c->fn);
    for (const func::input& i : c->fn->inputs()) {
      results.push_back(implement_copy(c->fn, assert_points(i.bounds), output.dims, *arg_i++, output_arg));
    }
    set_result(block::make(results));
  }
};  // namespace

}  // namespace

stmt implement_copies(const stmt& s, node_context& ctx) { return copy_implementer(ctx).mutate(s); }

}  // namespace slinky
