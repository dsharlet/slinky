#include "optimizations.h"

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

bool depends_on(const expr& e, std::span<const var> vars) {
  for (const var& i : vars) {
    if (depends_on(e, i.name())) {
      return true;
    }
  }
  return false;
}

class copy_implementer : public node_mutator {
  node_context& ctx;

  stmt implement_copy(
      const func* fn, std::vector<expr> in_x, std::vector<var> out_x, symbol_id in_arg, symbol_id out_arg) {
    // We're always going to have a call to copy at the innermost loop.
    stmt copy = call_func::make(
        [padding = fn->padding()](std::span<const index_t>, std::span<raw_buffer*> buffers) -> index_t {
          assert(buffers.size() == 2);
          const raw_buffer& in = *buffers[0];
          const raw_buffer& out = *buffers[1];
          slinky::copy(in, out, padding.empty() ? nullptr : padding.data());
          return 0;
        },
        {}, {in_arg, out_arg}, fn);

    // Make variables for the loops.
    std::vector<expr> loop_vars(out_x.size());
    symbol_map<expr> out_x_to_loop_vars;
    for (std::size_t od = 0; od < out_x.size(); ++od) {
      int uses_count = 0;
      for (const expr& i : in_x) {
        if (depends_on(i, out_x[od].name())) {
          uses_count++;
        }
      }

      if (od < in_x.size() && uses_count <= 1) {
        // This input dimension is accessed only by this output dimension. We might be able to let copy handle it.
        interval_expr bounds_x = bounds_of(in_x[od], {{out_x[od].name(), interval_expr::all()}});
        bounds_x.min = simplify(bounds_x.min);
        bounds_x.max = simplify(bounds_x.max);

        if (bounds_x.min.defined() && match(bounds_x.min, bounds_x.max) && !depends_on(bounds_x.min, out_x)) {
          // This dimension is a broadcast.
          // TODO: copy can handle this, but we need to set the stride to 0 somehow.
          // in_x[od] = expr();
          // continue;
        }

        // Simplify the input x assuming it is in bounds.
        expr without_bounds = simplify(in_x[od], {{out_x[od].name(), bounds_x}});
        if (match(without_bounds, out_x[od])) {
          // This dimension is a simple copy.
          in_x[od] = expr();

          // But did we clamp it?
          if (!is_negative_infinity(bounds_x.min) || !is_positive_infinity(bounds_x.max)) {
            copy = crop_dim::make(in_arg, od, bounds_x, copy);
          }
          continue;
        }

        // The above simplification doesn't actually handle clamps yet.
        // TODO: When simplify can simplify away redundant clamps, this case should be removed.
        symbol_map<expr> matches;
        var x(0), a(1), b(2);
        if (match(clamp(x, a, b), in_x[od], matches) && match(*matches[x], out_x[od])) {
          // This dimension is a cropped copy.
          in_x[od] = expr();

          copy = crop_dim::make(in_arg, od, {*matches[a], *matches[b]}, copy);
          continue;
        }
      }

      // TODO: If we decide we can assume that output::dims and input::bounds must use the same context as the rest of
      // the pipeline, we could use those variables here, which would make for more readable code.
      loop_vars[od] = variable::make(ctx.insert_unique());
      out_x_to_loop_vars[out_x[od]] = loop_vars[od];
    }

    // Slice the buffers.
    copy = slice_buffer::make(in_arg, substitute(in_x, out_x_to_loop_vars), copy);
    copy = slice_buffer::make(out_arg, loop_vars, copy);

    // Make the loops.
    for (int od = 0; od < static_cast<int>(out_x.size()); ++od) {
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
