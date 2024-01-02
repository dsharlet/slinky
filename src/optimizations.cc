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

// Check if an output loop can be eliminated (no inputs other than the same dimension
// use it).
bool can_eliminate_output_loop(const std::vector<expr>& in, var out, std::size_t d) {
  for (std::size_t i = 0; i < in.size(); ++i) {
    if (depends_on(in[i], out.name()) && i != d) {
      // An input that isn't in[d] depends on out, we can't eliminate it.
      return false;
    }
  }
  return true;
}

bool is_copy(expr in, var out, interval_expr& bounds) { 
  if (match(in, out)) {
    bounds = interval_expr::all();
    return true;
  }

  symbol_map<expr> matches;
  var x(0), a(1), b(2);
  if (match(clamp(x, a, b), in, matches) && match(*matches[x], out)) {
    bounds = {*matches[a], *matches[b]};
    return true;
  }
  
  return false;
}

bool is_broadcast(expr in, var out) {
  interval_expr bounds = bounds_of(in, {{out.name(), interval_expr::all()}});
  bounds.min = simplify(bounds.min);
  bounds.max = simplify(bounds.max);

  // This is a broadcast if the bounds are a single point.
  return bounds.min.defined() && match(bounds.min, bounds.max);
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
    for (index_t od = 0; od < static_cast<index_t>(out_x.size()); ++od) {
      // TODO: If we decide we can assume that output::dims and input::bounds must use the same context as the rest of
      // the pipeline, we could use those variables here, which would make for more readable code (both here and in the
      // generated stmt).
      var loop_var(ctx.insert_unique());
      loop_vars[od] = loop_var;

      // Replace the variables with our new ones.
      for (expr& i : in_x) {
        i = substitute(i, out_x[od].name(), loop_var);
      }
      out_x[od] = loop_var;
    }

    std::vector<dim_expr> in_dims, out_dims;
    expr out_buf = variable::make(out_arg);
    expr in_buf = variable::make(in_arg);
    index_t id = 0;
    for (index_t od = 0; od < static_cast<index_t>(out_x.size()); ++od) {
      // We implement copies by first assuming that we are going to call copy for each point in the output buffer,
      // and creating a single pointed buffer for each of these calls.
      dim_expr in_dim = {point(out_x[od]), buffer_stride(in_buf, id), buffer_fold_factor(in_buf, id)};
      interval_expr out_bounds = point(out_x[od]);

      if (!can_eliminate_output_loop(in_x, out_x[od], od)) {
        // We can't eliminate this dimension of the copy because another dimension uses it (e.g. a transpose).
        ++id;
      } else if (is_copy(in_x[od], out_x[od], in_dim.bounds)) {
        // copy can handle this copy loop, eliminate it.
        in_x[od] = expr();
        ++id;
        out_x[od] = var();
        loop_vars[od] = expr();

        // If the copy was clamped, clamp the input buffer accordingly.
        if (is_negative_infinity(in_dim.bounds.min)) {
          in_dim.bounds.min = buffer_min(out_buf, od);
        }
        if (is_positive_infinity(in_dim.bounds.max)) {
          in_dim.bounds.max = buffer_max(out_buf, od);
        }
        out_bounds = buffer_bounds(out_buf, od);
      } else if (is_broadcast(in_x[od], out_x[od])) {
        // copy can handle this broadcast loop, eliminate it.
        ++id;
        out_x[od] = var();
        loop_vars[od] = expr();

        out_bounds = buffer_bounds(out_buf, od);
        in_dim.bounds = out_bounds;
        in_dim.stride = 0;
      } else {
        ++id;
      }

      if (!out_bounds.min.same_as(out_bounds.max)) {
        out_dims.emplace_back(out_bounds, buffer_stride(out_buf, od), buffer_fold_factor(out_buf, od));
        // We want the output coordinates here, we adjust the base below.
        in_dims.push_back(in_dim);
      }
    }

    // Make the new buffers.
    copy = make_buffer::make(out_arg, buffer_at(out_buf, out_x), buffer_elem_size(out_buf), out_dims, copy);
    copy = make_buffer::make(in_arg, buffer_at(in_buf, in_x), buffer_elem_size(in_buf), in_dims, copy);

    // Make the loops.
    for (index_t od = 0; od < static_cast<index_t>(out_x.size()); ++od) {
      if (loop_vars[od].defined()) {
        interval_expr bounds = {buffer_min(out_buf, od), buffer_max(out_buf, od)};
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
