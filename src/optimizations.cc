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

    expr out_buf = variable::make(out_arg);
    expr in_buf = variable::make(in_arg);

    // Start out by describing the copy as a complete loop nest of scalar copies, where we compute the address of
    // each scalar for every element. We might need to insert fake dimenions here if the copy is a broadcast.
    std::vector<expr> loop_vars(out_x.size());
    symbol_map<expr> out_x_to_loop_vars;
    std::vector<dim_expr> in_dims(out_x.size());
    std::vector<dim_expr> out_dims(out_x.size());
    std::vector<index_t> in_dims_map(out_x.size());
    index_t id = 0;
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

      // To start with, we describe copies as a loop over scalar copies.
      out_dims[od] = {point(out_x[od]), buffer_stride(out_buf, od), buffer_fold_factor(out_buf, od)};
      if (in_x.size() < out_x.size() &&
          (od >= static_cast<index_t>(in_x.size()) || !depends_on(in_x[od], out_x[od].name()))) {
        // We want to rewrite copies like so:
        //
        //   out(x, y, z) = in(x, z)
        //
        // by inserting a dummy dimension of stride 0 to represent the broadcast. It doesn't matter where we insert it
        // for correctness, but we want to insert it where we maximize the likliehood of being able to use `copy` to
        // implement the loop.
        in_dims[od] = {point(0), 0, 0};
        in_dims_map[od] = id;
        in_x.insert(in_x.begin() + od, 0);
      } else {
        in_dims[od] = {point(out_x[od]), buffer_stride(in_buf, id), buffer_fold_factor(in_buf, id)};
        in_dims_map[od] = id++;
      }
    }

    // After we've built the complete copy loop nest, find dimensions that the copy call can handle, and eliminate
    // those loops from the loop nest.
    assert(in_x.size() == out_x.size());
    assert(in_dims.size() == out_dims.size());
    for (index_t d = static_cast<index_t>(out_x.size()) - 1; d >= 0; --d) {
      dim_expr& in_dim = in_dims[d];
      dim_expr& out_dim = out_dims[d];
      interval_expr& in_bounds = in_dim.bounds;
      interval_expr& out_bounds = out_dim.bounds;

      interval_expr clamp_bounds;

      // We implement copies by first assuming that we are going to call copy for each point in the output buffer,
      // and creating a single pointed buffer for each of these calls.
      if (!can_eliminate_output_loop(in_x, out_x[d], d)) {
        // We can't eliminate this dimension of the copy because another dimension uses it (e.g. a transpose).
      } else if (is_copy(in_x[d], out_x[d], clamp_bounds)) {
        // copy can handle this copy loop, eliminate it.
        in_x[d] = expr();
        out_x[d] = var();
        loop_vars[d] = expr();

        // If the copy was clamped, use the intersection of the clamp and the original bounds.
        out_bounds = buffer_bounds(out_buf, d);
        in_bounds = buffer_bounds(in_buf, in_dims_map[d]) & clamp_bounds;
      } else if (is_broadcast(in_x[d], out_x[d])) {
        // copy can handle this broadcast loop, eliminate it.
        in_x[d] = expr();
        out_x[d] = var();
        loop_vars[d] = expr();

        out_bounds = buffer_bounds(out_buf, d);
        in_bounds = out_bounds;
        in_dim.stride = 0;
      }

      if (out_bounds.min.same_as(out_bounds.max)) {
        out_dims.erase(out_dims.begin() + d);
        in_dims.erase(in_dims.begin() + d);
      }
    }

    // Make the new buffers.
    copy = make_buffer::make(out_arg, buffer_at(out_buf, out_x), buffer_elem_size(out_buf), out_dims, copy);
    copy = make_buffer::make(in_arg, buffer_at(in_buf, in_x), buffer_elem_size(in_buf), in_dims, copy);

    // Make the loops.
    for (index_t d = 0; d < static_cast<index_t>(out_x.size()); ++d) {
      if (loop_vars[d].defined()) {
        interval_expr bounds = {buffer_min(out_buf, d), buffer_max(out_buf, d)};
        copy = loop::make(*as_variable(loop_vars[d]), bounds, copy);
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
