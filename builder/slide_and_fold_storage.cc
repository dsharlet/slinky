#include "builder/slide_and_fold_storage.h"

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>

#include "builder/node_mutator.h"
#include "builder/simplify.h"
#include "builder/substitute.h"
#include "runtime/depends_on.h"
#include "runtime/expr.h"
#include "runtime/util.h"

namespace slinky {

namespace {

// Get a reference to `n`th vector element of v, resizing the vector if necessary.
template <typename T>
T& vector_at(std::vector<T>& v, std::size_t n) {
  if (n >= v.size()) {
    v.resize(n + 1);
  }
  return v[n];
}
template <typename T>
T& vector_at(std::optional<std::vector<T>>& v, std::size_t n) {
  if (!v) {
    v = std::vector<T>(n + 1);
  }
  return vector_at(*v, n);
}

void merge_crop(std::optional<box_expr>& bounds, int d, const interval_expr& new_bounds) {
  // Crops produce the intersection of the old bounds and the new bounds.
  // TODO: This is equivalent to vector_at(bounds, d) &= new_bounds, except for simplification, which makes
  // a huge difference in the cost of this.
  interval_expr& bounds_d = vector_at(bounds, d);
  if (bounds_d.min.defined() && new_bounds.min.defined()) {
    bounds_d.min = simplify(static_cast<const class max*>(nullptr), bounds_d.min, new_bounds.min);
  } else if (new_bounds.min.defined()) {
    bounds_d.min = new_bounds.min;
  }
  if (bounds_d.max.defined() && new_bounds.max.defined()) {
    bounds_d.max = simplify(static_cast<const class min*>(nullptr), bounds_d.max, new_bounds.max);
  } else if (new_bounds.max.defined()) {
    bounds_d.max = new_bounds.max;
  }
}

void merge_crop(std::optional<box_expr>& bounds, const box_expr& new_bounds) {
  if (!bounds) {
    bounds = box_expr();
  }
  for (int d = 0; d < static_cast<int>(new_bounds.size()); ++d) {
    merge_crop(bounds, d, new_bounds[d]);
  }
}

// Keep substituting substitutions until nothing happens.
std::vector<dim_expr> recursive_substitute(
    std::vector<dim_expr> dims, span<const std::pair<expr, expr>> substitutions) {
  while (true) {
    bool changed = false;
    for (dim_expr& dim : dims) {
      dim_expr new_dim = dim;
      for (const std::pair<expr, expr>& j : substitutions) {
        new_dim.bounds.min = substitute(new_dim.bounds.min, j.first, j.second);
        new_dim.bounds.max = substitute(new_dim.bounds.max, j.first, j.second);
        new_dim.stride = substitute(new_dim.stride, j.first, j.second);
        new_dim.fold_factor = substitute(new_dim.fold_factor, j.first, j.second);
      }
      if (!new_dim.same_as(dim)) {
        changed = true;
        dim = new_dim;
      }
    }
    if (!changed) return dims;
  }
}

void substitute_bounds(box_expr& bounds, const symbol_map<box_expr>& buffers) {
  for (symbol_id i = 0; i < buffers.size(); ++i) {
    if (!buffers[i]) continue;
    for (interval_expr& j : bounds) {
      if (j.min.defined()) j.min = substitute_bounds(j.min, i, *buffers[i]);
      if (j.max.defined()) j.max = substitute_bounds(j.max, i, *buffers[i]);
    }
  }
}

// Try to find cases where we can do "sliding window" or "line buffering" optimizations. When there
// is a producer that is consumed by a stencil operation in a loop, the producer can incrementally produce
// only the values required by the next iteration, and re-use the rest of the values from the previous iteration.
class slide_and_fold : public node_mutator {
public:
  node_context& ctx;
  symbol_map<std::vector<expr>> fold_factors;
  struct loop_info {
    symbol_id sym;
    expr orig_min;
    interval_expr bounds;
    expr step;
    int max_workers;
    std::unique_ptr<symbol_map<box_expr>> buffer_bounds;

    loop_info(symbol_id sym, expr orig_min, interval_expr bounds, expr step, int max_workers)
        : sym(sym), orig_min(orig_min), bounds(bounds), step(step), max_workers(max_workers),
          buffer_bounds(std::make_unique<symbol_map<box_expr>>()) {}
  };
  std::vector<loop_info> loops;

  // We need an unknown to make equations of.
  var x;

  symbol_map<box_expr>& current_buffer_bounds() { return *loops.back().buffer_bounds; }

  slide_and_fold(node_context& ctx) : ctx(ctx), x(ctx.insert_unique("_x")) {
    loops.emplace_back(0, expr(), interval_expr::none(), expr(), loop::serial);
  }

  void visit(const allocate* op) override {
    box_expr bounds;
    bounds.reserve(op->dims.size());
    for (const dim_expr& d : op->dims) {
      bounds.push_back(d.bounds);
    }
    auto set_buffer_bounds = set_value_in_scope(current_buffer_bounds(), op->sym, bounds);
    // Initialize the fold factors to infinity.
    auto set_fold_factors =
        set_value_in_scope(fold_factors, op->sym, std::vector<expr>(op->dims.size(), positive_infinity()));
    stmt body = mutate(op->body);

    // When we constructed the pipeline, the buffer dimensions were set to buffer_* calls.
    // (This is a little janky because the buffers they are loading from don't exist where they are used.)
    // Here, we are building a list of replacements for those expressions. This way, if the user did something
    // like buf->dim(0).extent = buf->dim(0).extent + 10 (i.e. pad the extent by 10), we'll add 10 to our
    // inferred value.
    // TODO: Is this actually a good design...?
    const std::vector<expr>& fold_info = *fold_factors[op->sym];
    std::vector<std::pair<expr, expr>> replacements;
    expr alloc_var = variable::make(op->sym);
    for (index_t d = 0; d < static_cast<index_t>(op->dims.size()); ++d) {
      replacements.emplace_back(buffer_fold_factor(alloc_var, d), fold_info[d]);
    }
    std::vector<dim_expr> dims = recursive_substitute(op->dims, replacements);
    // Replace infinite fold factors with undefined.
    for (dim_expr& d : dims) {
      if (is_positive_infinity(d.fold_factor)) d.fold_factor = expr();
    }

    set_result(allocate::make(op->sym, op->storage, op->elem_size, std::move(dims), body));
  }

  template <typename T>
  void visit_call_or_copy(const T* op, span<const symbol_id> outputs) {
    set_result(op);
    for (symbol_id output : outputs) {
      // Start from 1 to skip the 'outermost' loop.
      bool did_overlapped_fold = false;
      for (std::size_t loop_index = 1; loop_index < loops.size(); ++loop_index) {
        loop_info& loop = loops[loop_index];
        std::optional<box_expr>& bounds = (*loop.buffer_bounds)[output];
        if (!bounds) continue;

        if (loop.max_workers != loop::serial) {
          // TODO: We can handle parallel loops if we add some synchronization
          // https://github.com/dsharlet/slinky/issues/18
          continue;
        }

        expr loop_var = variable::make(loop.sym);

        for (int d = 0; d < static_cast<int>(bounds->size()); ++d) {
          interval_expr cur_bounds_d = (*bounds)[d];
          if (!depends_on(cur_bounds_d, loop.sym).any()) {
            // TODO: In this case, the func is entirely computed redundantly on every iteration. We should be able to
            // just compute it once.
            continue;
          }

          interval_expr prev_bounds_d = {
              substitute(cur_bounds_d.min, loop.sym, loop_var - loop.step),
              substitute(cur_bounds_d.max, loop.sym, loop_var - loop.step),
          };

          // A few things here struggle to simplify when there is a min(loop_max, x) expression involved, where x is
          // some expression that is bounded by the loop bounds. This min simplifies away if we know that x <= loop_max,
          // but the simplifier can't figure that out. As a hopefully temporary workaround, we can just substitute
          // infinity for the loop max.
          auto ignore_loop_max = [&](const expr& e) { return substitute(e, loop.bounds.max, positive_infinity()); };

          interval_expr overlap = prev_bounds_d & cur_bounds_d;
          if (prove_true(ignore_loop_max(overlap.empty()))) {
            // The bounds of each loop iteration do not overlap. We can't re-use work between loop iterations, but we
            // can fold the storage.
            expr fold_factor = simplify(bounds_of(ignore_loop_max(cur_bounds_d.extent())).max);
            if (!depends_on(fold_factor, loop.sym).any()) {
              vector_at(fold_factors[output], d) = fold_factor;
            } else {
              // The fold factor didn't simplify to something that doesn't depend on the loop variable.
            }
            continue;
          }

          // Allowing the leading edge to not change means that some calls may ask for empty buffers.
          expr is_monotonic_increasing = prev_bounds_d.min <= cur_bounds_d.min && prev_bounds_d.max <= cur_bounds_d.max;
          expr is_monotonic_decreasing = prev_bounds_d.min >= cur_bounds_d.min && prev_bounds_d.max >= cur_bounds_d.max;
          if (prove_true(ignore_loop_max(is_monotonic_increasing))) {
            // The bounds for each loop iteration overlap and are monotonically increasing,
            // so we can incrementally compute only the newly required bounds.
            expr old_min = cur_bounds_d.min;
            expr new_min = simplify(prev_bounds_d.max + 1);

            if (!did_overlapped_fold) {
              expr fold_factor = simplify(bounds_of(ignore_loop_max(cur_bounds_d.extent())).max);
              if (!depends_on(fold_factor, loop.sym).any()) {
                // Align the fold factor to the loop step size, so it doesn't try to crop across a folding boundary.
                fold_factor = simplify(align_up(fold_factor, loop.step));
                vector_at(fold_factors[output], d) = fold_factor;
                did_overlapped_fold = true;
              } else {
                // The fold factor didn't simplify to something that doesn't depend on the loop variable.
              }
            }

            // Now that we're only computing the newly required parts of the domain, we need
            // to move the loop min back so we compute the whole required region.
            expr new_min_at_new_loop_min = substitute(new_min, loop.sym, x);
            expr old_min_at_loop_min = substitute(old_min, loop.sym, loop.bounds.min);
            expr new_loop_min =
                where_true(ignore_loop_max(new_min_at_new_loop_min <= old_min_at_loop_min), x.sym()).max;
            if (!is_negative_infinity(new_loop_min)) {
              loop.bounds.min = new_loop_min;

              (*bounds)[d].min = new_min;
            } else {
              // We couldn't find the new loop min. We need to warm up the loop on (or before) the first iteration.
              // TODO(https://github.com/dsharlet/slinky/issues/118): If there is a mix of warmup strategies, this will
              // effectively not slide while running before the original loop min.
              (*bounds)[d].min = select(loop_var <= loop.orig_min, old_min, new_min);
            }
          } else if (prove_true(ignore_loop_max(is_monotonic_decreasing))) {
            // TODO: We could also try to slide when the bounds are monotonically
            // decreasing, but this is an unusual case.
          }
        }
      }
    }
  }

  void visit(const call_stmt* op) override { visit_call_or_copy(op, op->outputs); }
  void visit(const copy_stmt* op) override { visit_call_or_copy(op, {&op->dst, 1}); }

  void visit(const crop_buffer* op) override {
    std::optional<box_expr> bounds = current_buffer_bounds()[op->sym];
    merge_crop(bounds, op->bounds);
    if (bounds) {
      substitute_bounds(*bounds, current_buffer_bounds());
      // This simplify can be heavy, but is really useful in reducing the size of the
      // expressions.
      for (auto& b : *bounds) {
        b = simplify(b);
      }
    }
    auto set_bounds = set_value_in_scope(current_buffer_bounds(), op->sym, bounds);
    stmt body = mutate(op->body);
    if (current_buffer_bounds()[op->sym]) {
      // If we folded something, the bounds required may have shrank, update the crop.
      box_expr new_bounds = *current_buffer_bounds()[op->sym];
      set_result(crop_buffer::make(op->sym, std::move(new_bounds), std::move(body)));
    } else {
      set_result(crop_buffer::make(op->sym, op->bounds, std::move(body)));
    }
  }

  void visit(const crop_dim* op) override {
    std::optional<box_expr> bounds = current_buffer_bounds()[op->sym];
    merge_crop(bounds, op->dim, op->bounds);
    substitute_bounds(*bounds, current_buffer_bounds());
    // This simplify can be heavy, but is really useful in reducing the size of the
    // expressions.
    for (auto& b : *bounds) {
      b = simplify(b);
    }

    auto set_bounds = set_value_in_scope(current_buffer_bounds(), op->sym, bounds);
    stmt body = mutate(op->body);
    interval_expr new_bounds = (*current_buffer_bounds()[op->sym])[op->dim];

    if (body.same_as(op->body) && new_bounds.same_as(op->bounds)) {
      set_result(op);
    } else {
      set_result(crop_dim::make(op->sym, op->dim, std::move(new_bounds), std::move(body)));
    }
  }

  void visit(const slice_buffer* op) override {
    std::optional<box_expr> bounds = current_buffer_bounds()[op->sym];
    if (bounds) {
      for (int d = std::min(op->at.size(), bounds->size()) - 1; d >= 0; --d) {
        if (!op->at[d].defined()) continue;
        bounds->erase(bounds->begin() + d);
      }
    }

    auto set_bounds = set_value_in_scope(current_buffer_bounds(), op->sym, bounds);
    stmt body = mutate(op->body);
    // TODO: If the bounds of the sliced dimensions are modified, do we need to insert an "if" here?
    if (body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(clone_with_new_body(op, std::move(body)));
    }
  }
  void visit(const slice_dim* op) override {
    std::optional<box_expr> bounds = current_buffer_bounds()[op->sym];
    if (bounds && op->dim < static_cast<int>(bounds->size())) {
      bounds->erase(bounds->begin() + op->dim);
    }

    auto set_bounds = set_value_in_scope(current_buffer_bounds(), op->sym, bounds);
    stmt body = mutate(op->body);
    // TODO: If the bounds of the sliced dimensions are modified, do we need to insert an "if" here?
    if (body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(clone_with_new_body(op, std::move(body)));
    }
  }
  void visit(const truncate_rank*) override { std::abort(); }

  void visit(const loop* op) override {
    var orig_min(ctx, ctx.name(op->sym) + ".min_orig");

    symbol_map<box_expr> last_buffer_bounds = current_buffer_bounds();
    loops.emplace_back(op->sym, orig_min, bounds(orig_min, op->bounds.max), op->step, op->max_workers);
    current_buffer_bounds() = last_buffer_bounds;

    stmt body = mutate(op->body);
    expr loop_min = loops.back().bounds.min;
    loops.pop_back();

    if (loop_min.same_as(orig_min)) {
      loop_min = op->bounds.min;
    }

    if (!is_variable(loop_min, orig_min.sym()) || depends_on(body, orig_min.sym()).any()) {
      // We rewrote or used the loop min.
      stmt result = loop::make(op->sym, op->max_workers, {loop_min, op->bounds.max}, op->step, std::move(body));
      set_result(let_stmt::make(orig_min.sym(), op->bounds.min, result));
      return;
    }

    if (body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(loop::make(op->sym, op->max_workers, op->bounds, op->step, std::move(body)));
    }
  }

  void visit(const block* op) override {
    // Visit blocks in reverse order. TODO: Is this really sufficient?
    std::vector<stmt> stmts(op->stmts.size());
    bool changed = false;
    for (int i = static_cast<int>(op->stmts.size()) - 1; i >= 0; --i) {
      stmts[i] = mutate(op->stmts[i]);
      changed = changed || !stmts[i].same_as(op->stmts[i]);
    }
    if (!changed) {
      set_result(op);
    } else {
      set_result(block::make(std::move(stmts)));
    }
  }
};

}  // namespace

stmt slide_and_fold_storage(const stmt& s, node_context& ctx) { return slide_and_fold(ctx).mutate(s); }

}  // namespace slinky
