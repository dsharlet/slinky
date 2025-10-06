#include "builder/slide_and_fold_storage.h"

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>

#include "base/chrome_trace.h"
#include "builder/node_mutator.h"
#include "builder/simplify.h"
#include "builder/substitute.h"
#include "runtime/depends_on.h"
#include "runtime/evaluate.h"
#include "runtime/expr.h"

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
  bounds_d = simplify_intersection(std::move(bounds_d), new_bounds);
}

void merge_crop(std::optional<box_expr>& bounds, const box_expr& new_bounds) {
  if (!bounds) {
    bounds = box_expr();
  }
  for (int d = 0; d < static_cast<int>(new_bounds.size()); ++d) {
    merge_crop(bounds, d, new_bounds[d]);
  }
}

// Replace any undefined bounds, which can come from merge_crop above, with buffer_min/buffer_max.
void define_undef_bounds(box_expr& bounds, var sym) {
  for (int d = 0; d < static_cast<int>(bounds.size()); ++d) {
    interval_expr& bounds_d = bounds[d];
    if (!bounds_d.min.defined()) bounds_d.min = buffer_min(sym, d);
    if (!bounds_d.max.defined()) bounds_d.max = buffer_max(sym, d);
  }
}

// Keep substituting substitutions until nothing happens.
std::vector<dim_expr> recursive_substitute(
    std::vector<dim_expr> dims, span<const std::pair<expr, expr>> substitutions) {
  scoped_trace trace("recursive_substitute");
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

void substitute_bounds(interval_expr& bounds, const symbol_map<box_expr>& buffers) {
  scoped_trace trace("substitute_bounds");
  for (std::size_t i = 0; i < buffers.size(); ++i) {
    if (!buffers[i]) continue;
    bounds = substitute_buffer(bounds, var(i), make_dims_from_bounds(*buffers[i]));
  }
}

void substitute_bounds(box_expr& bounds, const symbol_map<box_expr>& buffers) {
  scoped_trace trace("substitute_bounds");
  for (std::size_t i = 0; i < buffers.size(); ++i) {
    if (!buffers[i]) continue;
    auto dims = make_dims_from_bounds(*buffers[i]);
    for (interval_expr& j : bounds) {
      j = substitute_buffer(j, var(i), dims);
    }
  }
}

void substitute_bounds(symbol_map<box_expr>& buffers, var buffer_id, const box_expr& bounds) {
  scoped_trace trace("substitute_bounds");
  auto dims = make_dims_from_bounds(bounds);
  for (std::size_t i = 0; i < buffers.size(); ++i) {
    if (!buffers[i]) continue;
    slinky::box_expr& b = *buffers[i];
    for (interval_expr& j : b) {
      j = substitute_buffer(j, buffer_id, dims);
    }
  }
}

// Check if the given buffer is produced inside of the statement.
class check_if_produced : public recursive_node_visitor {
  var v;

public:
  check_if_produced(var v) : v(v) {}
  bool found = false;

  void visit(const call_stmt* op) override {
    for (const auto& o : op->outputs) {
      found = found || (o == v);
    }
  }
  void visit(const copy_stmt* op) override { found = found || (op->dst == v); }
};

bool is_produced_by(var v, const stmt& body) {
  scoped_trace trace("is_produced_by");
  if (!body.defined()) return false;
  check_if_produced f(v);
  body.accept(&f);
  return f.found;
}

// Find a maximum value of x which makes `condition` expression true. The search goes
// backwards from initial_guess up to some fixed depth.
expr where_true_upper_bound(const expr& condition, var x, const expr& initial_guess, const bounds_map& expr_bounds,
    const symbol_map<modulus_remainder<index_t>>& expr_alignment) {
  scoped_trace trace("where_true_upper_bound");
  expr result = negative_infinity();

  // TODO: this search can be more efficient by trying to cover wider range of depth
  // using binary search or something similar.
  const int max_search_depth = 10;

  for (int ix = 0; ix < max_search_depth; ix++) {
    expr shifted = substitute(condition, x, (initial_guess - ix));
    if (prove_true(shifted, expr_bounds, expr_alignment)) {
      result = simplify(initial_guess - ix, expr_bounds, expr_alignment);
      break;
    }
  }

  return result;
}

// Try to find cases where we can do "sliding window" or "line buffering" optimizations. When there
// is a producer that is consumed by a stencil operation in a loop, the producer can incrementally produce
// only the values required by the next iteration, and re-use the rest of the values from the previous iteration.
class slide_and_fold : public stmt_mutator {
public:
  node_context& ctx;

  struct dim_fold_info {
    // The fold factor
    expr factor = positive_infinity();

    // Overlap between iteration i and i + 1.
    expr overlap;

    // Unique ID of the loop this fold is for.
    std::size_t loop_id;
  };
  symbol_map<std::vector<dim_fold_info>> fold_factors;

  // Counter for the number of loops we've seen.
  std::size_t loop_counter = 0;

  struct loop_info {
    var sym;
    expr orig_min;
    interval_expr bounds;
    expr step;
    expr max_workers = 0;
    bool data_parallel = true;
    std::unique_ptr<symbol_map<box_expr>> buffer_bounds = std::make_unique<symbol_map<box_expr>>();
    std::unique_ptr<symbol_map<interval_expr>> expr_bounds = std::make_unique<symbol_map<interval_expr>>();
    std::unique_ptr<symbol_map<modulus_remainder<index_t>>> expr_alignment =
        std::make_unique<symbol_map<modulus_remainder<index_t>>>();

    // The next few fields relate to implementing synchronization in pipelined loops. In a pipelined loop, we
    // treat a sequence of stmts as "stages" in the pipeline, where we add synchronization to cause the loop
    // to appear to be executed serially to the stages: a stage can assume the same stage for a previous iteration has
    // completed, and can assume that all previous stages for the same iteration have completed.
    var semaphores;
    var worker_count;

    // How many stages we've added synchronization for in total so far.
    int sync_stages = 0;
    // We only track the stage we're currently working on. This optional being present indicates the current stage needs
    // synchronization, and the value indicates which stage it is.
    std::optional<int> stage;

    // Unique loop ID.
    std::size_t loop_id = -1;

    bool add_synchronization() {
      if (prove_true(sync_stages + 1 >= max_workers)) {
        // It's pointless to add more stages to the loop, because we can't run then in parallel anyways, it would just
        // add more synchronization overhead.
        return false;
      }

      // We need synchronization, but we might already have it.
      if (!stage) {
        stage = sync_stages++;
      }
      return true;
    }

    loop_info() = default;

    loop_info(node_context& ctx, var sym, std::size_t loop_id, expr orig_min, interval_expr bounds, expr step,
        expr max_workers)
        : sym(sym), orig_min(orig_min), bounds(bounds), step(step), max_workers(max_workers),
          semaphores(ctx, ctx.name(sym) + "_semaphores"), worker_count(ctx, ctx.name(sym) + "_worker_count"),
          loop_id(loop_id) {}
  };
  std::vector<loop_info> loops;

  symbol_map<var> aliases;

  // We need an unknown to make equations of.
  var x;

  symbol_map<box_expr>& current_buffer_bounds() { return *loops.back().buffer_bounds; }
  symbol_map<interval_expr>& current_expr_bounds() { return *loops.back().expr_bounds; }
  symbol_map<modulus_remainder<index_t>>& current_expr_alignment() { return *loops.back().expr_alignment; }

  slide_and_fold(node_context& ctx) : ctx(ctx), x(ctx.insert_unique("_x")) { loops.emplace_back(loop_info()); }

  stmt mutate(const stmt& s) override {
    stmt result = stmt_mutator::mutate(s);

    // The loop at the back of the loops vector is the immediately containing loop. So, we know there are no
    // intervening loops, and we can add any synchronization that has been requested. Doing so completes the current
    // pipeline stage.
    loop_info& l = loops.back();
    if (l.stage) {
      result = block::make({
          // Wait for the previous iteration of this stage to complete.
          // The l.sym here is equal to l.min + x * l.step, so dividing l.sym by l.step we  get floor_div(l.min) + x.
          // This works even if l.min is not divisible by l.step, because it remains constant w.r.t to the loop index.
          check::make(semaphore_wait(buffer_at(l.semaphores, *l.stage, floor_div(expr(l.sym), l.step) - 1))),
          result,
          // Signal we've done this iteration.
          check::make(semaphore_signal(buffer_at(l.semaphores, *l.stage, floor_div(expr(l.sym), l.step)))),
      });
      l.stage = std::nullopt;
    }

    return result;
  }

  void visit(const let_stmt* op) override {
    auto& bounds = current_expr_bounds();
    std::vector<scoped_value_in_symbol_map<interval_expr>> values;
    values.reserve(op->lets.size());
    for (const auto& i : op->lets) {
      values.push_back(set_value_in_scope(bounds, i.first, bounds_of(i.second, bounds)));
    }
    stmt body = mutate(op->body);
    if (!body.same_as(op->body)) {
      set_result(let_stmt::make(op->lets, std::move(body)));
    } else {
      set_result(op);
    }
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
        set_value_in_scope(fold_factors, op->sym, std::vector<dim_fold_info>(op->dims.size(), dim_fold_info()));
    stmt body = mutate(op->body);

    // When we constructed the pipeline, the buffer dimensions were set to buffer_* calls.
    // (This is a little janky because the buffers they are loading from don't exist where they are used.)
    // Here, we are building a list of replacements for those expressions. This way, if the user did something
    // like buf->dim(0).extent = buf->dim(0).extent + 10 (i.e. pad the extent by 10), we'll add 10 to our
    // inferred value.
    // TODO: Is this actually a good design...?
    const std::vector<dim_fold_info>& fold_info = *fold_factors[op->sym];
    std::vector<std::pair<expr, expr>> replacements;
    for (index_t d = 0; d < static_cast<index_t>(op->dims.size()); ++d) {
      replacements.emplace_back(buffer_fold_factor(op->sym, d), fold_info[d].factor);
    }
    std::vector<dim_expr> dims = recursive_substitute(op->dims, replacements);
    // Replace infinite fold factors with undefined.
    for (dim_expr& d : dims) {
      if (is_positive_infinity(d.fold_factor)) d.fold_factor = expr();
    }

    set_result(allocate::make(op->sym, op->storage, op->elem_size, std::move(dims), body));
  }

  void slide_and_fold_buffer(const var& output, const stmt& body) {
    scoped_trace trace("slide_and_fold_buffer");
    // We only want to fold if we are inside of the loop and the cropped buffer
    // is produced there.
    if (loops.size() < 2 || !is_produced_by(output, body)) return;

    bool did_overlapped_fold = false;

    if (fold_factors[output]) {
      for (int d = 0; d < static_cast<int>(fold_factors[output]->size()); ++d) {
        expr fold = (*fold_factors[output])[d].factor;
        expr overlap = (*fold_factors[output])[d].overlap;
        if (!is_finite(fold)) continue;
        // If fold is finite and bounds don't overlap the fold and overlap
        // will be set to the same expr.
        did_overlapped_fold = did_overlapped_fold || !match(fold, overlap);
      }
    }

    loop_info& loop = loops.back();
    std::optional<box_expr>& bounds = (*loop.buffer_bounds)[output];
    if (!bounds) return;

    expr loop_var = variable::make(loop.sym);

    for (int d = 0; d < static_cast<int>(bounds->size()); ++d) {
      if (fold_factors[output] && (d < static_cast<int>(fold_factors[output]->size()))) {
        expr fold_factor = (*fold_factors[output])[d].factor;
        // Skip if we already folded this dimension.
        if (is_finite(fold_factor)) continue;
      }

      interval_expr cur_bounds_d = (*bounds)[d];
      if (!depends_on(cur_bounds_d, loop.sym).any()) {
        // TODO: In this case, the func is entirely computed redundantly on every iteration. We should be able to
        // just compute it once.
        continue;
      }

      // Some expressions which involve loop bounds are difficult to simplify, so let's try to do that using the latest
      // loop bounds.;
      cur_bounds_d = simplify(cur_bounds_d, *loop.expr_bounds, *loop.expr_alignment);

      interval_expr prev_bounds_d = {
          substitute(cur_bounds_d.min, loop.sym, loop_var - loop.step),
          substitute(cur_bounds_d.max, loop.sym, loop_var - loop.step),
      };

      interval_expr overlap = prev_bounds_d & cur_bounds_d;
      if (prove_true(overlap.empty(), *loop.expr_bounds, *loop.expr_alignment)) {
        // The bounds of each loop iteration do not overlap. We can't re-use work between loop iterations, but we
        // can fold the storage.
        expr fold_factor =
            simplify(bounds_of(cur_bounds_d.extent(), *loop.expr_bounds).max, *loop.expr_bounds, *loop.expr_alignment);
        fold_factor = simplify(constant_upper_bound(fold_factor), *loop.expr_bounds, *loop.expr_alignment);
        if (is_finite(fold_factor) && !depends_on(fold_factor, loop.sym).any()) {
          vector_at(fold_factors[output], d) = {fold_factor, fold_factor, loops.back().loop_id};
        } else {
          // The fold factor didn't simplify to something that doesn't depend on the loop variable.
        }
        continue;
      }

      // Allowing the leading edge to not change means that some calls may ask for empty buffers.
      expr is_monotonic_increasing = prev_bounds_d.min <= cur_bounds_d.min && prev_bounds_d.max <= cur_bounds_d.max;
      expr is_monotonic_decreasing = prev_bounds_d.min >= cur_bounds_d.min && prev_bounds_d.max >= cur_bounds_d.max;
      if (prove_true(is_monotonic_increasing, *loop.expr_bounds, *loop.expr_alignment)) {
        // The bounds for each loop iteration overlap and are monotonically increasing,
        // so we can incrementally compute only the newly required bounds.
        expr old_min = cur_bounds_d.min;
        expr new_min = simplify(prev_bounds_d.max + 1, *loop.expr_bounds, *loop.expr_alignment);

        if (!did_overlapped_fold) {
          expr fold_factor = simplify(bounds_of(cur_bounds_d.extent(), *loop.expr_bounds, *loop.expr_alignment).max,
              *loop.expr_bounds, *loop.expr_alignment);
          fold_factor = simplify(constant_upper_bound(fold_factor), *loop.expr_bounds, *loop.expr_alignment);
          if (is_finite(fold_factor) && !depends_on(fold_factor, loop.sym).any()) {
            // Align the fold factor to the loop step size, so it doesn't try to crop across a folding boundary.
            vector_at(fold_factors[output], d) = {simplify(fold_factor, *loop.expr_bounds, *loop.expr_alignment),
                simplify(constant_upper_bound(
                             bounds_of(cur_bounds_d.max - new_min + 1, *loop.expr_bounds, *loop.expr_alignment).max),
                    *loop.expr_bounds),
                loops.back().loop_id};
            did_overlapped_fold = true;
          } else {
            // The fold factor didn't simplify to something that doesn't depend on the loop variable.
          }
        }

        // Now that we're only computing the newly required parts of the domain, we need
        // to move the loop min back so we compute the whole required region.
        expr new_min_at_new_loop_min = substitute(new_min, loop.sym, x);
        expr old_min_at_loop_min = substitute(old_min, loop.sym, loop.bounds.min);
        expr new_loop_min = where_true_upper_bound(
            new_min <= old_min_at_loop_min, loop.sym, loop.bounds.min, *loop.expr_bounds, *loop.expr_alignment);

        if (!is_negative_infinity(new_loop_min)) {
          loop.bounds.min = new_loop_min;

          (*bounds)[d].min = new_min;
        } else {
          // We couldn't find the new loop min. We need to warm up the loop on (or before) the first iteration.
          // TODO(https://github.com/dsharlet/slinky/issues/118): If there is a mix of warmup strategies, this will
          // effectively not slide while running before the original loop min.
          (*bounds)[d].min = select(loop_var <= loop.orig_min, old_min, new_min);
        }

        // This loop has a dependency between loop iterations, mark it as not data parallel.
        loop.data_parallel = false;
      } else if (prove_true(is_monotonic_decreasing, *loop.expr_bounds, *loop.expr_alignment)) {
        // TODO: We could also try to slide when the bounds are monotonically
        // decreasing, but this is an unusual case.
      }
    }
  }

  void visit_call_or_copy(span<const var> inputs, span<const var> outputs) {
    scoped_trace trace("visit_call_or_copy");

    for (var input : inputs) {
      // Remove folding for an input that was folded in a more deeply nested loop than the current loop.
      // We need to do this for any aliased symbols of the input as well.
      var a = input;
      while (true) {
        if (fold_factors[a]) {
          for (dim_fold_info& i : *fold_factors[a]) {
            bool is_correct_fold = false;
            for (const loop_info& loop : loops) {
              is_correct_fold = is_correct_fold || loop.loop_id == i.loop_id;
            }
            if (!is_correct_fold) {
              // We found a consumer of a buffer outside the loop it was folded in, remove the folding.
              i = dim_fold_info();
            }
          }
        }
        std::optional<var> next_a = aliases[a];
        if (!next_a || *next_a == a) break;
        a = *next_a;
      }
    }

    for (var output : outputs) {
      for (loop_info& loop : loops) {
        if (!fold_factors[output]) continue;
        loop.add_synchronization();

        expr loop_var = variable::make(loop.sym);
        for (int d = 0; d < static_cast<int>(fold_factors[output]->size()); ++d) {
          if ((*fold_factors[output])[d].loop_id != loop.loop_id) {
            // This is a fold factor for a different loop.
            continue;
          }

          expr fold_factor = (*fold_factors[output])[d].factor;
          if (!is_finite(fold_factor)) {
            continue;
          }

          if (!depends_on(fold_factor, loop.sym).any()) {
            // We need an extra fold per worker when parallelizing the loop.
            // TODO: This extra folding seems excessive, it allows all workers to execute any stage.
            // If we can figure out how to add some synchronization to limit the number of workers that
            // work on a single stage at a time, we should be able to reduce this extra folding.
            // TODO: In this case, we currently need synchronization, but we should find a way to eliminate it.
            // This synchronization will cause the loop to run only as fast as the slowest stage, which is
            // unnecessary in the case of a fully data parallel loop. In order to avoid this, we need to avoid race
            // conditions. The synchronization avoids the race condition by only allowing a window of max_workers to
            // run at once, so the storage folding here works as intended. If we could instead find a way to give
            // each worker its own slice of this buffer, we could avoid this synchronization. I think this might be
            // doable by making the worker index available to the loop body, and using that to grab a slice of this
            // buffer, so each worker can get its own fold.

            fold_factor += (loop.worker_count - 1) * (*fold_factors[output])[d].overlap;
            vector_at(fold_factors[output], d).factor = simplify(fold_factor);
          }
        }
      }
    }
  }

  void visit(const call_stmt* op) override {
    set_result(op);
    visit_call_or_copy(op->inputs, op->outputs);
  }
  void visit(const copy_stmt* op) override {
    set_result(op);
    visit_call_or_copy({&op->src, 1}, {&op->dst, 1});
  }

  void visit(const crop_buffer* op) override {
    std::optional<box_expr> bounds = current_buffer_bounds()[op->src];
    merge_crop(bounds, op->bounds);
    if (bounds) {
      define_undef_bounds(*bounds, op->sym);
      substitute_bounds(*bounds, current_buffer_bounds());
      // This simplify can be heavy, but is really useful in reducing the size of the
      // expressions.
      for (auto& b : *bounds) {
        b = simplify(b);
      }

      // Now do the reverse substitution, because the updated bounds can be used in other
      // bounds.
      substitute_bounds(current_buffer_bounds(), op->sym, *bounds);
    }

    auto set_bounds = set_value_in_scope(current_buffer_bounds(), op->sym, bounds);
    auto set_alias = set_value_in_scope(aliases, op->sym, op->src);

    slide_and_fold_buffer(op->sym, op->body);

    stmt body = mutate(op->body);
    if (current_buffer_bounds()[op->sym]) {
      // If we folded something, the bounds required may have shrank, update the crop.
      box_expr new_bounds = *current_buffer_bounds()[op->sym];
      set_result(crop_buffer::make(op->sym, op->src, std::move(new_bounds), std::move(body)));
    } else {
      set_result(crop_buffer::make(op->sym, op->src, op->bounds, std::move(body)));
    }
  }

  void visit(const crop_dim* op) override {
    std::optional<box_expr> bounds = current_buffer_bounds()[op->src];
    merge_crop(bounds, op->dim, op->bounds);
    define_undef_bounds(*bounds, op->sym);
    substitute_bounds(*bounds, current_buffer_bounds());
    // This simplify can be heavy, but is really useful in reducing the size of the
    // expressions.
    for (auto& b : *bounds) {
      b = simplify(b);
    }

    // Now do the reverse substitution, because the updated bounds can be used in other
    // bounds.
    substitute_bounds(current_buffer_bounds(), op->sym, *bounds);

    auto set_bounds = set_value_in_scope(current_buffer_bounds(), op->sym, bounds);
    auto set_alias = set_value_in_scope(aliases, op->sym, op->src);

    slide_and_fold_buffer(op->sym, op->body);

    stmt body = mutate(op->body);
    interval_expr new_bounds = (*current_buffer_bounds()[op->sym])[op->dim];

    if (body.same_as(op->body) && new_bounds.same_as(op->bounds)) {
      set_result(op);
    } else {
      set_result(crop_dim::make(op->sym, op->src, op->dim, std::move(new_bounds), std::move(body)));
    }
  }

  void visit(const slice_buffer* op) override {
    std::optional<box_expr> bounds = current_buffer_bounds()[op->src];
    if (bounds) {
      for (int d = std::min(op->at.size(), bounds->size()) - 1; d >= 0; --d) {
        if (!op->at[d].defined()) continue;
        bounds->erase(bounds->begin() + d);
      }
    }

    auto set_bounds = set_value_in_scope(current_buffer_bounds(), op->sym, bounds);
    auto set_alias = set_value_in_scope(aliases, op->sym, op->src);
    stmt body = mutate(op->body);
    // TODO: If the bounds of the sliced dimensions are modified, do we need to insert an "if" here?
    if (body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(clone_with(op, std::move(body)));
    }
  }
  void visit(const slice_dim* op) override {
    std::optional<box_expr> bounds = current_buffer_bounds()[op->src];
    if (bounds && op->dim < static_cast<int>(bounds->size())) {
      bounds->erase(bounds->begin() + op->dim);
    }

    auto set_bounds = set_value_in_scope(current_buffer_bounds(), op->sym, bounds);
    auto set_alias = set_value_in_scope(aliases, op->sym, op->src);
    stmt body = mutate(op->body);
    // TODO: If the bounds of the sliced dimensions are modified, do we need to insert an "if" here?
    if (body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(clone_with(op, std::move(body)));
    }
  }
  void visit(const transpose*) override { SLINKY_UNREACHABLE << "transpose not handled by slide_and_fold_storage"; }
  void visit(const clone_buffer* op) override {
    auto set_alias = set_value_in_scope(aliases, op->sym, op->src);
    stmt_mutator::visit(op);
  }

  void visit(const loop* op) override {
    var orig_min(ctx, ctx.name(op->sym) + ".min_orig");

    symbol_map<box_expr> last_buffer_bounds = current_buffer_bounds();
    symbol_map<interval_expr> last_expr_bounds = current_expr_bounds();
    symbol_map<modulus_remainder<index_t>> last_expr_alignment = current_expr_alignment();

    interval_expr loop_bounds = op->bounds;
    // It's possible that after sliding some of the buffers the bounds of the
    // loop will need to include the region which is outside of the actual buffer bounds this loop
    // depends on. In this case buffer bounds will be clamped to the actual buffer bounds
    // which will make the loop bounds smaller than necessary, so in order to avoid this clamping
    // we substitute current buffer bounds into loop bounds.
    substitute_bounds(loop_bounds, current_buffer_bounds());

    loops.emplace_back(ctx, op->sym, loop_counter++, orig_min, loop_bounds, op->step, op->max_workers);
    current_buffer_bounds() = last_buffer_bounds;
    current_expr_bounds() = last_expr_bounds;
    current_expr_alignment() = last_expr_alignment;

    stmt body;
    {
      // We can use narrower bounds for the loop var, because the loop var not necessarily will reach max if step > 1.
      auto set_expr_bounds = set_value_in_scope(current_expr_bounds(), op->sym,
          {loop_bounds.min, loop_bounds.min + align_down(loop_bounds.extent() - 1, op->step)});

      auto maybe_min = as_constant(loop_bounds.min);
      auto maybe_step = as_constant(op->step);
      index_t modulus = maybe_step ? *maybe_step : 1;
      index_t remainder = (maybe_min && maybe_step) ? *maybe_min % *maybe_step : 0;

      auto set_expr_alignment = set_value_in_scope(current_expr_alignment(), op->sym, {modulus, remainder});
      body = mutate(op->body);
    }

    scoped_trace trace("visit(const loop*)");

    loop_bounds.min = loops.back().bounds.min;
    if (body.same_as(op->body) && loop_bounds.min.same_as(op->bounds.min) && loop_bounds.max.same_as(op->bounds.max)) {
      set_result(op);
      return;
    }

    const loop_info& l = loops.back();
    const int stage_count = l.sync_stages;
    expr max_workers = l.data_parallel ? op->max_workers : std::max(1, stage_count);
    stmt result = loop::make(op->sym, max_workers, loop_bounds, op->step, std::move(body));

    // Substitute the placeholder worker_count.
    result = substitute(result, l.worker_count, max_workers);
    // We need to do this in the fold factors too.
    for (std::optional<std::vector<dim_fold_info>>& i : fold_factors) {
      if (!i) continue;
      for (dim_fold_info& j : *i) {
        if (!depends_on(j.factor, l.worker_count).any()) continue;

        if (l.data_parallel && !prove_true(max_workers == loop::serial)) {
          // This is a data parallel loop, remove the folding.
          // TODO: We have other options that would be better:
          // - Move the allocation into the loop.
          // - Rewrite accesses to this dimension to be a function of a thread ID (and rewrite the fold factor to the
          // max thread ID).
          j.factor = expr();
        } else {
          // This is a serial or pipelined loop, we can still fold.
          j.factor = substitute(j.factor, l.worker_count, max_workers);
        }
      }
    }

    if (!l.data_parallel && stage_count > 1) {
      // We added synchronization in the loop, we need to allocate a buffer for the semaphores.
      interval_expr sem_bounds = {0, stage_count - 1};

      index_t sem_size = sizeof(index_t);
      call_stmt::attributes init_sems_attrs;
      init_sems_attrs.name = "init_semaphores";
      stmt init_sems = call_stmt::make(
          [stage_count](const call_stmt* s, eval_context& ctx) -> index_t {
            const buffer<index_t>& sems = *ctx.lookup_buffer<index_t>(s->outputs[0]);
            assert(sems.rank == 2);
            assert(sems.dim(0).min() == 0);
            assert(sems.dim(0).extent() == stage_count);
            memset(sems.base(), 0, sems.size_bytes());
            // Initialize the first semaphore for each stage (the one before the loop min) to 1,
            // unblocking the first iteration.
            assert(sems.dim(0).stride() == sizeof(index_t));
            std::fill_n(&sems(0, sems.dim(1).min()), stage_count, 1);
            return 0;
          },
          {}, {l.semaphores}, {}, std::move(init_sems_attrs));
      // We can fold the semaphores array by the number of threads we'll use.
      // TODO: Use the loop index and not the loop variable directly for semaphores so we don't need to do this.
      expr sem_fold_factor = stage_count;
      std::vector<dim_expr> sem_dims = {
          {sem_bounds, sem_size},
          // TODO: We should just let dimensions like this have undefined bounds.
          {{floor_div(loop_bounds.min, op->step) - 1, floor_div(loop_bounds.max, op->step)},
              sem_size * sem_bounds.extent(), sem_fold_factor},
      };
      result = allocate::make(
          l.semaphores, memory_type::stack, sem_size, std::move(sem_dims), block::make({init_sems, result}));
    } else {
      // We only have one stage, there's no need for semaphores.
      result = substitute(result, l.semaphores, expr());
    }

    if (!is_variable(loop_bounds.min, orig_min) || depends_on(result, orig_min).any()) {
      // We rewrote or used the loop min.
      result = let_stmt::make(orig_min, op->bounds.min, result);
    }

    set_result(std::move(result));
    loops.pop_back();
  }
};

}  // namespace

stmt slide_and_fold_storage(const stmt& s, node_context& ctx) {
  scoped_trace trace("slide_and_fold_storage");
  return slide_and_fold(ctx).mutate(s);
}

}  // namespace slinky
