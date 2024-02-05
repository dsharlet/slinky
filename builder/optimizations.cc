#include "builder/optimizations.h"

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "builder/node_mutator.h"
#include "builder/substitute.h"
#include "runtime/buffer.h"
#include "runtime/depends_on.h"
#include "runtime/evaluate.h"
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

void merge_crop(std::optional<box_expr>& bounds, int dim, const interval_expr& new_bounds) {
  if (new_bounds.min.defined()) {
    vector_at(bounds, dim).min = new_bounds.min;
  }
  if (new_bounds.max.defined()) {
    vector_at(bounds, dim).max = new_bounds.max;
  }
}

void merge_crop(std::optional<box_expr>& bounds, const box_expr& new_bounds) {
  for (int d = 0; d < static_cast<int>(new_bounds.size()); ++d) {
    merge_crop(bounds, d, new_bounds[d]);
  }
}

bool is_elementwise(const box_expr& in_x, symbol_id out) {
  for (index_t d = 0; d < static_cast<index_t>(in_x.size()); ++d) {
    // TODO: This is too lax, we really need to check for elementwise before we've computed the bounds of this
    // particular call, so we can check that a single point of the output is a function of the same point in the input
    // (and not a rectangle of output being a function of a rectangle of the input).
    if (!is_buffer_min(in_x[d].min, out, d)) return false;
    if (!is_buffer_max(in_x[d].max, out, d)) return false;
  }
  return true;
}

bool is_copy(expr in, var out, expr& offset) {
  static var x(0), dx(1), negative_dx(2);
  static expr patterns[] = {
      x, x + dx, x - negative_dx,
      // TODO: we could also handle scaling of x by multiplying the stride.
  };

  symbol_map<expr> matches;
  for (const expr& p : patterns) {
    matches.clear();
    if (match(p, in, matches) && match(*matches[x], out)) {
      offset = 0;
      // We found a pattern that is a copy. We don't care which one, we just need to look at the matches we have.
      if (matches[dx]) offset = *matches[dx];
      if (matches[negative_dx]) offset = -*matches[negative_dx];
      return true;
    }
  }

  return false;
}

bool is_copy(const copy_stmt* op, std::vector<expr>& offset) {
  if (op->src_x.size() != op->dst_x.size()) return false;
  offset.resize(op->dst_x.size());
  for (std::size_t d = 0; d < op->dst_x.size(); ++d) {
    if (!is_copy(op->src_x[d], op->dst_x[d], offset[d])) return false;
  }
  return true;
}

// Replaces `copy_stmt` with a call to `pad`.
class replace_copy_with_pad : public node_mutator {
  symbol_id src;
  symbol_id dst;

public:
  replace_copy_with_pad(symbol_id src, symbol_id dst) : src(src), dst(dst) {}

  void visit(const copy_stmt* op) {
    if (op->src == src && op->dst == dst) {
      if (op->padding.empty()) {
        set_result(stmt());
      } else {
        set_result(call_stmt::make(
            [src = op->src, dst = op->dst, padding = op->padding](const eval_context& ctx) -> index_t {
              const raw_buffer* src_buf = ctx.lookup_buffer(src);
              const raw_buffer* dst_buf = ctx.lookup_buffer(dst);
              pad(src_buf->dims, *dst_buf, padding.data());
              return 0;
            },
            {src}, {dst}));
      }
    } else {
      set_result(op);
    }
  }
};

class buffer_aliaser : public node_mutator {
  struct buffer_alias {
    std::vector<expr> offset;
  };

  class buffer_info {
  public:
    std::map<symbol_id, buffer_alias> can_alias_;
    std::set<symbol_id> cannot_alias_;

  public:
    const std::map<symbol_id, buffer_alias>& can_alias() const { return can_alias_; }

    void maybe_alias(symbol_id s, buffer_alias a) {
      if (!cannot_alias_.count(s)) {
        can_alias_[s] = std::move(a);
      }
    }

    void do_not_alias(symbol_id s) {
      can_alias_.erase(s);
      cannot_alias_.insert(s);
    }
  };
  symbol_map<buffer_info> alias_info;
  symbol_map<box_expr> buffer_bounds;
  symbol_map<std::size_t> elem_sizes;
  symbol_map<bool> do_not_alias;

public:
  void visit(const allocate* op) override {
    box_expr bounds;
    bounds.reserve(op->dims.size());
    bool do_not_alias_sym = false;
    for (const dim_expr& d : op->dims) {
      bounds.push_back(d.bounds);
      if (d.fold_factor.defined()) {
        // This buffer can't be aliased.
        do_not_alias_sym = true;
      }
    }
    auto set_do_not_alias = set_value_in_scope(do_not_alias, op->sym, do_not_alias_sym);
    auto set_buffer_bounds = set_value_in_scope(buffer_bounds, op->sym, bounds);
    auto set_elem_size = set_value_in_scope(elem_sizes, op->sym, op->elem_size);

    // When we allocate a buffer, we can look for all the uses of this buffer. If it is:
    // - consumed elemenwise,
    // - consumed by a producer that has an output that we can re-use,
    // - not consumed after the buffer it aliases to is produced,
    // - doesn't have any folded dimensions,
    // then we can alias it to the buffer produced by its consumer.

    // Start out by setting it to elementwise.
    auto s = set_value_in_scope(alias_info, op->sym, buffer_info());
    stmt body = mutate(op->body);
    const std::map<symbol_id, buffer_alias>& can_alias = alias_info[op->sym]->can_alias();

    if (!can_alias.empty()) {
      const std::pair<symbol_id, buffer_alias>& target = *can_alias.begin();
      var target_var(target.first);

      // Here, we're essentially constructing make_buffer(op->sym, ...) { crop_buffer(op->sym, dims_bounds(op->dims) {
      // ... } }, but we can't do that (and just rely on the simplifier) because translated crops might require a
      // buffer_at call that is out of bounds.
      std::vector<expr> at = target.second.offset;
      std::vector<dim_expr> dims = buffer_dims(target_var, op->dims.size());
      assert(at.size() <= dims.size());
      at.resize(dims.size());
      for (int d = 0; d < static_cast<int>(dims.size()); ++d) {
        if (!at[d].defined()) at[d] = 0;
        at[d] = max(buffer_min(target_var, d) - at[d], op->dims[d].bounds.min);
        dims[d].bounds &= op->dims[d].bounds;
      }
      stmt result = make_buffer::make(
          op->sym, buffer_at(target_var, at), static_cast<index_t>(op->elem_size), std::move(dims), std::move(body));
      // If we aliased the source and destination of a copy, replace the copy with a pad.
      stmt pad_result = replace_copy_with_pad(op->sym, target.first).mutate(result);
      if (pad_result.same_as(result)) {
        // This wasn't a copy, we actually did some computation in place. We can't alias another buffer to this target
        // without understanding the lifetimes more carefully.
        // TODO: I think this is a hack, but I'm not sure. I think maybe the proper thing to do is track a box_expr
        // of the region that has been aliased so far, and allow another alias as long as it does not intersect that
        // region. That will likely be very difficult to do symbolically.
        for (std::optional<buffer_info>& i : alias_info) {
          if (!i) continue;
          i->do_not_alias(target.first);
        }
      }
      set_result(pad_result);
    } else if (!body.same_as(op->body)) {
      set_result(clone_with_new_body(op, std::move(body)));
    } else {
      set_result(op);
    }
  }

  void visit(const call_stmt* op) override {
    set_result(op);
    for (symbol_id o : op->outputs) {
      std::optional<std::size_t> elem_size_o = elem_sizes[o];
      if (!elem_size_o) continue;

      std::optional<bool> no_alias = do_not_alias[o];
      if (no_alias && *no_alias) {
        continue;
      }

      for (symbol_id i : op->inputs) {
        const std::optional<box_expr>& in_x = buffer_bounds[i];
        std::optional<buffer_info>& info = alias_info[i];
        if (!info) continue;
        std::optional<std::size_t> elem_size_i = elem_sizes[i];
        if (!elem_size_i) continue;

        if (!in_x || *elem_size_o != *elem_size_i || !is_elementwise(*in_x, o)) {
          info->do_not_alias(o);
          return;
        }
        buffer_alias a;
        a.offset = {};
        info->maybe_alias(o, std::move(a));
      }
    }
  }

  void visit(const copy_stmt* op) override {
    set_result(op);

    std::optional<bool> no_alias = do_not_alias[op->dst];
    if (no_alias && *no_alias) {
      return;
    }

    std::optional<buffer_info>& info = alias_info[op->src];
    if (!info) {
      return;
    }

    buffer_alias a;
    if (!is_copy(op, a.offset)) {
      return;
    }
    info->maybe_alias(op->dst, std::move(a));
  }

  void visit(const crop_buffer* op) override {
    std::optional<box_expr> bounds = buffer_bounds[op->sym];
    merge_crop(bounds, op->bounds);
    auto set_bounds = set_value_in_scope(buffer_bounds, op->sym, bounds);
    node_mutator::visit(op);
  }

  void visit(const crop_dim* op) override {
    std::optional<box_expr> bounds = buffer_bounds[op->sym];
    merge_crop(bounds, op->dim, op->bounds);
    auto set_bounds = set_value_in_scope(buffer_bounds, op->sym, bounds);
    node_mutator::visit(op);
  }

  // TODO: Need to handle this?
  void visit(const slice_buffer*) override { std::abort(); }
  void visit(const slice_dim*) override { std::abort(); }
  void visit(const truncate_rank*) override { std::abort(); }

  // We need a better way to do this. `check` doesn't have a scope, so we need to be careful to only learn information
  // about buffers within the scope the check has been guaranteed to have run in. For now, only bother with checks at
  // the global scope.
  bool at_root_scope = true;

  void visit(const check* op) override {
    set_result(op);
    if (!at_root_scope) return;
    if (const equal* eq = op->condition.as<equal>()) {
      if (const call* op = eq->a.as<call>()) {
        if (op->intrinsic == intrinsic::buffer_elem_size) {
          const symbol_id* buf = as_variable(op->args[0]);
          const index_t* value = as_constant(eq->b);
          if (buf && value) {
            elem_sizes[*buf] = *value;
          }
        }
      }
    }
  }

  void visit(const block* op) override {
    std::vector<stmt> stmts;
    stmts.reserve(op->stmts.size());
    bool changed = false;
    for (const stmt& s : op->stmts) {
      if (!s.as<check>() && !s.as<block>()) {
        at_root_scope = false;
      }
      stmts.push_back(mutate(s));
      changed = changed || !stmts.back().same_as(s);
    }
    if (!changed) {
      set_result(op);
    } else {
      set_result(block::make(std::move(stmts)));
    }
  }
};

}  // namespace

stmt alias_buffers(const stmt& s) { return buffer_aliaser().mutate(s); }

namespace {

class copy_optimizer : public node_mutator {
  node_context& ctx;

public:
  copy_optimizer(node_context& ctx) : ctx(ctx) {}

  void visit(const copy_stmt* op) override {
    // Start by making a call to copy.
    stmt result = call_stmt::make(
        [src = op->src, dst = op->dst, padding = op->padding](const eval_context& ctx) -> index_t {
          const raw_buffer* src_buf = ctx.lookup_buffer(src);
          const raw_buffer* dst_buf = ctx.lookup_buffer(dst);
          copy(*src_buf, *dst_buf, padding.empty() ? nullptr : padding.data());
          return 0;
        },
        {op->src}, {op->dst});

    var src_var(op->src);
    var dst_var(op->dst);

    std::vector<expr> src_x = op->src_x;
    std::vector<dim_expr> src_dims;
    std::vector<std::pair<symbol_id, int>> dst_x;

    // If we just leave these two arrays alone, the copy will be correct, but slow.
    // We can speed it up by finding dimensions we can let pass through to the copy.
    for (int d = 0; d < static_cast<int>(op->dst_x.size()); ++d) {
      int dep_count = 0;
      int src_d = -1;
      for (int sd = 0; sd < static_cast<int>(src_x.size()); ++sd) {
        if (depends_on(src_x[sd], op->dst_x[d]).any()) {
          ++dep_count;
          src_d = sd;
        }
      }
      bool handled = false;
      if (dep_count == 0) {
        // This dimension is a broadcast. To handle this, we're going to add a dummy dimension to the input.
        // We can just always do this, regardless of whether this broadcast is implicit (the input has fewer
        // dimensions than the output) or not.
        src_dims.push_back({buffer_bounds(dst_var, d), 0, expr()});
        handled = true;
      } else if (dep_count == 1) {
        expr offset;
        if (is_copy(src_x[src_d], op->dst_x[d], offset)) {
          interval_expr dst_bounds = buffer_bounds(dst_var, d);
          interval_expr src_bounds = buffer_bounds(src_var, src_d) - offset;
          src_dims.push_back(
              {dst_bounds & src_bounds, buffer_stride(src_var, src_d), buffer_fold_factor(src_var, src_d)});
          src_x[src_d] = max(buffer_min(dst_var, d) + offset, buffer_min(src_var, src_d));
          handled = true;
        }
      }
      if (!handled) {
        dst_x.emplace_back(op->dst_x[d], d);
      }
    }

    // TODO: Try to optimize reshapes, where the index of the input is an "unpacking" of a flat index of the output.
    // This will require the simplifier to understand the constraints implied by the checks on the buffer metadata
    // at the beginning of the pipeline, e.g. that buffer_stride(dst_var, d) == buffer_stride(dst_var, d - 1) *
    // buffer_extent(dst_var, d - 1).

    // Rewrite the source buffer to be only the dimensions of the src we want to pass to copy.
    result = make_buffer::make(op->src, buffer_at(src_var, src_x), buffer_elem_size(src_var), src_dims, result);

    // Any dimensions left need loops and slices.
    // We're going to make slices here, which invalidates buffer metadata calls in the body. To avoid breaking
    // the body, we'll make lets of the buffer metadata outside the loops.
    // TODO: Is this really the right thing to do, or is it an artifact of a bad idea/implementation?
    std::vector<std::pair<symbol_id, expr>> lets;
    symbol_id let_id = ctx.insert_unique();
    auto do_substitute = [&](const expr& value) {
      stmt new_result = substitute(result, value, variable::make(let_id));
      if (!new_result.same_as(result)) {
        lets.push_back({let_id, value});
        let_id = ctx.insert_unique();
        result = std::move(new_result);
      }
    };
    for (int d = 0; d < static_cast<index_t>(op->dst_x.size()); ++d) {
      do_substitute(buffer_min(dst_var, d));
      do_substitute(buffer_max(dst_var, d));
      do_substitute(buffer_extent(dst_var, d));
      do_substitute(buffer_stride(dst_var, d));
      do_substitute(buffer_fold_factor(dst_var, d));
    }

    for (const std::pair<symbol_id, int>& d : dst_x) {
      result = slice_dim::make(op->dst, d.second, var(d.first), result);
      result = loop::make(d.first, loop_mode::serial, buffer_bounds(dst_var, d.second), 1, result);
    }
    for (const auto& i : lets) {
      result = let_stmt::make(i.first, i.second, result);
    }

    set_result(result);
  }
};

}  // namespace

stmt optimize_copies(const stmt& s, node_context& ctx) { return copy_optimizer(ctx).mutate(s); }

namespace {

// Split `body` into 3 parts:
// - stmts that don't depend on `vars`
// - stmts that do depend on `vars`
// - stmts that don't depend on `vars`
std::tuple<stmt, stmt, stmt> split_body(const stmt& body, span<const symbol_id> vars) {
  if (const block* b = body.as<block>()) {
    const auto depends_on_stmt = [&](const stmt& s) { return depends_on(s, vars).any(); };
    auto end_before = std::find_if(b->stmts.begin(), b->stmts.end(), depends_on_stmt);
    if (end_before != b->stmts.end()) {
      std::vector<stmt> before = {b->stmts.begin(), end_before};
      auto end_body = std::find_if(b->stmts.rbegin(), b->stmts.rend(), depends_on_stmt).base();
      std::vector<stmt> new_body = {end_before, end_body};
      std::vector<stmt> after = {end_body, b->stmts.end()};
      return {block::make(std::move(before)), block::make(std::move(new_body)), block::make(std::move(after))};
    } else {
      return {body, stmt{}, stmt{}};
    }
  } else if (depends_on(body, vars).any()) {
    return {stmt{}, body, stmt{}};
  } else {
    return {body, stmt{}, stmt{}};
  }
}

std::tuple<stmt, stmt, stmt> split_body(const stmt& body, symbol_id var) {
  symbol_id vars[] = {var};
  return split_body(body, vars);
}

class scope_reducer : public node_mutator {
public:
  template <typename T>
  void visit_stmt(const T* op) {
    stmt body = mutate(op->body);

    stmt before, new_body, after;
    std::tie(before, new_body, after) = split_body(body, op->sym);

    if (body.same_as(op->body) && !before.defined() && !after.defined()) {
      set_result(op);
    } else if (new_body.defined()) {
      stmt result = clone_with_new_body(op, std::move(new_body));
      set_result(block::make({std::move(before), std::move(result), std::move(after)}));
    } else {
      // The op was dead...?
      set_result(block::make({std::move(before), std::move(after)}));
    }
  }

  void visit(const let_stmt* op) override { visit_stmt(op); }
  void visit(const allocate* op) override { visit_stmt(op); }
  void visit(const make_buffer* op) override { visit_stmt(op); }
  void visit(const clone_buffer* op) override { visit_stmt(op); }
  void visit(const crop_buffer* op) override { visit_stmt(op); }
  void visit(const crop_dim* op) override { visit_stmt(op); }
  void visit(const slice_buffer* op) override { visit_stmt(op); }
  void visit(const slice_dim* op) override { visit_stmt(op); }
  void visit(const truncate_rank* op) override { visit_stmt(op); }
};

}  // namespace

stmt reduce_scopes(const stmt& s) { return scope_reducer().mutate(s); }

namespace {

class race_condition_fixer : public node_mutator {
  symbol_map<bool> mutated;

public:
  void visit(const loop* op) override {
    if (op->mode != loop_mode::parallel) {
      node_mutator::visit(op);
      return;
    }

    // We've hit a parallel loop. The buffers that are allocated outside this loop, but mutated inside this loop, will
    // be true in the mutated map. We need to make copies of these buffers upon entering the loop.
    stmt body = mutate(op->body);
    for (symbol_id i = 0; i < mutated.size(); ++i) {
      if (mutated[i] && *mutated[i]) {
        body = clone_buffer::make(i, i, body);
      }
    }
    if (body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(loop::make(op->sym, op->mode, op->bounds, op->step, std::move(body)));
    }
  }

  template <typename T>
  void visit_buffer_allocator(const T* op) {
    // Buffers start out not mutated.
    auto s = set_value_in_scope(mutated, op->sym, false);
    node_mutator::visit(op);
  }

  void visit(const allocate* op) override { visit_buffer_allocator(op); }
  void visit(const make_buffer* op) override { visit_buffer_allocator(op); }
  void visit(const clone_buffer* op) override { visit_buffer_allocator(op); }

  template <typename T>
  void visit_buffer_mutator(const T* op) {
    mutated[op->sym] = true;
    node_mutator::visit(op);
  }

  void visit(const crop_buffer* op) override { visit_buffer_mutator(op); }
  void visit(const crop_dim* op) override { visit_buffer_mutator(op); }
  void visit(const slice_buffer* op) override { visit_buffer_mutator(op); }
  void visit(const slice_dim* op) override { visit_buffer_mutator(op); }
  void visit(const truncate_rank* op) override { visit_buffer_mutator(op); }
};

}  // namespace

stmt fix_buffer_races(const stmt& s) { return race_condition_fixer().mutate(s); }

}  // namespace slinky
