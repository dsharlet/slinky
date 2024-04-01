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
#include "builder/simplify.h"
#include "builder/substitute.h"
#include "runtime/buffer.h"
#include "runtime/depends_on.h"
#include "runtime/evaluate.h"
#include "runtime/expr.h"
#include "runtime/util.h"

namespace slinky {

namespace {

// Checks if the copy operands `src_x` and `dst_x` represent a simple copy that can be handled by slinky::copy.
bool is_copy(expr src_x, var dst_x, expr& offset) {
  offset = simplify(src_x - dst_x);
  return !depends_on(offset, dst_x.sym()).any();
}

// Same as above, applied to each dimension of the copy.
bool is_copy(const copy_stmt* op, std::vector<expr>& offset) {
  if (op->src_x.size() != op->dst_x.size()) return false;
  offset.resize(op->dst_x.size());
  for (std::size_t d = 0; d < op->dst_x.size(); ++d) {
    if (!is_copy(op->src_x[d], op->dst_x[d], offset[d])) return false;
  }
  return true;
}

class buffer_aliaser : public node_mutator {
  struct buffer_alias {
    std::vector<expr> offset;
  };

  class buffer_info {
  public:
    std::map<symbol_id, buffer_alias> can_alias_;
    std::set<symbol_id> cannot_alias_;

  public:
    std::map<symbol_id, buffer_alias>& can_alias() { return can_alias_; }
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
  symbol_map<bool> do_not_alias;

public:
  void visit(const allocate* op) override {
    bool do_not_alias_sym = false;
    for (const dim_expr& d : op->dims) {
      if (d.fold_factor.defined()) {
        // This buffer can't be aliased.
        do_not_alias_sym = true;
      }
    }
    auto set_do_not_alias = set_value_in_scope(do_not_alias, op->sym, do_not_alias_sym);

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
      at.resize(std::max(at.size(), dims.size()));
      for (int d = 0; d < static_cast<int>(at.size()); ++d) {
        if (!at[d].defined()) at[d] = 0;
        if (d < static_cast<int>(op->dims.size())) {
          at[d] = max(buffer_min(target_var, d) - at[d], op->dims[d].bounds.min);
          dims[d].bounds &= op->dims[d].bounds;
        }
      }
      stmt result = make_buffer::make(
          op->sym, buffer_at(target_var, at), op->elem_size, std::move(dims), std::move(body));
      // If we aliased the source and destination of a copy, replace the copy with a pad.
      stmt pad_result = recursive_mutate<copy_stmt>(result, [src = op->sym, dst = target.first](const copy_stmt* op) {
        if (op->src != src || op->dst != dst) {
          // Not this copy.
          return stmt(op);
        }
        if (!op->padding || op->padding->empty()) {
          // No padding, this copy is now a no-op.
          return stmt();
        }
        // Make a call to `pad`.
        return call_stmt::make(
            [padding = *op->padding](const call_stmt* op, const eval_context& ctx) -> index_t {
              const raw_buffer* src_buf = ctx.lookup_buffer(op->inputs[0]);
              const raw_buffer* dst_buf = ctx.lookup_buffer(op->outputs[0]);
              ctx.pad(src_buf->dims, *dst_buf, padding.data());
              return 0;
            },
            {src}, {dst}, {});
      });

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
      std::optional<bool> no_alias = do_not_alias[o];
      if (no_alias && *no_alias) {
        continue;
      }

      for (symbol_id i : op->inputs) {
        std::optional<buffer_info>& info = alias_info[i];
        if (!info) continue;

        if (!op->attrs.allow_in_place) {
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

  void merge_alias_info(symbol_map<buffer_info> add) {
    for (symbol_id i = 0; i < add.size(); ++i) {
      if (!add[i]) continue;
      std::optional<buffer_info>& info = alias_info[i];
      if (!info) {
        info = std::move(add[i]);
      } else {
        for (auto& j : add[i]->can_alias()) {
          info->maybe_alias(j.first, std::move(j.second));
        }
      }
    }
  }

  void visit(const slice_buffer* op) override {
    // We need to know which alias candidates are added inside this slice.
    symbol_map<buffer_info> old_alias_info;
    std::swap(old_alias_info, alias_info);
    for (symbol_id i = 0; i < old_alias_info.size(); ++i) {
      if (old_alias_info[i]) {
        alias_info[i] = buffer_info();
      }
    }

    node_mutator::visit(op);

    // If we chose to alias this buffer, we need to insert offsets for where we sliced it.
    for (std::optional<buffer_info>& i : alias_info) {
      if (!i) continue;
      auto j = i->can_alias().find(op->sym);
      if (j != i->can_alias().end()) {
        std::vector<expr>& offset = j->second.offset;
        for (std::size_t d = 0; d < op->at.size(); ++d) {
          if (!op->at[d].defined()) continue;
          offset.insert(offset.begin() + d, op->at[d]);
        }
      }
    }

    // Add the old alias candidates back to the alias info.
    merge_alias_info(std::move(old_alias_info));
  }

  void visit(const slice_dim* op) override {
    // We need to know which alias candidates are added inside this slice.
    symbol_map<buffer_info> old_alias_info;
    std::swap(old_alias_info, alias_info);
    for (symbol_id i = 0; i < old_alias_info.size(); ++i) {
      if (old_alias_info[i]) {
        alias_info[i] = buffer_info();
      }
    }

    node_mutator::visit(op);

    // If we chose to alias this buffer, we need to insert offsets for where we sliced it.
    for (std::optional<buffer_info>& i : alias_info) {
      if (!i) continue;
      auto j = i->can_alias().find(op->sym);
      if (j != i->can_alias().end()) {
        std::vector<expr>& offset = j->second.offset;
        offset.insert(offset.begin() + op->dim, op->at);
      }
    }

    // Add the old alias candidates back to the alias info.
    merge_alias_info(std::move(old_alias_info));
  }

  void visit(const clone_buffer* op) override {
    auto set_info_sym = set_value_in_scope(alias_info, op->sym, alias_info[op->src]);
    node_mutator::visit(op);

    // Alias candidates for op->sym are also alias candidates for op->src.
    std::optional<buffer_info> sym_info = std::move(alias_info[op->sym]);
    if (sym_info) {
      std::optional<buffer_info>& src_info = alias_info[op->src];
      if (!src_info) {
        src_info = std::move(sym_info);
      } else {
        for (auto& j : sym_info->can_alias()) {
          src_info->maybe_alias(j.first, std::move(j.second));
        }
      }
    }
  }

  void visit(const truncate_rank*) override { std::abort(); }
};

}  // namespace

stmt alias_buffers(const stmt& s) { return buffer_aliaser().mutate(s); }

stmt implement_copy(const copy_stmt* op, node_context& ctx) {
  // Start by making a call to copy.
  stmt result = call_stmt::make(
      [padding = op->padding](const call_stmt* op, const eval_context& ctx) -> index_t {
        const raw_buffer* src_buf = ctx.lookup_buffer(op->inputs[0]);
        const raw_buffer* dst_buf = ctx.lookup_buffer(op->outputs[0]);
        const void* pad_value = (!padding || padding->empty()) ? nullptr : padding->data();
        ctx.copy(*src_buf, *dst_buf, pad_value);
        return 0;
      },
      {op->src}, {op->dst}, {});

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
    result = loop::make(d.first, loop::serial, buffer_bounds(dst_var, d.second), 1, result);
  }
  return let_stmt::make(std::move(lets), result);
}

stmt implement_copies(const stmt& s, node_context& ctx) {
  return recursive_mutate<copy_stmt>(s, [&](const copy_stmt* op) { return implement_copy(op, ctx); });
}

namespace {

class race_condition_fixer : public node_mutator {
  symbol_map<bool> mutated;

public:
  void visit(const loop* op) override {
    if (op->is_serial()) {
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
      set_result(loop::make(op->sym, op->max_workers, op->bounds, op->step, std::move(body)));
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
