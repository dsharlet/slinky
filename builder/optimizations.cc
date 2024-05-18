#include "builder/optimizations.h"

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <map>
#include <numeric>
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

namespace slinky {

namespace {

dim_expr select(const expr& c, dim_expr t, dim_expr f) {
  return {
      select(c, std::move(t.bounds), std::move(f.bounds)),
      select(c, std::move(t.stride), std::move(f.stride)),
      select(c, std::move(t.fold_factor), std::move(f.fold_factor)),
  };
}

// Checks if the copy operands `src_x` and `dst_x` represent a simple copy that can be handled by slinky::copy.
bool is_copy(var src, expr src_x, int src_d, var dst, var dst_x, int dst_d, expr& at, dim_expr& src_dim) {
  if (const class select* s = src_x.as<class select>()) {
    // The src is a select of two things that might both be copies.
    expr at_t = at;
    expr at_f = at;
    dim_expr src_dim_t = src_dim;
    dim_expr src_dim_f = src_dim;
    if (is_copy(src, s->true_value, src_d, dst, dst_x, dst_d, at_t, src_dim_t) &&
        is_copy(src, s->false_value, src_d, dst, dst_x, dst_d, at_f, src_dim_f)) {
      at = select(s->condition, at_t, at_f);
      src_dim = select(s->condition, src_dim_t, src_dim_f);
      return true;
    } else {
      return false;
    }
  } else if (!depends_on(src_x, dst_x).any()) {
    // This is a broadcast because the src_x is constant w.r.t. dst_x.
    at = src_x;
    src_dim.bounds = buffer_bounds(dst, dst_d);
    src_dim.stride = 0;
    src_dim.fold_factor = dim::unfolded;
    return true;
  } else {
    // Try to parse src_x = dst_x * scale + offset
    expr scale = 1;
    if (const class mul* s = src_x.as<class mul>()) {
      if (!depends_on(s->a, dst_x).any()) {
        scale = s->a;
        src_x = s->b;
      } else if (!depends_on(s->b, dst_x).any()) {
        scale = s->b;
        src_x = s->a;
      } else {
        return false;
      }
    }

    expr offset = simplify((src_x - dst_x) * scale);
    if (depends_on(offset, dst_x).any()) {
      // We don't understand this src_x as a copy.
      return false;
    }

    src_dim.bounds = (buffer_bounds(src, src_d) - offset) / scale;
    src_dim.stride = buffer_stride(src, src_d) * scale;
    src_dim.fold_factor = buffer_fold_factor(src, src_d);
    at = buffer_min(src, src_d) + offset * (scale - 1);

    // Alternative definitions that may be useful in the future and were difficult to determine:
    // src_dim.bounds = buffer_bounds(dst, dst_d);
    // at = buffer_min(dst, dst_d) * scale + offset;

    return true;
  }
}

bool is_copy(const copy_stmt* op, int src_d, int dst_d, expr& at, dim_expr& src_dim) {
  // We might not have an src dim if we're trying to broadcast.
  expr src_x = src_d >= 0 ? op->src_x[src_d] : expr();
  return is_copy(op->src, src_x, src_d, op->dst, op->dst_x[dst_d], dst_d, at, src_dim);
}

// `dst_d` may be a copy dim of `op` if it is used by exactly one src dim, where it might be a copy, or zero src dims,
// where it is a broadcast.
bool is_copy_dst_dim(const copy_stmt* op, int dst_d, int& src_d) {
  src_d = -1;
  for (int i = 0; i < static_cast<int>(op->src_x.size()); ++i) {
    if (depends_on(op->src_x[i], op->dst_x[dst_d]).any()) {
      if (src_d == -1) {
        src_d = i;
      } else {
        // dst_x[dst_d] is used by more than one src, we can't handle it with a copy.
        return false;
      }
    }
  }
  return true;
}

std::vector<expr> buffer_mins(var buf, std::size_t rank) {
  std::vector<expr> result(rank);
  for (int i = 0; i < static_cast<int>(rank); ++i) {
    result[i] = buffer_min(buf, i);
  }
  return result;
}

stmt replace_copy_with_pad(const stmt& s, var a, var b, const std::vector<int>& permutation) {
  return recursive_mutate<copy_stmt>(s, [a, b, &permutation](const copy_stmt* op) {
    if (!((op->src == a && op->dst == b) || (op->src == b && op->dst == a))) {
      // Not this copy.
      return stmt(op);
    }
    if (!op->padding || op->padding->empty()) {
      // No padding, this copy is now a no-op.
      return stmt();
    }

    // Make a call to `pad`.
    call_stmt::attributes pad_attrs;
    pad_attrs.name = "pad";
    stmt pad = call_stmt::make(
        [padding = *op->padding](const call_stmt* op, const eval_context& ctx) -> index_t {
          // TODO: This passes the src buffer as an output, not an input, because slinky thinks the bounds of inputs
          // don't matter. But in this case, they do...
          const raw_buffer* src_buf = ctx.lookup_buffer(op->outputs[0]);
          const raw_buffer* dst_buf = ctx.lookup_buffer(op->outputs[1]);
          ctx.pad(src_buf->dims, *dst_buf, padding.data());
          return 0;
        },
        {}, {op->src, op->dst}, std::move(pad_attrs));

    // The copy may have also be transposed.
    return transpose::make(op->dst, op->dst, permutation, pad);
  });
}

class buffer_aliaser : public node_mutator {
  node_context& ctx;

  struct buffer_alias {
    std::vector<dim_expr> dims;
    std::vector<int> permutation;
    std::vector<expr> at;
    bool assume_in_bounds = false;
  };

  class buffer_info {
  public:
    std::vector<dim_expr> dims;
    std::map<var, buffer_alias> can_alias_;
    std::set<var> cannot_alias_;

    // If we decided to alias this buffer, we might have grown the bounds. If so, we need to make a new allocation with
    // this symbol, but make a crop of it for the original bounds.
    var shared_alloc_sym;

  public:
    buffer_info(std::vector<dim_expr> dims) : dims(std::move(dims)) {}

    std::map<var, buffer_alias>& can_alias() { return can_alias_; }
    const std::map<var, buffer_alias>& can_alias() const { return can_alias_; }

    void maybe_alias(var s, buffer_alias a) {
      if (cannot_alias_.count(s)) {
        return;
      }

      can_alias_[s] = std::move(a);
    }

    void do_not_alias(var s) {
      can_alias_.erase(s);
      cannot_alias_.insert(s);
    }
  };
  symbol_map<buffer_info> alloc_info;
  symbol_map<bool> do_not_alias;

  // We need to map clones back to their original allocations.
  symbol_map<var> alloc_map;

  std::optional<buffer_info>& lookup_alloc(var x) {
    const std::optional<var>& mapped = alloc_map[x];
    if (mapped) {
      return lookup_alloc(*mapped);
    } else {
      return alloc_info[x];
    }
  }

  static bool alias_compatible(
      const allocate* op, const buffer_alias& alias, const std::optional<buffer_info>& target_info) {
    assert(op->dims.size() == alias.dims.size());
    for (std::size_t d = 0; d < op->dims.size(); ++d) {
      if (!alias.assume_in_bounds) {
        assert(alias.permutation.size() == op->dims.size());
        if (!prove_true(op->dims[d].bounds.min >= alias.dims[alias.permutation[d]].bounds.min) ||
            !prove_true(op->dims[d].bounds.max <= alias.dims[alias.permutation[d]].bounds.max)) {
          // We don't know if this target is big enough for this allocation.
          if (!target_info) {
            // We didn't allocate this buffer, we can't grow it to use this buffer as an alias target.
            return false;
          }
        }
      }
      if (op->dims[d].stride.defined()) {
        if (!prove_true(op->dims[d].stride == alias.dims[alias.permutation[d]].stride)) {
          // This alias would violate a constraint on the stride of the buffer.
          return false;
        }
      }
      if (op->dims[d].fold_factor.defined()) {
        if (target_info && target_info->dims[alias.permutation[d]].fold_factor.defined()) {
          // This alias is not compatible because of fold factors.
          // TODO: We should try to relax this constraint.
          return false;
        }
      }
    }
    return true;
  }

public:
  buffer_aliaser(node_context& ctx) : ctx(ctx) {}

  void visit(const allocate* op) override {
    bool do_not_alias_sym = false;
    for (const dim_expr& d : op->dims) {
      if (d.fold_factor.defined()) {
        // This buffer can't be aliased.
        do_not_alias_sym = true;
      }
    }
    auto set_do_not_alias = set_value_in_scope(do_not_alias, op->sym, do_not_alias_sym);

    auto s = set_value_in_scope(alloc_info, op->sym, buffer_info(op->dims));
    stmt body = mutate(op->body);
    buffer_info info = std::move(*alloc_info[op->sym]);

    // When an allocation goes out of scope, we should remove it as an aliasing candidate.
    for (std::optional<buffer_info>& i : alloc_info) {
      if (i) i->do_not_alias(op->sym);
    }

    box_expr op_dims_bounds = dims_bounds(op->dims);
    for (auto& target : info.can_alias()) {
      var target_var = target.first;
      buffer_alias& alias = target.second;

      var alloc_var = target_var;
      std::optional<buffer_info>& target_info = lookup_alloc(target_var);
      if (!alias_compatible(op, alias, target_info)) {
        continue;
      }

      // The alias might have used the bounds of this symbol, substitute them now.
      for (dim_expr& i : alias.dims) {
        i.bounds.min = substitute_bounds(i.bounds.min, op->sym, op_dims_bounds);
        i.bounds.max = substitute_bounds(i.bounds.max, op->sym, op_dims_bounds);
      }
      for (expr& i : alias.at) {
        i = substitute_bounds(i, op->sym, op_dims_bounds);
      }

      if (!alias.assume_in_bounds) {
        if (target_info) {
          // We allocated this buffer, make it big enough to share with this buffer.
          if (!target_info->shared_alloc_sym.defined()) {
            target_info->shared_alloc_sym = ctx.insert_unique(ctx.name(target_var) + "/" + ctx.name(op->sym));
            alloc_var = target_info->shared_alloc_sym;
          }
          for (std::size_t d = 0; d < op->dims.size(); ++d) {
            // TODO: We may have proven this is unnecessary in alias_compatible, we can avoid this in such cases.
            target_info->dims[d].bounds |= alias.dims[alias.permutation[d]].bounds;
          }
        } else {
          // In this case, alias_compatible must have determined that we do not need to grow the allocation.
        }
      }

      // Replace the allocation with a buffer using the dims the alias wants.
      stmt result =
          make_buffer::make(op->sym, buffer_at(alloc_var, alias.at), op->elem_size, alias.dims, std::move(body));
      // If we aliased the source and destination of a copy, replace the copy with a pad.
      stmt pad_result = replace_copy_with_pad(result, op->sym, target_var, alias.permutation);
      if (pad_result.same_as(result)) {
        // This wasn't a copy, we actually did some computation in place. We can't alias another buffer to this target
        // without understanding the lifetimes more carefully.
        // TODO: I think this is a hack, but I'm not sure. I think maybe the proper thing to do is track a box_expr
        // of the region that has been aliased so far, and allow another alias as long as it does not intersect that
        // region. That will likely be very difficult to do symbolically.
        for (std::optional<buffer_info>& i : alloc_info) {
          if (!i) continue;
          i->do_not_alias(target_var);
        }
      }
      // We may attempt to alias this both ways (src -> dst and dst -> src), we only want to do one of them.
      // TODO: Is this possible? The inner one should have gone out of scope.
      if (target_info) {
        target_info->do_not_alias(op->sym);
      }
      set_result(pad_result);
      return;
    }
    if (!body.same_as(op->body)) {
      if (info.shared_alloc_sym.defined()) {
        // This allocation's bounds were expanded to accommodate aliases. Make a new expanded allocation, and make the
        // original allocation a crop of the expanded allocation.
        stmt result = crop_buffer::make(op->sym, info.shared_alloc_sym, std::move(op_dims_bounds), std::move(body));
        result =
            allocate::make(info.shared_alloc_sym, op->storage, op->elem_size, std::move(info.dims), std::move(result));
        set_result(result);
      } else {
        set_result(clone_with_new_body(op, std::move(body)));
      }
    } else {
      set_result(op);
    }
  }

  bool can_alias(var x) {
    std::optional<bool> no_alias = do_not_alias[x];
    return !no_alias || !*no_alias;
  }

  void visit(const call_stmt* op) override {
    set_result(op);
    if (!op->attrs.allow_in_place) {
      // This call does not allow aliasing an input to an output.
      return;
    }
    for (var o : op->outputs) {
      if (!can_alias(o)) continue;
      for (var i : op->inputs) {
        std::optional<buffer_info>& input_info = lookup_alloc(i);
        if (input_info) {
          buffer_alias a;
          a.dims = buffer_dims(o, input_info->dims.size());
          a.at = buffer_mins(o, input_info->dims.size());
          // We assume that op->attrs.allow_in_place means that the input is in bounds of the entire output, and the
          // dimensions are not permuted.
          a.assume_in_bounds = true;
          a.permutation.resize(input_info->dims.size());
          std::iota(a.permutation.begin(), a.permutation.end(), 0);
          input_info->maybe_alias(o, std::move(a));
        }
      }
    }
  }

  void alias_copy_dst(const copy_stmt* op) {
    if (!lookup_alloc(op->dst) || !can_alias(op->src)) {
      // We didn't allocate the dst.
      return;
    }

    // We allocated the dst. We might be able to replace the allocation with an alias of the src.
    // This case is a straightforward use of is_copy, which produces the dims that should be the src of a copy, which
    // are the same dimensions we want the dst to be.
    std::optional<buffer_info>& info = lookup_alloc(op->dst);

    buffer_alias a;
    a.at.resize(op->src_x.size());
    a.permutation.resize(op->dst_x.size());
    a.dims = info->dims;
    for (int dst_d = 0; dst_d < static_cast<int>(op->dst_x.size()); ++dst_d) {
      int src_d;
      if (!is_copy_dst_dim(op, dst_d, src_d)) {
        return;
      }

      dim_expr src_dim;
      expr at;
      if (!is_copy(op, src_d, dst_d, at, src_dim)) {
        return;
      }

      // We want the bounds of the original dst dimension, but the memory layout of the src dimension. This may require
      // the allocation to be expanded to accommodate this alias.
      a.dims[dst_d] = {buffer_bounds(op->dst, dst_d), src_dim.stride, src_dim.fold_factor};
      a.permutation[dst_d] = src_d;
      if (at.defined()) {
        a.at[src_d] = at - src_dim.bounds.min + a.dims[dst_d].bounds.min;
      }
    }

    // If there is no padding, we can assume that the src is always in bounds of dst.
    a.assume_in_bounds = !op->padding || op->padding->empty();
    info->maybe_alias(op->src, std::move(a));
  }

  void alias_copy_src(const copy_stmt* op) {
    if (!lookup_alloc(op->src) || !can_alias(op->dst)) {
      // We didn't allocate the src.
      return;
    }

    // We allocated the src. We might be able to replace the allocation with an alias of the dst.
    // In this case, we're going to make the src an alias of another buffer. We're more limited in what we can do here
    // vs. the above case, because we can't expect producers to handle everything the copy is doing (such as
    // broadcasting).
    std::optional<buffer_info>& info = lookup_alloc(op->src);

    buffer_alias a;
    a.at.resize(op->dst_x.size());
    a.dims.resize(op->src_x.size());
    assert(op->src_x.size() == info->dims.size());
    for (int dst_d = 0; dst_d < static_cast<int>(op->dst_x.size()); ++dst_d) {
      int src_d;
      if (!is_copy_dst_dim(op, dst_d, src_d)) {
        return;
      }

      if (src_d < 0) {
        // We can't handle a broadcast here, because we can't ask the producer of our src to produce more dimensions.
        return;
      }

      expr offset = simplify(op->src_x[src_d] - op->dst_x[dst_d]);
      if (depends_on(offset, op->dst_x[dst_d]).any()) {
        // This is not a simple copy, we can't handle it here.
        return;
      }

      // We want the bounds of the src buffer, but with dst's memory layout.
      a.dims[src_d] = {
          info->dims[src_d].bounds,
          buffer_stride(op->dst, dst_d),
          buffer_fold_factor(op->dst, dst_d),
      };
      a.at[dst_d] = info->dims[src_d].bounds.min - offset;
    }

    for (const dim_expr& d : a.dims) {
      if (!d.stride.defined()) {
        // We didn't define all the dimensions of the buffer we want to replace.
        return;
      }
    }

    // In this case, we just want an identity permutation, because the alias dims are already in the src order.
    a.permutation.resize(op->dst_x.size());
    std::iota(a.permutation.begin(), a.permutation.end(), 0);

    info->maybe_alias(op->dst, std::move(a));
  }

  void visit(const copy_stmt* op) override {
    set_result(op);

    alias_copy_dst(op);
    alias_copy_src(op);
  }

  template <typename T, typename Fn>
  void visit_buffer_mutator(const T* op, Fn&& handler) {
    // We need to know which alias candidates are added inside this mutator.
    symbol_map<buffer_info> old_alloc_info(alloc_info.size());
    std::swap(old_alloc_info, alloc_info);
    for (std::size_t i = 0; i < old_alloc_info.size(); ++i) {
      if (old_alloc_info[i]) {
        alloc_info[i] = buffer_info(old_alloc_info[i]->dims);
      }
    }

    auto set_info_sym = set_value_in_scope(alloc_info, op->sym, lookup_alloc(op->src));
    node_mutator::visit(op);

    for (std::optional<buffer_info>& i : alloc_info) {
      if (!i) continue;
      auto j = i->can_alias().find(op->sym);
      if (j != i->can_alias().end()) {
        handler(j->second);
      }
    }

    // Add the old alias candidates back to the alias info.
    alloc_info.reserve(std::max(alloc_info.size(), old_alloc_info.size()));
    for (std::size_t i = 0; i < old_alloc_info.size(); ++i) {
      if (!old_alloc_info[i]) continue;
      std::optional<buffer_info>& info = alloc_info[i];
      if (!info) {
        info = std::move(old_alloc_info[i]);
      } else {
        for (auto& j : old_alloc_info[i]->can_alias()) {
          info->maybe_alias(j.first, std::move(j.second));
        }
      }
    }
  }

  void visit(const slice_buffer* op) override {
    visit_buffer_mutator(op, [=](buffer_alias& alias) {
      for (std::size_t d = 0; d < op->at.size(); ++d) {
        if (!op->at[d].defined()) continue;
        alias.at.insert(alias.at.begin() + d, op->at[d]);
      }
    });
  }

  void visit(const slice_dim* op) override {
    visit_buffer_mutator(op, [=](buffer_alias& alias) { alias.at.insert(alias.at.begin() + op->dim, op->at); });
  }

  void visit(const clone_buffer* op) override {
    auto set_mapped_sym = set_value_in_scope(alloc_map, op->sym, op->src);
    node_mutator::visit(op);
  }

  void visit(const transpose*) override { std::abort(); }
};

}  // namespace

stmt alias_buffers(const stmt& s, node_context& ctx) { return buffer_aliaser(ctx).mutate(s); }

stmt implement_copy(const copy_stmt* op, node_context& ctx) {
  // Start by making a call to copy.
  call_stmt::attributes copy_attrs;
  copy_attrs.name = "copy";
  stmt result = call_stmt::make(
      [padding = op->padding](const call_stmt* op, const eval_context& ctx) -> index_t {
        const raw_buffer* src_buf = ctx.lookup_buffer(op->inputs[0]);
        const raw_buffer* dst_buf = ctx.lookup_buffer(op->outputs[0]);
        const void* pad_value = (!padding || padding->empty()) ? nullptr : padding->data();
        ctx.copy(*src_buf, *dst_buf, pad_value);
        return 0;
      },
      {op->src}, {op->dst}, std::move(copy_attrs));

  std::vector<expr> src_x = op->src_x;
  std::vector<var> dst_x = op->dst_x;
  std::vector<dim_expr> src_dims;

  // If we just leave these two arrays alone, the copy will be correct, but slow.
  // We can speed it up by finding dimensions we can let pass through to the copy.
  for (int dst_d = 0; dst_d < static_cast<int>(dst_x.size()); ++dst_d) {
    int src_d;
    if (!is_copy_dst_dim(op, dst_d, src_d)) {
      continue;
    }

    dim_expr src_dim;
    expr at;
    if (is_copy(op, src_d, dst_d, at, src_dim)) {
      src_dims.push_back(src_dim);
      if (at.defined()) {
        src_x[src_d] = at;
      }
      dst_x[dst_d] = var();
    }
  }

  // TODO: Try to optimize reshapes, where the index of the input is an "unpacking" of a flat index of the output.
  // This will require the simplifier to understand the constraints implied by the checks on the buffer metadata
  // at the beginning of the pipeline, e.g. that buffer_stride(op->dst, d) == buffer_stride(op->dst, d - 1) *
  // buffer_extent(op->dst, d - 1).

  // Rewrite the source buffer to be only the dimensions of the src we want to pass to copy.
  result = make_buffer::make(op->src, buffer_at(op->src, src_x), buffer_elem_size(op->src), src_dims, result);

  // Any dimensions left need loops and slices.
  // We're going to make slices here, which invalidates buffer metadata calls in the body. To avoid breaking
  // the body, we'll make lets of the buffer metadata outside the loops.
  // TODO: Is this really the right thing to do, or is it an artifact of a bad idea/implementation?
  std::vector<std::pair<var, expr>> lets;
  var let_id = ctx.insert_unique();
  auto do_substitute = [&](const expr& value) {
    stmt new_result = substitute(result, value, variable::make(let_id));
    if (!new_result.same_as(result)) {
      lets.push_back({let_id, value});
      let_id = ctx.insert_unique();
      result = std::move(new_result);
    }
  };
  for (int d = 0; d < static_cast<index_t>(op->dst_x.size()); ++d) {
    do_substitute(buffer_min(op->dst, d));
    do_substitute(buffer_max(op->dst, d));
    do_substitute(buffer_stride(op->dst, d));
    do_substitute(buffer_fold_factor(op->dst, d));
  }

  for (int d = 0; d < static_cast<index_t>(dst_x.size()); ++d) {
    if (!dst_x[d].defined()) continue;
    result = slice_dim::make(op->dst, op->dst, d, dst_x[d], result);
    result = loop::make(dst_x[d], loop::serial, buffer_bounds(op->dst, d), 1, result);
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
    // TODO: This inserts clone_buffer ops even for pipelined loops that don't need them, because we know that that
    // particular stage of the pipeline will not be executed by more than one thread concurrently.
    if (op->is_serial()) {
      node_mutator::visit(op);
      return;
    }

    // We've hit a parallel loop. The buffers that are allocated outside this loop, but mutated inside this loop, will
    // be true in the mutated map. We need to make copies of these buffers upon entering the loop.
    stmt body = mutate(op->body);
    for (std::size_t i = 0; i < mutated.size(); ++i) {
      if (mutated[i] && *mutated[i]) {
        body = clone_buffer::make(var(i), var(i), body);
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
  void visit(const transpose* op) override { visit_buffer_mutator(op); }
};

}  // namespace

stmt fix_buffer_races(const stmt& s) { return race_condition_fixer().mutate(s); }

namespace {

class insert_free_into_allocate : public node_mutator {
  // Contains the sym of the allocate node + all other buffer nodes which reference it.
  std::vector<var> names;
  // If we found some statement which references anything from `names`.
  bool found = false;
  // We don't want to insert into the very last statement of the block,
  // so use this flag to mark that we visited at least one.
  bool visited_something = false;

public:
  insert_free_into_allocate(var name) { names.push_back(name); }

  void visit(const block* op) override {
    // Visit blocks in reverse order.
    std::vector<stmt> stmts(op->stmts.size());
    bool changed = false;
    for (int i = static_cast<int>(op->stmts.size()) - 1; i >= 0; --i) {
      // Don't mutate the rest if we changed one of the block's statements.
      if (changed || found) {
        stmts[i] = op->stmts[i];
      } else {
        stmts[i] = mutate(op->stmts[i]);
      }
      // We don't want to insert into the very last statement of the block,
      // so use this flag to mark that we visited at least one.
      visited_something = true;
      changed = changed || !stmts[i].same_as(op->stmts[i]);
    }
    if (!changed) {
      set_result(op);
    } else {
      set_result(block::make(std::move(stmts)));
    }
  }

  // Handler for the `terminal` nodes.
  void visit_terminal(const stmt& s) {
    stmt result = s;
    if (!found && depends_on(s, names).any()) {
      found = true;
      if (visited_something) {
        result = block::make({result, check::make(call::make(intrinsic::free, {names.front()}))});
      }
    }

    set_result(result);
  }

  void visit(const loop* op) override { visit_terminal(op); }
  void visit(const call_stmt* op) override { visit_terminal(op); }
  void visit(const copy_stmt* op) override { visit_terminal(op); }

  // Remaining functions collect all the buffer symbols which refer the original allocate
  // symbol or its dependencies.
  void visit(const make_buffer* op) override {
    bool base_depends = depends_on(op->base, names).any();
    found = found || base_depends;
    if (std::find(names.begin(), names.end(), op->sym) != names.end()) {
      // Don't look inside if shadowing names.
      set_result(op);
    } else {
      if (base_depends) {
        names.push_back(op->sym);
      }
      node_mutator::visit(op);
      if (base_depends) {
        names.pop_back();
      }
    }
  }

  template <typename T>
  void visit_buffer_mutator(const T* op) {
    auto n = std::find(names.begin(), names.end(), op->src);
    found = found || n != names.end();
    if (std::find(names.begin(), names.end(), op->sym) != names.end()) {
      // Don't look inside if shadowing names.
      set_result(op);
    } else {
      if (n != names.end()) {
        names.push_back(op->sym);
      }
      node_mutator::visit(op);
      if (n != names.end()) {
        names.pop_back();
      }
    }
  }

  void visit(const clone_buffer* op) override { visit_buffer_mutator(op); }
  void visit(const crop_buffer* op) override { visit_buffer_mutator(op); }
  void visit(const crop_dim* op) override { visit_buffer_mutator(op); }
  void visit(const slice_buffer* op) override { visit_buffer_mutator(op); }
  void visit(const slice_dim* op) override { visit_buffer_mutator(op); }
  void visit(const transpose* op) override { visit_buffer_mutator(op); }
};

class early_free_inserter : public node_mutator {
public:
  void visit(const allocate* op) override {
    stmt body = mutate(op->body);
    body = insert_free_into_allocate(op->sym).mutate(body);
    if (body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(allocate::make(op->sym, op->storage, op->elem_size, op->dims, body));
    }
  }
};

}  // namespace

stmt insert_early_free(const stmt& s) { return early_free_inserter().mutate(s); }

}  // namespace slinky
