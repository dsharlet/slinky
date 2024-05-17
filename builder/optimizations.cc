#include "builder/optimizations.h"

#include <cassert>
#include <cstddef>
#include <cstdlib>
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
    src_dim.stride = 0;
    src_dim.fold_factor = dim::unfolded;
    return true;
  } else {
    // check for f(x) = g(x * C)
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

    src_dim.stride *= scale;

    expr offset = simplify((src_x - dst_x) * scale);
    if (!depends_on(offset, dst_x).any()) {
      // The difference of src_x and dst_x does not depend on dst_x, it's a simple copy.
      if (is_zero(offset)) {
        // If the offset is zero, the index we want for the buffer_at call is buffer_min(src, src_d), which is
        // definitely in bounds, so we don't need to clamp it.
        src_dim.bounds = buffer_bounds(src, src_d) / scale;
        at = src_dim.bounds.min * scale;
      } else {
        // The offset is non-zero, we might go out of bounds with our buffer_at call. To avoid this, we need to
        // clamp to the intersection of the src and dst buffers, like copy would have done.
        src_dim.bounds &= (buffer_bounds(src, src_d) - offset) / scale;
        at = (src_dim.bounds.min + offset) * scale;
      }
      return true;
    }

    return false;
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

stmt replace_copy_with_pad(const stmt& s, var a, var b) {
  return recursive_mutate<copy_stmt>(s, [a, b](const copy_stmt* op) {
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
    return call_stmt::make(
        [padding = *op->padding](const call_stmt* op, const eval_context& ctx) -> index_t {
          const raw_buffer* src_buf = ctx.lookup_buffer(op->inputs[0]);
          const raw_buffer* dst_buf = ctx.lookup_buffer(op->outputs[0]);
          ctx.pad(src_buf->dims, *dst_buf, padding.data());
          return 0;
        },
        {op->src}, {op->dst}, std::move(pad_attrs));
  });
}

class buffer_aliaser : public node_mutator {
  struct buffer_alias {
    std::vector<dim_expr> dims;
    std::vector<expr> at;
  };

  class buffer_info {
  public:
    std::vector<dim_expr> dims;
    std::map<var, buffer_alias> can_alias_;
    std::set<var> cannot_alias_;

  public:
    buffer_info(std::vector<dim_expr> dims) : dims(std::move(dims)) {}

    std::map<var, buffer_alias>& can_alias() { return can_alias_; }
    const std::map<var, buffer_alias>& can_alias() const { return can_alias_; }

    void maybe_alias(var s, buffer_alias a) {
      if (cannot_alias_.count(s)) {
        return;
      }

      assert(dims.size() == a.dims.size());
      for (std::size_t d = 0; d < dims.size(); ++d) {
        if (dims[d].stride.defined()) {
          if (!prove_true(dims[d].stride == a.dims[d].stride)) {
            // This alias is not compatible because it would violate a constraint on the stride of the buffer.
            return;
          }
        }
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
    auto s = set_value_in_scope(alloc_info, op->sym, buffer_info(op->dims));
    stmt body = mutate(op->body);
    buffer_info info = std::move(*alloc_info[op->sym]);

    // When an allocation goes out of scope, we should remove it as an aliasing candidate.
    for (std::optional<buffer_info>& i : alloc_info) {
      if (i) i->do_not_alias(op->sym);
    }

    for (const auto& target : info.can_alias()) {
      var target_var = target.first;
      const buffer_alias& alias = target.second;

      // Replace the allocation with a buffer using the dims the alias wants.
      stmt result =
          make_buffer::make(op->sym, buffer_at(target_var, alias.at), op->elem_size, alias.dims, std::move(body));
      // If we aliased the source and destination of a copy, replace the copy with a pad.
      stmt pad_result = replace_copy_with_pad(result, op->sym, target_var);
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
      if (alloc_info[target_var]) {
        alloc_info[target_var]->do_not_alias(op->sym);
      }
      set_result(pad_result);
      return;
    }
    if (!body.same_as(op->body)) {
      set_result(clone_with_new_body(op, std::move(body)));
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
        std::optional<buffer_info>& input_info = alloc_info[i];
        if (input_info) {
          buffer_alias a;
          a.dims = buffer_dims(o, input_info->dims.size());
          a.at = buffer_mins(o, input_info->dims.size());
          input_info->maybe_alias(o, std::move(a));
        }
      }
    }
  }

  void alias_copy_dst(const copy_stmt* op) {
    if (!alloc_info[op->dst] || !can_alias(op->src)) {
      // We didn't allocate the dst.
      return;
    }

    // We allocated the dst. We might be able to replace the allocation with an alias of the src.
    // This case is a straightforward use of is_copy, which produces the dims that should be the src of a copy, which
    // are the same dimensions we want the dst to be.
    std::optional<buffer_info>& info = alloc_info[op->dst];

    buffer_alias a;
    a.at.resize(op->src_x.size());
    a.dims = info->dims;
    for (int dst_d = 0; dst_d < static_cast<int>(op->dst_x.size()); ++dst_d) {
      int src_d;
      if (!is_copy_dst_dim(op, dst_d, src_d)) {
        return;
      }

      expr at_unused;
      expr& at = src_d >= 0 ? a.at[src_d] : at_unused;
      a.dims[dst_d].stride = buffer_stride(op->src, src_d);
      a.dims[dst_d].fold_factor = buffer_fold_factor(op->src, src_d);
      if (!is_copy(op, src_d, dst_d, at, a.dims[dst_d])) {
        return;
      }
    }

    info->maybe_alias(op->src, std::move(a));
  }

  void alias_copy_src(const copy_stmt* op) {
    if (!alloc_info[op->src] || !can_alias(op->dst)) {
      // We didn't allocate the src.
      return;
    }

    // We allocated the src. We might be able to replace the allocation with an alias of the dst.
    // In this case, we're going to make the src an alias of another buffer. We're more limited in what we can do here
    // vs. the above case, because we can't expect producers to handle everything the copy is doing (such as
    // broadcasting).
    std::optional<buffer_info>& info = alloc_info[op->src];

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

      a.dims[src_d] = {
          (buffer_bounds(op->dst, dst_d) + offset) & info->dims[src_d].bounds,
          buffer_stride(op->dst, dst_d),
          buffer_fold_factor(op->dst, dst_d),
      };
      a.at[dst_d] = max(buffer_min(op->dst, dst_d), info->dims[src_d].bounds.min - offset);
    }

    for (const dim_expr& d : a.dims) {
      if (!d.stride.defined()) {
        // We didn't define all the dimensions of the buffer we want to replace.
        return;
      }
    }

    info->maybe_alias(op->dst, std::move(a));
  }

  void visit(const copy_stmt* op) override {
    set_result(op);

    alias_copy_dst(op);
    alias_copy_src(op);
  }

  void merge_alloc_info(symbol_map<buffer_info> add) {
    alloc_info.reserve(std::max(alloc_info.size(), add.size()));
    for (std::size_t i = 0; i < add.size(); ++i) {
      if (!add[i]) continue;
      std::optional<buffer_info>& info = alloc_info[i];
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
    symbol_map<buffer_info> old_alloc_info(alloc_info.size());
    std::swap(old_alloc_info, alloc_info);
    for (std::size_t i = 0; i < old_alloc_info.size(); ++i) {
      if (old_alloc_info[i]) {
        alloc_info[i] = buffer_info(old_alloc_info[i]->dims);
      }
    }

    auto set_info_sym = set_value_in_scope(alloc_info, op->sym, alloc_info[op->src]);
    node_mutator::visit(op);

    // If we chose to alias this buffer, we need to insert offsets for where we sliced it.
    for (std::optional<buffer_info>& i : alloc_info) {
      if (!i) continue;
      auto j = i->can_alias().find(op->sym);
      if (j != i->can_alias().end()) {
        std::vector<expr>& at = j->second.at;
        for (std::size_t d = 0; d < op->at.size(); ++d) {
          if (!op->at[d].defined()) continue;
          at.insert(at.begin() + d, op->at[d]);
        }
      }
    }

    // Add the old alias candidates back to the alias info.
    merge_alloc_info(std::move(old_alloc_info));
  }

  void visit(const slice_dim* op) override {
    // We need to know which alias candidates are added inside this slice.
    symbol_map<buffer_info> old_alloc_info(alloc_info.size());
    std::swap(old_alloc_info, alloc_info);
    for (std::size_t i = 0; i < old_alloc_info.size(); ++i) {
      if (old_alloc_info[i]) {
        alloc_info[i] = buffer_info(old_alloc_info[i]->dims);
      }
    }

    auto set_info_sym = set_value_in_scope(alloc_info, op->sym, alloc_info[op->src]);
    node_mutator::visit(op);

    // If we chose to alias this buffer, we need to insert offsets for where we sliced it.
    for (std::optional<buffer_info>& i : alloc_info) {
      if (!i) continue;
      auto j = i->can_alias().find(op->sym);
      if (j != i->can_alias().end()) {
        std::vector<expr>& at = j->second.at;
        at.insert(at.begin() + op->dim, op->at);
      }
    }

    // Add the old alias candidates back to the alias info.
    merge_alloc_info(std::move(old_alloc_info));
  }

  void visit(const clone_buffer* op) override {
    auto set_info_sym = set_value_in_scope(alloc_info, op->sym, alloc_info[op->src]);
    node_mutator::visit(op);

    // When a buffer goes out of scope, we should remove it as an aliasing candidate.
    for (std::optional<buffer_info>& i : alloc_info) {
      if (i) i->do_not_alias(op->sym);
    }

    // Alias candidates for op->sym are also alias candidates for op->src.
    std::optional<buffer_info> sym_info = std::move(alloc_info[op->sym]);
    if (sym_info) {
      std::optional<buffer_info>& src_info = alloc_info[op->src];
      if (!src_info) {
        src_info = std::move(sym_info);
      } else {
        for (auto& j : sym_info->can_alias()) {
          src_info->maybe_alias(j.first, std::move(j.second));
        }
      }
    }
  }

  void visit(const transpose*) override { std::abort(); }
};

}  // namespace

stmt alias_buffers(const stmt& s) { return buffer_aliaser().mutate(s); }

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

    dim_expr src_dim = {buffer_bounds(op->dst, dst_d), 0, dim::unfolded};
    if (src_d >= 0) {
      src_dim.stride = buffer_stride(op->src, src_d);
      src_dim.fold_factor = buffer_fold_factor(op->src, src_d);
    }
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
    if (base_depends) {
      names.push_back(op->sym);
    }
    node_mutator::visit(op);
    if (base_depends) {
      names.pop_back();
    }
  }

  template <typename T>
  void visit_buffer_mutator(const T* op) {
    auto n = std::find(names.begin(), names.end(), op->src);
    if (n != names.end()) {
      names.push_back(op->sym);
    }
    node_mutator::visit(op);
    if (n != names.end()) {
      names.pop_back();
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
