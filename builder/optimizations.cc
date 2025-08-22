#include "builder/optimizations.h"

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "base/chrome_trace.h"
#include "base/function_ref.h"
#include "base/set.h"
#include "builder/node_mutator.h"
#include "builder/pipeline.h"
#include "builder/simplify.h"
#include "builder/substitute.h"
#include "runtime/buffer.h"
#include "runtime/depends_on.h"
#include "runtime/evaluate.h"
#include "runtime/expr.h"
#include "runtime/print.h"
#include "runtime/stmt.h"

namespace slinky {

namespace {

// A select that simplifies assuming that if t or f are undefined, the value can be anything.
expr select_or_undef(const expr& c, expr t, expr f) {
  if (t.defined() && f.defined()) {
    return select(c, std::move(t), std::move(f));
  } else if (t.defined()) {
    return t;
  } else {
    return f;
  }
}

dim_expr select(const expr& c, dim_expr t, dim_expr f) {
  return {
      select(c, std::move(t.bounds), std::move(f.bounds)),
      select_or_undef(c, std::move(t.stride), std::move(f.stride)),
      select_or_undef(c, std::move(t.fold_factor), std::move(f.fold_factor)),
  };
}

// Try to find a, b such that y = a*x + b
bool is_linear(const expr& y, var x, expr& a, expr& b) {
  if (match(y, x)) {
    // y = x
    a = 1;
    b = 0;
    return true;
  } else if (!depends_on(y, x).var) {
    // y = b
    a = 0;
    b = y;
    return true;
  } else if (const add* op = y.as<add>()) {
    expr aa, ab;
    expr ba, bb;
    if (is_linear(op->a, x, aa, ab) && is_linear(op->b, x, ba, bb)) {
      a = aa + ba;
      b = ab + bb;
      return true;
    }
  } else if (const sub* op = y.as<sub>()) {
    expr aa, ab;
    expr ba, bb;
    if (is_linear(op->a, x, aa, ab) && is_linear(op->b, x, ba, bb)) {
      a = aa - ba;
      b = ab - bb;
      return true;
    }
  } else if (const mul* op = y.as<mul>()) {
    if (is_linear(op->a, x, a, b) && !depends_on(op->b, x).var) {
      a *= op->b;
      b *= op->b;
      return true;
    } else if (is_linear(op->b, x, a, b) && !depends_on(op->a, x).var) {
      a *= op->a;
      b *= op->a;
      return true;
    }
  }

  return false;
}

// Checks if the copy operands `src_x` and `dst_x[dst_d]` represent a simple copy that can be handled by slinky::copy.
// dst_x dimensions other than `dst_d` are assumed to be handled by a different `is_copy` call.
bool is_copy(var src, expr src_x, int src_d, var dst, span<const var> dst_x, int dst_d, expr& at, dim_expr& src_dim) {
  if (const class select* s = src_x.as<class select>()) {
    // The src is a select of two things that might both be copies.
    expr at_t;
    expr at_f;
    dim_expr src_dim_t;
    dim_expr src_dim_f;
    if (is_copy(src, s->true_value, src_d, dst, dst_x, dst_d, at_t, src_dim_t) &&
        is_copy(src, s->false_value, src_d, dst, dst_x, dst_d, at_f, src_dim_f)) {
      at = select(s->condition, at_t, at_f);
      src_dim = select(s->condition, src_dim_t, src_dim_f);
      return true;
    } else {
      return false;
    }
  } else if (!depends_on(src_x, dst_x).var) {
    // This is a broadcast because the src_x is constant w.r.t. dst_x.
    at = src_x;
    src_dim.bounds = buffer_bounds(dst, dst_d);
    src_dim.stride = 0;
    // It doesn't matter what we choose for the fold factor because the stride is 0.
    src_dim.fold_factor = expr();
    return true;
  } else {
    // If a src_x depends on multiple dst_x, only consider this dst dim for now.
    for (int i = 0; i < static_cast<int>(dst_x.size()); ++i) {
      if (i != dst_d) {
        src_x = substitute(src_x, dst_x[i], buffer_min(dst, i));
      }
    }
    src_x = simplify(src_x);

    // Try to parse src_x = dst_x * scale + offset
    expr scale, offset;
    if (!is_linear(src_x, dst_x[dst_d], scale, offset)) {
      return false;
    }
    scale = simplify(scale);
    offset = simplify(offset);

    std::optional<index_t> scale_bound = evaluate_constant_lower_bound(scale);
    if (!scale_bound || *scale_bound < 0) {
      // TODO: Maybe we could handle negative stride copies.
      return false;
    }

    src_dim.stride = buffer_stride(src, src_d) * scale;
    src_dim.fold_factor = buffer_fold_factor(src, src_d);

    // Both branches of this if should be equivalent if the copy is fully in bounds (not padded).
    if (!depends_on(offset, dst).any()) {
      src_dim.bounds = (buffer_bounds(src, src_d) - offset) / scale;
      at = buffer_min(src, src_d) + offset * (scale - 1);
    } else {
      // This formulation is simpler and would be nice to use all the time, however, it assumes there is no padding on
      // the copy.
      src_dim.bounds = buffer_bounds(dst, dst_d);
      at = buffer_min(dst, dst_d) * scale + offset;
    }

    return true;
  }
}

bool is_copy(const copy_stmt* op, int src_d, int dst_d, expr& at, dim_expr& src_dim) {
  // We might not have an src dim if we're trying to broadcast.
  expr src_x = src_d >= 0 ? op->src_x[src_d] : expr();
  return is_copy(op->src, src_x, src_d, op->dst, op->dst_x, dst_d, at, src_dim);
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

// Replace copies between buffers a and b with calls to pad.
class copy_remover : public stmt_mutator {
  // Track all names of a and b as we go.
  std::vector<var> as;
  std::vector<var> bs;

  bool is_a_and_b(var x, var y) const {
    return (std::find(as.begin(), as.end(), x) != as.end() && std::find(bs.begin(), bs.end(), y) != bs.end()) ||
           (std::find(as.begin(), as.end(), y) != as.end() && std::find(bs.begin(), bs.end(), x) != bs.end());
  }

public:
  copy_remover(var a, var b) : as({a}), bs({b}) {}

  void visit(const copy_stmt* op) override {
    if (!is_a_and_b(op->src, op->dst)) {
      // Not this copy.
      set_result(op);
      return;
    }
    if (!op->pad.defined()) {
      // No padding, this copy is now a no-op.
      set_result(stmt());
      return;
    }
    set_result(op);
  }

  void visit(const call_stmt* op) override {
    if (op->attrs.name == "memcpy" && op->inputs.size() == 1 && op->outputs.size() == 1 &&
        is_a_and_b(op->inputs[0], op->outputs[0])) {
      expr input_size = variable::make(op->inputs[0], buffer_field::size_bytes);
      expr output_size = variable::make(op->outputs[0], buffer_field::size_bytes);
      set_result(check::make(input_size == output_size));
    } else {
      set_result(op);
    }
  }

  template <typename T>
  void visit_buffer_mutator(const T* op) {
    bool a_contains = std::find(as.begin(), as.end(), op->src) != as.end();
    bool b_contains = std::find(bs.begin(), bs.end(), op->src) != bs.end();
    if (a_contains) as.push_back(op->sym);
    if (b_contains) bs.push_back(op->sym);
    stmt_mutator::visit(op);
    if (a_contains) as.pop_back();
    if (b_contains) bs.pop_back();
  }

  void visit(const crop_dim* op) override { visit_buffer_mutator(op); }
  void visit(const crop_buffer* op) override { visit_buffer_mutator(op); }
  void visit(const slice_dim* op) override { visit_buffer_mutator(op); }
  void visit(const slice_buffer* op) override { visit_buffer_mutator(op); }
  void visit(const clone_buffer* op) override { visit_buffer_mutator(op); }
  void visit(const transpose* op) override { visit_buffer_mutator(op); }
};

stmt remove_copy(const stmt& s, var a, var b) {
  scoped_trace trace("remove_copy");
  return copy_remover(a, b).mutate(s);
}

bool dim_has_stride(const dim_expr& d) { return d.stride.defined(); }
bool any_stride_defined(span<const dim_expr> dims) { return std::any_of(dims.begin(), dims.end(), dim_has_stride); }

class copy_aliaser : public stmt_mutator {
  node_context& ctx;

  struct alias_info {
    var target;

    // Parameters for this alias's make_buffer call.
    std::vector<dim_expr> dims;
    expr elem_size;

    // Coordinates to pass to buffer_at to get the base pointer for the alias.
    std::vector<expr> at;

    // Mapping of dimensions of the alias to the original buffer's dimensions.
    std::vector<int> permutation;

    // If true, we know this alias is a subset of the aliased buffer.
    bool assume_in_bounds = false;

    bool is_contiguous_copy = false;

    bool disabled = false;
  };

  class buffer_info {
  public:
    // The buffer allocation parameters.
    std::vector<dim_expr> dims;
    expr elem_size;

    bool is_input;
    bool is_output;

    int producers = 0;

    // Possible aliases of this allocation.
    std::vector<alias_info> aliases;

    // If we decided to alias this buffer, we might have grown the bounds. If so, we need to make a new allocation with
    // this symbol, but make a crop of it for the original bounds.
    var shared_alloc_sym;

    buffer_info(std::vector<dim_expr> dims, expr elem_size, bool is_input = false, bool is_output = false)
        : dims(std::move(dims)), elem_size(std::move(elem_size)), is_input(is_input), is_output(is_output) {}

    void do_not_alias(var t) {
      for (alias_info& i : aliases) {
        if (i.target == t) {
          i.disabled = true;
        }
      }
    }
  };
  symbol_map<buffer_info> buffers;

  // Checks if `op` can safely be replaced by `alias` targeting a buffer `target` described by `target_info`.
  // This updates some fields of `alias` from what we learn from `target_info`.
  static bool alias_compatible(
      const buffer_info& alloc_info, alias_info& alias, var target, const buffer_info& target_info) {
    scoped_trace trace("alias_compatible");
    assert(alloc_info.dims.size() == alias.dims.size());

    if (alias.is_contiguous_copy) {
      assert(alias.assume_in_bounds);
      // We just assume flat copies are OK.
      return true;
    }
    const bool target_has_stride = any_stride_defined(target_info.dims);
    const bool alloc_has_stride = any_stride_defined(alloc_info.dims);

    // Figure out what the actual alias dimension will be by substituting the target into it.
    std::vector<dim_expr> alloc_dims = alloc_info.dims;

    for (dim_expr& i : alloc_dims) {
      i.bounds = substitute_buffer(i.bounds, target, target_info.dims);
      i.stride = substitute_buffer(i.stride, target, target_info.dims);
      i.fold_factor = substitute_buffer(i.fold_factor, target, target_info.dims);
    }

    std::vector<dim_expr> alias_dims = alias.dims;

    for (dim_expr& i : alias_dims) {
      i.bounds = substitute_buffer(i.bounds, target, target_info.dims);
      i.stride = substitute_buffer(i.stride, target, target_info.dims);
      i.fold_factor = substitute_buffer(i.fold_factor, target, target_info.dims);
    }

    if (!alias.assume_in_bounds) {
      bool in_bounds = true;
      for (std::size_t d = 0; d < alloc_dims.size(); ++d) {
        const dim_expr& alias_dim = alias_dims[d];
        if (!prove_true(alloc_dims[d].bounds.min >= alias_dim.bounds.min) ||
            !prove_true(alloc_dims[d].bounds.max <= alias_dim.bounds.max)) {
          // We don't know if this target is big enough for this allocation.
          if (target_info.is_input || target_info.is_output) {
            // We can't reallocate this buffer.
            return false;
          }
          in_bounds = false;
        }
      }

      alias.assume_in_bounds = in_bounds;
    }

    // If the target doesn't have strides, we can propagate our strides there, but we need to make sure there are no
    // contradictions when we do so. This tracks what we want the strides to be.
    std::vector<expr> target_stride(target_info.dims.size());
    for (std::size_t d = 0; d < alloc_dims.size(); ++d) {
      const dim_expr& alias_dim = alias_dims[d];
      if (!alias.assume_in_bounds) {
        // The alias might grow the target allocation, so we can't use the target's strides.
        if (alloc_dims[d].stride.defined()) {
          if (!prove_true(alloc_dims[d].stride == alias_dim.stride)) {
            // This alias would violate a constraint on the stride of the buffer.
            return false;
          }
        }
      } else if (!alloc_has_stride) {
        // The allocation doesn't care what the strides are.
      } else if (!target_has_stride) {
        if (!alloc_info.dims[d].stride.defined()) continue;
        int target_d = alias.permutation[d];
        if (target_d >= 0) {
          expr& target_stride_d = target_stride[target_d];
          if (target_stride_d.defined() && !prove_true(target_stride_d == alloc_info.dims[d].stride)) {
            // We tried to set the same dimension to two different strides.
            return false;
          }
          target_stride_d = alloc_info.dims[d].stride;
        }
      } else {
        // There are strides on both the allocation and the target, they must be equal.
        if (!prove_true(alloc_dims[d].stride == alias_dim.stride)) {
          // This alias would violate a constraint on the stride of the buffer.
          return false;
        }
      }

      if (alloc_dims[d].fold_factor.defined()) {
        if (!alias_dim.fold_factor.defined() || is_constant(alias_dim.fold_factor, dim::unfolded)) {
          // The target isn't folded, we can alias this buffer. We lose our fold factor, but it's not going to occupy
          // any memory anyways if it's an alias.
        } else if (!prove_true(alias_dim.fold_factor >= alloc_dims[d].fold_factor)) {
          // The fold factor of this allocation does not evenly divide the target fold factor.
          // TODO: We could increase the fold factor like we do the bounds.
          return false;
        } else if (!prove_true((alias_dim.bounds.min % alias_dim.fold_factor) ==
                               (alloc_dims[d].bounds.min % alloc_dims[d].fold_factor))) {
          // The mins of folded buffers are not aligned.
          return false;
        }
      } else if ((alias_dim.fold_factor.defined() && !is_constant(alias_dim.fold_factor, dim::unfolded)) &&
                 !prove_true(alloc_dims[d].extent() <= alias_dim.fold_factor)) {
        // If the target is folded, but the op is not, we can only alias it if the extent of this dimension
        // is less than the fold factor.
        return false;
      }
    }
    return true;
  }

public:
  copy_aliaser(
      node_context& ctx, const std::vector<buffer_expr_ptr>& inputs, const std::vector<buffer_expr_ptr>& outputs)
      : ctx(ctx) {
    for (const buffer_expr_ptr& i : inputs) {
      buffers[i->sym()] = buffer_info(i->dims(), i->elem_size(), /*is_input=*/true, /*is_output=*/false);
    }
    for (const buffer_expr_ptr& i : outputs) {
      buffers[i->sym()] = buffer_info(i->dims(), i->elem_size(), /*is_input=*/false, /*is_output=*/true);
    }
  }

  void visit(const constant_buffer* op) override {
    // Constants are similar to inputs in that they cannot be mutated.
    auto s = set_value_in_scope(
        buffers, op->sym, buffer_info(buffer_dims(*op->value), op->value->elem_size, /*is_input=*/true));
    stmt_mutator::visit(op);

    // When an allocation goes out of scope, we should remove it as an aliasing candidate.
    for (std::optional<buffer_info>& i : buffers) {
      if (i) i->do_not_alias(op->sym);
    }
  }

  void visit(const allocate* op) override {
    auto s = set_value_in_scope(buffers, op->sym, buffer_info(op->dims, op->elem_size));
    stmt body = mutate(op->body);

    scoped_trace trace("visit(const allocate*)");
    buffer_info info = std::move(*buffers[op->sym]);
    var sym = info.shared_alloc_sym.defined() ? info.shared_alloc_sym : op->sym;

    // When an allocation goes out of scope, we should remove it as an aliasing candidate.
    for (std::optional<buffer_info>& i : buffers) {
      if (i) i->do_not_alias(op->sym);
    }

    // Make a set of dims for substituting the bounds (but not the stride or fold factor) of this symbol.
    std::vector<dim_expr> op_dims_bounds = info.dims;
    for (dim_expr& i : op_dims_bounds) {
      i.stride = expr();
      i.fold_factor = expr();
    }
    if (info.producers > 1) {
      // Don't try to alias a buffer with more than one producer (e.g. copies that concatenate).
      // TODO: We might be able to handle this case, if all the producers have a compatible alias with the same target?
    } else {
      for (alias_info& alias : info.aliases) {
        if (alias.disabled) {
          continue;
        }

        var target_var = alias.target;
        std::optional<buffer_info>& target_info = buffers[target_var];
        assert(target_info);

        if (!alias_compatible(info, alias, target_var, *target_info)) {
          continue;
        }

        // The alias might have used the bounds of this symbol, substitute them now.
        for (dim_expr& i : alias.dims) {
          i.bounds = substitute_buffer(i.bounds, op->sym, op_dims_bounds);
        }
        for (expr& i : alias.at) {
          i = substitute_buffer(i, op->sym, op_dims_bounds);
        }

        var alloc_var = target_var;
        if (!alias.assume_in_bounds) {
          assert(!target_info->is_output);
          assert(!target_info->is_input);  // We shouldn't be trying to write to an input anyways.
          // We allocated this buffer, make it big enough to share with this buffer.
          std::string old_name =
              ctx.name(target_info->shared_alloc_sym.defined() ? target_info->shared_alloc_sym : target_var);
          target_info->shared_alloc_sym = ctx.insert_unique(old_name + "/" + ctx.name(sym));
          alloc_var = target_info->shared_alloc_sym;
          assert(target_info->dims.size() == alias.at.size());
          for (std::size_t d = 0; d < target_info->dims.size(); ++d) {
            // TODO: We may have proven this is unnecessary in alias_compatible, we can avoid this in such cases.
            // We need the bounds of the alias, as it exists in the target buffer. `alias.at` tells us where this alias
            // starts.
            int alias_d = alias.permutation[d];
            if (alias_d >= 0) {
              target_info->dims[d].bounds |= alias.at[d] + min_extent(0, alias.dims[alias_d].bounds.extent());
            }
          }
        } else if (!any_stride_defined(target_info->dims) && !alias.is_contiguous_copy) {
          assert(info.dims.size() == alias.permutation.size());
          // The target doesn't have any strides, we might have some strides we assumed we could propagate.
          for (std::size_t d = 0; d < info.dims.size(); ++d) {
            if (!info.dims[d].stride.defined()) continue;
            int alias_d = alias.permutation[d];
            if (alias_d >= 0) {
              assert(alias_d < static_cast<int>(target_info->dims.size()));
              if (target_info->dims[alias_d].stride.defined()) {
                assert(prove_true(target_info->dims[alias_d].stride == info.dims[d].stride));
              } else {
                target_info->dims[alias_d].stride = info.dims[d].stride;
              }
            }
          }
        }

        // Replace the allocation with a buffer using the dims (and maybe elem_size) the alias wants.
        expr elem_size = alias.elem_size.defined() ? alias.elem_size : op->elem_size;
        if (sym != op->sym) {
          body = crop_buffer::make(op->sym, sym, dims_bounds(op->dims), std::move(body));
        }
        stmt result = make_buffer::make(sym, buffer_at(alloc_var, alias.at), elem_size, alias.dims, std::move(body));
        // Wrap with the original buffer in case we want to use the metadata in the construction of the buffer.
        result = make_buffer::make(sym, expr(), elem_size, op->dims, std::move(result));

        for (auto& i : target_info->aliases) {
          i.assume_in_bounds = i.assume_in_bounds && alias.assume_in_bounds;
        }

        if (elem_size.defined()) {
          result = block::make({check::make(elem_size == op->elem_size), result});
        }

        // If we aliased the source and destination of a copy with no padding, the copy can be removed.
        result = remove_copy(result, op->sym, target_var);

        set_result(std::move(result));
        return;
      }
    }
    if (!body.same_as(op->body)) {
      if (info.shared_alloc_sym.defined()) {
        // This allocation's bounds were expanded to accommodate aliases. Make a new expanded allocation, and make the
        // original allocation a crop of the expanded allocation.
        body = crop_buffer::make(op->sym, info.shared_alloc_sym, dims_bounds(op->dims), std::move(body));
      }
      stmt result = allocate::make(sym, op->storage, op->elem_size, std::move(info.dims), std::move(body));
      // Wrap with the original buffer in case we want to use the metadata in the construction of the buffer.
      result = make_buffer::make(op->sym, expr(), op->elem_size, op->dims, std::move(result));
      set_result(std::move(result));
    } else {
      set_result(op);
    }
  }

  void visit(const loop* op) override {
    stmt body = mutate(op->body);
    if (body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(clone_with(op, std::move(body)));
    }

    symbol_map<interval_expr> loop_bounds;
    loop_bounds[op->sym] = op->bounds;

    for (std::optional<buffer_info>& i : buffers) {
      if (!i) continue;
      for (dim_expr& d : i->dims) {
        if (depends_on(d.bounds, op->sym).any()) {
          d.bounds = bounds_of(d.bounds, loop_bounds);
        }
      }
    }
  }

  // Make dimensions that assign strides that are contiguous and ascending.
  static std::vector<dim_expr> make_contiguous_dims(var buf, std::size_t rank) {
    std::vector<dim_expr> dims(rank);
    expr stride = buffer_elem_size(buf);
    for (int i = 0; i < static_cast<int>(rank); ++i) {
      dims[i].bounds = buffer_bounds(buf, i);
      dims[i].stride = stride;
      stride *= dims[i].bounds.extent();
    }
    return dims;
  }

  void visit(const call_stmt* op) override {
    set_result(op);
    if (op->attrs.name == "memcpy") {
      assert(op->inputs.size() == 1);
      assert(op->outputs.size() == 1);
      var in = op->inputs[0];
      var out = op->outputs[0];
      std::optional<buffer_info>& input_info = buffers[in];
      std::optional<buffer_info>& output_info = buffers[out];
      if (output_info) output_info->producers++;
      if (input_info && output_info) {
        alias_info fwd;
        fwd.target = out;
        fwd.dims = make_contiguous_dims(in, input_info->dims.size());
        fwd.at = buffer_mins(out, output_info->dims.size());
        fwd.is_contiguous_copy = true;
        fwd.assume_in_bounds = true;
        input_info->aliases.push_back(std::move(fwd));

        alias_info back;
        back.target = in;
        back.dims = make_contiguous_dims(out, output_info->dims.size());
        back.at = buffer_mins(in, input_info->dims.size());
        back.is_contiguous_copy = true;
        back.assume_in_bounds = true;
        output_info->aliases.push_back(std::move(back));
      }
    }
  }

  void alias_copy_dst(const copy_stmt* op) {
    scoped_trace trace("alias_copy_dst");
    std::optional<buffer_info>& info = buffers[op->dst];
    if (info) info->producers++;
    if (!info || info->is_output) {
      // We didn't allocate the dst.
      return;
    }
    // We allocated the dst. We might be able to replace the allocation with an alias of the src.
    // This case is a straightforward use of is_copy, which produces the dims that should be the src of a copy, which
    // are the same dimensions we want the dst to be.

    alias_info a;
    a.target = op->src;
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

      // We want the bounds of the original dst dimension, but the memory layout of the src dimension. This may
      // require the allocation to be expanded to accommodate this alias.
      a.dims[dst_d] = {buffer_bounds(op->dst, dst_d), src_dim.stride, src_dim.fold_factor};
      a.permutation[dst_d] = src_d;
      if (at.defined()) {
        a.at[src_d] = at - src_dim.bounds.min + a.dims[dst_d].bounds.min;
      }
    }

    // If there is no padding, we can assume that the src is always in bounds of dst.
    a.assume_in_bounds = !op->pad.defined();

    a.elem_size = buffer_elem_size(op->src);

    info->aliases.push_back(std::move(a));
  }

  void alias_copy_src(const copy_stmt* op) {
    scoped_trace trace("alias_copy_src");
    std::optional<buffer_info>& info = buffers[op->src];
    if (!info || info->is_input) {
      // We didn't allocate the src.
      return;
    }
    // We allocated the src. We might be able to replace the allocation with an alias of the dst.
    // In this case, we're going to make the src an alias of another buffer. We're more limited in what we can do here
    // vs. the above case, because we can't expect producers to handle everything the copy is doing (such as
    // broadcasting).

    alias_info a;
    a.target = op->dst;
    a.at.resize(op->dst_x.size());
    a.dims.resize(op->src_x.size());
    a.permutation.resize(op->dst_x.size());
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

      // We want the intersection of the bounds of the src and dst, but with dst's memory layout.
      // Conceptually, we really just want the bounds of the src, but including the bounds of the dst enables us to
      // detect when the dst is smaller than the src, which makes the alias invalid.
      a.dims[src_d] = {
          info->dims[src_d].bounds & (buffer_bounds(op->dst, dst_d) + offset),
          buffer_stride(op->dst, dst_d),
          buffer_fold_factor(op->dst, dst_d),
      };
      a.at[dst_d] = info->dims[src_d].bounds.min - offset;
      a.permutation[dst_d] = src_d;
    }

    for (const dim_expr& d : a.dims) {
      if (!d.stride.defined()) {
        // We didn't define all the dimensions of the buffer we want to replace.
        return;
      }
    }

    a.assume_in_bounds = false;
    a.elem_size = buffer_elem_size(op->dst);

    info->aliases.push_back(std::move(a));
  }

  void visit(const copy_stmt* op) override {
    set_result(op);

    alias_copy_dst(op);
    alias_copy_src(op);
  }

  void merge_buffer_info(
      symbol_map<buffer_info>& old_buffers, var sym, var src, function_ref<void(alias_info&)> handler) {
    for (std::optional<buffer_info>& i : buffers) {
      if (!i) continue;
      for (auto& a : i->aliases) {
        if (a.target == sym) {
          handler(a);
        }
        // We need to substitute uses of sym with uses of src in the aliases we added here.
        for (dim_expr& d : a.dims) {
          d.bounds = substitute(d.bounds, sym, src);
          d.stride = substitute(d.stride, sym, src);
          d.fold_factor = substitute(d.fold_factor, sym, src);
        }
        a.elem_size = substitute(a.elem_size, sym, src);
        for (expr& i : a.at) {
          i = substitute(i, sym, src);
        }
      }
    }

    // Add the old alias candidates back to the alias info.
    old_buffers.reserve(std::max(buffers.size(), old_buffers.size()));
    for (std::size_t i = 0; i < buffers.size(); ++i) {
      if (!buffers[i]) continue;
      std::optional<buffer_info> info = std::move(buffers[i]);
      std::optional<buffer_info>& old_info = old_buffers[var(i) != sym ? var(i) : src];
      if (!old_info) {
        old_info = buffer_info(info->dims, info->elem_size, info->is_input, info->is_output);
      } else {
        old_info->dims = std::move(info->dims);
        old_info->elem_size = std::move(info->elem_size);
        old_info->producers += info->producers;
      }
      if (info->shared_alloc_sym.defined()) {
        assert(!old_info->shared_alloc_sym.defined() || old_info->shared_alloc_sym == info->shared_alloc_sym);
        old_info->shared_alloc_sym = info->shared_alloc_sym;
      }
      for (alias_info& a : info->aliases) {
        if (a.target == sym) {
          a.target = src;
        }
        old_info->aliases.push_back(std::move(a));
      }
    }
    std::swap(old_buffers, buffers);
  }

  template <typename T>
  void visit_buffer_mutator(const T* op, function_ref<void(alias_info&)> handler) {
    // We need to know which alias candidates are added inside this mutator.
    symbol_map<buffer_info> old_buffers(buffers.size());
    std::swap(old_buffers, buffers);
    // Copy the buffer info, but not alias candidates, we'll copy those back later below.
    for (std::size_t i = 0; i < old_buffers.size(); ++i) {
      if (old_buffers[i]) {
        // TODO: I think slices need to slice this info here, and unslice it upon exiting this mutator.
        buffers[i] = buffer_info(
            old_buffers[i]->dims, old_buffers[i]->elem_size, old_buffers[i]->is_input, old_buffers[i]->is_output);
        buffers[i]->shared_alloc_sym = old_buffers[i]->shared_alloc_sym;
      }
    }

    auto set_info_sym = set_value_in_scope(buffers, op->sym, buffers[op->src]);
    stmt_mutator::visit(op);

    scoped_trace trace("visit_buffer_mutator");
    merge_buffer_info(old_buffers, op->sym, op->src, handler);
  }

  void substitute_crop_into_allocs(var sym, var src, const std::vector<dim_expr>& dims) {
    for (std::optional<buffer_info>& i : buffers) {
      if (!i) continue;
      for (dim_expr& d : i->dims) {
        d.bounds = substitute_buffer(d.bounds, sym, dims, src);
      }
    }
  }

  void visit(const slice_buffer* op) override {
    visit_buffer_mutator(op, [=](alias_info& alias) {
      for (std::size_t d = 0; d < op->at.size(); ++d) {
        if (!op->at[d].defined()) continue;
        alias.at.insert(alias.at.begin() + d, op->at[d]);
      }
    });
  }

  void visit(const slice_dim* op) override {
    visit_buffer_mutator(op, [=](alias_info& alias) { alias.at.insert(alias.at.begin() + op->dim, op->at); });
  }

  void visit(const crop_buffer* op) override {
    visit_buffer_mutator(op, [](alias_info&) {});

    std::vector<dim_expr> subs(op->bounds.size());
    for (std::size_t i = 0; i < subs.size(); ++i) {
      subs[i].bounds = op->bounds[i] & buffer_bounds(op->src, i);
    }
    substitute_crop_into_allocs(op->sym, op->src, subs);
  }

  void visit(const crop_dim* op) override {
    visit_buffer_mutator(op, [](alias_info&) {});

    std::vector<dim_expr> subs(op->dim + 1);
    subs[op->dim].bounds = op->bounds & buffer_bounds(op->src, op->dim);
    substitute_crop_into_allocs(op->sym, op->src, subs);
  }

  void visit(const clone_buffer* op) override {
    visit_buffer_mutator(op, [](alias_info&) {});
  }

  void visit(const transpose*) override {
    // TODO: We should be able to handle this.
    SLINKY_UNREACHABLE << "transpose not handled by buffer_aliaser";
  }
};

}  // namespace

stmt alias_copies(const stmt& s, node_context& ctx, const std::vector<buffer_expr_ptr>& inputs,
    const std::vector<buffer_expr_ptr>& outputs) {
  scoped_trace trace("alias_copies");
  return copy_aliaser(ctx, inputs, outputs).mutate(s);
}

namespace {

class in_place_aliaser : public stmt_mutator {
  struct buffer_info {
    // The name of the allocation or output variable that this buffer is derived from.
    var root;

    std::vector<dim_expr> dims;

    // Which loop level this allocation appears at. We assume that buffers allocated outside the pipeline
    // get this default, which indicates that the buffer is effectively inside any possible loop, allowing it to be
    // reused as an output.
    int loop_level = std::numeric_limits<int>::max();

    bool allow_alias = true;
  };
  // Tracks buffer symbols that are actually the same buffer.
  symbol_map<buffer_info> buffers;

  // Tracks buffers that we intend to replace with a crop.
  symbol_map<var> backward;
  symbol_map<var> forward;

  // Tracks if a buffer is used. Buffers start out unused, and we visit block stmts in reverse order, so the first use
  // we encounter is the last use of the buffer.
  symbol_map<int> use_count;

  int loop_level = 0;

public:
  in_place_aliaser(const std::vector<buffer_expr_ptr>& outputs) {
    for (const buffer_expr_ptr& i : outputs) {
      buffers[i->sym()] = {i->sym()};
      use_count[i->sym()] = 0;
    }
  }

  bool fold_factors_strides_same(const std::vector<dim_expr>& alloc_dims, const std::vector<dim_expr>& alias_dims) {
    if (alloc_dims.size() > alias_dims.size()) {
      return !(std::any_of(alloc_dims.begin(), alloc_dims.end(),
          [&](const dim_expr& i) { return i.stride.defined() || i.fold_factor.defined(); }));
    }
    for (std::size_t ix = 0; ix < alloc_dims.size(); ++ix) {
      if (alloc_dims[ix].stride.defined() && !prove_true(alloc_dims[ix].stride == alias_dims[ix].stride)) return false;
      if (alloc_dims[ix].fold_factor.defined() && !prove_true(alloc_dims[ix].fold_factor == alias_dims[ix].fold_factor))
        return false;
    }
    return true;
  }

  void visit(const allocate* op) override {
    auto set_buffer = set_value_in_scope(buffers, op->sym, {op->sym, op->dims, loop_level});
    auto set_back = set_value_in_scope(backward, op->sym, var());
    auto set_fwd = set_value_in_scope(forward, op->sym, var());
    auto set_used = set_value_in_scope(use_count, op->sym, 0);
    stmt body = mutate(op->body);

    std::optional<var> back = backward.lookup(op->sym);
    std::optional<var> fwd = forward.lookup(op->sym);
    int uses = *use_count.lookup(op->sym);

    bool can_alias = true;
    if (uses != 1) {
      // TODO: Try to relax constraint that there is only one use. We already limit aliases to be the last use. The
      // problem with multiple uses is the buffer we use instead of this allocation might be bigger, and the other use
      // needs those values missing from this allocation.
      can_alias = false;
    }

    if (can_alias && back && back->defined() && buffers.lookup(*back) &&
        fold_factors_strides_same(op->dims, buffers[*back]->dims)) {
      forward.erase(*back);
      set_result(crop_buffer::make(op->sym, *back, dims_bounds(op->dims), std::move(body)));
    } else if (can_alias && fwd && fwd->defined() && buffers.lookup(*fwd) &&
               fold_factors_strides_same(op->dims, buffers[*fwd]->dims)) {
      backward.erase(*fwd);

      stmt cropped = crop_buffer::make(op->sym, op->sym, dims_bounds(op->dims), std::move(body));
      set_result(clone_buffer::make(op->sym, *fwd, std::move(cropped)));
    } else if (!body.same_as(op->body)) {
      set_result(clone_with(op, std::move(body)));
    } else {
      set_result(op);
    }
  }

  void add_use(var buf) {
    std::optional<int>& uses = use_count[buf];
    if (!uses) uses = 0;
    ++(*uses);
  }

  void visit(const call_stmt* op) override {
    set_result(op);

    for (var i : op->inputs) {
      std::optional<buffer_info>& input_alloc = buffers[i];
      if (!input_alloc) continue;
      add_use(input_alloc->root);
    }

    if (op->attrs.name == "memcpy") {
      // We can't handle this, it should have been handled by copy_aliaser if it could be aliased.
      return;
    }

    for (std::size_t o = 0; o < op->outputs.size(); ++o) {
      std::optional<buffer_info> output_alloc = buffers.lookup(op->outputs[o]);
      if (!output_alloc || !output_alloc->allow_alias) {
        // We don't know how to alias this buffer.
        continue;
      }
      if (output_alloc->loop_level < loop_level) {
        // This allocation is at an outer loop level.
        // TODO: We could still alias this if we can prove that this allocation is written everywhere it's read inside
        // this loop.
        continue;
      }
      for (std::size_t i = 0; i < op->inputs.size(); ++i) {
        const std::size_t bit = o * op->inputs.size() + i;
        if (bit > 31) break;
        if ((op->attrs.allow_in_place & (1 << bit)) == 0) {
          continue;
        }

        std::optional<buffer_info>& input_alloc = buffers[op->inputs[i]];
        if (!input_alloc || !input_alloc->allow_alias || !use_count[input_alloc->root] ||
            *use_count[input_alloc->root] > 1) {
          // We're traversing blocks backwards, if we already had a use, this is not the last use of the buffer, we
          // can't alias it.
          continue;
        }
        if (input_alloc->loop_level < loop_level) {
          // This allocation is at an outer loop level.
          continue;
        }

        backward[output_alloc->root] = input_alloc->root;
        forward[input_alloc->root] = output_alloc->root;
        break;
      }
    }
  }

  void visit(const copy_stmt* op) override {
    set_result(op);
    std::optional<buffer_info> src = buffers.lookup(op->src);
    if (src) add_use(src->root);
  }

  // TODO: We can handle some of these buffer mutators here.
  template <typename T>
  void visit_opaque_buffer_decl(const T* op, var src) {
    // These are buffer declarations that we don't want to allow aliasing through.
    std::optional<buffer_info> info = buffers.lookup(src);
    if (info) info->allow_alias = false;
    auto s = set_value_in_scope(buffers, op->sym, std::move(info));
    stmt_mutator::visit(op);
  }

  void visit(const make_buffer* op) override { visit_opaque_buffer_decl(op, find_buffer_data_dependency(op->base)); }
  void visit(const slice_buffer* op) override { visit_opaque_buffer_decl(op, op->src); }
  void visit(const slice_dim* op) override { visit_opaque_buffer_decl(op, op->src); }
  void visit(const transpose* op) override { visit_opaque_buffer_decl(op, op->src); }

  template <typename T>
  void visit_buffer_decl(const T* op, var src) {
    auto s = set_value_in_scope(buffers, op->sym, buffers.lookup(src));
    stmt_mutator::visit(op);
  }

  void visit(const crop_buffer* op) override { visit_buffer_decl(op, op->src); }
  void visit(const crop_dim* op) override { visit_buffer_decl(op, op->src); }
  void visit(const clone_buffer* op) override { visit_buffer_decl(op, op->src); }

  void visit(const block* op) override {
    std::vector<stmt> stmts;
    stmts.reserve(op->stmts.size());
    bool changed = false;
    for (auto i = op->stmts.rbegin(); i != op->stmts.rend(); ++i) {
      stmts.push_back(mutate(*i));
      changed = changed || !stmts.back().same_as(*i);
    }
    if (!changed) {
      set_result(op);
    } else {
      std::reverse(stmts.begin(), stmts.end());
      set_result(block::make(std::move(stmts)));
    }
  }

  void visit(const loop* op) override {
    ++loop_level;
    stmt_mutator::visit(op);
    --loop_level;
  }
};

}  // namespace

stmt alias_in_place(const stmt& s, const std::vector<buffer_expr_ptr>& outputs) {
  scoped_trace trace("alias_in_place");
  return in_place_aliaser(outputs).mutate(s);
}

namespace {

template <typename T>
bool match(span<const T> a, span<const T> b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!match(a[i], b[i])) return false;
  }
  return true;
}

class sibling_fuser : public stmt_mutator {
  // Sibling buffer declarations can be fused if they produce the same buffer (same parameters).
  static bool can_fuse(const allocate* a, const allocate* b) {
    return a->storage == b->storage && match(a->elem_size, b->elem_size) && match<dim_expr>(a->dims, b->dims);
  }
  static bool can_fuse(const make_buffer* a, const make_buffer* b) {
    return match(a->base, b->base) && match(a->elem_size, b->elem_size) && match<dim_expr>(a->dims, b->dims);
  }
  static bool can_fuse(const crop_dim* a, const crop_dim* b) {
    return a->src == b->src && a->dim == b->dim && match(a->bounds, b->bounds);
  }
  static bool can_fuse(const crop_buffer* a, const crop_buffer* b) {
    return a->src == b->src && match<interval_expr>(a->bounds, b->bounds);
  }
  static bool can_fuse(const slice_dim* a, const slice_dim* b) {
    return a->src == b->src && a->dim == b->dim && match(a->at, b->at);
  }
  static bool can_fuse(const slice_buffer* a, const slice_buffer* b) {
    return a->src == b->src && match<expr>(a->at, b->at);
  }
  static bool can_fuse(const transpose* a, const transpose* b) { return a->src == b->src && a->dims == b->dims; }

  template <typename T>
  static bool fuse(const T* a, const T* b, stmt& result) {
    if (!a || !b || !can_fuse(a, b)) return false;

    stmt body = block::make({a->body, substitute(b->body, b->sym, a->sym)});
    result = clone_with(a, std::move(body));
    return true;
  }

  static bool fuse(stmt& a, const stmt& b) {
    return fuse(a.as<allocate>(), b.as<allocate>(), a) || fuse(a.as<make_buffer>(), b.as<make_buffer>(), a) ||
           fuse(a.as<crop_dim>(), b.as<crop_dim>(), a) || fuse(a.as<crop_buffer>(), b.as<crop_buffer>(), a) ||
           fuse(a.as<slice_dim>(), b.as<slice_dim>(), a) || fuse(a.as<slice_buffer>(), b.as<slice_buffer>(), a) ||
           fuse(a.as<transpose>(), b.as<transpose>(), a);
  }

public:
  void visit(const block* op) override {
    if (op->stmts.empty()) {
      set_result(op);
      return;
    }
    std::vector<stmt> result;
    result.reserve(op->stmts.size());

    // TODO: This currently only looks for immediately adjacent nodes that can be fused. We can also try to fuse
    // ops with intervening ops, but this isn't obviously a simplification, and in the case of allocations, may
    // increase peak memory usage.
    result.push_back(op->stmts.front());
    bool changed = false;
    auto mutate_back = [&]() {
      stmt m = mutate(result.back());
      if (!m.same_as(result.back())) {
        result.back() = std::move(m);
        changed = true;
      }
    };
    for (std::size_t i = 1; i < op->stmts.size(); ++i) {
      if (!fuse(result.back(), op->stmts[i])) {
        mutate_back();
        result.push_back(op->stmts[i]);
      } else {
        changed = true;
      }
    }
    mutate_back();

    if (changed) {
      set_result(block::make(std::move(result)));
    } else {
      set_result(op);
    }
  }
};

}  // namespace

stmt fuse_siblings(const stmt& s) {
  scoped_trace trace("fuse_siblings");
  return sibling_fuser().mutate(s);
}

stmt implement_copy(const copy_stmt* op, node_context& ctx) {
  scoped_trace trace("implement_copy");
  // Start by making a call to copy.
  // We're going to slice this buffer, to avoid messing with metadata in the user expressions, work on a clone instead.
  var dst = ctx.insert_unique(ctx.name(op->dst) + ".sliced");
  call_stmt::attributes copy_attrs;
  copy_attrs.name = "copy";
  stmt result = call_stmt::make(
      [impl = op->impl](const call_stmt* op, const eval_context& ctx) -> index_t {
        // TODO: This passes the src buffer as an output, not an input, because slinky thinks the bounds of inputs
        // don't matter. But in this case, they do...
        const raw_buffer* src_buf = ctx.lookup_buffer(op->outputs[0]);
        const raw_buffer* dst_buf = ctx.lookup_buffer(op->outputs[1]);
        const raw_buffer* pad_buf = op->outputs[2].defined() ? ctx.lookup_buffer(op->outputs[2]) : &no_padding;
        assert(src_buf);
        assert(dst_buf);
        assert(pad_buf);
        impl(*src_buf, *dst_buf, *pad_buf);
        return 0;
      },
      {}, {op->src, dst, op->pad}, std::move(copy_attrs));

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
  // at the beginning of the pipeline, e.g. that
  // buffer_stride(dst, d) == buffer_stride(dst, d - 1) * buffer_extent(dst, d - 1).

  // Rewrite the source buffer to be only the dimensions of the src we want to pass to copy.
  result = make_buffer::make(op->src, buffer_at(op->src, src_x), buffer_elem_size(op->src), src_dims, result);

  // Any dimensions left need loops and slices.
  for (int d = 0; d < static_cast<index_t>(dst_x.size()); ++d) {
    if (!dst_x[d].defined()) continue;
    result = slice_dim::make(dst, dst, d, dst_x[d], result);
    result = loop::make(dst_x[d], loop::serial, buffer_bounds(dst, d), 1, result);
  }
  return clone_buffer::make(dst, op->dst, std::move(result));
}

stmt implement_copies(const stmt& s, node_context& ctx) {
  scoped_trace trace("implement_copies");
  return recursive_mutate<copy_stmt>(s, [&](const copy_stmt* op) { return implement_copy(op, ctx); });
}

namespace {

class insert_free_into_allocate : public stmt_mutator {
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
  void visit_terminal(stmt s) {
    if (!found && depends_on(s, names).any()) {
      found = true;
      if (visited_something) {
        s = block::make({std::move(s), check::make(call::make(intrinsic::free, {names.front()}))});
      }
    }

    set_result(std::move(s));
  }

  void visit(const loop* op) override { visit_terminal(stmt(op)); }
  void visit(const call_stmt* op) override { visit_terminal(stmt(op)); }
  void visit(const copy_stmt* op) override { visit_terminal(stmt(op)); }
  void visit(const check* op) override { visit_terminal(stmt(op)); }
  void visit(const let_stmt* op) override { visit_terminal(stmt(op)); }

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
      stmt_mutator::visit(op);
      if (base_depends) {
        names.pop_back();
      }
    }
  }

  template <typename T>
  void visit_buffer_mutator(const T* op) {
    const bool found_src = std::find(names.begin(), names.end(), op->src) != names.end();
    found = found || found_src;
    if (std::find(names.begin(), names.end(), op->sym) != names.end()) {
      // Don't look inside if shadowing names.
      set_result(op);
    } else {
      if (found_src) {
        names.push_back(op->sym);
      }
      stmt_mutator::visit(op);
      if (found_src) {
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

class early_free_inserter : public stmt_mutator {
public:
  void visit(const allocate* op) override {
    stmt body = mutate(op->body);
    if (op->storage == memory_type::heap) {
      body = insert_free_into_allocate(op->sym).mutate(body);
    }
    if (body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(allocate::make(op->sym, op->storage, op->elem_size, op->dims, body));
    }
  }
};

}  // namespace

stmt insert_early_free(const stmt& s) {
  scoped_trace trace("insert_early_free");
  return early_free_inserter().mutate(s);
}

namespace {

class pure_dims_remover : public stmt_mutator {
  // Track dimensions of buffers that are provably one.
  using sliceable_dims = std::bitset<64>;
  symbol_map<sliceable_dims> symbol_to_sliceable_dims_;
  symbol_map<box_expr> buffer_bounds;

  static sliceable_dims single_dim(int d) {
    sliceable_dims ret;
    ret.set(d);
    return ret;
  }

  static sliceable_dims erase_dim(sliceable_dims src_sd, int d) {
    sliceable_dims sd = {};
    for (int dim = 0; dim < d; ++dim) {
      sd[dim] = src_sd[dim];
    }
    for (int dim = d; dim + 1 < static_cast<int>(sd.size()); ++dim) {
      sd[dim] = src_sd[dim + 1];
    }
    return sd;
  }

  static bool is_extent_one(const interval_expr& interval) {
    return interval.min.defined() && interval.max.defined() &&
           (interval.is_point() || match(interval.min, interval.max));
  }
  static bool is_extent_one(const dim_expr& d) { return is_extent_one(d.bounds); }

  template <typename T>
  static sliceable_dims find_sliceable(const std::vector<T>& dims) {
    sliceable_dims result = {};
    for (std::size_t i = 0; i < dims.size(); ++i) {
      if (is_extent_one(dims[i])) {
        result.set(i);
      }
    }
    return result;
  }

  void merge_crop(std::optional<box_expr>& bounds, int d, const interval_expr& new_bounds) {
    // Crops produce the intersection of the old bounds and the new bounds.
    interval_expr& bounds_d = (*bounds)[d];
    bounds_d = simplify_intersection(std::move(bounds_d), new_bounds);
  }

public:
  pure_dims_remover() = default;

  void visit(const crop_dim* op) override {
    if (!is_extent_one(op->bounds)) {
      return stmt_mutator::visit(op);
    }
    auto sd = symbol_to_sliceable_dims_.lookup(op->src, {});
    sd.set(op->dim);
    auto s = set_value_in_scope(symbol_to_sliceable_dims_, op->sym, sd);
    std::optional<box_expr> bounds = buffer_bounds.lookup(op->src);
    if (bounds) {
      merge_crop(bounds, op->dim, op->bounds);
    }
    auto set_buffer_bounds = set_value_in_scope(buffer_bounds, op->sym, std::move(bounds));
    stmt_mutator::visit(op);
  }

  void visit(const crop_buffer* op) override {
    sliceable_dims sd = symbol_to_sliceable_dims_.lookup(op->src, {});
    sd |= find_sliceable(op->bounds);
    auto s = set_value_in_scope(symbol_to_sliceable_dims_, op->sym, sd);

    std::optional<box_expr> bounds = buffer_bounds.lookup(op->src);

    if (bounds) {
      for (size_t i = 0; i < op->bounds.size(); ++i) {
        merge_crop(bounds, i, op->bounds[i]);
      }
    }
    auto set_buffer_bounds = set_value_in_scope(buffer_bounds, op->sym, std::move(bounds));

    stmt_mutator::visit(op);
  }

  void visit(const allocate* op) override {
    sliceable_dims sd = find_sliceable(op->dims);
    auto s = set_value_in_scope(symbol_to_sliceable_dims_, op->sym, sd);
    box_expr bounds;
    bounds.reserve(op->dims.size());
    for (const dim_expr& d : op->dims) {
      bounds.push_back(d.bounds);
    }
    auto set_buffer_bounds = set_value_in_scope(buffer_bounds, op->sym, std::move(bounds));
    stmt_mutator::visit(op);
  }

  void visit(const make_buffer* op) override {
    sliceable_dims sd = find_sliceable(op->dims);
    auto s = set_value_in_scope(symbol_to_sliceable_dims_, op->sym, sd);
    box_expr bounds;
    bounds.reserve(op->dims.size());
    for (const dim_expr& d : op->dims) {
      bounds.push_back(d.bounds);
    }
    auto set_buffer_bounds = set_value_in_scope(buffer_bounds, op->sym, std::move(bounds));
    stmt_mutator::visit(op);
  }

  void visit(const constant_buffer* op) override {
    sliceable_dims sd = {};
    for (std::size_t dim = 0; dim < op->value->rank; ++dim) {
      if (op->value->dim(dim).extent() == 1) {
        sd.set(dim);
      }
    }
    auto s = set_value_in_scope(symbol_to_sliceable_dims_, op->sym, sd);
    stmt_mutator::visit(op);
  }

  void visit(const clone_buffer* op) override {
    sliceable_dims sd = symbol_to_sliceable_dims_.lookup(op->src, {});
    auto s = set_value_in_scope(symbol_to_sliceable_dims_, op->sym, sd);
    auto set_buffer_bounds = set_value_in_scope(buffer_bounds, op->sym, buffer_bounds[op->src]);
    stmt_mutator::visit(op);
  }

  void visit(const transpose* op) override {
    sliceable_dims src_sd = symbol_to_sliceable_dims_.lookup(op->src, {});
    sliceable_dims sd = {};
    for (std::size_t dim = 0; dim < op->dims.size(); ++dim) {
      int src_dim = op->dims[dim];
      sd[dim] = src_sd[src_dim];
    }
    auto s = set_value_in_scope(symbol_to_sliceable_dims_, op->sym, sd);

    std::optional<box_expr> bounds = buffer_bounds.lookup(op->src);
    box_expr new_bounds(op->dims.size());

    if (bounds) {
      for (size_t d = 0; d < op->dims.size(); d++) {
        new_bounds[d] = (*bounds)[op->dims[d]];
      }
    }
    auto set_buffer_bounds =
        set_value_in_scope(buffer_bounds, op->sym, bounds ? std::move(new_bounds) : std::move(bounds));

    stmt_mutator::visit(op);
  }

  void visit(const slice_dim* op) override {
    sliceable_dims src_sd = symbol_to_sliceable_dims_.lookup(op->src, {});
    sliceable_dims sd = erase_dim(src_sd, op->dim);
    auto s = set_value_in_scope(symbol_to_sliceable_dims_, op->sym, sd);
    std::optional<box_expr> bounds = buffer_bounds.lookup(op->src);
    if (bounds) {
      bounds->erase(bounds->begin() + op->dim);
    }
    auto set_buffer_bounds = set_value_in_scope(buffer_bounds, op->sym, std::move(bounds));
    stmt_mutator::visit(op);
  }

  void visit(const slice_buffer* op) override {
    sliceable_dims sd = symbol_to_sliceable_dims_.lookup(op->src, {});
    for (int dim = std::min(sd.size(), op->at.size()) - 1; dim >= 0; --dim) {
      if (op->at[dim].defined()) {
        sd = erase_dim(sd, dim);
      }
    }
    auto s = set_value_in_scope(symbol_to_sliceable_dims_, op->sym, sd);
    std::optional<box_expr> bounds = buffer_bounds.lookup(op->src);
    if (bounds) {
      for (int d = std::min(op->at.size(), bounds->size()) - 1; d >= 0; --d) {
        if (!op->at[d].defined()) continue;
        bounds->erase(bounds->begin() + d);
      }
    }
    auto set_buffer_bounds = set_value_in_scope(buffer_bounds, op->sym, std::move(bounds));
    stmt_mutator::visit(op);
  }

  void visit(const call_stmt* op) override {
    if (op->attrs.min_rank == std::numeric_limits<int>::max()) {
      return set_result(stmt{op});
    }

    sliceable_dims sliceable = -1;
    for (var i : op->outputs) {
      sliceable &= symbol_to_sliceable_dims_.lookup(i, {});
    }

    std::vector<expr> slices(sliceable.size());
    for (int d = slices.size() - 1; d >= op->attrs.min_rank; --d) {
      if (sliceable.test(d)) {
        if (buffer_bounds[op->outputs.front()]) {
          slices[d] = (*buffer_bounds[op->outputs.front()])[d].min;
        } else {
          slices[d] = buffer_min(op->outputs.front(), d);
        }
      }
    }
    // Remove trailing slices we didn't set.
    while (!slices.empty() && !slices.back().defined()) {
      slices.pop_back();
    }
    if (slices.empty()) {
      set_result(op);
      return;
    }

    stmt result{op};
    std::set<var> sliced;
    for (var i : op->outputs) {
      if (sliced.insert(i).second) {
        result = slice_buffer::make(i, i, slices, result);
      }
    }
    for (var i : op->inputs) {
      if (sliced.insert(i).second) {
        result = slice_buffer::make(i, i, slices, result);
      }
    }
    set_result(result);
  }

  void visit(const copy_stmt* op) override { set_result(op); }
};

}  // namespace

stmt remove_pure_dims(const stmt& s) {
  scoped_trace trace("remove_pure_dims");
  return pure_dims_remover{}.mutate(s);
}

namespace {

class deshadower : public substitutor {
  node_context& ctx;
  symbol_map<var> symbols;
  var in_loop;

  std::vector<scoped_value_in_symbol_map<var>> decls;

public:
  deshadower(node_context& ctx, span<var> external_symbols) : ctx(ctx) {
    for (var i : external_symbols) {
      symbols[i] = i;
    }
  }

  var rename(var x) {
    std::string suffix = in_loop.defined() ? "." + ctx.name(in_loop) : "";
    return ctx.insert_unique(ctx.name(x) + suffix);
  }

  var visit_symbol(var x) override {
    std::optional<var> new_x = symbols.lookup(x);
    return new_x ? *new_x : x;
  }

  var enter_decl(var x) override {
    var renamed = symbols.contains(x) ? rename(x) : x;
    decls.push_back(set_value_in_scope(symbols, x, renamed));
    return renamed;
  }

  void exit_decls(int n) override { decls.erase(decls.begin() + decls.size() - n, decls.end()); }

  void visit(const loop* op) override {
    interval_expr bounds = mutate(op->bounds);
    expr step = mutate(op->step);
    expr max_workers = mutate(op->max_workers);
    var sym = symbols.contains(op->sym) ? rename(op->sym) : op->sym;
    auto s = set_value_in_scope(symbols, op->sym, sym);
    var old_in_loop = in_loop;
    in_loop = sym;
    stmt body = mutate(op->body);
    in_loop = old_in_loop;
    if (sym == op->sym && bounds.same_as(op->bounds) && step.same_as(op->step) &&
        max_workers.same_as(op->max_workers) && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(loop::make(sym, std::move(max_workers), std::move(bounds), std::move(step), std::move(body)));
    }
  }

  void visit(const allocate* op) override {
    expr elem_size = mutate(op->elem_size);
    std::vector<dim_expr> dims;
    dims.reserve(op->dims.size());
    bool changed = false;
    for (const dim_expr& i : op->dims) {
      dims.push_back({mutate(i.bounds), mutate(i.stride), mutate(i.fold_factor)});
      changed = changed || !dims.back().same_as(i);
    }
    // We don't rename allocations.
    // TODO: We don't want to rename allocations that shadow make_buffer (we rename make_buffer instead below), but we
    // do want to rename allocations that shadow anything else.
    auto s = set_value_in_scope(symbols, op->sym, op->sym);
    stmt body = mutate(op->body);
    if (!changed && elem_size.same_as(op->elem_size) && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(allocate::make(op->sym, op->storage, std::move(elem_size), std::move(dims), std::move(body)));
    }
  }

  void visit(const make_buffer* op) override {
    expr base = mutate(op->base);
    expr elem_size = mutate(op->elem_size);
    std::vector<dim_expr> dims;
    dims.reserve(op->dims.size());
    for (const dim_expr& i : op->dims) {
      dims.push_back({mutate(i.bounds), mutate(i.stride), mutate(i.fold_factor)});
    }
    // We want to keep the name of allocates that shadow make_buffers, so rename the make_buffer instead.
    // TODO: We should only do this if there is actually an allocate shadowing this buffer.
    var sym = rename(op->sym);
    auto s = set_value_in_scope(symbols, op->sym, sym);
    stmt body = mutate(op->body);
    set_result(make_buffer::make(sym, std::move(base), std::move(elem_size), std::move(dims), std::move(body)));
  }

  using node_mutator::visit;
};

// This mutator attempts to re-write buffer mutators to be performed in-place when possible. Most mutators are more
// efficient when performed in place.
class reuse_shadows : public stmt_mutator {
  // Buffers that can be mutated in place are true in this map.
  symbol_map<bool> can_mutate;

public:
  template <typename T>
  void visit_buffer_mutator(const T* op) {
    stmt body = op->body;
    var sym = op->sym;
    // If we don't know about a buffer, we assume we cannot mutate it in place. This covers input and output buffers.
    bool can_mutate_src = can_mutate[op->src] && *can_mutate[op->src];
    // TODO: This mutator has quadratic complexity. It's hard to do this in one pass...
    if (can_mutate_src && !depends_on(body, op->src).any()) {
      // We can re-use the src because the body doesn't use it, and it will not create a data race.
      sym = op->src;
      body = substitute(body, op->sym, sym);
    }

    // Buffers start out mutable.
    can_mutate[op->sym] = true;

    body = mutate(body);

    if (sym == op->sym && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(clone_with(op, sym, std::move(body)));
    }
  }

  template <typename T>
  void visit_buffer_decl(const T* op, bool decl_mutable = true) {
    can_mutate[op->sym] = decl_mutable;
    stmt_mutator::visit(op);
  }

  stmt mutate_closure(stmt s) {
    // We're entering a stmt executed in parallel. All the buffers in scope cannot be mutated in this scope.
    symbol_map<bool> old_can_mutate;
    std::swap(can_mutate, old_can_mutate);
    s = mutate(s);
    can_mutate = std::move(old_can_mutate);

    std::vector<var> referenced = find_dependencies(s);
    std::vector<std::pair<var, expr>> lets;
    for (var i : referenced) {
      lets.push_back({i, expr(i)});
    }
    return let_stmt::make(std::move(lets), std::move(s), /*is_closure=*/true);
  }

  void visit(const loop* op) override {
    if (!prove_true(op->max_workers == loop::serial)) {
      stmt body = mutate_closure(op->body);
      set_result(loop::make(op->sym, op->max_workers, op->bounds, op->step, std::move(body)));
    } else {
      stmt_mutator::visit(op);
    }
  }

  void visit(const async* op) override {
    // We're entering an async task. All the buffers in scope cannot be mutated in this scope.
    stmt task = mutate_closure(op->task);
    stmt body = mutate_closure(op->body);
    set_result(async::make(op->sym, std::move(task), std::move(body)));
  }

  void visit(const allocate* op) override { visit_buffer_decl(op); }
  void visit(const make_buffer* op) override { visit_buffer_decl(op); }
  void visit(const constant_buffer* op) override {
    // Constant buffers are not mutable, because the raw_buffer object we use is not allocated by a declaration.
    visit_buffer_decl(op, false);
  }

  void visit(const crop_buffer* op) override { visit_buffer_mutator(op); }
  void visit(const crop_dim* op) override { visit_buffer_mutator(op); }
  void visit(const slice_buffer* op) override { visit_buffer_mutator(op); }
  void visit(const slice_dim* op) override { visit_buffer_mutator(op); }
  void visit(const transpose* op) override { visit_buffer_mutator(op); }
};

}  // namespace

stmt deshadow(const stmt& s, span<var> symbols, node_context& ctx) {
  scoped_trace trace("deshadow");
  return deshadower(ctx, symbols).mutate(s);
}
stmt optimize_symbols(const stmt& s, node_context& ctx) {
  scoped_trace trace("optimize_symbols");
  return reuse_shadows().mutate(s);
}

namespace {

class node_canonicalizer : public node_mutator {
  std::map<expr, expr, node_less> exprs;
  std::map<stmt, stmt, node_less> stmts;

public:
  using node_mutator::mutate;

  stmt mutate(const stmt& s) override {
    if (s.as<call_stmt>()) {
      // calls can capture state in their target that we can't compare for equality here.
      return s;
    }
    stmt& result = stmts[s];
    if (!result.defined()) result = node_mutator::mutate(s);
    return result;
  }

  expr mutate(const expr& e) override {
    expr& result = exprs[e];
    if (!result.defined()) result = node_mutator::mutate(e);
    return result;
  }
};

}  // namespace

expr canonicalize_nodes(const expr& e) { return node_canonicalizer().mutate(e); }
stmt canonicalize_nodes(const stmt& s) {
  scoped_trace trace("canonicalize_nodes");
  return node_canonicalizer().mutate(s);
}

namespace {

class task_parallelizer : public node_mutator {
  std::set<var> consumed, produced;
  bool barrier = false;
  symbol_map<var> aliases;

  var lookup_alias(var x) {
    auto alias = aliases.lookup(x);
    return alias ? *alias : x;
  }

  template <typename T>
  static void remove_from_set(std::set<T>& set, const T& x) {
    auto it = set.find(x);
    if (it != set.end()) set.erase(it);
  }

  static bool should_async(const stmt& s) {
    if (const block* b = s.as<block>()) {
      for (const stmt& i : b->stmts) {
        if (should_async(i)) return true;
      }
      return false;
    } else if (s.as<check>()) {
      return false;
    } else {
      return true;
    }
  }

  // Here we are going to construct a DAG of stmts with their dependencies determining the edges.
  struct stage {
    stmt body;

    // Set of buffers consumed (read by a call or copy), or produced (written by a call or copy) in this stage.
    std::set<var> consumed, produced;

    // Stages consuming something we produce, and producing something we consume, respectively.
    std::set<stage*> consumers, producers;

    // Fuse the producer `p` into this stage, the consumer, making `p` a no-op.
    void fuse(stage* p) {
      body = block::make(std::move(p->body), std::move(body));
      for (stage* i : p->consumers) {
        remove_from_set(i->producers, p);
        i->producers.insert(this);
      }
      for (stage* i : p->producers) {
        remove_from_set(i->consumers, p);
        i->consumers.insert(this);
      }
      consumers.insert(p->consumers.begin(), p->consumers.end());
      producers.insert(p->producers.begin(), p->producers.end());
      consumed.insert(p->consumed.begin(), p->consumed.end());
      produced.insert(p->produced.begin(), p->produced.end());

      remove_from_set(consumers, this);
      remove_from_set(consumers, p);
      remove_from_set(producers, this);
      remove_from_set(producers, p);
    }
  };

  stage mutate_stage(const stmt& s) {
    consumed.clear();
    produced.clear();
    stage result = {mutate(s)};
    result.consumed = std::move(consumed);
    result.produced = std::move(produced);
    return result;
  }

public:
  void visit(const block* b) override {
    std::vector<stage> stages;
    stages.reserve(b->stmts.size());

    // Mutating the statements in the block, and determine which stages are producers for prior stages.
    bool changed = false;
    for (const stmt& i : b->stmts) {
      stages.push_back(mutate_stage(i));
      changed = changed || !stages.back().body.same_as(i);
      for (stage& s : stages) {
        if (&s == &stages.back()) continue;
        if (!empty_intersection(s.produced, stages.back().consumed) ||
            !empty_intersection(s.produced, stages.back().produced) ||
            !empty_intersection(s.consumed, stages.back().produced)) {
          // We are a consumer of this stage, or we are a producer of something this stage consumes.
          stages.back().producers.insert(&s);
          s.consumers.insert(&stages.back());
        }
      }
    }

    // Populate the consumers, and the outer produced/consumed set result for this block.
    produced.clear();
    consumed.clear();
    for (stage& s : stages) {
      consumed.insert(s.consumed.begin(), s.consumed.end());
      produced.insert(s.produced.begin(), s.produced.end());
    }

    if (barrier) {
      // For now, give up and don't parallelize anything. We could do better here, and parallelize between barriers.
      // Currently, the only barriers are `semaphore_*` intrinsics, which we hope to get rid of anyways.
      if (changed) {
        std::vector<stmt> stmts;
        stmts.reserve(stages.size());
        for (stage& i : stages) {
          stmts.push_back(std::move(i.body));
        }
        set_result(block::make(std::move(stmts)));
      } else {
        set_result(b);
      }
      return;
    }

    // From here onwards, we track which nodes we've produced.
    std::set<stage*> produced;
    auto produce = [&](stage* s) {
      for (stage* j : s->producers) {
        remove_from_set(j->consumers, s);
      }
      produced.insert(s);
    };

    // Fuse nodes that we don't want to compute in their own task.
    for (stage& p : stages) {
      stage* fuse = nullptr;
      if (p.consumers.size() == 1) {
        // This producer only has one consumer, fuse into the consumer if it also only has one producer.
        fuse = *p.consumers.begin();
        if (fuse->producers.size() == 1) {
          // This edge should be collapsed into one task.
          fuse->fuse(&p);
          produced.insert(&p);
        }
      }
    }

    // Build the result. We start at the end of the body, and produce everything that has no consumers. Producing these
    // stages results in other stages having no consumers, which can then run on the next iteration.
    // TODO: This synchronizes between each group of tasks that can be executed, which is excessive synchronization,
    // we should be able to wait for the specific tasks that we need to complete using `wait_for`.
    stmt result;
    while (true) {
      // All of the stages that are ready to compute (have no remaining consumers) either go in this set, or go in the
      // `synchronous` vector, for stmts we don't want to bother computing asynchronously.
      std::set<stage*, std::greater<stage*>> to_produce;
      std::vector<stmt> synchronous;
      for (stage& s : stages) {
        if (produced.count(&s)) continue;

        assert(s.body.defined());
        if (s.consumers.empty()) {
          // This stage can start now.
          if (!should_async(s.body)) {
            synchronous.push_back(std::move(s.body));
            produce(&s);
          } else {
            to_produce.insert(&s);
          }
        }
      }

      // Put the synchronous stages together with the first task to compute asynchronously.
      if (!to_produce.empty()) {
        stage* s = *to_produce.begin();
        synchronous.push_back(std::move(s->body));
        produce(s);
        to_produce.erase(to_produce.begin());
      }

      stmt tasks = block::make(std::move(synchronous));

      // Build the rest of the tasks asynchronously.
      for (stage* s : to_produce) {
        assert(tasks.defined());
        tasks = async::make(var(), std::move(s->body), std::move(tasks));
        produce(s);
      }
      if (!tasks.defined()) {
        break;
      }
      result = block::make(std::move(tasks), std::move(result));
    }
    assert(produced.size() == stages.size());
    set_result(std::move(result));
  }

  void visit(const call_stmt* op) override {
    for (var i : op->inputs) {
      consumed.insert(lookup_alias(i));
    }
    for (var i : op->outputs) {
      produced.insert(lookup_alias(i));
    }
    set_result(op);
  }

  void visit(const copy_stmt* op) override {
    consumed.insert(lookup_alias(op->src));
    produced.insert(lookup_alias(op->dst));
    set_result(op);
  }

  bool in_check = false;

  void visit(const variable* op) override {
    set_result(op);
    if (in_check && op->field != buffer_field::none) {
      // Treat buffers accessed by a check as being produced, so we don't compute reorder a check w.r.t. the things its
      // checking.
      produced.insert(lookup_alias(op->sym));
    }
  }

  static bool is_barrier(intrinsic fn) {
    switch (fn) {
    case intrinsic::semaphore_init:
    case intrinsic::semaphore_wait:
    case intrinsic::semaphore_signal: return true;
    default: return false;
    }
  }

  void visit(const call* op) override {
    node_mutator::visit(op);
    if (op->intrinsic == intrinsic::buffer_at) {
      assert(op->args.size() >= 1);
      auto buf = as_variable(op->args[0]);
      // Assume we are both producing and consuming this buffer.
      consumed.insert(lookup_alias(*buf));
      produced.insert(lookup_alias(*buf));
    }
    barrier = barrier || is_barrier(op->intrinsic);
  }

  void visit(const check* op) override {
    in_check = true;
    node_mutator::visit(op);
    in_check = false;
  }

  template <typename T>
  void visit_buffer_decl(const T* op, var src = var()) {
    if (src.defined()) {
      // Just remember what this is an alias of.
      auto s = set_value_in_scope(aliases, op->sym, lookup_alias(src));
      node_mutator::visit(op);
    } else {
      // Handle shadowing of op->sym by saving the state of op->sym being produced or consumed, then clearing it.
      auto consumed_i = consumed.find(op->sym);
      auto produced_i = produced.find(op->sym);
      bool was_consumed = consumed_i != consumed.end();
      bool was_produced = produced_i != produced.end();
      if (was_consumed) consumed.erase(consumed_i);
      if (was_produced) produced.erase(produced_i);

      node_mutator::visit(op);

      // Restore the state of op->sym being produced or consumed.
      if (was_produced) {
        produced.insert(op->sym);
      } else {
        auto i = produced.find(op->sym);
        if (i != produced.end()) produced.erase(i);
      }
      if (was_consumed) {
        consumed.insert(op->sym);
      } else {
        auto i = consumed.find(op->sym);
        if (i != consumed.end()) consumed.erase(i);
      }
    }
  }

  void visit(const allocate* op) override { visit_buffer_decl(op); }
  void visit(const constant_buffer* op) override { visit_buffer_decl(op); }
  void visit(const make_buffer* op) override { visit_buffer_decl(op, find_buffer_data_dependency(op->base)); }
  void visit(const clone_buffer* op) override { visit_buffer_decl(op, op->src); }
  void visit(const crop_dim* op) override { visit_buffer_decl(op, op->src); }
  void visit(const crop_buffer* op) override { visit_buffer_decl(op, op->src); }
  void visit(const slice_dim* op) override { visit_buffer_decl(op, op->src); }
  void visit(const slice_buffer* op) override { visit_buffer_decl(op, op->src); }
  void visit(const transpose* op) override { visit_buffer_decl(op, op->src); }

  using node_mutator::visit;
};

}  // namespace

stmt parallelize_tasks(const stmt& s) {
  scoped_trace trace("parallelize_tasks");
  return task_parallelizer().mutate(s);
}

namespace {
class semaphore_cleaner : public node_mutator {
  struct alloc_info {
    // Is this allocation semaphores?
    bool is_semaphores = false;

    // Is this allocation used asynchronously anywhere?
    bool used_async = false;

    // Depth of this allocation node.
    int async_depth = 0;
  };

  // Each async stmt increases the depth by one. This includes loops and async
  // nodes.
  int async_depth = 0;
  symbol_map<alloc_info> allocations;

public:
  using node_mutator::visit;

  void visit(const call_stmt* op) override {
    if (op->attrs.name == "init_semaphores") {
      allocations[op->outputs[0]]->is_semaphores = true;
    }

    node_mutator::visit(op);
  }

  void visit(const allocate* op) override {
    auto s = set_value_in_scope(allocations, op->sym, alloc_info{false, false, async_depth});
    stmt body = mutate(op->body);
    auto info = allocations.lookup(op->sym);
    if (info->is_semaphores && !info->used_async) {
      // These semaphores are not needed.
      const block* maybe_block = body.as<block>();
      if (maybe_block) {
        // Remove the initialization call, because otherwise the following
        // substitute call will fail.
        const call_stmt* maybe_init = maybe_block->stmts[0].as<call_stmt>();
        if (maybe_init && maybe_init->attrs.name == "init_semaphores") {
          std::vector<stmt> new_block(maybe_block->stmts.begin() + 1, maybe_block->stmts.end());
          body = block::make(new_block);
          body = substitute(body, op->sym, expr());
          set_result(std::move(body));
          return;
        }
      }
    }

    if (body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(allocate::make(op->sym, op->storage, std::move(op->elem_size), std::move(op->dims), std::move(body)));
    }
  }

  void visit(const loop* op) override {
    int op_async_depth = !is_constant(op->max_workers, 1);
    async_depth += op_async_depth;
    node_mutator::visit(op);
    async_depth -= op_async_depth;
  }

  void visit(const async* op) override {
    async_depth++;
    node_mutator::visit(op);
    async_depth--;
  }

  void visit(const call* op) override {
    if (op->intrinsic == intrinsic::semaphore_wait || op->intrinsic == intrinsic::semaphore_signal) {
      for (size_t sem_arg = 0; sem_arg < op->args.size(); sem_arg += 2) {
        auto buffer_at = as_intrinsic(op->args[sem_arg], intrinsic::buffer_at);
        assert(buffer_at);
        auto var = as_variable(buffer_at->args[0]);
        assert(var);
        alloc_info& info = *allocations[*var];
        info.used_async = info.used_async || info.async_depth != async_depth;
      }
    }

    node_mutator::visit(op);
    return;
  }
};
}  // namespace

stmt cleanup_semaphores(const stmt& s) {
  scoped_trace trace("cleanup_semaphores");
  return semaphore_cleaner().mutate(s);
}

}  // namespace slinky
