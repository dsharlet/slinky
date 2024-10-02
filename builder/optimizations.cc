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

#include "base/chrome_trace.h"
#include "builder/node_mutator.h"
#include "builder/simplify.h"
#include "builder/substitute.h"
#include "runtime/buffer.h"
#include "runtime/depends_on.h"
#include "runtime/evaluate.h"
#include "runtime/expr.h"

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
    // It doesn't matter what we choose for the fold factor because the stride is 0.
    src_dim.fold_factor = expr();
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

// Replace copies between buffers a and b with calls to pad.
class copy_remover : public node_mutator {
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
    if (!op->padding || op->padding->empty()) {
      // No padding, this copy is now a no-op.
      set_result(stmt());
      return;
    }
    set_result(op);
  }

  void visit(const call_stmt* op) override {
    if (op->attrs.name == "memcpy" && op->inputs.size() == 1 && op->outputs.size() == 1 &&
        is_a_and_b(op->inputs[0], op->outputs[0])) {
      expr input_size = call::make(intrinsic::buffer_size_bytes, {op->inputs[0]});
      expr output_size = call::make(intrinsic::buffer_size_bytes, {op->outputs[0]});
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
    node_mutator::visit(op);
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

class buffer_aliaser : public node_mutator {
  node_context& ctx;

  struct alias_info {
    // Parameters for this alias's make_buffer call.
    std::vector<dim_expr> dims;
    expr elem_size;

    // Coordinates to pass to buffer_at to get the base pointer for the alias.
    std::vector<expr> at;

    // Mapping of dimensions of the alias to the original buffer's dimensions.
    std::vector<int> permutation;

    // If true, we know this alias is a subset of the aliased buffer.
    bool assume_in_bounds = false;

    bool is_copy = false;

    bool is_contiguous_copy = false;
  };

  class buffer_info {
  public:
    // The buffer allocation parameters.
    std::vector<dim_expr> dims;
    expr elem_size;

    bool is_input;
    bool is_output;

    // Possible aliases of this allocation.
    std::map<var, alias_info> aliases;

    // If we decided to alias this buffer, we might have grown the bounds. If so, we need to make a new allocation with
    // this symbol, but make a crop of it for the original bounds.
    var shared_alloc_sym;

  public:
    buffer_info(std::vector<dim_expr> dims, expr elem_size, bool is_input = false, bool is_output = false)
        : dims(std::move(dims)), elem_size(std::move(elem_size)), is_input(is_input), is_output(is_output) {}

    void maybe_alias(var s, alias_info a) {
      assert(aliases.count(s) == 0);
      aliases[s] = std::move(a);
    }
    void do_not_alias(var s) { aliases.erase(s); }
  };
  symbol_map<buffer_info> buffers;

  static bool alias_compatible(
      const allocate* op, const alias_info& alias, var target, const buffer_info& target_info) {
    scoped_trace trace("alias_compatible");
    assert(op->dims.size() == alias.dims.size());

    if (alias.is_contiguous_copy) {
      // We just assume flat copies are OK.
      return true;
    }
    for (std::size_t d = 0; d < op->dims.size(); ++d) {
      if (alias.permutation[d] < 0) {
        // This dimension must be a broadcast.
        continue;
      }
      const dim_expr& alias_dim = alias.dims[alias.permutation[d]];
      if (!alias.assume_in_bounds) {
        assert(alias.permutation.size() == op->dims.size());
        if (!prove_true(op->dims[d].bounds.min >= alias_dim.bounds.min) ||
            !prove_true(op->dims[d].bounds.max <= alias_dim.bounds.max)) {
          // We don't know if this target is big enough for this allocation.
          if (target_info.is_input || target_info.is_output) {
            // We can't reallocate this buffer.
            return false;
          }
        }
      }
      if (op->dims[d].stride.defined()) {
        if (!prove_true(op->dims[d].stride == alias_dim.stride)) {
          // This alias would violate a constraint on the stride of the buffer.
          return false;
        }
      }

      const expr& target_fold_factor = target_info.dims[alias.permutation[d]].fold_factor;
      if (op->dims[d].fold_factor.defined()) {
        if (!target_fold_factor.defined() || is_constant(target_fold_factor, dim::unfolded)) {
          // The target isn't folded, we can alias this buffer. We lose our fold factor, but it's not going to occupy
          // any memory anyways if it's an alias.
        } else if (!prove_true(target_fold_factor >= op->dims[d].fold_factor)) {
          // The fold factor of this allocation does not evenly divide the target fold factor.
          // TODO: We could increase the fold factor like we do the bounds.
          return false;
        } else if (!prove_true((target_info.dims[alias.permutation[d]].bounds.min % target_fold_factor) ==
                               (op->dims[d].bounds.min % op->dims[d].fold_factor))) {
          // The mins of folded buffers are not aligned.
          return false;
        }
      } else if ((target_fold_factor.defined() && !is_constant(target_fold_factor, dim::unfolded)) &&
                 !prove_true(op->dims[d].extent() <= target_fold_factor)) {
        // If the target is folded, but the op is not, we can only alias it if the extent of this dimension
        // is less than the fold factor.
        return false;
      }
    }
    return true;
  }

public:
  buffer_aliaser(
      node_context& ctx, const std::vector<buffer_expr_ptr>& inputs, const std::vector<buffer_expr_ptr>& outputs)
      : ctx(ctx) {
    for (const buffer_expr_ptr& i : inputs) {
      buffers[i->sym()] = buffer_info(i->dims(), i->elem_size(), /*is_input=*/true, /*is_output=*/false);
    }
    for (const buffer_expr_ptr& i : outputs) {
      buffers[i->sym()] = buffer_info(i->dims(), i->elem_size(), /*is_input=*/false, /*is_output=*/true);
    }
  }

  void visit(const allocate* op) override {
    auto s = set_value_in_scope(buffers, op->sym, buffer_info(op->dims, op->elem_size));
    stmt body = mutate(op->body);

    scoped_trace trace("visit(const allocate*)");
    buffer_info info = std::move(*buffers[op->sym]);

    // When an allocation goes out of scope, we should remove it as an aliasing candidate.
    for (std::optional<buffer_info>& i : buffers) {
      if (i) i->do_not_alias(op->sym);
    }

    box_expr op_dims_bounds = dims_bounds(op->dims);
    for (auto& target : info.aliases) {
      var target_var = target.first;
      alias_info& alias = target.second;

      var alloc_var = target_var;
      std::optional<buffer_info>& target_info = buffers[target_var];
      assert(target_info);

      if (!alias_compatible(op, alias, target_var, *target_info)) {
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

      if (!alias.assume_in_bounds && !alias.is_contiguous_copy) {
        if (!target_info->is_output) {
          assert(!target_info->is_input);  // We shouldn't be trying to write to an input anyways.
          // We allocated this buffer, make it big enough to share with this buffer.
          std::string old_name =
              ctx.name(target_info->shared_alloc_sym.defined() ? target_info->shared_alloc_sym : target_var);
          target_info->shared_alloc_sym = ctx.insert_unique(old_name + "/" + ctx.name(op->sym));
          alloc_var = target_info->shared_alloc_sym;
          for (std::size_t d = 0; d < op->dims.size(); ++d) {
            // TODO: We may have proven this is unnecessary in alias_compatible, we can avoid this in such cases.
            // We need the bounds of the alias, as it exists in the target buffer. `alias.at` tells us where this alias
            // starts.
            target_info->dims[d].bounds |=
                alias.at[alias.permutation[d]] + min_extent(0, alias.dims[alias.permutation[d]].bounds.extent());
          }
        } else {
          // In this case, alias_compatible must have determined that we do not need to grow the allocation.
        }
      }

      // Replace the allocation with a buffer using the dims (and maybe elem_size) the alias wants.
      expr elem_size = alias.elem_size.defined() ? alias.elem_size : op->elem_size;
      var sym = info.shared_alloc_sym.defined() ? info.shared_alloc_sym : op->sym;
      if (sym != op->sym) {
        body = clone_buffer::make(op->sym, sym, std::move(body));
      }
      stmt result = make_buffer::make(sym, buffer_at(alloc_var, alias.at), elem_size, alias.dims, std::move(body));
      // Wrap with the original buffer in case we want to use the metadata in the construction of the buffer.
      result = make_buffer::make(sym, expr(), elem_size, op->dims, result);

      if (elem_size.defined()) {
        result = block::make({check::make(elem_size == op->elem_size), result});
      }

      // If we aliased the source and destination of a copy with no padding, the copy can be removed.
      result = remove_copy(result, op->sym, target_var);

      if (!alias.is_copy) {
        // This wasn't a copy, we actually did some computation in place. We can't alias another buffer to this target
        // without understanding the lifetimes more carefully.
        // TODO: I think this is a hack, but I'm not sure. I think maybe the proper thing to do is track a box_expr
        // of the region that has been aliased so far, and allow another alias as long as it does not intersect that
        // region. That will likely be very difficult to do symbolically.
        for (std::optional<buffer_info>& i : buffers) {
          if (!i) continue;
          i->do_not_alias(target_var);
        }
      }

      set_result(std::move(result));
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
        set_result(clone_with(op, std::move(body)));
      }
    } else {
      set_result(op);
    }
  }

  // Make dimensions that assign strides that are contiguous and ascending.
  static std::vector<dim_expr> make_contiguous_dims(var buf, std::size_t rank) {
    std::vector<dim_expr> dims(rank);
    expr stride = buffer_elem_size(buf);
    for (std::size_t i = 0; i < rank; ++i) {
      dims[i].bounds = buffer_bounds(buf, i);
      dims[i].stride = stride;
      stride *= dims[i].bounds.extent();
    }
    return dims;
  }

  void visit(const call_stmt* op) override {
    scoped_trace trace("visit(const call_stmt*)");
    set_result(op);
    if (op->attrs.name == "memcpy") {
      assert(op->inputs.size() == 1);
      assert(op->outputs.size() == 1);
      var in = op->inputs[0];
      var out = op->outputs[0];
      std::optional<buffer_info>& input_info = buffers[in];
      std::optional<buffer_info>& output_info = buffers[out];
      if (input_info && output_info) {
        alias_info fwd;
        fwd.dims = make_contiguous_dims(in, input_info->dims.size());
        fwd.at = buffer_mins(out, output_info->dims.size());
        fwd.is_contiguous_copy = true;
        input_info->maybe_alias(out, std::move(fwd));

        alias_info back;
        back.dims = make_contiguous_dims(out, output_info->dims.size());
        back.at = buffer_mins(in, input_info->dims.size());
        back.is_contiguous_copy = true;
        output_info->maybe_alias(in, std::move(back));
      }
    } else if (op->attrs.allow_in_place) {
      // If input is repeated, we don't want to add into the alias info again.
      std::set<var> unique_inputs(op->inputs.begin(), op->inputs.end());
      for (var i : unique_inputs) {
        std::optional<buffer_info>& input_info = buffers[i];
        if (!input_info || input_info->is_input) {
          // We can't write to this buffer.
          continue;
        }
        for (var o : op->outputs) {
          std::optional<buffer_info>& output_info = buffers[o];
          if (!output_info) continue;

          if (input_info->dims.size() != output_info->dims.size()) {
            // This is allow_in_place, but appears to not be elementwise?
            continue;
          }
          size_t rank = input_info->dims.size();

          alias_info fwd;
          fwd.dims = buffer_dims(o, rank);
          fwd.at = buffer_mins(i, rank);
          fwd.assume_in_bounds = true;
          fwd.permutation.resize(rank);
          std::iota(fwd.permutation.begin(), fwd.permutation.end(), 0);
          input_info->maybe_alias(o, std::move(fwd));

          alias_info back;
          // Use the bounds of the output, but the memory layout of the input.
          back.dims.resize(rank);
          for (size_t d = 0; d < rank; ++d) {
            back.dims[d] = {buffer_bounds(o, d), buffer_stride(i, d), buffer_fold_factor(i, d)};
          }
          back.at = buffer_mins(o, rank);
          back.assume_in_bounds = true;
          back.permutation.resize(rank);
          std::iota(back.permutation.begin(), back.permutation.end(), 0);
          output_info->maybe_alias(i, std::move(back));
        }
      }
    }
  }

  void alias_copy_dst(const copy_stmt* op) {
    scoped_trace trace("alias_copy_dst");
    std::optional<buffer_info>& info = buffers[op->dst];
    if (!info || info->is_output) {
      // We didn't allocate the dst.
      return;
    }
    // We allocated the dst. We might be able to replace the allocation with an alias of the src.
    // This case is a straightforward use of is_copy, which produces the dims that should be the src of a copy, which
    // are the same dimensions we want the dst to be.

    alias_info a;
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
    a.is_copy = true;

    a.elem_size = buffer_elem_size(op->src);

    info->maybe_alias(op->src, std::move(a));
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

    a.is_copy = true;
    a.elem_size = buffer_elem_size(op->dst);

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
    symbol_map<buffer_info> old_buffers(buffers.size());
    std::swap(old_buffers, buffers);
    for (std::size_t i = 0; i < old_buffers.size(); ++i) {
      if (old_buffers[i]) {
        buffers[i] = buffer_info(
            old_buffers[i]->dims, old_buffers[i]->elem_size, old_buffers[i]->is_input, old_buffers[i]->is_output);
        buffers[i]->shared_alloc_sym = old_buffers[i]->shared_alloc_sym;
      }
    }

    auto set_info_sym = set_value_in_scope(buffers, op->sym, buffers[op->src]);
    node_mutator::visit(op);

    scoped_trace trace("visit_buffer_mutator");

    for (std::optional<buffer_info>& i : buffers) {
      if (!i) continue;
      auto j = i->aliases.find(op->sym);
      if (j != i->aliases.end()) {
        handler(j->second);
      }
      for (auto& a : i->aliases) {
        // We need to substitute uses of sym with uses of src in the aliases we added here.
        for (dim_expr& d : a.second.dims) {
          d.bounds = substitute(d.bounds, op->sym, op->src);
          d.stride = substitute(d.stride, op->sym, op->src);
          d.fold_factor = substitute(d.fold_factor, op->sym, op->src);
        }
        a.second.elem_size = substitute(a.second.elem_size, op->sym, op->src);
        for (expr& i : a.second.at) {
          i = substitute(i, op->sym, op->src);
        }
      }
    }

    // Add the old alias candidates back to the alias info.
    old_buffers.reserve(std::max(buffers.size(), old_buffers.size()));
    for (std::size_t i = 0; i < buffers.size(); ++i) {
      if (!buffers[i]) continue;
      std::optional<buffer_info> info = std::move(buffers[i]);
      std::optional<buffer_info>& old_info = old_buffers[var(i) != op->sym ? var(i) : op->src];
      if (!old_info) {
        old_info = buffer_info(info->dims, info->elem_size, info->is_input, info->is_output);
      } else {
        old_info->dims = std::move(info->dims);
        old_info->elem_size = std::move(info->elem_size);
      }
      if (info->shared_alloc_sym.defined()) {
        assert(!old_info->shared_alloc_sym.defined() || old_info->shared_alloc_sym == info->shared_alloc_sym);
        old_info->shared_alloc_sym = info->shared_alloc_sym;
      }
      for (auto& j : info->aliases) {
        old_info->maybe_alias(j.first == op->sym ? op->src : j.first, std::move(j.second));
      }
    }
    std::swap(old_buffers, buffers);
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
  }

  void visit(const crop_dim* op) override {
    visit_buffer_mutator(op, [](alias_info&) {});
  }

  void visit(const clone_buffer* op) override {
    visit_buffer_mutator(op, [](alias_info&) {});
  }

  void visit(const transpose*) override {
    // TODO: We should be able to handle this.
    std::abort();
  }
};

}  // namespace

stmt alias_buffers(const stmt& s, node_context& ctx, const std::vector<buffer_expr_ptr>& inputs,
    const std::vector<buffer_expr_ptr>& outputs) {
  scoped_trace trace("alias_buffers");
  return buffer_aliaser(ctx, inputs, outputs).mutate(s);
}

stmt implement_copy(const copy_stmt* op, node_context& ctx) {
  scoped_trace trace("implement_copy");
  // Start by making a call to copy.
  call_stmt::attributes copy_attrs;
  copy_attrs.name = "copy";
  stmt result = call_stmt::make(
      [padding = op->padding](const call_stmt* op, const eval_context& ctx) -> index_t {
        // TODO: This passes the src buffer as an output, not an input, because slinky thinks the bounds of inputs
        // don't matter. But in this case, they do...
        const raw_buffer* src_buf = ctx.lookup_buffer(op->outputs[0]);
        const raw_buffer* dst_buf = ctx.lookup_buffer(op->outputs[1]);
        const void* pad_value = (!padding || padding->empty()) ? nullptr : padding->data();
        ctx.copy(*src_buf, *dst_buf, pad_value);
        return 0;
      },
      {}, {op->src, op->dst}, std::move(copy_attrs));

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
  scoped_trace trace("implement_copies");
  return recursive_mutate<copy_stmt>(s, [&](const copy_stmt* op) { return implement_copy(op, ctx); });
}

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
  void visit(const check* op) override { visit_terminal(op); }
  void visit(const let_stmt* op) override { visit_terminal(op); }

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
    const bool found_src = std::find(names.begin(), names.end(), op->src) != names.end();
    found = found || found_src;
    if (std::find(names.begin(), names.end(), op->sym) != names.end()) {
      // Don't look inside if shadowing names.
      set_result(op);
    } else {
      if (found_src) {
        names.push_back(op->sym);
      }
      node_mutator::visit(op);
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

stmt insert_early_free(const stmt& s) {
  scoped_trace trace("insert_early_free");
  return early_free_inserter().mutate(s);
}

namespace {

class deshadower : public node_mutator {
  node_context& ctx;
  symbol_map<bool> symbols;
  var in_loop;

public:
  deshadower(node_context& ctx) : ctx(ctx) {}

  void visit_symbol(var x) { symbols[x] = true; }

  void visit(const variable* op) override {
    visit_symbol(op->sym);
    set_result(op);
  }

  var rename(var x) {
    std::string suffix = in_loop.defined() ? "." + ctx.name(in_loop) : "";
    return ctx.insert_unique(ctx.name(x) + suffix);
  }

  template <typename T>
  void visit_decl(const T* op) {
    stmt result = op;
    const std::optional<bool>& sym_defined = symbols[op->sym];
    var sym = op->sym;
    if (sym_defined && *sym_defined) {
      sym = rename(op->sym);
      result = clone_with(op, sym, substitute(op->body, op->sym, sym));
    }
    auto s = set_value_in_scope(symbols, sym, true);
    node_mutator::visit(result.as<T>());
  }

  void visit(const loop* op) override {
    stmt result = op;
    const std::optional<bool>& sym_defined = symbols[op->sym];
    var sym = op->sym;
    if (sym_defined && *sym_defined) {
      sym = rename(op->sym);
      result = clone_with(op, sym, substitute(op->body, op->sym, sym));
    }
    var old_in_loop = in_loop;
    in_loop = sym;
    auto s = set_value_in_scope(symbols, sym, true);
    node_mutator::visit(result.as<loop>());
    in_loop = old_in_loop;
  }
  void visit(const allocate* op) override { visit_decl(op); }
  void visit(const make_buffer* op) override {
    stmt result = op;
    // We want to keep the name of allocates that shadow make_buffers, so rename the make_buffer instead.
    // TODO: We should only do this if there is actually an allocate shadowing this buffer.
    var sym = rename(op->sym);
    result = clone_with(op, sym, substitute(op->body, op->sym, sym));
    auto s = set_value_in_scope(symbols, sym, true);
    node_mutator::visit(result.as<make_buffer>());
  }
  void visit(const crop_buffer* op) override {
    visit_symbol(op->src);
    visit_decl(op);
  }
  void visit(const crop_dim* op) override {
    visit_symbol(op->src);
    visit_decl(op);
  }
  void visit(const slice_buffer* op) override {
    visit_symbol(op->src);
    visit_decl(op);
  }
  void visit(const slice_dim* op) override {
    visit_symbol(op->src);
    visit_decl(op);
  }
  void visit(const transpose* op) override {
    visit_symbol(op->src);
    visit_decl(op);
  }
  void visit(const clone_buffer* op) override {
    visit_symbol(op->src);
    visit_decl(op);
  }
};

// This mutator attempts to re-write buffer mutators to be performed in-place when possible. Most mutators are more
// efficient when performed in place.
class reuse_shadows : public node_mutator {
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
  void visit_buffer_decl(const T* op) {
    // Buffers start out mutable.
    can_mutate[op->sym] = true;
    node_mutator::visit(op);
  }

  void visit(const loop* op) override {
    if (op->max_workers != loop::serial) {
      // We're entering a parallel loop. All the buffers in scope cannot be mutated in this scope.
      symbol_map<bool> old_can_mutate;
      std::swap(can_mutate, old_can_mutate);
      node_mutator::visit(op);
      can_mutate = std::move(old_can_mutate);
    } else {
      node_mutator::visit(op);
    }
  }

  void visit(const allocate* op) override { visit_buffer_decl(op); }
  void visit(const make_buffer* op) override { visit_buffer_decl(op); }

  void visit(const crop_buffer* op) override { visit_buffer_mutator(op); }
  void visit(const crop_dim* op) override { visit_buffer_mutator(op); }
  void visit(const slice_buffer* op) override { visit_buffer_mutator(op); }
  void visit(const slice_dim* op) override { visit_buffer_mutator(op); }
  void visit(const transpose* op) override { visit_buffer_mutator(op); }
};

}  // namespace

stmt deshadow(const stmt& s, node_context& ctx) {
  scoped_trace trace("deshadow");
  return deshadower(ctx).mutate(s);
}
stmt optimize_symbols(const stmt& s, node_context& ctx) {
  scoped_trace trace("optimize_symbols");
  return reuse_shadows().mutate(s);
}

}  // namespace slinky
