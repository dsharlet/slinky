#include "optimizations.h"

#include <cassert>
#include <iostream>
#include <map>
#include <set>

#include "evaluate.h"
#include "node_mutator.h"
#include "print.h"
#include "simplify.h"
#include "substitute.h"

namespace slinky {

namespace {

stmt clone_with_new_body(const let_stmt* op, stmt new_body) {
  return let_stmt::make(op->sym, op->value, std::move(new_body));
}
stmt clone_with_new_body(const allocate* op, stmt new_body) {
  return allocate::make(op->storage, op->sym, op->elem_size, op->dims, std::move(new_body));
}
stmt clone_with_new_body(const make_buffer* op, stmt new_body) {
  return make_buffer::make(op->sym, op->base, op->elem_size, op->dims, std::move(new_body));
}
stmt clone_with_new_body(const crop_buffer* op, stmt new_body) {
  return crop_buffer::make(op->sym, op->bounds, std::move(new_body));
}
stmt clone_with_new_body(const crop_dim* op, stmt new_body) {
  return crop_dim::make(op->sym, op->dim, op->bounds, std::move(new_body));
}
stmt clone_with_new_body(const slice_buffer* op, stmt new_body) {
  return slice_buffer::make(op->sym, op->at, std::move(new_body));
}
stmt clone_with_new_body(const slice_dim* op, stmt new_body) {
  return slice_dim::make(op->sym, op->dim, op->at, std::move(new_body));
}
stmt clone_with_new_body(const truncate_rank* op, stmt new_body) {
  return truncate_rank::make(op->sym, op->rank, std::move(new_body));
}

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
  expr out_var = variable::make(out);
  for (index_t d = 0; d < static_cast<index_t>(in_x.size()); ++d) {
    // TODO: This is too lax, we really need to check for elementwise before we've computed the bounds of this
    // particular call, so we can check that a single point of the output is a function of the same point in the input
    // (and not a rectangle of output being a function of a rectangle of the input).
    if (!match(in_x[d].min, buffer_min(out_var, d))) return false;
    if (!match(in_x[d].max, buffer_max(out_var, d))) return false;
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

bool is_copy(const copy_stmt* op, box_expr& bounds) {
  if (op->src_x.size() != op->dst_x.size()) return false;
  bounds.resize(op->dst_x.size());
  for (std::size_t d = 0; d < op->dst_x.size(); ++d) {
    if (!is_copy(op->src_x[d], op->dst_x[d], bounds[d])) return false;
  }
  return true;
}

std::vector<expr> mins(const box_expr& b) {
  std::vector<expr> r;
  r.reserve(b.size());
  for (const interval_expr& d : b) {
    r.push_back(d.min);
  }
  return r;
}

// Replaces `copy_stmt` with a call to `pad`.
class replace_copy_with_pad : public node_mutator {
  symbol_id src;
  symbol_id dst;

public:
  replace_copy_with_pad(symbol_id src, symbol_id dst) : src(src), dst(dst) {}

  void visit(const copy_stmt* op) {
    if (op->src == src && op->dst == dst) {
      stmt result;
      if (!op->padding.empty()) {
        result = call_stmt::make(
            [src = op->src, dst = op->dst, padding = op->padding](const eval_context& ctx) -> index_t {
              const raw_buffer* src_buf = ctx.lookup_buffer(src);
              const raw_buffer* dst_buf = ctx.lookup_buffer(dst);
              pad(src_buf->dims, *dst_buf, padding.data());
              return 0;
            },
            {src}, {dst});
      }
      set_result(result);
    } else {
      set_result(op);
    }
  }
};

class buffer_aliaser : public node_mutator {
  struct buffer_alias {
    box_expr bounds;
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

public:
  void visit(const allocate* op) override {
    box_expr bounds;
    bounds.reserve(op->dims.size());
    for (const dim_expr& d : op->dims) {
      bounds.push_back(d.bounds);
    }
    auto set_buffer_bounds = set_value_in_scope(buffer_bounds, op->sym, bounds);
    auto set_elem_size = set_value_in_scope(elem_sizes, op->sym, op->elem_size);

    // When we allocate a buffer, we can look for all the uses of this buffer. If it is:
    // - consumed elemenwise,
    // - consumed by a producer that has an output that we can re-use,
    // - not consumed after the buffer it aliases to is produced,
    // then we can alias it to the buffer produced by its consumer.

    // Start out by setting it to elementwise.
    auto s = set_value_in_scope(alias_info, op->sym, buffer_info());
    stmt body = mutate(op->body);
    const std::map<symbol_id, buffer_alias>& can_alias = alias_info[op->sym]->can_alias();

    if (!can_alias.empty()) {
      const std::pair<symbol_id, buffer_alias>& target = *can_alias.begin();
      var target_var(target.first);
      assert(target.second.bounds.size() == op->dims.size());
      std::vector<dim_expr> dims(target.second.bounds.size());
      for (int d = 0; d < static_cast<int>(target.second.bounds.size()); ++d) {
        dims[d] = {target.second.bounds[d], buffer_stride(target_var, d), buffer_fold_factor(target_var, d)};
      }
      expr base = buffer_at(target_var, target.second.offset);
      index_t elem_size = op->elem_size;
      stmt result = make_buffer::make(op->sym, base, elem_size, std::move(dims), std::move(body));
      // If we aliased the source and destination of a copy, replace the copy with a pad.
      result = replace_copy_with_pad(op->sym, target.first).mutate(result);
      set_result(result);
      // Remove this as a candidate for other aliases.
      for (std::optional<buffer_info>& i : alias_info) {
        if (!i) continue;
        i->do_not_alias(target.first);
      }
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
        info->maybe_alias(o, {*in_x, mins(*in_x)});
      }
    }
  }

  void visit(const copy_stmt* op) override {
    set_result(op);

    box_expr bounds;
    if (!is_copy(op, bounds)) {
      return;
    }

    std::optional<buffer_info>& info = alias_info[op->src];
    if (!info) {
      return;
    }

    const std::optional<box_expr>& in_x = buffer_bounds[op->src];
    info->maybe_alias(op->dst, {*in_x, mins(*in_x)});
  }

  void visit(const crop_buffer* c) override {
    std::optional<box_expr> bounds = buffer_bounds[c->sym];
    merge_crop(bounds, c->bounds);
    auto set_bounds = set_value_in_scope(buffer_bounds, c->sym, bounds);
    node_mutator::visit(c);
  }

  void visit(const crop_dim* c) override {
    std::optional<box_expr> bounds = buffer_bounds[c->sym];
    merge_crop(bounds, c->dim, c->bounds);
    auto set_bounds = set_value_in_scope(buffer_bounds, c->sym, bounds);
    node_mutator::visit(c);
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
      if (const call* c = eq->a.as<call>()) {
        if (c->intrinsic == intrinsic::buffer_elem_size) {
          const symbol_id* buf = as_variable(c->args[0]);
          const index_t* value = as_constant(eq->b);
          if (buf && value) {
            elem_sizes[*buf] = *value;
          }
        }
      }
    }
  }

  void visit(const block* op) override {
    stmt a, b;
    if (op->a.defined()) {
      if (!op->a.as<check>() && !op->a.as<block>()) {
        at_root_scope = false;
      }
      a = mutate(op->a);
    }
    if (op->b.defined()) {
      if (!op->b.as<check>() && !op->b.as<block>()) {
        at_root_scope = false;
      }
      b = mutate(op->b);
    }
    if (a.same_as(op->a) && b.same_as(op->b)) {
      set_result(op);
    } else {
      set_result(block::make(std::move(a), std::move(b)));
    }
  }
};

}  // namespace

stmt alias_buffers(const stmt& s) { return buffer_aliaser().mutate(s); }

namespace {
/*

bool is_broadcast(expr in, var out) {
  interval_expr bounds = bounds_of(in, {{out.sym(), interval_expr::all()}});
  bounds.min = simplify(bounds.min);
  bounds.max = simplify(bounds.max);

  // This is a broadcast if the bounds are a single point.
  return bounds.min.defined() && match(bounds.min, bounds.max);
}
*/
class copy_optimizer : public node_mutator {
public:
  void visit(const copy_stmt* c) override {
    // Start by making a call to copy.
    stmt result = call_stmt::make(
        [src = c->src, dst = c->dst, padding = c->padding](const eval_context& ctx) -> index_t {
          const raw_buffer* src_buf = ctx.lookup_buffer(src);
          const raw_buffer* dst_buf = ctx.lookup_buffer(dst);
          copy(*src_buf, *dst_buf, padding.empty() ? nullptr : padding.data());
          return 0;
        },
        {c->src}, {c->dst});

    box_expr bounds;
    if (is_copy(c, bounds)) {
      result = crop_buffer::make(c->src, bounds, result);
      set_result(result);
      return;
    }

    var src_var(c->src);
    var dst_var(c->dst);

    std::vector<expr> src_min = c->src_x;
    std::vector<std::pair<symbol_id, int>> dst_x;

    // If we just leave these two arrays alone, the copy will be correct, but slow.
    // We can speed it up by finding dimensions we can let pass through to the copy.
    for (int d = 0; d < static_cast<int>(c->dst_x.size()); ++d) {
      int dep_count = 0;
      for (int src_d = 0; src_d < static_cast<int>(src_min.size()); ++src_d) {
        if (depends_on(src_min[src_d], c->dst_x[d])) {
          ++dep_count;
        }
      }
      bool handled = false;
      if (dep_count == 1) {
        // TODO: Try to handle this dimension by passing it to copy.
      }
      if (!handled) {
        dst_x.emplace_back(c->dst_x[d], d);
      }
    }

    // Any dimensions left need loops and slices.
    result = slice_buffer::make(c->src, src_min, result);
    for (const std::pair<symbol_id, int>& d : dst_x) {
      result = slice_dim::make(c->dst, d.second, var(d.first), result);
      result = loop::make(d.first, buffer_bounds(dst_var, d.second), 1, result);
    }

    set_result(result);
  }
};

}  // namespace

stmt optimize_copies(const stmt& s) { return copy_optimizer().mutate(s); }

namespace {

// Traverse stmts in a block in order.
template <typename Fn>
void for_each_stmt_forward(const stmt& s, const Fn& fn) {
  if (const block* b = s.as<block>()) {
    for_each_stmt_forward(b->a, fn);
    for_each_stmt_forward(b->b, fn);
  } else {
    fn(s);
  }
}

template <typename Fn>
void for_each_stmt_backward(const stmt& s, const Fn& fn) {
  if (const block* b = s.as<block>()) {
    for_each_stmt_backward(b->b, fn);
    for_each_stmt_backward(b->a, fn);
  } else {
    fn(s);
  }
}

// Split `body` into 3 parts:
// - stmts that don't depend on `vars`
// - stmts that do depend on `vars`
// - stmts that don't depend on `vars`
std::tuple<stmt, stmt, stmt> split_body(const stmt& body, std::span<const symbol_id> vars) {
  stmt before;
  stmt new_body_after;
  bool depended_on = false;
  // First, split the body into the before, and the new body + after.
  for_each_stmt_forward(body, [&](const stmt& s) {
    if (depended_on || depends_on(s, vars)) {
      new_body_after = block::make({new_body_after, s});
      depended_on = true;
    } else {
      before = block::make({before, s});
    }
  });

  // Now, split the new body + after into the new body and the after.
  stmt new_body;
  stmt after;
  depended_on = false;
  for_each_stmt_backward(new_body_after, [&](const stmt& s) {
    if (!depended_on && !depends_on(s, vars)) {
      after = block::make({s, after});
    } else {
      new_body = block::make({s, new_body});
      depended_on = true;
    }
  });

  return {before, new_body, after};
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
      set_result(block::make({before, result, after}));
    } else {
      // The op was dead...?
      set_result(block::make({before, after}));
    }
  }

  void visit(const let_stmt* op) override { visit_stmt(op); }
  void visit(const allocate* op) override { visit_stmt(op); }
  void visit(const make_buffer* op) override { visit_stmt(op); }
  void visit(const crop_buffer* op) override { visit_stmt(op); }
  void visit(const crop_dim* op) override { visit_stmt(op); }
  void visit(const slice_buffer* op) override { visit_stmt(op); }
  void visit(const slice_dim* op) override { visit_stmt(op); }
  void visit(const truncate_rank* op) override { visit_stmt(op); }
};

}  // namespace

stmt reduce_scopes(const stmt& s) { return scope_reducer().mutate(s); }

}  // namespace slinky
