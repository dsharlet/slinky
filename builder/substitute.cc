#include "builder/substitute.h"

#include <cassert>
#include <cstddef>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "base/chrome_trace.h"
#include "builder/node_mutator.h"
#include "runtime/depends_on.h"
#include "runtime/expr.h"

namespace slinky {

class matcher : public expr_visitor, public stmt_visitor {
  // In this class, we visit the pattern, and manually traverse the expression being matched.
  union {
    void* self = nullptr;
    const base_expr_node* self_expr;
    const base_stmt_node* self_stmt;
  };

public:
  int match = 0;

  template <typename T>
  bool try_match(T self, T op) {
    assert(match == 0);
    if (self < op) {
      match = -1;
    } else if (op < self) {
      match = 1;
    }
    return match == 0;
  }

  bool try_match(const var& self, const var& op) { return try_match(self.id, op.id); }

  // Skip the visitor pattern (two virtual function calls) for a few node types that are very frequently visited.
  void visit(const base_expr_node* op) {
    switch (op->type) {
    case expr_node_type::variable: visit(reinterpret_cast<const variable*>(op)); return;
    case expr_node_type::add: visit(reinterpret_cast<const add*>(op)); return;
    case expr_node_type::min: visit(reinterpret_cast<const class min*>(op)); return;
    case expr_node_type::max: visit(reinterpret_cast<const class max*>(op)); return;
    default: op->accept(this);
    }
  }

  bool try_match(const base_expr_node* e, const base_expr_node* op) {
    assert(match == 0);
    if (!e && !op) {
    } else if (!e) {
      match = -1;
    } else if (!op) {
      match = 1;
    } else if (e->type < op->type) {
      match = -1;
    } else if (e->type > op->type) {
      match = 1;
    } else {
      self_expr = e;
      visit(op);
    }
    return match == 0;
  }
  bool try_match(const expr& e, const expr& op) { return try_match(e.get(), op.get()); }

  bool try_match(const base_stmt_node* s, const base_stmt_node* op) {
    assert(match == 0);
    if (!s && !op) {
    } else if (!s) {
      match = -1;
    } else if (!op) {
      match = 1;
    } else if (s->type < op->type) {
      match = -1;
    } else if (s->type > op->type) {
      match = 1;
    } else {
      self_stmt = s;
      op->accept(this);
    }
    return match == 0;
  }
  bool try_match(const stmt& s, const stmt& op) { return try_match(s.get(), op.get()); }

  bool try_match(const interval_expr& self, const interval_expr& op) {
    if (!try_match(self.min, op.min)) return false;
    if (!try_match(self.max, op.max)) return false;
    return true;
  }

  bool try_match(const dim_expr& self, const dim_expr& op) {
    if (!try_match(self.bounds, op.bounds)) return false;
    if (!try_match(self.stride, op.stride)) return false;
    if (!try_match(self.fold_factor, op.fold_factor)) return false;
    return true;
  }

  template <typename A, typename B>
  bool try_match(const std::pair<A, B>& self, const std::pair<A, B>& op) {
    if (!try_match(self.first, op.first)) return false;
    if (!try_match(self.second, op.second)) return false;
    return true;
  }

  template <typename T>
  bool try_match(const std::vector<T>& self, const std::vector<T>& op) {
    if (!try_match(self.size(), op.size())) return false;
    for (std::size_t i = 0; i < self.size(); ++i) {
      if (!try_match(self[i], op[i])) return false;
    }

    return true;
  }

  template <typename T>
  void match_binary(const T* op) {
    const T* ex = static_cast<const T*>(self);

    if (!try_match(ex->a, op->a)) return;
    if (!try_match(ex->b, op->b)) return;
  }

  void visit(const variable* op) override {
    const variable* ev = static_cast<const variable*>(self);
    try_match(ev->sym, op->sym);
  }

  void visit(const constant* op) override {
    const constant* ec = static_cast<const constant*>(self);
    try_match(ec->value, op->value);
  }

  template <typename T>
  void visit_let(const T* op) {
    const T* el = static_cast<const T*>(self);

    if (!try_match(el->lets, op->lets)) return;
    if (!try_match(el->body, op->body)) return;
  }

  void visit(const let* op) override { visit_let(op); }
  void visit(const add* op) override { match_binary(op); }
  void visit(const sub* op) override { match_binary(op); }
  void visit(const mul* op) override { match_binary(op); }
  void visit(const div* op) override { match_binary(op); }
  void visit(const mod* op) override { match_binary(op); }
  void visit(const class min* op) override { match_binary(op); }
  void visit(const class max* op) override { match_binary(op); }
  void visit(const equal* op) override { match_binary(op); }
  void visit(const not_equal* op) override { match_binary(op); }
  void visit(const less* op) override { match_binary(op); }
  void visit(const less_equal* op) override { match_binary(op); }
  void visit(const logical_and* op) override { match_binary(op); }
  void visit(const logical_or* op) override { match_binary(op); }

  void visit(const logical_not* op) override {
    const class logical_not* ne = static_cast<const logical_not*>(self);

    try_match(ne->a, op->a);
  }

  void visit(const class select* op) override {
    const class select* se = static_cast<const class select*>(self);

    if (!try_match(se->condition, op->condition)) return;
    if (!try_match(se->true_value, op->true_value)) return;
    if (!try_match(se->false_value, op->false_value)) return;
  }

  void visit(const call* op) override {
    const call* c = static_cast<const call*>(self);

    if (!try_match(c->intrinsic, op->intrinsic)) return;
    if (!try_match(c->args, op->args)) return;
  }

  void visit(const let_stmt* op) override { visit_let(static_cast<const let_stmt*>(self)); }

  void visit(const block* op) override {
    const block* bs = static_cast<const block*>(self);

    if (!try_match(bs->stmts, op->stmts)) return;
  }

  void visit(const loop* op) override {
    const loop* ls = static_cast<const loop*>(self);

    if (!try_match(ls->sym, op->sym)) return;
    if (!try_match(ls->bounds, op->bounds)) return;
    if (!try_match(ls->step, op->step)) return;
    if (!try_match(ls->body, op->body)) return;
  }

  void visit(const call_stmt* op) override {
    if (match) return;
    const call_stmt* cs = static_cast<const call_stmt*>(self);
    assert(cs);

    if (!try_match(cs->inputs, op->inputs)) return;
    if (!try_match(cs->outputs, op->outputs)) return;
  }

  void visit(const copy_stmt* op) override {
    const copy_stmt* cs = static_cast<const copy_stmt*>(self);
    assert(cs);

    if (!try_match(cs->src, op->src)) return;
    if (!try_match(cs->src_x, op->src_x)) return;
    if (!try_match(cs->dst, op->dst)) return;
    if (!try_match(cs->dst_x, op->dst_x)) return;
    if (!try_match(cs->padding, op->padding)) return;
  }

  void visit(const allocate* op) override {
    const allocate* as = static_cast<const allocate*>(self);
    assert(as);

    if (!try_match(as->sym, op->sym)) return;
    if (!try_match(as->elem_size, op->elem_size)) return;
    if (!try_match(as->dims, op->dims)) return;
    if (!try_match(as->body, op->body)) return;
  }

  void visit(const make_buffer* op) override {
    const make_buffer* mbs = static_cast<const make_buffer*>(self);
    assert(mbs);

    if (!try_match(mbs->sym, op->sym)) return;
    if (!try_match(mbs->base, op->base)) return;
    if (!try_match(mbs->elem_size, op->elem_size)) return;
    if (!try_match(mbs->dims, op->dims)) return;
    if (!try_match(mbs->body, op->body)) return;
  }

  void visit(const clone_buffer* op) override {
    const clone_buffer* mbs = static_cast<const clone_buffer*>(self);
    assert(mbs);

    if (!try_match(mbs->sym, op->sym)) return;
    if (!try_match(mbs->src, op->src)) return;
    if (!try_match(mbs->body, op->body)) return;
  }

  void visit(const crop_buffer* op) override {
    const crop_buffer* cbs = static_cast<const crop_buffer*>(self);
    assert(cbs);

    if (!try_match(cbs->sym, op->sym)) return;
    if (!try_match(cbs->src, op->src)) return;
    if (!try_match(cbs->bounds, op->bounds)) return;
    if (!try_match(cbs->body, op->body)) return;
  }

  void visit(const crop_dim* op) override {
    const crop_dim* cds = static_cast<const crop_dim*>(self);
    assert(cds);

    if (!try_match(cds->sym, op->sym)) return;
    if (!try_match(cds->src, op->src)) return;
    if (!try_match(cds->dim, op->dim)) return;
    if (!try_match(cds->bounds, op->bounds)) return;
    if (!try_match(cds->body, op->body)) return;
  }

  void visit(const slice_buffer* op) override {
    const slice_buffer* cbs = static_cast<const slice_buffer*>(self);
    assert(cbs);

    if (!try_match(cbs->sym, op->sym)) return;
    if (!try_match(cbs->src, op->src)) return;
    if (!try_match(cbs->at, op->at)) return;
    if (!try_match(cbs->body, op->body)) return;
  }

  void visit(const slice_dim* op) override {
    const slice_dim* cds = static_cast<const slice_dim*>(self);
    assert(cds);

    if (!try_match(cds->sym, op->sym)) return;
    if (!try_match(cds->src, op->src)) return;
    if (!try_match(cds->dim, op->dim)) return;
    if (!try_match(cds->at, op->at)) return;
    if (!try_match(cds->body, op->body)) return;
  }

  void visit(const transpose* op) override {
    const transpose* trs = static_cast<const transpose*>(self);
    assert(trs);

    if (!try_match(trs->sym, op->sym)) return;
    if (!try_match(trs->src, op->src)) return;
    if (!try_match(trs->dims, op->dims)) return;
    if (!try_match(trs->body, op->body)) return;
  }

  void visit(const check* op) override {
    const check* cs = static_cast<const check*>(self);
    assert(cs);

    try_match(cs->condition, op->condition);
  }
};

bool match(const expr& a, const expr& b) { return compare(a, b) == 0; }
bool match(const stmt& a, const stmt& b) { return compare(a, b) == 0; }
bool match(const interval_expr& a, const interval_expr& b) { return match(a.min, b.min) && match(a.max, b.max); }
bool match(const dim_expr& a, const dim_expr& b) {
  return match(a.bounds, b.bounds) && match(a.stride, b.stride) && match(a.fold_factor, b.fold_factor);
}

const call* match_call(const expr& x, intrinsic fn, var a) {
  const call* c = as_intrinsic(x, fn);
  if (!c) return nullptr;

  assert(c->args.size() >= 1);
  const var* av = as_variable(c->args[0]);
  if (!av || *av != a) return nullptr;

  return c;
}

const call* match_call(const expr& x, intrinsic fn, var a, index_t b) {
  const call* c = match_call(x, fn, a);
  if (!c) return nullptr;

  assert(c->args.size() >= 2);
  const index_t* bv = as_constant(c->args[1]);
  if (!bv || *bv != b) return nullptr;

  return c;
}

int compare(const var& a, const var& b) {
  matcher m;
  m.try_match(a, b);
  return m.match;
}
int compare(const expr& a, const expr& b) { return compare(a.get(), b.get()); }

int compare(const base_expr_node* a, const base_expr_node* b) {
  matcher m;
  m.try_match(a, b);
  return m.match;
}

int compare(const stmt& a, const stmt& b) {
  matcher m;
  m.try_match(a, b);
  return m.match;
}

namespace {

expr eval_buffer_intrinsic(intrinsic fn, const dim_expr& d) {
  switch (fn) {
  case intrinsic::buffer_min: return d.bounds.min;
  case intrinsic::buffer_max: return d.bounds.max;
  case intrinsic::buffer_stride: return d.stride;
  case intrinsic::buffer_fold_factor: return d.fold_factor;
  default: std::abort();
  }
}

// This base class helps substitute implementations handle shadowing correctly.
class substitutor : public node_mutator {
public:
  // Track newly declared variables that might shadow the variables we want to replace.
  std::vector<var> shadowed;

public:
  bool is_shadowed(var x) const { return std::find(shadowed.begin(), shadowed.end(), x) != shadowed.end(); }

  // Implementation of substitution for vars.
  virtual var visit_symbol(var x) { return x; }

  // Implementation of substitution for slice bodies.
  virtual stmt mutate_slice_body(var sym, var src, span<const int> slices, stmt body) = 0;

  // Implementation of substitution for buffer intrinsics.
  virtual expr mutate_buffer_intrinsic(intrinsic fn, var buf, span<const expr> args) { return expr(); }

  // The implementation must provide the maximum rank of any substitution of buffer metadata for x.
  virtual std::size_t get_target_buffer_rank(var x) { return 0; }

  template <typename T>
  T mutate_decl_body(var sym, const T& x) {
    shadowed.push_back(sym);
    T result = mutate(x);
    shadowed.pop_back();
    return result;
  }

  template <typename T>
  auto mutate_let(const T* op) {
    std::vector<std::pair<var, expr>> lets;
    lets.reserve(op->lets.size());
    bool changed = false;
    for (const auto& s : op->lets) {
      lets.emplace_back(s.first, mutate(s.second));
      shadowed.push_back(s.first);
      changed = changed || !lets.back().second.same_as(s.second);
    }

    auto body = mutate(op->body);
    shadowed.resize(shadowed.size() - lets.size());
    changed = changed || !body.same_as(op->body);

    if (!changed) {
      return decltype(body){op};
    } else {
      return T::make(std::move(lets), std::move(body));
    }
  }

  void visit(const let* op) override { set_result(mutate_let(op)); }
  void visit(const let_stmt* op) override { set_result(mutate_let(op)); }

  void visit(const loop* op) override {
    interval_expr bounds = {mutate(op->bounds.min), mutate(op->bounds.max)};
    expr step = mutate(op->step);
    stmt body = mutate_decl_body(op->sym, op->body);
    if (bounds.same_as(op->bounds) && step.same_as(op->step) && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(loop::make(op->sym, op->max_workers, std::move(bounds), std::move(step), std::move(body)));
    }
  }
  void visit(const allocate* op) override {
    expr elem_size = mutate(op->elem_size);
    std::vector<dim_expr> dims;
    dims.reserve(op->dims.size());
    bool changed = false;
    for (const dim_expr& i : op->dims) {
      interval_expr bounds = {mutate(i.bounds.min), mutate(i.bounds.max)};
      dims.push_back({std::move(bounds), mutate(i.stride), mutate(i.fold_factor)});
      changed = changed || !dims.back().same_as(i);
    }
    stmt body = mutate_decl_body(op->sym, op->body);
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
    bool changed = false;
    for (const dim_expr& i : op->dims) {
      interval_expr bounds = {mutate(i.bounds.min), mutate(i.bounds.max)};
      dims.push_back({std::move(bounds), mutate(i.stride), mutate(i.fold_factor)});
      changed = changed || !dims.back().same_as(i);
    }
    stmt body = mutate_decl_body(op->sym, op->body);
    if (!changed && base.same_as(op->base) && elem_size.same_as(op->elem_size) && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(make_buffer::make(op->sym, std::move(base), std::move(elem_size), std::move(dims), std::move(body)));
    }
  }

  void visit(const slice_buffer* op) override {
    // Slices do not shadow, so we should substitute sym as well.
    // TODO: This seems sketchy.
    var sym = visit_symbol(op->sym);
    var src = visit_symbol(op->src);
    std::vector<expr> at(op->at.size());
    at.reserve(op->at.size());
    bool changed = false;
    std::vector<int> dims;
    for (int d = 0; d < static_cast<int>(op->at.size()); ++d) {
      at[d] = mutate(op->at[d]);
      changed = changed || !at[d].same_as(op->at[d]);
      if (at[d].defined()) {
        dims.push_back(d);
      }
    }
    stmt body = mutate_slice_body(op->sym, op->src, dims, op->body);
    if (!changed && sym == op->sym && src == op->src && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(slice_buffer::make(sym, src, std::move(at), std::move(body)));
    }
  }
  void visit(const slice_dim* op) override {
    // Slices do not shadow, so we should substitute sym as well.
    // TODO: This seems sketchy.
    var sym = visit_symbol(op->sym);
    var src = visit_symbol(op->src);
    expr at = mutate(op->at);
    int slices[] = {op->dim};
    stmt body = mutate_slice_body(op->sym, op->src, slices, op->body);
    if (sym == op->sym && src == op->src && at.same_as(op->at) && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(slice_dim::make(sym, src, op->dim, std::move(at), std::move(body)));
    }
  }

  interval_expr substitute_crop_bounds(var new_src, var src, int dim, const interval_expr& bounds) {
    // When substituting crop bounds, we need to apply the implicit clamp, which uses buffer_min(sym, dim) and
    // buffer_max(src, dim).
    interval_expr old_bounds = buffer_bounds(src, dim);
    interval_expr new_bounds = {mutate(old_bounds.min), mutate(old_bounds.max)};
    interval_expr result = {mutate(bounds.min), mutate(bounds.max)};
    if (!old_bounds.min.same_as(new_bounds.min) && !match_call(new_bounds.min, intrinsic::buffer_min, new_src, dim)) {
      // The substitution changed the implicit clamp, include it.
      result.min = max(result.min, new_bounds.min);
    }
    if (!old_bounds.max.same_as(new_bounds.max) && !match_call(new_bounds.max, intrinsic::buffer_max, new_src, dim)) {
      // The substitution changed the implicit clamp, include it.
      result.max = min(result.max, new_bounds.max);
    }
    return result;
  }

  void visit(const crop_buffer* op) override {
    var src = visit_symbol(op->src);
    box_expr bounds(op->bounds.size());
    bool changed = false;
    for (std::size_t i = 0; i < op->bounds.size(); ++i) {
      bounds[i] = substitute_crop_bounds(src, op->src, i, op->bounds[i]);
      changed = changed || !bounds[i].same_as(op->bounds[i]);
    }
    stmt body = mutate_decl_body(op->sym, op->body);
    if (changed || src != op->src || !body.same_as(op->body)) {
      set_result(crop_buffer::make(op->sym, src, std::move(bounds), std::move(body)));
    } else {
      set_result(op);
    }
  }

  void visit(const crop_dim* op) override {
    var src = visit_symbol(op->src);
    interval_expr bounds = substitute_crop_bounds(src, op->src, op->dim, op->bounds);
    stmt body = mutate_decl_body(op->sym, op->body);
    if (src == op->src && bounds.same_as(op->bounds) && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(crop_dim::make(op->sym, src, op->dim, std::move(bounds), std::move(body)));
    }
  }

  void visit(const call* op) override {
    std::vector<expr> args;
    args.reserve(op->args.size());
    bool changed = false;
    for (const expr& i : op->args) {
      args.push_back(mutate(i));
      changed = changed || !args.back().same_as(i);
    }
    if (is_buffer_intrinsic(op->intrinsic) && !args.empty() && args.front().defined()) {
      const var* buf = as_variable(args[0]);
      assert(buf);
      if (!is_shadowed(*buf)) {
        if (op->intrinsic == intrinsic::buffer_at) {
          const std::size_t buf_rank = get_target_buffer_rank(*buf);
          for (std::size_t d = 0; d < buf_rank; ++d) {
            if (d + 1 >= args.size() || !args[d + 1].defined()) {
              // buffer_at has an implicit buffer_min if it is not defined.
              expr min_args[] = {static_cast<index_t>(d)};
              expr min = mutate_buffer_intrinsic(intrinsic::buffer_min, *buf, min_args);
              if (min.defined()) {
                args.resize(std::max(args.size(), d + 2));
                args[d + 1] = min;
                changed = true;
              }
            }
          }
        }

        expr result = mutate_buffer_intrinsic(op->intrinsic, *buf, span<const expr>(args).subspan(1));
        if (result.defined()) {
          set_result(result);
          return;
        }
      }
    }
    if (changed) {
      set_result(call::make(op->intrinsic, std::move(args)));
    } else {
      set_result(op);
    }
  }

  void visit(const transpose* op) override {
    // TODO: transpose is a bit tricky, the replacements for expressions might be invalid if they access truncated
    // dims.
    var src = visit_symbol(op->src);
    stmt body = mutate_decl_body(op->sym, op->body);
    if (src != op->src || !body.same_as(op->body)) {
      set_result(transpose::make(op->sym, src, op->dims, std::move(body)));
    } else {
      set_result(op);
    }
  }

  void visit(const call_stmt* op) override {
    call_stmt::symbol_list inputs(op->inputs.size());
    call_stmt::symbol_list outputs(op->outputs.size());
    bool changed = false;
    for (std::size_t i = 0; i < op->inputs.size(); ++i) {
      inputs[i] = visit_symbol(op->inputs[i]);
      changed = changed || inputs[i] != op->inputs[i];
    }
    for (std::size_t i = 0; i < op->outputs.size(); ++i) {
      outputs[i] = visit_symbol(op->outputs[i]);
      changed = changed || outputs[i] != op->outputs[i];
    }
    if (changed) {
      set_result(call_stmt::make(op->target, std::move(inputs), std::move(outputs), op->attrs));
    } else {
      set_result(op);
    }
  }

  void visit(const copy_stmt* op) override {
    var src = visit_symbol(op->src);
    var dst = visit_symbol(op->dst);

    // copy_stmt is effectively a declaration of the dst_x symbols for the src_x expressions.
    shadowed.insert(shadowed.end(), op->dst_x.begin(), op->dst_x.end());
    std::vector<expr> src_x(op->src_x.size());
    bool changed = false;
    for (std::size_t i = 0; i < op->src_x.size(); ++i) {
      src_x[i] = mutate(op->src_x[i]);
      changed = changed || !src_x[i].same_as(op->src_x[i]);
    }
    shadowed.resize(shadowed.size() - op->dst_x.size());
    if (changed || src != op->src || dst != op->dst) {
      set_result(copy_stmt::make(src, std::move(src_x), dst, op->dst_x, op->padding));
    } else {
      set_result(op);
    }
  }

  void visit(const clone_buffer* op) override {
    var src = visit_symbol(op->src);
    stmt body = mutate_decl_body(op->sym, op->body);
    if (src != op->src || !body.same_as(op->body)) {
      set_result(clone_buffer::make(op->sym, src, std::move(body)));
    } else {
      set_result(op);
    }
  }
};

// A substutitor implementation for target vars
class var_substitutor : public substitutor {
public:
  const symbol_map<expr>* replacements = nullptr;
  var target;
  expr replacement;

public:
  var_substitutor(const symbol_map<expr>& replacements) : replacements(&replacements) {}
  var_substitutor(var target, const expr& replacement) : target(target), replacement(replacement) {}

  void visit(const variable* v) override {
    if (is_shadowed(v->sym)) {
      // This variable has been shadowed, don't substitute it.
    } else if (v->sym == target && !depends_on(replacement, shadowed).any()) {
      set_result(replacement);
      return;
    } else if (replacements) {
      std::optional<expr> r = replacements->lookup(v->sym);
      if (r && !depends_on(*r, shadowed).any()) {
        set_result(*r);
        return;
      }
    }
    set_result(v);
  }
  using substitutor::visit;

  static var replacement_symbol(const expr& r) {
    const var* s = as_variable(r);
    assert(s);
    return *s;
  }

  var visit_symbol(var x) override {
    if (is_shadowed(x)) {
      // This variable has been shadowed, don't substitute it.
      return x;
    } else if (x == target && !depends_on(replacement, shadowed).any()) {
      return replacement_symbol(replacement);
    } else if (replacements) {
      std::optional<expr> r = replacements->lookup(x);
      if (r && !depends_on(*r, shadowed).any()) {
        return replacement_symbol(*r);
      }
    }
    return x;
  }

  stmt mutate_slice_body(var sym, var src, span<const int> slices, stmt body) override {
    // Remember the replacements from before the slice.
    const symbol_map<expr>* old_replacements = replacements;
    expr old_replacement = replacement;

    // Update the replacements for slices.
    symbol_map<expr> new_replacements;
    if (replacements) {
      new_replacements = *replacements;
      for (std::optional<expr>& i : new_replacements) {
        if (i) i = update_sliced_buffer_metadata(*i, sym, slices);
      }
      replacements = &new_replacements;
    }

    replacement = update_sliced_buffer_metadata(replacement, sym, slices);

    // Mutate the slice
    if (sym != src) shadowed.push_back(sym);
    body = mutate(body);
    if (sym != src) shadowed.pop_back();

    // Restore the old replacements.
    replacements = old_replacements;
    replacement = old_replacement;

    return body;
  }
};

// A substitutor implementation for target exprs.
class expr_substitutor : public substitutor {
public:
  expr target;
  expr replacement;

public:
  expr_substitutor(expr target, expr replacement) : target(target), replacement(replacement) {
    assert(!as_variable(target));
  }

  expr mutate(const expr& op) override {
    if (!depends_on(target, shadowed).any() && match(op, target) && !depends_on(replacement, shadowed).any()) {
      return replacement;
    }
    return node_mutator::mutate(op);
  }
  using node_mutator::mutate;

  stmt mutate_slice_body(var sym, var src, span<const int> slices, stmt body) override {
    // Remember the replacements from before the slice.
    expr old_target = target;
    expr old_replacement = replacement;

    // Update the replacements for slices.
    target = update_sliced_buffer_metadata(target, sym, slices);
    replacement = update_sliced_buffer_metadata(replacement, sym, slices);

    // Mutate the slice
    if (sym != src) shadowed.push_back(sym);
    body = mutate(body);
    if (sym != src) shadowed.pop_back();

    // Restore the old replacements.
    target = old_target;
    replacement = old_replacement;

    return body;
  }

  std::size_t get_target_buffer_rank(var x) override {
    if (const call* c = target.as<call>()) {
      if (is_buffer_dim_intrinsic(c->intrinsic) && is_variable(c->args[0], x)) {
        const index_t* dim = as_constant(c->args[1]);
        assert(dim);
        return *dim + 1;
      }
    }
    return 0;
  }

  expr mutate_buffer_intrinsic(intrinsic fn, var buf, span<const expr> args) override {
    const call* c = as_intrinsic(target, fn);
    if (!c) return expr();
    if (!match(c->args[0], buf)) return expr();
    if (c->args.size() != args.size() + 1) return expr();
    for (std::size_t d = 0; d < args.size(); ++d) {
      if (!match(c->args[d + 1], args[d])) return expr();
    }
    return replacement;
  }
};

// A substitutor implementation for target buffers.
class buffer_substitutor : public substitutor {
public:
  var target;
  expr elem_size;
  span<const dim_expr> dims;

public:
  buffer_substitutor(var target, expr elem_size, span<const dim_expr> dims)
      : target(target), elem_size(elem_size), dims(dims) {}

  stmt mutate_slice_body(var sym, var src, span<const int> slices, stmt body) override {
    // Remember the replacements from before the slice.
    expr old_elem_size = elem_size;
    span<const dim_expr> old_dims = dims;

    // Update the replacements for slices.
    elem_size = update_sliced_buffer_metadata(elem_size, sym, slices);
    std::vector<dim_expr> new_dims;
    new_dims.reserve(dims.size());
    for (std::size_t d = 0; d < dims.size(); ++d) {
      if (target != sym || std::find(slices.begin(), slices.end(), d) == slices.end()) {
        new_dims.push_back(update_sliced_buffer_metadata(dims[d], sym, slices));
      }
    }
    dims = span<const dim_expr>(new_dims);

    // Mutate the slice
    if (sym != src) shadowed.push_back(sym);
    body = mutate(body);
    if (sym != src) shadowed.pop_back();

    // Restore the old replacements.
    elem_size = old_elem_size;
    dims = old_dims;

    return body;
  }

  std::size_t get_target_buffer_rank(var x) override { return x == target ? dims.size() : 0; }

  expr mutate_buffer_intrinsic(intrinsic fn, var buf, span<const expr> args) override {
    if (buf != target) return expr();
    
    if (fn == intrinsic::buffer_elem_size) {
      if (elem_size.defined()) {
        return elem_size;
      }
    } else if (is_buffer_dim_intrinsic(fn)) {
      assert(args.size() == 1);
      const index_t* dim = as_constant(args[0]);
      assert(dim);
      if (*dim < static_cast<index_t>(dims.size())) {
        expr result = eval_buffer_intrinsic(fn, dims[*dim]);
        if (result.defined()) {
          return result;
        }
      }
    }
    return expr();
  }
};

template <typename T>
T substitute_bounds_impl(const T& op, var buffer, int dim, const interval_expr& bounds) {
  std::vector<dim_expr> dims(dim + 1);
  dims[dim].bounds = bounds;
  return substitute_buffer(op, buffer, expr(), dims);
}

template <typename T>
T substitute_bounds_impl(const T& op, var buffer, const box_expr& bounds) {
  std::vector<dim_expr> dims(bounds.size());
  for (index_t d = 0; d < static_cast<index_t>(bounds.size()); ++d) {
    dims[d].bounds = bounds[d];
  }
  return substitute_buffer(op, buffer, expr(), dims);
}

}  // namespace

expr substitute(const expr& e, const symbol_map<expr>& replacements) { return var_substitutor(replacements).mutate(e); }
stmt substitute(const stmt& s, const symbol_map<expr>& replacements) {
  scoped_trace trace("substitute");
  return var_substitutor(replacements).mutate(s);
}

expr substitute(const expr& e, var target, const expr& replacement) {
  return var_substitutor(target, replacement).mutate(e);
}
stmt substitute(const stmt& s, var target, const expr& replacement) {
  scoped_trace trace("substitute");
  return var_substitutor(target, replacement).mutate(s);
}
interval_expr substitute(const interval_expr& x, var target, const expr& replacement) {
  if (x.is_point()) {
    return point(substitute(x.min, target, replacement));
  } else {
    return {substitute(x.min, target, replacement), substitute(x.max, target, replacement)};
  }
}

expr substitute(const expr& e, const expr& target, const expr& replacement) {
  if (const var* v = as_variable(target)) {
    return var_substitutor(*v, replacement).mutate(e);
  } else {
    return expr_substitutor(target, replacement).mutate(e);
  }
}
stmt substitute(const stmt& s, const expr& target, const expr& replacement) {
  scoped_trace trace("substitute");
  if (const var* v = as_variable(target)) {
    return var_substitutor(*v, replacement).mutate(s);
  } else {
    return expr_substitutor(target, replacement).mutate(s);
  }
}

expr substitute_buffer(const expr& e, var buffer, const expr& elem_size, const std::vector<dim_expr>& dims) {
  return buffer_substitutor(buffer, elem_size, dims).mutate(e);
}
stmt substitute_buffer(const stmt& s, var buffer, const expr& elem_size, const std::vector<dim_expr>& dims) {
  scoped_trace trace("substitute_buffer");
  return buffer_substitutor(buffer, elem_size, dims).mutate(s);
}
expr substitute_bounds(const expr& e, var buffer, const box_expr& bounds) {
  return substitute_bounds_impl(e, buffer, bounds);
}
stmt substitute_bounds(const stmt& s, var buffer, const box_expr& bounds) {
  scoped_trace trace("substitute_bounds");
  return substitute_bounds_impl(s, buffer, bounds);
}
expr substitute_bounds(const expr& e, var buffer, int dim, const interval_expr& bounds) {
  return substitute_bounds_impl(e, buffer, dim, bounds);
}
stmt substitute_bounds(const stmt& s, var buffer, int dim, const interval_expr& bounds) {
  scoped_trace trace("substitute_bounds");
  return substitute_bounds_impl(s, buffer, dim, bounds);
}

namespace {

class slice_updater : public node_mutator {
  var sym;
  span<const int> slices;

public:
  slice_updater(var sym, span<const int> slices) : sym(sym), slices(slices) {}

  void visit(const call* op) override {
    switch (op->intrinsic) {
    case intrinsic::buffer_min:
    case intrinsic::buffer_max:
    case intrinsic::buffer_stride:
    case intrinsic::buffer_fold_factor:
      if (is_variable(op->args[0], sym)) {
        const index_t* dim = as_constant(op->args[1]);
        assert(dim);
        index_t new_dim = *dim;
        for (int i = static_cast<int>(slices.size()) - 1; i >= 0; --i) {
          if (slices[i] == new_dim) {
            // This dimension is gone.
            set_result(expr());
            return;
          } else if (slices[i] < new_dim) {
            --new_dim;
          }
        }
        if (new_dim != *dim) {
          set_result(call::make(op->intrinsic, {op->args[0], new_dim}));
        } else {
          set_result(op);
        }
        return;
      }
      break;
    case intrinsic::buffer_at:
      if (is_variable(op->args[0], sym)) {
        std::vector<expr> args = op->args;
        for (int i = static_cast<int>(slices.size()) - 1; i >= 0; --i) {
          if (slices[i] + 1 < static_cast<int>(args.size())) {
            args.erase(args.begin() + slices[i] + 1);
          }
        }
        bool changed = args.size() < op->args.size();
        for (expr& i : args) {
          expr new_i = mutate(i);
          changed = changed || !new_i.same_as(i);
          i = new_i;
        }
        if (changed) {
          set_result(call::make(intrinsic::buffer_at, std::move(args)));
        } else {
          set_result(op);
        }
        return;
      }
      break;
    default: break;
    }
    node_mutator::visit(op);
  }
};

}  // namespace

expr update_sliced_buffer_metadata(const expr& e, var buf, span<const int> slices) {
  return slice_updater(buf, slices).mutate(e);
}

interval_expr update_sliced_buffer_metadata(const interval_expr& x, var buf, span<const int> slices) {
  slice_updater m(buf, slices);
  if (x.is_point()) {
    return point(m.mutate(x.min));
  } else {
    return {m.mutate(x.min), m.mutate(x.max)};
  }
}

dim_expr update_sliced_buffer_metadata(const dim_expr& x, var buf, span<const int> slices) {
  return {
      update_sliced_buffer_metadata(x.bounds, buf, slices),
      update_sliced_buffer_metadata(x.stride, buf, slices),
      update_sliced_buffer_metadata(x.fold_factor, buf, slices),
  };
}

}  // namespace slinky
