#include "builder/substitute.h"

#include <cassert>
#include <cstddef>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "builder/node_mutator.h"
#include "runtime/depends_on.h"
#include "runtime/expr.h"
#include "runtime/util.h"

namespace slinky {

class matcher : public expr_visitor, public stmt_visitor {
  // In this class, we visit the pattern, and manually traverse the expression being matched.
  union {
    void* self = nullptr;
    const base_expr_node* self_expr;
    const base_stmt_node* self_stmt;
  };
  symbol_map<expr>* matches;

public:
  int match = 0;

  matcher(symbol_map<expr>* matches = nullptr) : matches(matches) {}

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
    } else if (matches && op->type == expr_node_type::variable) {
      // When we are matching with variables as wildcards, the type doesn't need to match.
      self_expr = e;
      visit(reinterpret_cast<const variable*>(op));
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
    if (self.size() < op.size()) {
      match = -1;
      return false;
    } else if (self.size() > op.size()) {
      match = 1;
      return false;
    }

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
    if (matches) {
      std::optional<expr>& matched = (*matches)[op->sym];
      if (matched) {
        // We already matched this variable. The expression must match.
        if (!matched->same_as(self_expr)) {
          symbol_map<expr>* old_matches = matches;
          matches = nullptr;
          try_match(matched->get(), self_expr);
          matches = old_matches;
        }
      } else {
        // This is a new match.
        matched = self_expr;
      }
    } else {
      const variable* ev = static_cast<const variable*>(self);
      try_match(ev->sym, op->sym);
    }
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
    if (!try_match(cbs->bounds, op->bounds)) return;
    if (!try_match(cbs->body, op->body)) return;
  }

  void visit(const crop_dim* op) override {
    const crop_dim* cds = static_cast<const crop_dim*>(self);
    assert(cds);

    if (!try_match(cds->sym, op->sym)) return;
    if (!try_match(cds->dim, op->dim)) return;
    if (!try_match(cds->bounds, op->bounds)) return;
    if (!try_match(cds->body, op->body)) return;
  }

  void visit(const slice_buffer* op) override {
    const slice_buffer* cbs = static_cast<const slice_buffer*>(self);
    assert(cbs);

    if (!try_match(cbs->sym, op->sym)) return;
    if (!try_match(cbs->at, op->at)) return;
    if (!try_match(cbs->body, op->body)) return;
  }

  void visit(const slice_dim* op) override {
    const slice_dim* cds = static_cast<const slice_dim*>(self);
    assert(cds);

    if (!try_match(cds->sym, op->sym)) return;
    if (!try_match(cds->dim, op->dim)) return;
    if (!try_match(cds->at, op->at)) return;
    if (!try_match(cds->body, op->body)) return;
  }

  void visit(const truncate_rank* op) override {
    const truncate_rank* trs = static_cast<const truncate_rank*>(self);
    assert(trs);

    if (!try_match(trs->sym, op->sym)) return;
    if (!try_match(trs->rank, op->rank)) return;
    if (!try_match(trs->body, op->body)) return;
  }

  void visit(const check* op) override {
    const check* cs = static_cast<const check*>(self);
    assert(cs);

    try_match(cs->condition, op->condition);
  }
};

bool match(const expr& p, const expr& e, symbol_map<expr>& matches) {
  matcher m(&matches);
  m.try_match(e, p);
  return m.match == 0;
}

bool match(const expr& a, const expr& b) { return compare(a, b) == 0; }
bool match(const stmt& a, const stmt& b) { return compare(a, b) == 0; }
bool match(const interval_expr& a, const interval_expr& b) { return match(a.min, b.min) && match(a.max, b.max); }
bool match(const dim_expr& a, const dim_expr& b) {
  return match(a.bounds, b.bounds) && match(a.stride, b.stride) && match(a.fold_factor, b.fold_factor);
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

class substitutor : public node_mutator {
  const symbol_map<expr>* replacements = nullptr;
  symbol_id target_var;
  expr replacement;
  span<const std::pair<expr, expr>> expr_replacements;

  // Track newly declared variables that might shadow the variables we want to replace.
  std::vector<symbol_id> shadowed;

public:
  substitutor(const symbol_map<expr>& replacements) : replacements(&replacements) {}
  substitutor(symbol_id target, const expr& replacement) : target_var(target), replacement(replacement) {}
  substitutor(span<const std::pair<expr, expr>> expr_replacements) : expr_replacements(expr_replacements) {}

  expr mutate(const expr& op) override {
    for (const auto& i : expr_replacements) {
      if (!depends_on(i.first, shadowed).any() && match(op, i.first) && !depends_on(i.second, shadowed).any()) {
        return i.second;
      }
    }
    return node_mutator::mutate(op);
  }
  using node_mutator::mutate;

  void visit(const variable* v) override {
    if (std::find(shadowed.begin(), shadowed.end(), v->sym) != shadowed.end()) {
      // This variable has been shadowed, don't substitute it.
    } else if (v->sym == target_var && !depends_on(replacement, shadowed).any()) {
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

  static symbol_id replacement_symbol(const expr& r) {
    const symbol_id* s = as_variable(r);
    assert(s);
    return *s;
  }

  symbol_id visit_symbol(symbol_id x) {
    if (std::find(shadowed.begin(), shadowed.end(), x) != shadowed.end()) {
      // This variable has been shadowed, don't substitute it.
      return x;
    } else if (x == target_var && !depends_on(replacement, shadowed).any()) {
      return replacement_symbol(replacement);
    } else if (replacements) {
      std::optional<expr> r = replacements->lookup(x);
      if (r && !depends_on(*r, shadowed).any()) {
        return replacement_symbol(*r);
      }
    }
    for (const auto& i : expr_replacements) {
      const symbol_id* target = as_variable(i.first);
      if (!target) continue;
      if (*target == x) {
        return replacement_symbol(i.second);
      }
    }
    return x;
  }

  template <typename T>
  T mutate_decl_body(symbol_id sym, const T& x) {
    shadowed.push_back(sym);
    T result = mutate(x);
    shadowed.pop_back();
    return result;
  }

  template <typename T>
  auto mutate_let(const T* op) {
    std::vector<std::pair<symbol_id, expr>> lets;
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

  stmt mutate_slice_body(symbol_id sym, span<const int> slices, stmt body) {
    const symbol_map<expr>* old_replacements = replacements;
    expr old_replacement = replacement;
    span<const std::pair<expr, expr>> old_expr_replacements = expr_replacements;

    symbol_map<expr> new_replacements;
    if (replacements) {
      new_replacements = *replacements;
      for (std::optional<expr>& i : new_replacements) {
        if (i) i = update_sliced_buffer_metadata(*i, sym, slices);
      }
      replacements = &new_replacements;
    }

    replacement = update_sliced_buffer_metadata(replacement, sym, slices);

    std::vector<std::pair<expr, expr>> new_expr_replacements;
    new_expr_replacements.reserve(expr_replacements.size());
    for (const std::pair<expr, expr>& i : expr_replacements) {
      new_expr_replacements.emplace_back(
          update_sliced_buffer_metadata(i.first, sym, slices), update_sliced_buffer_metadata(i.second, sym, slices));
    }
    expr_replacements = span<const std::pair<expr, expr>>(new_expr_replacements);

    body = mutate(body);

    replacements = old_replacements;
    replacement = old_replacement;
    expr_replacements = old_expr_replacements;

    return body;
  }
  void visit(const slice_buffer* op) override {
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
    stmt body = mutate_slice_body(op->sym, dims, op->body);
    if (!changed && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(slice_buffer::make(op->sym, std::move(at), std::move(body)));
    }
  }
  void visit(const slice_dim* op) override {
    expr at = mutate(op->at);
    int slices[] = {op->dim};
    stmt body = mutate_slice_body(op->sym, slices, op->body);
    if (at.same_as(op->at) && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(slice_dim::make(op->sym, op->dim, std::move(at), std::move(body)));
    }
  }

  interval_expr substitute_crop_bounds(symbol_id sym, int dim, const interval_expr& bounds) {
    // When substituting crop bounds, we need to apply the implicit clamp, which uses buffer_min(sym, dim) and
    // buffer_max(sym, dim).
    expr buf_var = variable::make(sym);
    interval_expr old_bounds = {buffer_min(buf_var, dim), buffer_max(buf_var, dim)};
    interval_expr new_bounds = {mutate(old_bounds.min), mutate(old_bounds.max)};
    interval_expr result = {mutate(bounds.min), mutate(bounds.max)};
    if (!old_bounds.min.same_as(new_bounds.min)) {
      // The substitution changed the implicit clamp, include it.
      result.min = max(result.min, new_bounds.min);
    }
    if (!old_bounds.max.same_as(new_bounds.max)) {
      // The substitution changed the implicit clamp, include it.
      result.max = min(result.max, new_bounds.max);
    }
    return result;
  }

  void visit(const crop_buffer* op) override {
    box_expr bounds(op->bounds.size());
    bool changed = false;
    for (std::size_t i = 0; i < op->bounds.size(); ++i) {
      bounds[i] = substitute_crop_bounds(op->sym, i, op->bounds[i]);
      changed = changed || !bounds[i].same_as(op->bounds[i]);
    }
    stmt body = mutate_decl_body(op->sym, op->body);
    if (changed || !body.same_as(op->body)) {
      set_result(crop_buffer::make(op->sym, std::move(bounds), std::move(body)));
    } else {
      set_result(op);
    }
  }

  void visit(const crop_dim* op) override {
    interval_expr bounds = substitute_crop_bounds(op->sym, op->dim, op->bounds);
    stmt body = mutate_decl_body(op->sym, op->body);
    if (bounds.same_as(op->bounds) && body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(crop_dim::make(op->sym, op->dim, std::move(bounds), std::move(body)));
    }
  }

  void visit(const call* op) override {
    if (op->intrinsic == intrinsic::buffer_at) {
      std::vector<expr> args;
      args.reserve(op->args.size());
      bool changed = false;
      for (const expr& i : op->args) {
        args.push_back(mutate(i));
        changed = changed || !args.back().same_as(i);
      }
      for (const auto& i : expr_replacements) {
        if (const call* c = i.first.as<call>()) {
          if (c->intrinsic == intrinsic::buffer_min && match(c->args[0], op->args[0])) {
            const index_t* dim = as_constant(c->args[1]);
            assert(dim);
            if (*dim + 1 >= static_cast<index_t>(args.size())) {
              args.resize(*dim + 2);
            }
            expr& arg_d = args[*dim + 1];
            if (!arg_d.defined()) {
              // This is implicitly buffer_min(...)
              arg_d = i.second;
              changed = true;
            }
          }
        }
      }
      if (changed) {
        set_result(call::make(intrinsic::buffer_at, std::move(args)));
        return;
      }
    }
    node_mutator::visit(op);
  }
  // truncate_rank, clone_buffer, not treated here because references to dimensions of these
  // operations are still valid.
  // TODO: truncate_rank is a bit tricky, the replacements for expressions might be invalid if they access truncated
  // dims.

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
    symbol_id src = visit_symbol(op->src);
    symbol_id dst = visit_symbol(op->dst);

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
    symbol_id src = visit_symbol(op->src);
    stmt body = mutate_decl_body(op->sym, op->body);
    if (src != op->src || !body.same_as(op->body)) {
      set_result(clone_buffer::make(op->sym, src, std::move(body)));
    } else {
      set_result(op);
    }
  }
};

template <typename T>
T substitute_bounds_impl(T op, symbol_id buffer, int dim, const interval_expr& bounds) {
  expr buf_var = variable::make(buffer);
  std::vector<std::pair<expr, expr>> subs;
  subs.reserve(2);
  if (bounds.min.defined()) subs.emplace_back(buffer_min(buf_var, dim), bounds.min);
  if (bounds.max.defined()) subs.emplace_back(buffer_max(buf_var, dim), bounds.max);
  return substitutor(subs).mutate(op);
}

template <typename T>
T substitute_bounds_impl(T op, symbol_id buffer, const box_expr& bounds) {
  expr buf_var = variable::make(buffer);
  std::vector<std::pair<expr, expr>> subs;
  subs.reserve(bounds.size() * 2);
  for (index_t d = 0; d < static_cast<index_t>(bounds.size()); ++d) {
    if (bounds[d].min.defined()) subs.emplace_back(buffer_min(buf_var, d), bounds[d].min);
    if (bounds[d].max.defined()) subs.emplace_back(buffer_max(buf_var, d), bounds[d].max);
  }
  return substitutor(subs).mutate(op);
}

}  // namespace

expr substitute(const expr& e, const symbol_map<expr>& replacements) { return substitutor(replacements).mutate(e); }
stmt substitute(const stmt& s, const symbol_map<expr>& replacements) { return substitutor(replacements).mutate(s); }

expr substitute(const expr& e, symbol_id target, const expr& replacement) {
  return substitutor(target, replacement).mutate(e);
}
stmt substitute(const stmt& s, symbol_id target, const expr& replacement) {
  return substitutor(target, replacement).mutate(s);
}

expr substitute(const expr& e, const expr& target, const expr& replacement) {
  std::pair<expr, expr> subs[] = {{target, replacement}};
  return substitutor(subs).mutate(e);
}
stmt substitute(const stmt& s, const expr& target, const expr& replacement) {
  std::pair<expr, expr> subs[] = {{target, replacement}};
  return substitutor(subs).mutate(s);
}

expr substitute_bounds(const expr& e, symbol_id buffer, const box_expr& bounds) {
  return substitute_bounds_impl(e, buffer, bounds);
}
stmt substitute_bounds(const stmt& s, symbol_id buffer, const box_expr& bounds) {
  return substitute_bounds_impl(s, buffer, bounds);
}
expr substitute_bounds(const expr& e, symbol_id buffer, int dim, const interval_expr& bounds) {
  return substitute_bounds_impl(e, buffer, dim, bounds);
}
stmt substitute_bounds(const stmt& s, symbol_id buffer, int dim, const interval_expr& bounds) {
  return substitute_bounds_impl(s, buffer, dim, bounds);
}

namespace {

class slice_updater : public node_mutator {
  symbol_id sym;
  span<const int> slices;

public:
  slice_updater(symbol_id sym, span<const int> slices) : sym(sym), slices(slices) {}

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

expr update_sliced_buffer_metadata(const expr& e, symbol_id buf, span<const int> slices) {
  return slice_updater(buf, slices).mutate(e);
}

}  // namespace slinky
