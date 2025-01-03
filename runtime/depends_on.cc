#include "runtime/depends_on.h"

#include <cassert>

#include "base/chrome_trace.h"
#include "runtime/expr.h"

namespace slinky {

namespace {

class dependencies : public recursive_node_visitor {
public:
  bool is_pure = true;

  // This works by keeping track of dependencies for a set of vars. The order is important, later entries in this vector
  // will shadow earlier entries.
  // The size of this vector is important for performance, so if we encounter a var that shadows a var we care about, we
  // want to add a dummy to this.
  std::vector<std::pair<var, depends_on_result*>> var_deps;
  depends_on_result dummy_deps;

  dependencies() {}
  dependencies(std::vector<std::pair<var, depends_on_result*>> var_deps) : var_deps(var_deps) {}
  dependencies(span<const std::pair<var, depends_on_result&>> deps) {
    var_deps.reserve(deps.size());
    for (const auto& i : deps) {
      var_deps.push_back({i.first, &i.second});
    }
  }

  depends_on_result* find_deps(var s) {
    // Go in reverse order to handle shadowed declarations properly.
    for (auto i = var_deps.rbegin(); i != var_deps.rend(); ++i) {
      if (i->first == s) return i->second;
    }
    return nullptr;
  }

  depends_on_result* no_dummy(depends_on_result* deps) const { return deps != &dummy_deps ? deps : nullptr; }

  void visit(const variable* op) override {
    if (depends_on_result* deps = find_deps(op->sym)) {
      deps->var = true;
    }
  }
  void visit(const call* op) override {
    if (is_buffer_intrinsic(op->intrinsic)) {
      is_pure = false;
      assert(op->args.size() >= 1);
      if (op->args[0].defined()) {
        auto buf = as_variable(op->args[0]);
        assert(buf);
        if (depends_on_result* deps = find_deps(*buf)) {
          if (op->intrinsic == intrinsic::buffer_min || op->intrinsic == intrinsic::buffer_max) {
            deps->buffer_bounds = true;
          }
          if (is_buffer_dim_intrinsic(op->intrinsic)) {
            deps->buffer_dims = true;
          }
          if (op->intrinsic == intrinsic::buffer_at) {
            deps->buffer_base = true;
          }
          if (op->intrinsic == intrinsic::buffer_size_bytes) {
            deps->var = true;
          }
        }

        for (std::size_t i = 1; i < op->args.size(); ++i) {
          if (op->args[i].defined()) op->args[i].accept(this);
        }
      }
    } else {
      recursive_node_visitor::visit(op);
    }
  }

  template <typename T>
  void visit_let(const T* op) {
    size_t var_deps_count = var_deps.size();
    for (const auto& p : op->lets) {
      p.second.accept(this);
      if (no_dummy(find_deps(p.first))) {
        var_deps.push_back({p.first, &dummy_deps});
      }
    }
    op->body.accept(this);
    var_deps.resize(var_deps_count);
  }

  void visit_sym_body(var sym, depends_on_result* src_deps, const stmt& body) {
    if (!body.defined()) return;
    size_t var_deps_count = var_deps.size();
    if (no_dummy(src_deps)) {
      // We have src_deps we want to transitively add to via this declaration.
      var_deps.push_back({sym, src_deps});
    } else if (no_dummy(find_deps(sym))) {
      // We are shadowing something we are finding the dependencies of. Point at the dummy instead to avoid
      // contaminating the dependencies.
      var_deps.push_back({sym, &dummy_deps});
    }
    body.accept(this);
    var_deps.resize(var_deps_count);
  }

  void visit_sym_body(var sym, var src, depends_on_result* src_deps, const stmt& body) {
    if (sym == src) {
      if (!body.defined()) return;
      body.accept(this);
    } else {
      visit_sym_body(sym, src_deps, body);
    }
  }

  void visit(const loop* op) override {
    op->bounds.min.accept(this);
    op->bounds.max.accept(this);
    if (op->step.defined()) op->step.accept(this);

    visit_sym_body(op->sym, nullptr, op->body);
  }

  void visit(const call_stmt* op) override {
    for (var i : op->inputs) {
      if (depends_on_result* deps = find_deps(i)) {
        deps->var = true;
        deps->buffer_input = true;
        deps->buffer_dims = true;
      }
    }
    for (var i : op->outputs) {
      if (depends_on_result* deps = find_deps(i)) {
        deps->var = true;
        deps->buffer_output = true;
        deps->buffer_bounds = true;
        deps->buffer_dims = true;
      }
    }
  }

  void visit(const copy_stmt* op) override {
    if (depends_on_result* deps = find_deps(op->src)) {
      deps->var = true;
      deps->buffer_src = true;
      if (op->padding) {
        deps->buffer_bounds = true;
      }
      deps->buffer_dims = true;
    }
    if (depends_on_result* deps = find_deps(op->dst)) {
      deps->var = true;
      deps->buffer_dst = true;
      deps->buffer_bounds = true;
      deps->buffer_dims = true;
    }

    // copy_stmt is effectively a declaration of the dst_x symbols for the src_x expressions.
    size_t var_deps_count = var_deps.size();
    for (std::size_t i = 0; i < op->dst_x.size(); ++i) {
      if (no_dummy(find_deps(op->dst_x[i]))) {
        var_deps.push_back({op->dst_x[i], &dummy_deps});
      }
    }
    for (const expr& i : op->src_x) {
      i.accept(this);
    }
    var_deps.resize(var_deps_count);
  }

  void visit(const clone_buffer* op) override { visit_sym_body(op->sym, op->src, find_deps(op->src), op->body); }

  void visit(const allocate* op) override {
    op->elem_size.accept(this);
    for (const dim_expr& i : op->dims) {
      i.bounds.min.accept(this);
      i.bounds.max.accept(this);
      if (i.stride.defined()) i.stride.accept(this);
      if (i.fold_factor.defined()) i.fold_factor.accept(this);
    }
    visit_sym_body(op->sym, nullptr, op->body);
  }
  void visit(const make_buffer* op) override {
    if (op->base.defined()) op->base.accept(this);
    if (op->elem_size.defined()) op->elem_size.accept(this);
    for (const dim_expr& i : op->dims) {
      if (i.bounds.min.defined()) i.bounds.min.accept(this);
      if (i.bounds.max.defined()) i.bounds.max.accept(this);
      if (i.stride.defined()) i.stride.accept(this);
      if (i.fold_factor.defined()) i.fold_factor.accept(this);
    }
    visit_sym_body(op->sym, nullptr, op->body);
  }
  void visit(const crop_buffer* op) override {
    for (const interval_expr& i : op->bounds) {
      if (i.min.defined()) i.min.accept(this);
      if (i.max.defined()) i.max.accept(this);
    }
    depends_on_result* deps = find_deps(op->src);
    if (deps) {
      deps->buffer_bounds = true;
      deps->buffer_dims = true;
    }
    visit_sym_body(op->sym, op->src, deps, op->body);
  }
  void visit(const crop_dim* op) override {
    if (op->bounds.min.defined()) op->bounds.min.accept(this);
    if (op->bounds.max.defined()) op->bounds.max.accept(this);
    depends_on_result* deps = find_deps(op->src);
    if (deps) {
      deps->buffer_bounds = true;
      deps->buffer_dims = true;
    }
    visit_sym_body(op->sym, op->src, deps, op->body);
  }
  void visit(const slice_buffer* op) override {
    for (const expr& i : op->at) {
      if (i.defined()) i.accept(this);
    }
    depends_on_result* deps = find_deps(op->src);
    if (deps) {
      deps->buffer_bounds = true;
      deps->buffer_dims = true;
    }
    visit_sym_body(op->sym, op->src, deps, op->body);
  }
  void visit(const slice_dim* op) override {
    op->at.accept(this);
    depends_on_result* deps = find_deps(op->src);
    if (deps) {
      deps->buffer_bounds = true;
      deps->buffer_dims = true;
    }
    visit_sym_body(op->sym, op->src, deps, op->body);
  }
  void visit(const transpose* op) override {
    depends_on_result* deps = find_deps(op->src);
    if (deps) {
      deps->buffer_bounds = true;  // TODO: Maybe not?
      deps->buffer_dims = true;
    }
    visit_sym_body(op->sym, op->src, deps, op->body);
  }
};

}  // namespace

void depends_on(expr_ref e, span<const std::pair<var, depends_on_result&>> var_deps) {
  if (var_deps.empty()) return;
  dependencies v(var_deps);
  if (e.defined()) e.accept(&v);
}

void depends_on(stmt_ref s, span<const std::pair<var, depends_on_result&>> var_deps) {
  scoped_trace trace("depends_on");
  if (var_deps.empty()) return;
  dependencies v(var_deps);
  if (s.defined()) s.accept(&v);
}

void depends_on(expr_ref e, var x, depends_on_result& deps) {
  std::pair<var, depends_on_result&> var_deps[] = {{x, deps}};
  depends_on(e, var_deps);
}

void depends_on(stmt_ref s, var x, depends_on_result& deps) {
  std::pair<var, depends_on_result&> var_deps[] = {{x, deps}};
  depends_on(s, var_deps);
}

depends_on_result depends_on(expr_ref e, var x) {
  depends_on_result r;
  depends_on(e, x, r);
  return r;
}

depends_on_result depends_on(const interval_expr& e, var x) {
  depends_on_result r;
  depends_on(e.min, x, r);
  depends_on(e.max, x, r);
  return r;
}

depends_on_result depends_on(stmt_ref s, var x) {
  depends_on_result r;
  depends_on(s, x, r);
  return r;
}

depends_on_result depends_on(expr_ref e, span<const var> xs) {
  depends_on_result r;
  std::vector<std::pair<var, depends_on_result&>> var_deps;
  for (var x : xs) {
    var_deps.push_back({x, r});
  }
  depends_on(e, var_deps);
  return r;
}

depends_on_result depends_on(stmt_ref s, span<const var> xs) {
  depends_on_result r;
  std::vector<std::pair<var, depends_on_result&>> var_deps;
  for (var x : xs) {
    var_deps.push_back({x, r});
  }
  depends_on(s, var_deps);
  return r;
}

bool can_substitute_buffer(const depends_on_result& r) { return !(r.buffer_data() || r.var); }

bool is_pure(expr_ref x) {
  dependencies v;
  x.accept(&v);
  return v.is_pure;
}

}  // namespace slinky
