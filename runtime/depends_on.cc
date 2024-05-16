#include "runtime/depends_on.h"

#include <cassert>

#include "runtime/expr.h"

namespace slinky {

namespace {

class dependencies : public recursive_node_visitor {
public:
  std::vector<std::pair<var, depends_on_result*>> var_deps;

  dependencies(std::vector<std::pair<var, depends_on_result*>> var_deps) : var_deps(var_deps) {}
  dependencies(span<const std::pair<var, depends_on_result&>> deps) {
    var_deps.reserve(deps.size());
    for (const auto& i : deps) {
      var_deps.push_back({i.first, &i.second});
    }
  }

  template <typename Fn>
  void update_deps(var s, Fn fn) {
    // Go in reverse order to handle shadowed declarations properly.
    for (auto i = var_deps.rbegin(); i != var_deps.rend(); ++i) {
      if (i->first == s) {
        fn(*i->second);
        return;
      }
    }
  }

  // If a symbol is used as a buffer input or output, that should propagate through aliases of the symbol.
  void propagate_deps(const depends_on_result& deps, var to) {
    update_deps(to, [&](depends_on_result& to_deps) {
      to_deps.buffer_input = to_deps.buffer_input || deps.buffer_input;
      to_deps.buffer_output = to_deps.buffer_output || deps.buffer_output;
      to_deps.buffer_src = to_deps.buffer_src || deps.buffer_src;
      to_deps.buffer_dst = to_deps.buffer_dst || deps.buffer_dst;
    });
  }

  void visit(const variable* op) override {
    update_deps(op->sym, [](depends_on_result& deps) { deps.var = true; });
  }
  void visit(const call* op) override {
    if (is_buffer_intrinsic(op->intrinsic)) {
      assert(op->args.size() >= 1);
      if (op->args[0].defined()) {
        const var* buf = as_variable(op->args[0]);
        assert(buf);
        update_deps(*buf, [](depends_on_result& deps) { deps.buffer_meta = true; });

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
    std::vector<depends_on_result> let_deps;
    let_deps.reserve(op->lets.size());
    for (const auto& p : op->lets) {
      p.second.accept(this);
      let_deps.push_back({});
      var_deps.push_back({p.first, &let_deps.back()});
    }
    op->body.accept(this);
    for (const auto& p : op->lets) {
      var_deps.pop_back();
    }
  }

  template <typename T>
  depends_on_result visit_sym_body(const T* op) {
    if (!op->body.defined()) return depends_on_result{};
    depends_on_result sym_deps;
    var_deps.push_back({op->sym, &sym_deps});
    op->body.accept(this);
    var_deps.pop_back();
    return sym_deps;
  }

  void visit(const loop* op) override {
    op->bounds.min.accept(this);
    op->bounds.max.accept(this);
    if (op->step.defined()) op->step.accept(this);

    visit_sym_body(op);
  }

  void visit(const call_stmt* op) override {
    for (var i : op->inputs) {
      update_deps(i, [](depends_on_result& deps) { deps.buffer_input = true; });
    }
    for (var i : op->outputs) {
      update_deps(i, [](depends_on_result& deps) {
        deps.buffer_output = true;
        deps.buffer_meta = true;
      });
    }
  }

  void visit(const copy_stmt* op) override {
    update_deps(op->src, [](depends_on_result& deps) {
      deps.buffer_src = true;
      deps.buffer_meta = true;
    });
    update_deps(op->dst, [](depends_on_result& deps) {
      deps.buffer_dst = true;
      deps.buffer_meta = true;
    });

    // copy_stmt is effectively a declaration of the dst_x symbols for the src_x expressions.
    depends_on_result* sym_deps = SLINKY_ALLOCA(depends_on_result, op->dst_x.size());
    for (std::size_t i = 0; i < op->dst_x.size(); ++i) {
      var_deps.push_back({op->dst_x[i], &sym_deps[i]});
    }
    for (const expr& i : op->src_x) {
      i.accept(this);
    }
    for (std::size_t i = 0; i < op->dst_x.size(); ++i) {
      var_deps.pop_back();
    }
  }

  void visit(const clone_buffer* op) override {
    update_deps(op->src, [](depends_on_result& deps) { deps.buffer_meta = true; });
    depends_on_result sym_deps = visit_sym_body(op);
    propagate_deps(sym_deps, op->src);
  }

  void visit(const allocate* op) override {
    op->elem_size.accept(this);
    for (const dim_expr& i : op->dims) {
      i.bounds.min.accept(this);
      i.bounds.max.accept(this);
      if (i.stride.defined()) i.stride.accept(this);
      if (i.fold_factor.defined()) i.fold_factor.accept(this);
    }
    visit_sym_body(op);
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
    visit_sym_body(op);
  }
  void visit(const crop_buffer* op) override {
    for (const interval_expr& i : op->bounds) {
      if (i.min.defined()) i.min.accept(this);
      if (i.max.defined()) i.max.accept(this);
    }
    update_deps(op->src, [](depends_on_result& deps) { deps.buffer_meta = true; });
    depends_on_result sym_deps = visit_sym_body(op);
    propagate_deps(sym_deps, op->src);
  }
  void visit(const crop_dim* op) override {
    if (op->bounds.min.defined()) op->bounds.min.accept(this);
    if (op->bounds.max.defined()) op->bounds.max.accept(this);
    update_deps(op->src, [](depends_on_result& deps) { deps.buffer_meta = true; });
    depends_on_result sym_deps = visit_sym_body(op);
    propagate_deps(sym_deps, op->src);
  }
  void visit(const slice_buffer* op) override {
    for (const expr& i : op->at) {
      if (i.defined()) i.accept(this);
    }
    update_deps(op->src, [](depends_on_result& deps) { deps.buffer_meta = true; });
    depends_on_result sym_deps = visit_sym_body(op);
    propagate_deps(sym_deps, op->src);
  }
  void visit(const slice_dim* op) override {
    op->at.accept(this);
    update_deps(op->src, [](depends_on_result& deps) { deps.buffer_meta = true; });
    depends_on_result sym_deps = visit_sym_body(op);
    propagate_deps(sym_deps, op->src);
  }
  void visit(const transpose* op) override {
    update_deps(op->src, [](depends_on_result& deps) { deps.buffer_meta = true; });
    depends_on_result sym_deps = visit_sym_body(op);
    propagate_deps(sym_deps, op->src);
  }
};

}  // namespace

void depends_on(const expr& e, span<const std::pair<var, depends_on_result&>> var_deps) {
  dependencies v(var_deps);
  if (e.defined()) e.accept(&v);
}

void depends_on(const stmt& s, span<const std::pair<var, depends_on_result&>> var_deps) {
  dependencies v(var_deps);
  if (s.defined()) s.accept(&v);
}

void depends_on(const expr& e, var x, depends_on_result& deps) {
  std::pair<var, depends_on_result&> var_deps[] = {{x, deps}};
  depends_on(e, var_deps);
}

void depends_on(const stmt& s, var x, depends_on_result& deps) {
  std::pair<var, depends_on_result&> var_deps[] = {{x, deps}};
  depends_on(s, var_deps);
}

depends_on_result depends_on(const expr& e, var x) {
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

depends_on_result depends_on(const stmt& s, var x) {
  depends_on_result r;
  depends_on(s, x, r);
  return r;
}

depends_on_result depends_on(const expr& e, span<const var> xs) {
  depends_on_result r;
  std::vector<std::pair<var, depends_on_result&>> var_deps;
  for (var x : xs) {
    var_deps.push_back({x, r});
  }
  depends_on(e, var_deps);
  return r;
}

depends_on_result depends_on(const stmt& s, span<const var> xs) {
  depends_on_result r;
  std::vector<std::pair<var, depends_on_result&>> var_deps;
  for (var x : xs) {
    var_deps.push_back({x, r});
  }
  depends_on(s, var_deps);
  return r;
}

}  // namespace slinky
