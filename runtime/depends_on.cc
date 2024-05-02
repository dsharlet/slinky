#include "runtime/depends_on.h"

#include <cassert>

#include "runtime/expr.h"

namespace slinky {

namespace {

class dependencies : public recursive_node_visitor {
public:
  span<const std::pair<var, depends_on_result&>> var_deps;
  std::vector<var> shadowed;

  dependencies(span<const std::pair<var, depends_on_result&>> var_deps) : var_deps(var_deps) {}

  template <typename Fn>
  void update_deps(var i, Fn fn) {
    if (std::find(shadowed.begin(), shadowed.end(), i) != shadowed.end()) return;
    for (const auto& v : var_deps) {
      if (v.first == i) {
        fn(v.second);
        v.second.ref_count++;
      }
    }
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
        update_deps(*buf, [](depends_on_result& deps) { deps.buffer_meta_read = true; });

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
    for (const auto& p : op->lets) {
      p.second.accept(this);
      shadowed.push_back(p.first);
    }
    op->body.accept(this);
    for (const auto& p : op->lets) {
      shadowed.pop_back();
    }
  }

  template <typename T>
  void visit_sym_body(const T* op, bool shadow = true) {
    if (!op->body.defined()) return;
    if (shadow) shadowed.push_back(op->sym);
    op->body.accept(this);
    if (shadow) shadowed.pop_back();
  }

  void visit(const loop* op) override {
    op->bounds.min.accept(this);
    op->bounds.max.accept(this);
    if (op->step.defined()) op->step.accept(this);

    std::vector<int> old_ref_count(var_deps.size());
    for (std::size_t i = 0; i < var_deps.size(); ++i) {
      old_ref_count[i] = var_deps[i].second.ref_count;
    }

    visit_sym_body(op);

    for (std::size_t i = 0; i < var_deps.size(); ++i) {
      if (var_deps[i].second.ref_count > old_ref_count[i]) {
        var_deps[i].second.used_in_loop = true;
      }
    }
  }

  void visit(const call_stmt* op) override {
    for (var i : op->inputs) {
      update_deps(i, [](depends_on_result& deps) {
        deps.buffer_input = true;
        deps.buffer_meta_read = true;
      });
    }
    for (var i : op->outputs) {
      update_deps(i, [](depends_on_result& deps) {
        deps.buffer_output = true;
        deps.buffer_meta_read = true;
      });
    }
  }

  void visit(const copy_stmt* op) override {
    update_deps(op->src, [](depends_on_result& deps) {
      deps.buffer_src = true;
      deps.buffer_meta_read = true;
    });
    update_deps(op->dst, [](depends_on_result& deps) {
      deps.buffer_dst = true;
      deps.buffer_meta_read = true;
    });
    for (const expr& i : op->src_x) {
      i.accept(this);
    }
  }

  void visit(const clone_buffer* op) override {
    update_deps(op->src, [](depends_on_result& deps) { deps.buffer_meta_read = true; });
    visit_sym_body(op);
  }

  void visit(const allocate* op) override {
    op->elem_size.accept(this);
    for (const dim_expr& i : op->dims) {
      i.bounds.min.accept(this);
      i.bounds.max.accept(this);
      i.stride.accept(this);
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
    update_deps(op->sym, [](depends_on_result& deps) {
      deps.buffer_meta_read = true;
      deps.buffer_meta_mutated = true;
    });
    visit_sym_body(op, /*shadow=*/false);
  }
  void visit(const crop_dim* op) override {
    if (op->bounds.min.defined()) op->bounds.min.accept(this);
    if (op->bounds.max.defined()) op->bounds.max.accept(this);
    update_deps(op->sym, [](depends_on_result& deps) {
      deps.buffer_meta_read = true;
      deps.buffer_meta_mutated = true;
    });
    visit_sym_body(op, /*shadow=*/false);
  }
  void visit(const slice_buffer* op) override {
    for (const expr& i : op->at) {
      if (i.defined()) i.accept(this);
    }
    update_deps(op->sym, [](depends_on_result& deps) { deps.buffer_meta_mutated = true; });
    visit_sym_body(op, /*shadow=*/false);
  }
  void visit(const slice_dim* op) override {
    op->at.accept(this);
    update_deps(op->sym, [](depends_on_result& deps) { deps.buffer_meta_mutated = true; });
    visit_sym_body(op, /*shadow=*/false);
  }
  void visit(const truncate_rank* op) override {
    update_deps(op->sym, [](depends_on_result& deps) { deps.buffer_meta_mutated = true; });
    visit_sym_body(op, /*shadow=*/false);
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
