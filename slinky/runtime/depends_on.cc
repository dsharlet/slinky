#include "slinky/runtime/depends_on.h"

#include <cassert>
#include <cstddef>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "slinky/base/chrome_trace.h"
#include "slinky/base/span.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

namespace slinky {

namespace {

class dependencies {
public:
  bool is_pure = true;

  // This works by keeping track of dependencies for a set of vars. The order is important, later entries in this vector
  // will shadow earlier entries, and entries in `var_deps` shadow entries in `base_deps`.
  // The size of this vector is important for performance, so if we encounter a var that shadows a var we care about, we
  // want to add a dummy to this.
  span<std::pair<var, depends_on_result&>> base_deps;
  std::vector<std::pair<var, depends_on_result*>> var_deps;
  depends_on_result dummy_deps;

  // If non-null, we'll also add dependencies of variables as we find them.
  std::map<var, depends_on_result>* unknown_deps = nullptr;

  dependencies() {}
  dependencies(std::map<var, depends_on_result>& unknown_deps) : unknown_deps(&unknown_deps) {}
  dependencies(span<std::pair<var, depends_on_result&>> deps) : base_deps(deps) {}

  depends_on_result* find_deps(var s) {
    if (!s.defined()) return nullptr;

    // Go in reverse order to handle shadowed declarations properly.
    for (auto i = var_deps.rbegin(); i != var_deps.rend(); ++i) {
      if (i->first == s) return i->second;
    }
    for (auto i = base_deps.rbegin(); i != base_deps.rend(); ++i) {
      if (i->first == s) return &i->second;
    }

    if (unknown_deps) {
      return &(*unknown_deps)[s];
    }
    return nullptr;
  }

  // Check if we need to shadow a declaration.
  bool decl_needs_shadow(var s) const {
    // Go in reverse order to handle shadowed declarations properly.
    for (auto i = var_deps.rbegin(); i != var_deps.rend(); ++i) {
      if (i->first == s) return i->second != &dummy_deps;
    }
    for (auto i = base_deps.rbegin(); i != base_deps.rend(); ++i) {
      if (i->first == s) return true;
    }

    return unknown_deps != nullptr;
  }

  depends_on_result* no_dummy(depends_on_result* deps) const { return deps != &dummy_deps ? deps : nullptr; }

  void visit(const variable* op) {
    if (is_pure && op->field != buffer_field::none) is_pure = false;
    if (depends_on_result* deps = find_deps(op->sym)) {
      switch (op->field) {
      case buffer_field::none: deps->var = true; break;
      case buffer_field::min:
      case buffer_field::max:
        deps->buffer_bounds = true;
        deps->buffer_dims = true;
        break;
      case buffer_field::stride:
      case buffer_field::fold_factor: deps->buffer_dims = true; break;
      case buffer_field::rank:
      case buffer_field::elem_size: deps->var = true; break;
      }
    }
  }
  void visit(const call* op) {
    if (op->intrinsic == intrinsic::buffer_at) {
      is_pure = false;
      assert(op->args.size() >= 1);
      if (auto buf = as_variable(op->args[0])) {
        if (depends_on_result* deps = find_deps(*buf)) {
          deps->buffer_base = true;
          deps->buffer_dims = true;
          deps->buffer_bounds = true;
        }
      }

      for (std::size_t i = 1; i < op->args.size(); ++i) {
        visit(op->args[i]);
      }
    } else {
      if (op->intrinsic == intrinsic::buffer_size_bytes || op->intrinsic == intrinsic::validate_buffer) {
        is_pure = false;
      }
      for (const expr& i : op->args) {
        visit(i);
      }
    }
  }

  template <typename T>
  void visit_let(const T* op) {
    size_t var_deps_count = var_deps.size();
    for (const auto& p : op->lets) {
      visit(p.second);
      if (decl_needs_shadow(p.first)) {
        var_deps.push_back({p.first, &dummy_deps});
      }
    }
    visit(op->body);
    var_deps.resize(var_deps_count);
  }

  void visit(const let* op) { visit_let(op); }
  void visit(const let_stmt* op) { visit_let(op); }

  void visit(const binary_op* op) {
    visit(op->a);
    visit(op->b);
  }
  void visit(const logical_not* op) { visit(op->a); }
  void visit(const class select* op) {
    visit(op->condition);
    visit(op->true_value);
    visit(op->false_value);
  }

  void visit(const block* op) {
    for (const stmt& s : op->stmts) {
      visit(s);
    }
  }

  void visit_sym_body(var sym, depends_on_result* src_deps, const stmt& body) {
    size_t var_deps_count = var_deps.size();
    if (no_dummy(src_deps)) {
      // We have src_deps we want to transitively add to via this declaration.
      var_deps.push_back({sym, src_deps});
    } else if (decl_needs_shadow(sym)) {
      if (!unknown_deps && var_deps.size() + base_deps.size() == 1) {
        // The only dep we are looking for is shadowed by this declaration, so the body cannot depend on it.
        return;
      }
      // We are shadowing something we are finding the dependencies of. Point at the dummy instead to avoid
      // contaminating the dependencies.
      var_deps.push_back({sym, &dummy_deps});
    }
    visit(body);
    var_deps.resize(var_deps_count);
  }

  void visit_sym_body(var sym, var src, depends_on_result* src_deps, const stmt& body) {
    if (sym == src) {
      visit(body);
    } else {
      visit_sym_body(sym, src_deps, body);
    }
  }

  void visit(const loop* op) {
    visit(op->bounds.min);
    visit(op->bounds.max);
    visit(op->step);
    visit(op->max_workers);

    visit_sym_body(op->sym, nullptr, op->body);
  }

  void visit(const call_stmt* op) {
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
    for (const expr& i : op->scalars) {
      visit(i);
    }
  }

  void visit(const copy_stmt* op) {
    if (depends_on_result* deps = find_deps(op->src)) {
      deps->var = true;
      deps->buffer_src = true;
      if (op->pad.defined()) {
        deps->buffer_bounds = true;
      }
      deps->buffer_dims = true;
    }
    if (depends_on_result* deps = find_deps(op->pad)) {
      deps->var = true;
      deps->buffer_src = true;
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
      if (decl_needs_shadow(op->dst_x[i])) {
        var_deps.push_back({op->dst_x[i], &dummy_deps});
      }
    }
    for (const expr& i : op->src_x) {
      visit(i);
    }
    var_deps.resize(var_deps_count);
  }

  void visit(const clone_buffer* op) { visit_sym_body(op->sym, op->src, find_deps(op->src), op->body); }

  void visit(const allocate* op) {
    visit(op->elem_size);
    for (const dim_expr& i : op->dims) {
      visit(i.bounds.min);
      visit(i.bounds.max);
      visit(i.stride);
      visit(i.fold_factor);
    }
    visit_sym_body(op->sym, nullptr, op->body);
  }
  void visit(const make_buffer* op) {
    visit(op->base);
    visit(op->elem_size);
    for (const dim_expr& i : op->dims) {
      visit(i.bounds.min);
      visit(i.bounds.max);
      visit(i.stride);
      visit(i.fold_factor);
    }
    visit_sym_body(op->sym, nullptr, op->body);
  }
  void visit(const crop_buffer* op) {
    for (const interval_expr& i : op->bounds) {
      visit(i.min);
      visit(i.max);
    }
    depends_on_result* deps = find_deps(op->src);
    if (deps) {
      deps->buffer_bounds = true;
      deps->buffer_dims = true;
    }
    visit_sym_body(op->sym, op->src, deps, op->body);
  }
  void visit(const crop_dim* op) {
    visit(op->bounds.min);
    visit(op->bounds.max);
    depends_on_result* deps = find_deps(op->src);
    if (deps) {
      deps->buffer_bounds = true;
      deps->buffer_dims = true;
    }
    visit_sym_body(op->sym, op->src, deps, op->body);
  }
  void visit(const slice_buffer* op) {
    for (const expr& i : op->at) {
      visit(i);
    }
    depends_on_result* deps = find_deps(op->src);
    if (deps) {
      deps->buffer_bounds = true;
      deps->buffer_dims = true;
    }
    visit_sym_body(op->sym, op->src, deps, op->body);
  }
  void visit(const slice_dim* op) {
    visit(op->at);
    depends_on_result* deps = find_deps(op->src);
    if (deps) {
      deps->buffer_bounds = true;
      deps->buffer_dims = true;
    }
    visit_sym_body(op->sym, op->src, deps, op->body);
  }
  void visit(const transpose* op) {
    depends_on_result* deps = find_deps(op->src);
    if (deps) {
      deps->buffer_bounds = true;  // TODO: Maybe not?
      deps->buffer_dims = true;
    }
    visit_sym_body(op->sym, op->src, deps, op->body);
  }

  void visit(const async* op) {
    visit(op->task);
    visit(op->body);
  }
  void visit(const check* op) { visit(op->condition); }

  void visit(expr_ref e) {
    switch (e.type()) {
    case expr_node_type::variable: return visit(static_cast<const variable*>(e.get()));
    case expr_node_type::let: return visit(static_cast<const let*>(e.get()));
    case expr_node_type::add:
    case expr_node_type::sub:
    case expr_node_type::mul:
    case expr_node_type::div:
    case expr_node_type::mod:
    case expr_node_type::min:
    case expr_node_type::max:
    case expr_node_type::equal:
    case expr_node_type::not_equal:
    case expr_node_type::less:
    case expr_node_type::less_equal:
    case expr_node_type::logical_and:
    case expr_node_type::logical_or: return visit(static_cast<const binary_op*>(e.get()));
    case expr_node_type::logical_not: return visit(static_cast<const logical_not*>(e.get()));
    case expr_node_type::select: return visit(static_cast<const class select*>(e.get()));
    case expr_node_type::call: return visit(static_cast<const call*>(e.get()));
    case expr_node_type::constant:
    case expr_node_type::constant_buffer:
    case expr_node_type::none: return;
    }
  }

  void visit(stmt_ref s) {
    switch (s.type()) {
    case stmt_node_type::call_stmt: return visit(static_cast<const call_stmt*>(s.get()));
    case stmt_node_type::copy_stmt: return visit(static_cast<const copy_stmt*>(s.get()));
    case stmt_node_type::let_stmt: return visit(static_cast<const let_stmt*>(s.get()));
    case stmt_node_type::block: return visit(static_cast<const block*>(s.get()));
    case stmt_node_type::loop: return visit(static_cast<const loop*>(s.get()));
    case stmt_node_type::allocate: return visit(static_cast<const allocate*>(s.get()));
    case stmt_node_type::make_buffer: return visit(static_cast<const make_buffer*>(s.get()));
    case stmt_node_type::clone_buffer: return visit(static_cast<const clone_buffer*>(s.get()));
    case stmt_node_type::crop_buffer: return visit(static_cast<const crop_buffer*>(s.get()));
    case stmt_node_type::crop_dim: return visit(static_cast<const crop_dim*>(s.get()));
    case stmt_node_type::slice_buffer: return visit(static_cast<const slice_buffer*>(s.get()));
    case stmt_node_type::slice_dim: return visit(static_cast<const slice_dim*>(s.get()));
    case stmt_node_type::transpose: return visit(static_cast<const transpose*>(s.get()));
    case stmt_node_type::async: return visit(static_cast<const async*>(s.get()));
    case stmt_node_type::check: return visit(static_cast<const check*>(s.get()));
    case stmt_node_type::none: return;
    }
  }
};

}  // namespace

void depends_on(expr_ref e, span<std::pair<var, depends_on_result&>> var_deps) {
  if (var_deps.empty()) return;
  dependencies v(var_deps);
  v.visit(e);
}

void depends_on(stmt_ref s, span<std::pair<var, depends_on_result&>> var_deps) {
  scoped_trace trace("depends_on");
  if (var_deps.empty()) return;
  dependencies v(var_deps);
  v.visit(s);
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

depends_on_result depends_on(expr_ref e, span<var> xs) {
  depends_on_result r;
  std::vector<std::pair<var, depends_on_result&>> var_deps;
  for (var x : xs) {
    var_deps.push_back({x, r});
  }
  depends_on(e, var_deps);
  return r;
}

depends_on_result depends_on(stmt_ref s, span<var> xs) {
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
  v.visit(x);
  return v.is_pure;
}

namespace {

std::vector<var> keys(const std::map<var, depends_on_result>& m) {
  std::vector<var> result;
  result.reserve(m.size());
  for (const auto& i : m) {
    result.push_back(i.first);
  }
  return result;
}

}  // namespace

std::vector<var> find_dependencies(expr_ref e) {
  std::map<var, depends_on_result> deps;
  dependencies v(deps);
  v.visit(e);
  return keys(deps);
}
std::vector<var> find_dependencies(stmt_ref s) {
  std::map<var, depends_on_result> deps;
  dependencies v(deps);
  v.visit(s);
  return keys(deps);
}

std::vector<var> find_buffer_dependencies(stmt_ref s) { return find_buffer_dependencies(s, true, true, true, true); }

std::vector<var> find_buffer_dependencies(stmt_ref s, bool input, bool output) {
  return find_buffer_dependencies(s, input, input, output, output);
}

std::vector<var> find_buffer_dependencies(stmt_ref s, bool input, bool src, bool output, bool dst) {
  std::map<var, depends_on_result> deps;
  dependencies v(deps);
  v.visit(s);

  std::vector<var> result;
  result.reserve(deps.size());
  for (const auto& i : deps) {
    if ((input && i.second.buffer_input) || (src && i.second.buffer_src) || (output && i.second.buffer_output) ||
        (dst && i.second.buffer_dst)) {
      result.push_back(i.first);
    }
  }
  return result;
}

var find_buffer_data_dependency(expr_ref e) {
  std::map<var, depends_on_result> deps;
  dependencies v(deps);
  v.visit(e);

  var result;
  for (const auto& i : deps) {
    if (i.second.buffer_base) {
      if (result.defined()) {
        // More than one data dependency
        return var();
      } else {
        result = i.first;
      }
    }
  }
  return result;
}

namespace {
class side_effect_checker : public recursive_node_visitor {

public:
  bool has_side_effects = false;

  void visit(const call* op) override {
    switch (op->intrinsic) {
    case intrinsic::none:
    case intrinsic::semaphore_init:
    case intrinsic::semaphore_signal:
    case intrinsic::semaphore_wait:
    case intrinsic::wait_for:
    case intrinsic::trace_begin:
    case intrinsic::trace_end:
    case intrinsic::free: has_side_effects = true; return;
    case intrinsic::abs:
    case intrinsic::negative_infinity:
    case intrinsic::positive_infinity:
    case intrinsic::indeterminate:
    case intrinsic::and_then:
    case intrinsic::or_else:
    case intrinsic::buffer_at:
    case intrinsic::buffer_size_bytes:
    case intrinsic::validate_buffer: break;
    };

    recursive_node_visitor::visit(op);
  }
};

}  // namespace

bool has_side_effects(expr_ref x) {
  side_effect_checker checker;
  x.accept(&checker);
  return checker.has_side_effects;
}

}  // namespace slinky
