#include "runtime/validate.h"

#include "base/span.h"
#include "runtime/depends_on.h"
#include "runtime/expr.h"
#include "runtime/print.h"
#include "runtime/stmt.h"

namespace slinky {

namespace {

class validator : public expr_visitor, public stmt_visitor {
  enum variable_state {
    unknown,
    pointer,
    arithmetic,
  };

  const node_context* symbols;

  struct symbol_info {
    variable_state state;
    stmt decl_stmt;
    expr decl_expr;

    symbol_info(variable_state state) : state(state) {}
    symbol_info(variable_state state, const base_stmt_node* s) : state(state), decl_stmt(s) {}
    symbol_info(variable_state state, const base_expr_node* e) : state(state), decl_expr(e) {}
  };
  symbol_map<symbol_info> ctx;

  variable_state state = unknown;

  void print_symbol_info(const symbol_info& s) {
    if (s.decl_stmt.defined()) {
      std::cerr << "Declared by:" << std::endl;
      print(std::cerr, s.decl_stmt, symbols);
    } else if (s.decl_expr.defined()) {
      std::cerr << "Declared by:" << std::endl;
      print(std::cerr, s.decl_expr, symbols);
    } else {
      std::cerr << "Externally defined symbol" << std::endl;
    }
  }

public:
  bool error = false;

  validator(span<var> external, const node_context* symbols) : symbols(symbols) {
    for (var i : external) {
      ctx[i] = unknown;
    }
  }

  void visit(const variable* x) override {
    std::optional<symbol_info> x_state = ctx.lookup(x->sym);
    if (!x_state) {
      std::cerr << "Undefined variable ";
      print(std::cerr, expr_ref(x), symbols);
      std::cerr << " in context" << std::endl;
      error = true;
    }
    state = x_state->state;
  }
  void visit(const constant*) override { state = unknown; }

  template <typename T>
  void visit_let(const T* x) {
    std::vector<scoped_value_in_symbol_map<symbol_info>> lets;
    lets.reserve(x->lets.size());

    for (size_t i = 0; i < x->lets.size(); ++i) {
      check(x->lets[i].second);
      lets.push_back(set_value_in_scope(ctx, x->lets[i].first, symbol_info(state, x)));
    }
    if (!x->body.defined()) {
      std::cerr << "Undefined let body in context" << std::endl;
      error = true;
      return;
    }
    x->body.accept(this);
  }

  void visit(const let* x) override { visit_let(x); }

  void check_arithmetic(const expr& x, bool required = true) {
    if (error) return;

    if (x.defined()) {
      x.accept(this);
      if (state == pointer) {
        std::cerr << "Arithmetic on pointer value: ";
        print(std::cerr, x, symbols);
        std::cerr << std::endl << "In context:" << std::endl;
        error = true;
        return;
      }
    } else if (required) {
      std::cerr << "Undefined arithmetic expression in context:" << std::endl;
      error = true;
      return;
    }
    state = arithmetic;
  }

  void check(const expr& x, bool required = true) {
    if (error) return;
    if (x.defined()) {
      x.accept(this);
    } else if (required) {
      std::cerr << "Undefined expression in context:" << std::endl;
      error = true;
    }
  }

  template <typename T>
  void visit_binary_arithmetic(const T* x) {
    check_arithmetic(x->a, false);
    check_arithmetic(x->b, false);
  }

  template <typename T>
  void visit_binary(const T* x) {
    check(x->a, false);
    check(x->b, false);
  }

  void visit(const add* x) override { visit_binary_arithmetic(x); }
  void visit(const sub* x) override { visit_binary_arithmetic(x); }
  void visit(const mul* x) override { visit_binary_arithmetic(x); }
  void visit(const div* x) override { visit_binary_arithmetic(x); }
  void visit(const mod* x) override { visit_binary_arithmetic(x); }
  void visit(const class min* x) override { visit_binary_arithmetic(x); }
  void visit(const class max* x) override { visit_binary_arithmetic(x); }
  void visit(const equal* x) override { visit_binary(x); }
  void visit(const not_equal* x) override { visit_binary(x); }
  void visit(const less* x) override { visit_binary_arithmetic(x); }
  void visit(const less_equal* x) override { visit_binary_arithmetic(x); }
  void visit(const logical_and* x) override { visit_binary(x); }
  void visit(const logical_or* x) override { visit_binary(x); }
  void visit(const logical_not* x) override { check(x->a); }
  void visit(const class select* x) override {
    check(x->condition);
    check(x->true_value, false);
    check(x->false_value, false);
  }

  void check_pointer(const expr& x) {
    if (error) return;

    if (x.defined()) {
      x.accept(this);
      if (state == arithmetic) {
        std::cerr << "Expression " << x << " is arithmetic, expected pointer" << std::endl;
        std::cerr << std::endl << "In context:" << std::endl;
        error = true;
      }
    } else {
      std::cerr << "Undefined pointer expression in context:" << std::endl;
    }
  }

  void check_pointer(var sym) {
    std::optional<symbol_info> state = ctx.lookup(sym);
    if (state && state->state == arithmetic) {
      std::cerr << "Arithmetic symbol ";
      print(std::cerr, var(sym), symbols);
      std::cerr << " used as a pointer" << std::endl;
      print_symbol_info(*state);
      error = true;
    }
    state = pointer;
  }

  void visit(const call* x) override {
    if (error) return;

    switch (x->intrinsic) {
    case intrinsic::negative_infinity:
    case intrinsic::positive_infinity:
    case intrinsic::indeterminate:
      std::cerr << "Cannot evaluate " << x->intrinsic << std::endl;
      error = true;
      return;

    case intrinsic::abs: check_arithmetic(x->args[0]); return;

    case intrinsic::buffer_rank:
    case intrinsic::buffer_elem_size:
    case intrinsic::buffer_size_bytes:
      if (x->args.size() != 1) {
        std::cerr << "Wrong number of arguments for buffer intrinsic " << x->intrinsic << std::endl;
        error = true;
        return;
      }
      check_pointer(x->args[0]);
      state = arithmetic;
      return;
    case intrinsic::buffer_min:
    case intrinsic::buffer_max:
    case intrinsic::buffer_stride:
    case intrinsic::buffer_fold_factor:
      if (x->args.size() != 2) {
        std::cerr << "Wrong number of arguments for buffer intrinsic " << x->intrinsic << std::endl;
        error = true;
        return;
      }
      check_pointer(x->args[0]);
      check_arithmetic(x->args[1]);
      state = arithmetic;
      return;

    case intrinsic::buffer_at:
      check_pointer(x->args[0]);
      for (size_t i = 1; i < x->args.size(); ++i) {
        check_arithmetic(x->args[i], false);
      }
      state = arithmetic;
      return;
    case intrinsic::free:
      if (x->args.size() != 1) {
        std::cerr << "Wrong number of arguments for buffer intrinsic " << x->intrinsic << std::endl;
        error = true;
        return;
      }
      check_pointer(x->args[0]);
      state = arithmetic;
      return;
    case intrinsic::and_then:
    case intrinsic::or_else:
      for (const expr& i : x->args) {
        check_arithmetic(i);
      }
      state = arithmetic;
      return;
    case intrinsic::define_undef:
      if (x->args.size() != 2) {
        std::cerr << "Wrong number of arguments for buffer intrinsic " << x->intrinsic << std::endl;
        error = true;
        return;
      }
      check(x->args[0]);
      check(x->args[1]);
      return;
    case intrinsic::semaphore_init:
      if (x->args.size() != 2) {
        std::cerr << "Wrong number of arguments for buffer intrinsic " << x->intrinsic << std::endl;
        error = true;
        return;
      }
      check(x->args[0]);
      check_arithmetic(x->args[1], /*required=*/false);
      state = arithmetic;
      return;
    case intrinsic::semaphore_wait:
    case intrinsic::semaphore_signal:
      if (x->args.size() % 2 != 0) {
        std::cerr << "Wrong number of arguments for buffer intrinsic " << x->intrinsic << std::endl;
        error = true;
        return;
      }
      for (std::size_t i = 0; i < x->args.size(); i += 2) {
        check(x->args[i]);
        check_arithmetic(x->args[i + 1], /*required=*/false);
      }
      return;
    case intrinsic::trace_begin:
      if (x->args.size() != 1) {
        std::cerr << "Wrong number of arguments for buffer intrinsic " << x->intrinsic << std::endl;
        error = true;
        return;
      }
      return;
    case intrinsic::trace_end:
      if (x->args.size() != 1) {
        std::cerr << "Wrong number of arguments for buffer intrinsic " << x->intrinsic << std::endl;
        error = true;
        return;
      }
      state = arithmetic;
      return;
    }
  }

  void visit(const let_stmt* x) override { visit_let(x); }

  void check(const stmt& s, bool required = true) {
    if (s.defined()) {
      s.accept(this);
    } else if (required) {
      std::cerr << "Undefined statement " << std::endl;
      error = true;
    }
    if (error) {
      std::cerr << "  " << s.type() << std::endl;
    }
  }

  void check(const stmt& s, var sym, bool required = true) {
    if (s.defined()) {
      s.accept(this);
    } else if (required) {
      std::cerr << "Undefined statement " << std::endl;
      error = true;
    }
    if (error) {
      std::cerr << "  " << s.type() << " " << sym << std::endl;
    }
  }

  void visit(const block* x) override {
    for (const stmt& i : x->stmts) {
      check(i);
    }
  }
  void visit(const loop* x) override {
    check_arithmetic(x->bounds.min);
    check_arithmetic(x->bounds.max);
    check_arithmetic(x->step);
    auto s = set_value_in_scope(ctx, x->sym, {arithmetic, x});
    check(x->body);
  }
  void visit(const call_stmt* x) override {
    for (var b : x->inputs) {
      check_pointer(b);
    }
    for (var b : x->outputs) {
      check_pointer(b);
    }
  }
  void visit(const copy_stmt* x) override {
    check_pointer(x->src);
    check_pointer(x->dst);
  }

  void check_arithmetic(const interval_expr& b, bool required = true) {
    check_arithmetic(b.min, required);
    check_arithmetic(b.max, required);
  }

  void visit(const allocate* x) override {
    for (const dim_expr& d : x->dims) {
      check_arithmetic(d.bounds, /*required=*/true);
      check_arithmetic(d.stride, /*required=*/false);
      check_arithmetic(d.fold_factor, /*required=*/false);
    }
    auto s = set_value_in_scope(ctx, x->sym, {pointer, x});
    check(x->body, x->sym);
  }
  void visit(const make_buffer* x) override {
    check_arithmetic(x->base);  // We treat pointers to data as arithmetic
    check_arithmetic(x->elem_size);
    for (const dim_expr& d : x->dims) {
      check_arithmetic(d.bounds, /*required=*/true);
      check_arithmetic(d.stride, /*required=*/true);
      check_arithmetic(d.fold_factor, /*required=*/false);
    }
    auto s = set_value_in_scope(ctx, x->sym, {pointer, x});
    check(x->body, x->sym);
  }

  void visit(const crop_buffer* x) override {
    check_pointer(x->src);
    for (const interval_expr& i : x->bounds) {
      check_arithmetic(i, /*required=*/false);
    }
    auto s = set_value_in_scope(ctx, x->sym, {pointer, x});
    check(x->body, x->sym);
  }
  void visit(const crop_dim* x) override {
    check_pointer(x->src);
    check_arithmetic(x->bounds);
    auto s = set_value_in_scope(ctx, x->sym, {pointer, x});
    check(x->body, x->sym);
  }
  void visit(const slice_buffer* x) override {
    check_pointer(x->src);
    for (const expr& i : x->at) {
      check_arithmetic(i, /*required=*/false);
    }
    auto s = set_value_in_scope(ctx, x->sym, {pointer, x});
    check(x->body, x->sym);
  }
  void visit(const slice_dim* x) override {
    check_pointer(x->src);
    check_arithmetic(x->at);
    auto s = set_value_in_scope(ctx, x->sym, {pointer, x});
    check(x->body, x->sym);
  }
  void visit(const transpose* x) override {
    check_pointer(x->src);
    auto s = set_value_in_scope(ctx, x->sym, {pointer, x});
    check(x->body, x->sym);
  }
  void visit(const clone_buffer* x) override {
    check_pointer(x->src);
    auto s = set_value_in_scope(ctx, x->sym, {pointer, x});
    check(x->body, x->sym);
  }
  void visit(const class check* x) override { check(x->condition); }
};

}  // namespace

bool is_valid(const expr& e, span<var> external, const node_context* symbols) {
  validator v(external, symbols);
  e.accept(&v);
  return !v.error;
}

bool is_valid(const stmt& s, span<var> external, const node_context* symbols) {
  validator v(external, symbols);
  s.accept(&v);
  return !v.error;
}

}  // namespace slinky
