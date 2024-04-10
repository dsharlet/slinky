#include "evaluate.h"

#include <cassert>
#include <iostream>

#include "print.h"
#include "simplify.h"
#include "substitute.h"

namespace slinky {

bool can_evaluate(intrinsic fn) {
  switch (fn) {
  case intrinsic::abs: return true;
  default: return false;
  }
}

void dump_context_for_expr(
    std::ostream& s, const symbol_map<index_t>& ctx, const expr& deps_of, const node_context* symbols = nullptr) {
  for (symbol_id i = 0; i < ctx.size(); ++i) {
    std::string name = symbols ? symbols->name(i) : "<" + std::to_string(i) + ">";
    if (!deps_of.defined() || depends_on_variable(deps_of, i)) {
      if (ctx.contains(i)) {
        s << "  " << name << " = " << *ctx.lookup(i) << std::endl;
      } else {
        s << "  " << name << " = <>" << std::endl;
      }
    } else if (!deps_of.defined() || depends_on_buffer(deps_of, i)) {
      if (ctx.contains(i)) {
        const raw_buffer* buf = reinterpret_cast<const raw_buffer*>(*ctx.lookup(i));
        s << "  " << name << " = {base=" << buf->base << ", elem_size=" << buf->elem_size << ", dims={";
        for (std::size_t d = 0; d < buf->rank; ++d) {
          const dim& dim = buf->dims[d];
          s << "{min=" << dim.min() << ", max=" << dim.max() << ", extent=" << dim.extent()
            << ", stride=" << dim.stride();
          if (dim.fold_factor() > 0) {
            s << ", fold_factor=" << dim.fold_factor();
          }
          s << "}";
          if (d + 1 < buf->rank) {
            s << ",";
          }
        }
        s << "}" << std::endl;
      }
    }
  }
}

// TODO(https://github.com/dsharlet/slinky/issues/2): I think the T::accept/node_visitor::visit
// overhead (two virtual function calls per node) might be significant. This could be implemented
// as a switch statement instead.
class evaluator : public node_visitor {
public:
  index_t result = 0;
  eval_context& context;

  evaluator(eval_context& context) : context(context) {}

  // Skip the visitor pattern (two virtual function calls) for some frequently used node types.
  void visit(const expr& x) {
    switch (x.type()) {
    case node_type::variable: visit(reinterpret_cast<const variable*>(x.get())); return;
    case node_type::constant: visit(reinterpret_cast<const constant*>(x.get())); return;
    default: x.accept(this);
    }
  }

  // Assume `e` is defined, evaluate it and return the result.
  index_t eval_expr(const expr& e) {
    visit(e);
    index_t r = result;
    result = 0;
    return r;
  }

  // If `e` is defined, evaluate it and return the result. Otherwise, return default `def`.
  index_t eval_expr(const expr& e, index_t def) {
    if (e.defined()) {
      return eval_expr(e);
    } else {
      return def;
    }
  }

  void visit(const variable* v) override {
    auto value = context.lookup(v->name);
    assert(value);
    result = *value;
  }

  void visit(const wildcard* w) override {
    // Maybe evaluating this should just be an error.
    auto value = context.lookup(w->name);
    assert(value);
    result = *value;
  }

  void visit(const constant* c) override { result = c->value; }

  template <typename T>
  void visit_let(const T* l) {
    auto set_value = set_value_in_scope(context, l->name, eval_expr(l->value));
    l->body.accept(this);
  }

  void visit(const let* l) override { visit_let(l); }
  void visit(const let_stmt* l) override { visit_let(l); }

  void visit(const add* x) override { result = eval_expr(x->a) + eval_expr(x->b); }
  void visit(const sub* x) override { result = eval_expr(x->a) - eval_expr(x->b); }
  void visit(const mul* x) override { result = eval_expr(x->a) * eval_expr(x->b); }
  void visit(const div* x) override { result = euclidean_div(eval_expr(x->a), eval_expr(x->b)); }
  void visit(const mod* x) override { result = euclidean_mod(eval_expr(x->a), eval_expr(x->b)); }
  void visit(const class min* x) override { result = std::min(eval_expr(x->a), eval_expr(x->b)); }
  void visit(const class max* x) override { result = std::max(eval_expr(x->a), eval_expr(x->b)); }
  void visit(const equal* x) override { result = eval_expr(x->a) == eval_expr(x->b); }
  void visit(const not_equal* x) override { result = eval_expr(x->a) != eval_expr(x->b); }
  void visit(const less* x) override { result = eval_expr(x->a) < eval_expr(x->b); }
  void visit(const less_equal* x) override { result = eval_expr(x->a) <= eval_expr(x->b); }
  void visit(const logical_and* x) override { result = eval_expr(x->a) != 0 && eval_expr(x->b) != 0; }
  void visit(const logical_or* x) override { result = eval_expr(x->a) != 0 || eval_expr(x->b) != 0; }
  void visit(const logical_not* x) override { result = eval_expr(x->x) == 0; }

  void visit(const class select* x) override {
    if (eval_expr(x->condition)) {
      result = eval_expr(x->true_value);
    } else {
      result = eval_expr(x->false_value);
    }
  }

  index_t eval_buffer_metadata(const call* x) {
    assert(x->args.size() == 1);
    raw_buffer* buf = reinterpret_cast<raw_buffer*>(eval_expr(x->args[0]));
    assert(buf);
    switch (x->intrinsic) {
    case intrinsic::buffer_rank: return buf->rank;
    case intrinsic::buffer_elem_size: return buf->elem_size;
    case intrinsic::buffer_base: return reinterpret_cast<index_t>(buf->base);
    case intrinsic::buffer_size_bytes: return buf->size_bytes();
    default: std::abort();
    }
  }

  index_t eval_dim_metadata(const call* x) {
    assert(x->args.size() == 2);
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(eval_expr(x->args[0]));
    assert(buffer);
    index_t d = eval_expr(x->args[1]);
    assert(d < static_cast<index_t>(buffer->rank));
    const slinky::dim& dim = buffer->dim(d);
    switch (x->intrinsic) {
    case intrinsic::buffer_min: return dim.min();
    case intrinsic::buffer_max: return dim.max();
    case intrinsic::buffer_extent: return dim.extent();
    case intrinsic::buffer_stride: return dim.stride();
    case intrinsic::buffer_fold_factor: return dim.fold_factor();
    default: std::abort();
    }
  }

  void* eval_buffer_at(const call* x) {
    assert(x->args.size() >= 1);
    raw_buffer* buf = reinterpret_cast<raw_buffer*>(eval_expr(x->args[0]));
    void* result = buf->base;
    assert(x->args.size() <= buf->rank + 1);
    for (std::size_t d = 0; d < x->args.size() - 1; ++d) {
      if (x->args[d + 1].defined()) {
        result = offset_bytes(result, buf->dims[d].flat_offset_bytes(eval_expr(x->args[d + 1])));
      }
    }
    return result;
  }

  void visit(const call* x) override {
    switch (x->intrinsic) {
    case intrinsic::positive_infinity: std::cerr << "Cannot evaluate positive_infinity" << std::endl; std::abort();
    case intrinsic::negative_infinity: std::cerr << "Cannot evaluate negative_infinity" << std::endl; std::abort();
    case intrinsic::indeterminate: std::cerr << "Cannot evaluate indeterminate" << std::endl; std::abort();

    case intrinsic::abs:
      assert(x->args.size() == 1);
      result = std::abs(eval_expr(x->args[0]));
      return;

    case intrinsic::buffer_rank:
    case intrinsic::buffer_elem_size:
    case intrinsic::buffer_base:
    case intrinsic::buffer_size_bytes: 
      result = eval_buffer_metadata(x); 
      return;

    case intrinsic::buffer_min:
    case intrinsic::buffer_max:
    case intrinsic::buffer_extent:
    case intrinsic::buffer_stride:
    case intrinsic::buffer_fold_factor: 
      result = eval_dim_metadata(x); 
      return;

    case intrinsic::buffer_at: 
      result = reinterpret_cast<index_t>(eval_buffer_at(x)); 
      return;
    default: 
      std::cerr << "Unknown intrinsic: " << x->intrinsic << std::endl; 
      std::abort();
    }
  }

  void visit(const block* b) override {
    if (result == 0) b->a.accept(this);
    if (result == 0) b->b.accept(this);
  }

  void visit(const loop* l) override {
    index_t min = eval_expr(l->bounds.min);
    index_t max = eval_expr(l->bounds.max);
    // TODO(https://github.com/dsharlet/slinky/issues/3): We don't get a reference to context[l->name] here
    // because the context could grow and invalidate the reference. This could be fixed by having evaluate
    // fully traverse the expression to find the max symbol_id, and pre-allocate the context up front. It's
    // not clear this optimization is necessary yet.
    std::optional<index_t> old_value = context[l->name];
    for (index_t i = min; result == 0 && i <= max; ++i) {
      context[l->name] = i;
      l->body.accept(this);
    }
    context[l->name] = old_value;
  }

  void visit(const if_then_else* n) override {
    if (eval_expr(n->condition)) {
      n->true_body.accept(this);
    } else if (n->false_body.defined()) {
      n->false_body.accept(this);
    }
  }

  void visit(const call_func* n) override {
    index_t* scalars = reinterpret_cast<index_t*>(alloca(n->scalar_args.size() * sizeof(index_t)));
    for (std::size_t i = 0; i < n->scalar_args.size(); ++i) {
      scalars[i] = eval_expr(n->scalar_args[i]);
    }

    raw_buffer** buffers = reinterpret_cast<raw_buffer**>(alloca(n->buffer_args.size() * sizeof(raw_buffer*)));
    for (std::size_t i = 0; i < n->buffer_args.size(); ++i) {
      buffers[i] = reinterpret_cast<raw_buffer*>(*context.lookup(n->buffer_args[i]));
    }

    std::span<const index_t> scalars_span(scalars, n->scalar_args.size());
    std::span<raw_buffer*> buffers_span(buffers, n->buffer_args.size());
    result = n->target(scalars_span, buffers_span);
    if (result) {
      if (context.call_failed) {
        context.call_failed(n);
      } else {
        std::cerr << "call_func failed: " << stmt(n) << "->" << result << std::endl;
        std::abort();
      }
    }
  }

  void visit(const allocate* n) override {
    std::size_t rank = n->dims.size();
    // Allocate a buffer with space for its dims on the stack.
    char* storage = reinterpret_cast<char*>(alloca(sizeof(raw_buffer) + sizeof(dim) * rank));
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(&storage[0]);
    buffer->elem_size = n->elem_size;
    buffer->rank = rank;
    buffer->dims = reinterpret_cast<dim*>(&storage[sizeof(raw_buffer)]);

    for (std::size_t i = 0; i < rank; ++i) {
      slinky::dim& dim = buffer->dim(i);
      dim.set_bounds(eval_expr(n->dims[i].min()), eval_expr(n->dims[i].max()));
      dim.set_stride(eval_expr(n->dims[i].stride));
      dim.set_fold_factor(eval_expr(n->dims[i].fold_factor));
    }

    if (n->storage == memory_type::stack) {
      buffer->base = alloca(buffer->size_bytes());
    } else {
      assert(n->storage == memory_type::heap);
      buffer->allocation = nullptr;
      if (context.allocate) {
        assert(context.free);
        context.allocate(n->name, buffer);
      } else {
        buffer->allocate();
      }
    }

    auto set_buffer = set_value_in_scope(context, n->name, reinterpret_cast<index_t>(buffer));
    n->body.accept(this);

    if (n->storage == memory_type::heap) {
      if (context.free) {
        assert(context.allocate);
        context.free(n->name, buffer);
      } else {
        buffer->free();
      }
    }
  }

  void visit(const make_buffer* n) override {
    std::size_t rank = n->dims.size();
    // Allocate a buffer with space for its dims on the stack.
    char* storage = reinterpret_cast<char*>(alloca(sizeof(raw_buffer) + sizeof(dim) * rank));
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(&storage[0]);
    buffer->elem_size = eval_expr(n->elem_size);
    buffer->base = reinterpret_cast<void*>(eval_expr(n->base));
    buffer->rank = rank;
    buffer->dims = reinterpret_cast<dim*>(&storage[sizeof(raw_buffer)]);

    for (std::size_t i = 0; i < rank; ++i) {
      slinky::dim& dim = buffer->dim(i);
      dim.set_bounds(eval_expr(n->dims[i].min()), eval_expr(n->dims[i].max()));
      dim.set_stride(eval_expr(n->dims[i].stride));
      dim.set_fold_factor(eval_expr(n->dims[i].fold_factor));
    }

    auto set_buffer = set_value_in_scope(context, n->name, reinterpret_cast<index_t>(buffer));
    n->body.accept(this);
  }

  void visit(const crop_buffer* n) override {
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(*context.lookup(n->name));

    struct range {
      index_t min;
      index_t extent;
    };

    std::size_t crop_rank = n->bounds.size();
    range* old_bounds = reinterpret_cast<range*>(alloca(sizeof(range) * crop_rank));

    index_t offset = 0;
    for (std::size_t d = 0; d < crop_rank; ++d) {
      slinky::dim& dim = buffer->dims[d];
      old_bounds[d].min = dim.min();
      old_bounds[d].extent = dim.extent();

      // Allow these expressions to be undefined, and if so, they default to their existing values.
      index_t min = eval_expr(n->bounds[d].min, dim.min());
      index_t max = eval_expr(n->bounds[d].max, dim.max());
      offset += dim.flat_offset_bytes(min);

      dim.set_bounds(min, max);
    }

    void* old_base = buffer->base;
    buffer->base = offset_bytes(buffer->base, offset);

    n->body.accept(this);

    buffer->base = old_base;
    for (std::size_t d = 0; d < crop_rank; ++d) {
      slinky::dim& dim = buffer->dims[d];
      dim.set_min_extent(old_bounds[d].min, old_bounds[d].extent);
    }
  }

  void visit(const crop_dim* n) override {
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(*context.lookup(n->name));
    slinky::dim& dim = buffer->dims[n->dim];

    void* old_base = buffer->base;
    index_t old_min = dim.min();
    index_t old_extent = dim.extent();

    index_t min = eval_expr(n->bounds.min);
    buffer->base = offset_bytes(buffer->base, dim.flat_offset_bytes(min));
    if (n->bounds.min.same_as(n->bounds.max)) {
      // Crops to a single element are common, we can optimize them a little bit by re-using the min
      dim.set_point(min);
    } else {
      dim.set_bounds(min, eval_expr(n->bounds.max));
    }

    n->body.accept(this);

    buffer->base = old_base;
    dim.set_min_extent(old_min, old_extent);
  }

  void visit(const slice_buffer* n) override {
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(*context.lookup(n->name));

    // The rank of the result is equal to the current rank, less any sliced dimensions.
    std::size_t old_rank = buffer->rank;
    dim* old_dims = buffer->dims;

    // TODO: If we really care about stack usage here, we could find the number of dimensions we actually need first.
    buffer->dims = reinterpret_cast<dim*>(alloca(sizeof(dim) * old_rank));

    buffer->rank = 0;
    index_t offset = 0;
    for (std::size_t d = 0; d < old_rank; ++d) {
      if (d < n->at.size() && n->at[d].defined()) {
        offset += old_dims[d].flat_offset_bytes(eval_expr(n->at[d]));
      } else {
        buffer->dims[buffer->rank++] = old_dims[d];
      }
    }

    void* old_base = buffer->base;
    buffer->base = offset_bytes(buffer->base, offset);

    n->body.accept(this);

    buffer->base = old_base;
    buffer->rank = old_rank;
    buffer->dims = old_dims;
  }

  void visit(const slice_dim* n) override {
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(*context.lookup(n->name));

    // The rank of the result is equal to the current rank, less any sliced dimensions.
    dim* old_dims = buffer->dims;

    buffer->dims = reinterpret_cast<dim*>(alloca(sizeof(dim) * (buffer->rank - 1)));

    index_t at = eval_expr(n->at);
    index_t offset = old_dims[n->dim].flat_offset_bytes(at);
    void* old_base = buffer->base;
    buffer->base = offset_bytes(buffer->base, offset);

    for (int d = 0; d < n->dim; ++d) {
      buffer->dims[d] = old_dims[d];
    }
    for (int d = n->dim + 1; d < static_cast<int>(buffer->rank); ++d) {
      buffer->dims[d - 1] = old_dims[d];
    }
    buffer->rank -= 1;

    n->body.accept(this);

    buffer->base = old_base;
    buffer->rank += 1;
    buffer->dims = old_dims;
  }

  void visit(const truncate_rank* n) override {
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(*context.lookup(n->name));

    std::size_t old_rank = buffer->rank;
    buffer->rank = n->rank;

    n->body.accept(this);

    buffer->rank = old_rank;
  }

  void visit(const check* n) override {
    result = eval_expr(n->condition, 0) != 0 ? 0 : 1;
    if (result) {
      if (context.check_failed) {
        context.check_failed(n->condition);
      } else {
        std::cerr << "Check failed: " << n->condition << std::endl;
        std::cerr << "Context: " << std::endl;
        dump_context_for_expr(std::cerr, context, n->condition);
        std::abort();
      }
    }
  }
};

index_t evaluate(const expr& e, eval_context& context) {
  evaluator eval(context);
  e.accept(&eval);
  return eval.result;
}

index_t evaluate(const stmt& s, eval_context& context) {
  evaluator eval(context);
  s.accept(&eval);
  return eval.result;
}

index_t evaluate(const expr& e) {
  eval_context ctx;
  return evaluate(e, ctx);
}

index_t evaluate(const stmt& s) {
  eval_context ctx;
  return evaluate(s, ctx);
}

namespace {

class validator : public node_visitor {
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
    symbol_info(variable_state state, stmt s) : state(state), decl_stmt(std::move(s)) {}
    symbol_info(variable_state state, expr e) : state(state), decl_expr(std::move(e)) {}
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

  validator(const std::vector<symbol_id>& inputs, const node_context* symbols) : symbols(symbols) {
    for (symbol_id i : inputs) {
      ctx[i] = unknown;
    }
  }

  void visit(const variable* x) override {
    if (!ctx.contains(x->name)) {
      std::cerr << "Undefined variable ";
      print(std::cerr, x, symbols);
      std::cerr << " in context" << std::endl;
      error = true;
    }
    state = unknown;
  }
  void visit(const wildcard* x) override {
    if (!ctx.contains(x->name)) {
      std::cerr << "Undefined wildcard ";
      print(std::cerr, x, symbols);
      std::cerr << std::endl << "In context:" << std::endl;
      error = true;
    }
  }
  void visit(const constant*) override { state = unknown; }

  template <typename T>
  void visit_let(const T* x) {
    x->value.accept(this);
    auto s = set_value_in_scope(ctx, x->name, {state, x});
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
      }
    } else if (required) {
      std::cerr << "Undefined expression in context:" << std::endl;
    }
  }

  void check(const expr& x) {
    if (error) return;
    x.accept(this);
  }

  template <typename T>
  void visit_binary_arithmetic(const T* x) {
    check_arithmetic(x->a);
    check_arithmetic(x->b);
  }

  template <typename T>
  void visit_binary(const T* x) {
    check(x->a);
    check(x->b);
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
  void visit(const logical_not* x) override { check(x->x); }
  void visit(const class select* x) override {
    check(x->condition);
    check(x->true_value);
    check(x->false_value);
  }

  void check_pointer(const expr& x) {
    x.accept(this);
    if (state == arithmetic) {
      std::cerr << "Expression " << x << " is arithmetic, expected pointer" << std::endl;
      std::cerr << std::endl << "In context:" << std::endl;
      error = true;
    }
  }

  void check_pointer(symbol_id name) {
    std::optional<symbol_info> state = ctx.lookup(name);
    if (state && state->state == arithmetic) {
      std::cerr << "Arithmetic symbol ";
      print(std::cerr, var(name), symbols);
      std::cerr << " used as a pointer" << std::endl;
      print_symbol_info(*state);
      error = true;
    }
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
    case intrinsic::buffer_base:  // We treat pointers to data as arithmetic
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
    case intrinsic::buffer_extent:
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
      for (std::size_t i = 1; i < x->args.size(); ++i) {
        check_arithmetic(x->args[i]);
      }
      state = arithmetic;
      return;
    }
  }

  void visit(const let_stmt* x) override { visit_let(x); }

  void check(const stmt& s, bool required = true) {
    if (error) return;

    if (s.defined()) {
      s.accept(this);
    } else if (required) {
      std::cerr << "Undefined statement " << std::endl;
      error = true;
    }
  }

  void visit(const block* x) override {
    check(x->a, /*required=*/false);
    check(x->b, /*required=*/false);
  }
  void visit(const loop* x) override {
    check_arithmetic(x->bounds.min);
    check_arithmetic(x->bounds.max);
    auto s = set_value_in_scope(ctx, x->name, {arithmetic, x});
    check(x->body);
  }
  void visit(const if_then_else* x) override {
    check_arithmetic(x->condition);
    check(x->true_body, /*required=*/false);
    check(x->false_body, /*required=*/false);
  }
  void visit(const call_func* x) override {
    for (symbol_id b : x->buffer_args) {
      check_pointer(b);
    }
  }

  void check_arithmetic(const interval_expr& b, bool required = true) {
    check_arithmetic(b.min, required);
    check_arithmetic(b.max, required);
  }
  void check_arithmetic(const dim_expr& d, bool required = true) {
    check_arithmetic(d.bounds, required);
    check_arithmetic(d.stride, required);
    check_arithmetic(d.fold_factor, required);
  }

  void visit(const allocate* x) override {
    for (const dim_expr& i : x->dims) {
      check_arithmetic(i);
    }
    auto s = set_value_in_scope(ctx, x->name, {pointer, x});
    check(x->body);
  }
  void visit(const make_buffer* x) override {
    check_arithmetic(x->base);  // We treat pointers to data as arithmetic
    check_arithmetic(x->elem_size);
    for (const dim_expr& i : x->dims) {
      check_arithmetic(i);
    }
    auto s = set_value_in_scope(ctx, x->name, {pointer, x});
    check(x->body);
  }
  void visit(const crop_buffer* x) override {
    check_pointer(x->name);
    for (const interval_expr& i : x->bounds) {
      check_arithmetic(i, /*required=*/false);
    }
    check(x->body);
  }
  void visit(const crop_dim* x) override {
    check_pointer(x->name);
    check_arithmetic(x->bounds);
    check(x->body);
  }
  void visit(const slice_buffer* x) override {
    check_pointer(x->name);
    for (const expr& i : x->at) {
      check_arithmetic(i, /*required=*/false);
    }
    check(x->body);
  }
  void visit(const slice_dim* x) override {
    check_pointer(x->name);
    check_arithmetic(x->at);
    check(x->body);
  }
  void visit(const truncate_rank* x) override { check_pointer(x->name); }
  void visit(const class check* x) override { check(x->condition); }
};

}  // namespace

bool is_valid(const expr& e, const std::vector<symbol_id>& inputs, const node_context* symbols) {
  validator v(inputs, symbols);
  e.accept(&v);
  return !v.error;
}

bool is_valid(const stmt& s, const std::vector<symbol_id>& inputs, const node_context* symbols) {
  validator v(inputs, symbols);
  s.accept(&v);
  return !v.error;
}

}  // namespace slinky
