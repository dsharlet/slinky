#include "evaluate.h"

#include <cassert>
#include <iostream>

#include "print.h"

namespace slinky {

// TODO(https://github.com/dsharlet/slinky/issues/2): I think the T::accept/node_visitor::visit
// overhead (two virtual function calls per node) might be significant. This could be implemented
// as a switch statement instead.
class evaluator : public node_visitor {
public:
  index_t result = 0;
  eval_context& context;

  evaluator(eval_context& context) : context(context) {}

  // Assume `e` is defined, evaluate it and return the result.
  index_t eval_expr(const expr& e) {
    e.accept(this);
    return result;
  }

  // If `e` is defined, evaluate it and return the result. Otherwise, return default `def`.
  index_t eval_expr(const expr& e, index_t def) {
    if (e.defined()) {
      e.accept(this);
      return result;
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
  void visit(const bitwise_and* x) override { result = eval_expr(x->a) & eval_expr(x->b); }
  void visit(const bitwise_or* x) override { result = eval_expr(x->a) | eval_expr(x->b); }
  void visit(const bitwise_xor* x) override { result = eval_expr(x->a) ^ eval_expr(x->b); }
  void visit(const logical_and* x) override { result = eval_expr(x->a) != 0 && eval_expr(x->b) != 0; }
  void visit(const logical_or* x) override { result = eval_expr(x->a) != 0 || eval_expr(x->b) != 0; }
  void visit(const shift_left* x) override { result = eval_expr(x->a) << eval_expr(x->b); }
  void visit(const shift_right* x) override { result = eval_expr(x->a) >> eval_expr(x->b); }

  void visit(const class select* x) override {
    if (eval_expr(x->condition)) {
      result = eval_expr(x->true_value);
    } else {
      result = eval_expr(x->false_value);
    }
  }

  void visit(const load_buffer_meta* x) override {
    buffer_base* buffer = reinterpret_cast<buffer_base*>(eval_expr(x->buffer));
    assert(buffer);
    if (x->meta == buffer_meta::rank) {
      result = buffer->rank;
    } else if (x->meta == buffer_meta::base) {
      result = reinterpret_cast<index_t>(buffer->base);
    } else if (x->meta == buffer_meta::elem_size) {
      result = buffer->elem_size;
    } else {
      index_t d = eval_expr(x->dim);
      assert(d < buffer->rank);
      const slinky::dim& dim = buffer->dim(d);
      switch (x->meta) {
      case buffer_meta::min: result = dim.min(); return;
      case buffer_meta::max: result = dim.max(); return;
      case buffer_meta::extent: result = dim.extent(); return;
      case buffer_meta::stride_bytes: result = dim.stride_bytes(); return;
      case buffer_meta::fold_factor: result = dim.fold_factor(); return;
      default: std::abort();  // Should be handled above.
      }
    }
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
    }
  }

  void visit(const block* b) override {
    b->a.accept(this);
    b->b.accept(this);
  }

  void visit(const loop* l) override {
    index_t begin = eval_expr(l->begin, 0);
    index_t end = eval_expr(l->end);
    // TODO(https://github.com/dsharlet/slinky/issues/3): We don't get a reference to context[l->name] here
    // because the context could grow and invalidate the reference. This could be fixed by having evaluate
    // fully traverse the expression to find the max symbol_id, and pre-allocate the context up front. It's
    // not clear this optimization is necessary yet.
    std::optional<index_t> old_value = context[l->name];
    for (index_t i = begin; i < end; ++i) {
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

    buffer_base** buffers = reinterpret_cast<buffer_base**>(alloca(n->buffer_args.size() * sizeof(buffer_base*)));
    for (std::size_t i = 0; i < n->buffer_args.size(); ++i) {
      buffers[i] = reinterpret_cast<buffer_base*>(*context.lookup(n->buffer_args[i]));
    }

    std::span<const index_t> scalars_span(scalars, n->scalar_args.size());
    std::span<buffer_base*> buffers_span(buffers, n->buffer_args.size());
    result = n->target(scalars_span, buffers_span);
    if (result) {
      std::cerr << "call_func failed: " << stmt(n) << "->" << result << std::endl;
      std::abort();
    }
  }

  void visit(const allocate* n) override {
    std::size_t rank = n->dims.size();
    // Allocate a buffer with space for its dims on the stack.
    char* storage = reinterpret_cast<char*>(alloca(sizeof(buffer_base) + sizeof(dim) * rank));
    buffer_base* buffer = reinterpret_cast<buffer_base*>(&storage[0]);
    buffer->elem_size = n->elem_size;
    buffer->rank = rank;
    buffer->dims = reinterpret_cast<dim*>(&storage[sizeof(buffer_base)]);

    for (std::size_t i = 0; i < rank; ++i) {
      slinky::dim& dim = buffer->dim(i);
      dim.set_bounds(eval_expr(n->dims[i].min()), eval_expr(n->dims[i].max()));
      dim.set_stride_bytes(eval_expr(n->dims[i].stride_bytes));
      dim.set_fold_factor(eval_expr(n->dims[i].fold_factor));
    }

    std::size_t size = buffer->size_bytes();

    eval_context::memory_info::alloc_tracker tracker;
    if (n->type == memory_type::stack) {
      buffer->base = alloca(size);
      tracker = context.stack.track_allocate(size);
    } else {
      assert(n->type == memory_type::heap);
      buffer->base = malloc(size);
      tracker = context.heap.track_allocate(size);
    }

    auto set_buffer = set_value_in_scope(context, n->name, reinterpret_cast<index_t>(buffer));
    n->body.accept(this);

    if (n->type == memory_type::heap) {
      free(buffer->base);
    }
  }

  void visit(const make_buffer* n) override {
    std::size_t rank = n->dims.size();
    // Allocate a buffer with space for its dims on the stack.
    char* storage = reinterpret_cast<char*>(alloca(sizeof(buffer_base) + sizeof(dim) * rank));
    buffer_base* buffer = reinterpret_cast<buffer_base*>(&storage[0]);
    buffer->elem_size = n->elem_size;
    buffer->base = reinterpret_cast<void*>(eval_expr(n->base));
    buffer->rank = rank;
    buffer->dims = reinterpret_cast<dim*>(&storage[sizeof(buffer_base)]);

    for (std::size_t i = 0; i < rank; ++i) {
      slinky::dim& dim = buffer->dim(i);
      dim.set_bounds(eval_expr(n->dims[i].min()), eval_expr(n->dims[i].max()));
      dim.set_stride_bytes(eval_expr(n->dims[i].stride_bytes));
      dim.set_fold_factor(eval_expr(n->dims[i].fold_factor));
    }

    auto set_buffer = set_value_in_scope(context, n->name, reinterpret_cast<index_t>(buffer));
    n->body.accept(this);
  }

  void visit(const crop_dim* n) override {
    buffer_base* buffer = reinterpret_cast<buffer_base*>(*context.lookup(n->name));
    slinky::dim& dim = buffer->dims[n->dim];

    void* old_base = buffer->base;
    index_t old_min = dim.min();
    index_t old_extent = dim.extent();

    index_t min = eval_expr(n->min);
    index_t extent = eval_expr(n->extent);

    buffer->base = offset_bytes(buffer->base, dim.flat_offset_bytes(min));
    dim.set_min_extent(min, extent);

    n->body.accept(this);

    buffer->base = old_base;
    dim.set_min_extent(old_min, old_extent);
  }

  void visit(const crop_buffer* n) override {
    buffer_base* buffer = reinterpret_cast<buffer_base*>(*context.lookup(n->name));

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

  void visit(const check* n) override {
    result = eval_expr(n->condition);
    if (!result) {
      std::cerr << "check failed: " << n->condition << std::endl;
      std::abort();
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

}  // namespace slinky
