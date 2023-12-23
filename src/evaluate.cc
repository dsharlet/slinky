#include "evaluate.h"

#include <cassert>
#include <iostream>

#include "print.h"

namespace slinky {

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
    scoped_value<index_t> set_value(context, l->name, eval_expr(l->value));
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
  void visit(const logical_and* x) override { result = eval_expr(x->a) && eval_expr(x->b); }
  void visit(const logical_or* x) override { result = eval_expr(x->a) || eval_expr(x->b); }
  void visit(const shift_left* x) override { result = eval_expr(x->a) << eval_expr(x->b); }
  void visit(const shift_right* x) override { result = eval_expr(x->a) >> eval_expr(x->b); }

  void visit(const load_buffer_meta* x) override {
    buffer_base* buffer = reinterpret_cast<buffer_base*>(eval_expr(x->buffer));
    if (x->meta == buffer_meta::base) {
      result = reinterpret_cast<index_t>(buffer->base);
    } else {
      index_t dim = eval_expr(x->dim);
      switch (x->meta) {
      case buffer_meta::min: result = buffer->dims[dim].min; return;
      case buffer_meta::max: result = buffer->dims[dim].max(); return;
      case buffer_meta::extent: result = buffer->dims[dim].extent; return;
      case buffer_meta::stride_bytes: result = buffer->dims[dim].stride_bytes; return;
      case buffer_meta::fold_factor: result = buffer->dims[dim].fold_factor; return;
      case buffer_meta::base: std::abort();  // Handled above.
      }
    }
  }

  void visit(const block* b) override {
    b->a.accept(this);
    b->b.accept(this);
  }

  void visit(const loop* l) override {
    index_t begin = eval_expr(l->begin, 0);
    index_t end = eval_expr(l->end);
    std::optional<index_t>& value = context[l->name];
    std::optional<index_t> old_value;
    old_value = value;
    for (index_t i = begin; i < end; ++i) {
      value = i;
      l->body.accept(this);
    }
    value = old_value;
  }

  void visit(const if_then_else* n) override {
    if (eval_expr(n->condition)) {
      n->true_body.accept(this);
    } else if (n->false_body.defined()) {
      n->false_body.accept(this);
    }
  }

  void visit(const call* n) override {
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
      std::cerr << "call failed: " << stmt(n) << "->" << result << std::endl;
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
      buffer->dims[i].min = eval_expr(n->dims[i].min);
      buffer->dims[i].extent = eval_expr(n->dims[i].extent);
      buffer->dims[i].stride_bytes = eval_expr(n->dims[i].stride_bytes);
      buffer->dims[i].fold_factor = eval_expr(n->dims[i].fold_factor);
    }

    std::size_t size = buffer->size_bytes();

    if (n->type == memory_type::stack) {
      buffer->base = alloca(size);
    } else {
      assert(n->type == memory_type::heap);
      buffer->base = malloc(size);
    }

    scoped_value<index_t> set_buffer(context, n->name, reinterpret_cast<index_t>(buffer));
    n->body.accept(this);

    if (n->type == memory_type::heap) { free(buffer->base); }
  }

  void visit(const crop* n) override {
    buffer_base* buffer = reinterpret_cast<buffer_base*>(*context.lookup(n->name));

    void* old_base = buffer->base;
    index_t old_min = buffer->dims[n->dim].min;
    index_t old_extent = buffer->dims[n->dim].extent;

    index_t min = eval_expr(n->min);
    index_t extent = eval_expr(n->extent);

    buffer->base = offset_bytes(buffer->base, buffer->dims[n->dim].flat_offset_bytes(min));
    buffer->dims[n->dim].min = min;
    buffer->dims[n->dim].extent = extent;

    n->body.accept(this);

    buffer->base = old_base;
    buffer->dims[n->dim].min = old_min;
    buffer->dims[n->dim].extent = old_extent;
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
