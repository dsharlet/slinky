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
    std::string sym = symbols ? symbols->name(i) : "<" + std::to_string(i) + ">";
    if (!deps_of.defined() || depends_on_variable(deps_of, i)) {
      if (ctx.contains(i)) {
        s << "  " << sym << " = " << *ctx.lookup(i) << std::endl;
      } else {
        s << "  " << sym << " = <>" << std::endl;
      }
    } else if (!deps_of.defined() || depends_on_buffer(deps_of, i)) {
      if (ctx.contains(i)) {
        const raw_buffer* buf = reinterpret_cast<const raw_buffer*>(*ctx.lookup(i));
        s << "  " << sym << " = {base=" << buf->base << ", elem_size=" << buf->elem_size << ", dims={";
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

namespace {

// This is a very slow implementation of copy_stmt. The expectation is that copies will have been lowered to aliases or
// calls to `copy` in buffer.h/cc instead of relying on this implementation.
void copy_stmt_impl(
    eval_context& ctx, const raw_buffer& src, const dim* dst_dims, void* dst_base, const copy_stmt& c, int dim) {
  const class dim& dst_dim = dst_dims[dim];
  index_t dst_stride = dst_dim.stride();
  for (index_t dst_x = dst_dim.begin(); dst_x < dst_dim.end(); ++dst_x) {
    auto s = set_value_in_scope(ctx, c.dst_x[dim], dst_x);
    if (dim == 0) {
      const void* src_base = src.base;
      for (std::size_t d = 0; d < src.rank; ++d) {
        const class dim& src_dim = src.dims[d];

        index_t src_x = evaluate(c.src_x[d], ctx);
        if (src_dim.contains(src_x)) {
          src_base = offset_bytes(src_base, src_dim.flat_offset_bytes(src_x));
        } else {
          src_base = nullptr;
          break;
        }
      }
      if (src_base) {
        memcpy(dst_base, src_base, src.elem_size);
      } else if (!c.padding.empty()) {
        memcpy(dst_base, c.padding.data(), src.elem_size);
      } else {
        // Leave unmodified.
      }
    } else {
      copy_stmt_impl(ctx, src, dst_dims, dst_base, c, dim - 1);
    }
    dst_base = offset_bytes(dst_base, dst_stride);
  }
}

void copy_stmt_impl(eval_context& ctx, const raw_buffer& src, const raw_buffer& dst, const copy_stmt& c) {
  assert(c.src_x.size() == src.rank);
  assert(c.dst_x.size() == dst.rank);
  assert(dst.elem_size == src.elem_size);
  assert(c.padding.empty() || dst.elem_size == c.padding.size());
  if (dst.rank == 0) {
    // The buffer is scalar.
    assert(src.rank == 0);
    memcpy(dst.base, src.base, dst.elem_size);
  } else {
    copy_stmt_impl(ctx, src, dst.dims, dst.base, c, dst.rank - 1);
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
  void visit(const expr& op) {
    switch (op.type()) {
    case node_type::variable: visit(reinterpret_cast<const variable*>(op.get())); return;
    case node_type::constant: visit(reinterpret_cast<const constant*>(op.get())); return;
    default: op.accept(this);
    }
  }

  void visit(const stmt& op) {
    switch (op.type()) {
    // case node_type::call_stmt: visit(reinterpret_cast<const call_stmt*>(op.get())); return;
    // case node_type::crop_dim: visit(reinterpret_cast<const crop_dim*>(op.get())); return;
    // case node_type::slice_dim: visit(reinterpret_cast<const slice_dim*>(op.get())); return;
    // case node_type::block: visit(reinterpret_cast<const block*>(op.get())); return;
    default: op.accept(this);
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
    auto value = context.lookup(v->sym);
    assert(value);
    result = *value;
  }

  void visit(const wildcard* w) override {
    // Maybe evaluating this should just be an error.
    auto value = context.lookup(w->sym);
    assert(value);
    result = *value;
  }

  void visit(const constant* c) override { result = c->value; }

  template <typename T>
  void visit_let(const T* l) {
    auto set_value = set_value_in_scope(context, l->sym, eval_expr(l->value));
    visit(l->body);
  }

  void visit(const let* l) override { visit_let(l); }
  void visit(const let_stmt* l) override { visit_let(l); }

  void visit(const add* op) override { result = eval_expr(op->a) + eval_expr(op->b); }
  void visit(const sub* op) override { result = eval_expr(op->a) - eval_expr(op->b); }
  void visit(const mul* op) override { result = eval_expr(op->a) * eval_expr(op->b); }
  void visit(const div* op) override { result = euclidean_div(eval_expr(op->a), eval_expr(op->b)); }
  void visit(const mod* op) override { result = euclidean_mod(eval_expr(op->a), eval_expr(op->b)); }
  void visit(const class min* op) override { result = std::min(eval_expr(op->a), eval_expr(op->b)); }
  void visit(const class max* op) override { result = std::max(eval_expr(op->a), eval_expr(op->b)); }
  void visit(const equal* op) override { result = eval_expr(op->a) == eval_expr(op->b); }
  void visit(const not_equal* op) override { result = eval_expr(op->a) != eval_expr(op->b); }
  void visit(const less* op) override { result = eval_expr(op->a) < eval_expr(op->b); }
  void visit(const less_equal* op) override { result = eval_expr(op->a) <= eval_expr(op->b); }
  void visit(const logical_and* op) override { result = eval_expr(op->a) != 0 && eval_expr(op->b) != 0; }
  void visit(const logical_or* op) override { result = eval_expr(op->a) != 0 || eval_expr(op->b) != 0; }
  void visit(const logical_not* op) override { result = eval_expr(op->a) == 0; }

  void visit(const class select* op) override {
    if (eval_expr(op->condition)) {
      result = eval_expr(op->true_value);
    } else {
      result = eval_expr(op->false_value);
    }
  }

  index_t eval_buffer_metadata(const call* op) {
    assert(op->args.size() == 1);
    raw_buffer* buf = reinterpret_cast<raw_buffer*>(eval_expr(op->args[0]));
    assert(buf);
    switch (op->intrinsic) {
    case intrinsic::buffer_rank: return buf->rank;
    case intrinsic::buffer_elem_size: return buf->elem_size;
    case intrinsic::buffer_base: return reinterpret_cast<index_t>(buf->base);
    case intrinsic::buffer_size_bytes: return buf->size_bytes();
    default: std::abort();
    }
  }

  index_t eval_dim_metadata(const call* op) {
    assert(op->args.size() == 2);
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(eval_expr(op->args[0]));
    assert(buffer);
    index_t d = eval_expr(op->args[1]);
    assert(d < static_cast<index_t>(buffer->rank));
    const slinky::dim& dim = buffer->dim(d);
    switch (op->intrinsic) {
    case intrinsic::buffer_min: return dim.min();
    case intrinsic::buffer_max: return dim.max();
    case intrinsic::buffer_extent: return dim.extent();
    case intrinsic::buffer_stride: return dim.stride();
    case intrinsic::buffer_fold_factor: return dim.fold_factor();
    default: std::abort();
    }
  }

  void* eval_buffer_at(const call* op) {
    assert(op->args.size() >= 1);
    raw_buffer* buf = reinterpret_cast<raw_buffer*>(eval_expr(op->args[0]));
    void* result = buf->base;
    assert(op->args.size() <= buf->rank + 1);
    for (std::size_t d = 0; d < op->args.size() - 1; ++d) {
      if (op->args[d + 1].defined()) {
        result = offset_bytes(result, buf->dims[d].flat_offset_bytes(eval_expr(op->args[d + 1])));
      }
    }
    return result;
  }

  void visit(const call* op) override {
    switch (op->intrinsic) {
    case intrinsic::positive_infinity: std::cerr << "Cannot evaluate positive_infinity" << std::endl; std::abort();
    case intrinsic::negative_infinity: std::cerr << "Cannot evaluate negative_infinity" << std::endl; std::abort();
    case intrinsic::indeterminate: std::cerr << "Cannot evaluate indeterminate" << std::endl; std::abort();

    case intrinsic::abs:
      assert(op->args.size() == 1);
      result = std::abs(eval_expr(op->args[0]));
      return;

    case intrinsic::buffer_rank:
    case intrinsic::buffer_elem_size:
    case intrinsic::buffer_base:
    case intrinsic::buffer_size_bytes: result = eval_buffer_metadata(op); return;

    case intrinsic::buffer_min:
    case intrinsic::buffer_max:
    case intrinsic::buffer_extent:
    case intrinsic::buffer_stride:
    case intrinsic::buffer_fold_factor: result = eval_dim_metadata(op); return;

    case intrinsic::buffer_at: result = reinterpret_cast<index_t>(eval_buffer_at(op)); return;
    default: std::cerr << "Unknown intrinsic: " << op->intrinsic << std::endl; std::abort();
    }
  }

  void visit(const block* b) override {
    if (result == 0) visit(b->a);
    if (result == 0) visit(b->b);
  }

  void visit(const loop* l) override {
    index_t min = eval_expr(l->bounds.min);
    index_t max = eval_expr(l->bounds.max);
    index_t step = eval_expr(l->step, 1);
    // TODO(https://github.com/dsharlet/slinky/issues/3): We don't get a reference to context[l->sym] here
    // because the context could grow and invalidate the reference. This could be fixed by having evaluate
    // fully traverse the expression to find the max symbol_id, and pre-allocate the context up front. It's
    // not clear this optimization is necessary yet.
    std::optional<index_t> old_value = context[l->sym];
    for (index_t i = min; result == 0 && min <= i && i <= max; i += step) {
      context[l->sym] = i;
      visit(l->body);
    }
    context[l->sym] = old_value;
  }

  void visit(const if_then_else* n) override {
    if (eval_expr(n->condition)) {
      if (n->true_body.defined()) {
        visit(n->true_body);
      }
    } else {
      if (n->false_body.defined()) {
        visit(n->false_body);
      }
    }
  }

  void visit(const call_stmt* n) override {
    result = n->target(context);
    if (result) {
      if (context.call_failed) {
        context.call_failed(n);
      } else {
        std::cerr << "call_stmt failed: " << stmt(n) << "->" << result << std::endl;
        std::abort();
      }
    }
  }

  void visit(const copy_stmt* n) override {
    const raw_buffer* src = reinterpret_cast<raw_buffer*>(context.lookup(n->src, 0));
    const raw_buffer* dst = reinterpret_cast<raw_buffer*>(context.lookup(n->dst, 0));

    copy_stmt_impl(context, *src, *dst, *n);
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
        context.allocate(n->sym, buffer);
      } else {
        buffer->allocate();
      }
    }

    auto set_buffer = set_value_in_scope(context, n->sym, reinterpret_cast<index_t>(buffer));
    visit(n->body);

    if (n->storage == memory_type::heap) {
      if (context.free) {
        assert(context.allocate);
        context.free(n->sym, buffer);
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

    auto set_buffer = set_value_in_scope(context, n->sym, reinterpret_cast<index_t>(buffer));
    visit(n->body);
  }

  void visit(const crop_buffer* n) override {
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(*context.lookup(n->sym));
    assert(buffer);

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
      index_t min = std::max(dim.min(), eval_expr(n->bounds[d].min, dim.min()));
      index_t max = std::min(dim.max(), eval_expr(n->bounds[d].max, dim.max()));
      offset += dim.flat_offset_bytes(min);

      dim.set_bounds(min, max);
    }

    void* old_base = buffer->base;
    buffer->base = offset_bytes(buffer->base, offset);

    visit(n->body);

    buffer->base = old_base;
    for (std::size_t d = 0; d < crop_rank; ++d) {
      slinky::dim& dim = buffer->dims[d];
      dim.set_min_extent(old_bounds[d].min, old_bounds[d].extent);
    }
  }

  void visit(const crop_dim* n) override {
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(*context.lookup(n->sym));
    assert(buffer);
    slinky::dim& dim = buffer->dims[n->dim];

    void* old_base = buffer->base;
    index_t old_min = dim.min();
    index_t old_extent = dim.extent();

    index_t min = std::max(dim.min(), eval_expr(n->bounds.min));
    buffer->base = offset_bytes(buffer->base, dim.flat_offset_bytes(min));
    if (n->bounds.is_point()) {
      // Crops to a single element are common, we can optimize them a little bit by re-using the min
      dim.set_point(min);
    } else {
      dim.set_bounds(min, std::min(dim.max(), eval_expr(n->bounds.max)));
    }

    visit(n->body);

    buffer->base = old_base;
    dim.set_min_extent(old_min, old_extent);
  }

  void visit(const slice_buffer* n) override {
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(*context.lookup(n->sym));
    assert(buffer);

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

    visit(n->body);

    buffer->base = old_base;
    buffer->rank = old_rank;
    buffer->dims = old_dims;
  }

  void visit(const slice_dim* n) override {
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(*context.lookup(n->sym));
    assert(buffer);

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

    visit(n->body);

    buffer->base = old_base;
    buffer->rank += 1;
    buffer->dims = old_dims;
  }

  void visit(const truncate_rank* n) override {
    raw_buffer* buffer = reinterpret_cast<raw_buffer*>(*context.lookup(n->sym));
    assert(buffer);

    std::size_t old_rank = buffer->rank;
    buffer->rank = n->rank;

    visit(n->body);

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

}  // namespace

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
