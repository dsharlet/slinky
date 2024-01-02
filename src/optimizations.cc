#include "optimizations.h"

#include <cassert>
#include <iostream>

#include "evaluate.h"
#include "node_mutator.h"
#include "pipeline.h"
#include "print.h"
#include "simplify.h"
#include "substitute.h"

namespace slinky {

namespace {

struct copy_info {
  std::vector<expr> src_x;
  std::vector<var> dst_x;
  std::vector<char> padding;
};

// This is very slow, used as a reference implementation.
void copy(
    eval_context& ctx, const raw_buffer& src, const dim* dst_dims, void* dst_base, const copy_info& info, int dim) {
  const class dim& dst_dim = dst_dims[dim];
  index_t dst_stride = dst_dim.stride();
  for (index_t dst_x = dst_dim.begin(); dst_x < dst_dim.end(); ++dst_x) {
    auto s = set_value_in_scope(ctx, info.dst_x[dim].sym(), dst_x);
    if (dim == 0) {
      const void* src_base = src.base;
      for (int d = 0; d < src.rank; ++d) {
        const class dim& src_dim = src.dims[d];

        index_t src_x = evaluate(info.src_x[d], ctx);
        if (src_dim.contains(src_x)) {
          src_base = offset_bytes(src_base, src_dim.flat_offset_bytes(src_x));
        } else {
          src_base = nullptr;
          break;
        }
      }
      if (src_base) {
        memcpy(dst_base, src_base, src.elem_size);
      } else if (!info.padding.empty()) {
        memcpy(dst_base, info.padding.data(), src.elem_size);
      } else {
        // Leave unmodified.
      }
    } else {
      copy(ctx, src, dst_dims, dst_base, info, dim - 1);
    }
    dst_base = offset_bytes(dst_base, dst_stride);
  }
}

void copy(eval_context& ctx, const raw_buffer& src, const raw_buffer& dst, const copy_info& info) {
  assert(info.src_x.size() == src.rank);
  assert(info.dst_x.size() == dst.rank);
  assert(dst.elem_size == src.elem_size);
  assert(info.padding.empty() || dst.elem_size == info.padding.size());
  if (dst.rank == 0) {
    // The buffer is scalar.
    assert(src.rank == 0);
    memcpy(dst.base, src.base, dst.elem_size);
  } else {
    copy(ctx, src, dst.dims, dst.base, info, dst.rank - 1);
  }
}

std::vector<expr> assert_points(std::span<const interval_expr> bounds) {
  std::vector<expr> result;
  result.reserve(bounds.size());
  for (const interval_expr& i : bounds) {
    if (!i.min.same_as(i.max)) {
      std::cerr << "Bounds must be a single point." << std::endl;
      std::abort();
    }
    result.push_back(i.min);
  }
  return result;
}

class copy_implementer : public node_mutator {
  node_context& ctx;

  stmt implement_copy(
      const func* fn, std::vector<expr> src_x, std::vector<var> dst_x, symbol_id src_arg, symbol_id dst_arg) {
    // We're always going to have a call to copy at the innermost loop.
    copy_info info;
    info.src_x = src_x;
    info.dst_x = dst_x;
    info.padding = fn->padding();

    stmt result = call_func::make(
        [info, src_arg, dst_arg](eval_context& ctx) -> index_t {
          const raw_buffer& src = *ctx.lookup_buffer(src_arg);
          const raw_buffer& dst = *ctx.lookup_buffer(dst_arg);
          copy(ctx, src, dst, info);
          return 0;
        },
        fn);

    return result;
  }

public:
  copy_implementer(node_context& ctx) : ctx(ctx) {}

  void visit(const call_func* c) override {
    if (c->target) {
      // This call is not a copy.
      set_result(c);
      return;
    }

    assert(c->fn->outputs().size() == 1);
    const func::output& output = c->fn->outputs().front();

    std::vector<var> dims = output.dims;

    // We're going to implement multiple input copies by simply copying the input to the output each time, assuming
    // the padding is not replaced.
    // TODO: We could be smarter about this, and not have this limitation.
    assert(c->fn->inputs().size() == 1 || c->fn->padding().empty());
    std::vector<stmt> results;
    results.reserve(c->fn->inputs().size());

    assert(c->fn->outputs().size() == 1);
    symbol_id output_arg = c->fn->outputs()[0].sym();

    assert(c->fn);
    for (const func::input& i : c->fn->inputs()) {
      results.push_back(implement_copy(c->fn, assert_points(i.bounds), output.dims, i.sym(), output_arg));
    }
    set_result(block::make(results));
  }
};  // namespace

}  // namespace

stmt implement_copies(const stmt& s, node_context& ctx) { return copy_implementer(ctx).mutate(s); }

}  // namespace slinky
