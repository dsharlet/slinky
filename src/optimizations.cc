#include "optimizations.h"

#include <cassert>
#include <iostream>

#include "evaluate.h"
#include "node_mutator.h"
#include "print.h"
#include "simplify.h"
#include "substitute.h"

namespace slinky {

namespace {

struct copy_info {
  std::vector<expr> src_x;
  std::vector<symbol_id> dst_x;
  std::vector<char> padding;
};

// This is very slow, used as a reference implementation.
void copy(
    eval_context& ctx, const raw_buffer& src, const dim* dst_dims, void* dst_base, const copy_info& info, int dim) {
  const class dim& dst_dim = dst_dims[dim];
  index_t dst_stride = dst_dim.stride();
  for (index_t dst_x = dst_dim.begin(); dst_x < dst_dim.end(); ++dst_x) {
    auto s = set_value_in_scope(ctx, info.dst_x[dim], dst_x);
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

class copy_implementer : public node_mutator {
  node_context& ctx;

  stmt implement_copy(const copy_stmt* c) {
    // We're always going to have a call to copy at the innermost loop.
    copy_info info;
    info.src_x = c->src_x;
    info.dst_x = c->dst_x;
    info.padding = c->padding;

    stmt result = call_stmt::make(
        [info, src = c->src, dst = c->dst](eval_context& ctx) -> index_t {
          const raw_buffer& src_buf = *ctx.lookup_buffer(src);
          const raw_buffer& dst_buf = *ctx.lookup_buffer(dst);
          copy(ctx, src_buf, dst_buf, info);
          return 0;
        }, {c->src}, {c->dst});

    return result;
  }

public:
  copy_implementer(node_context& ctx) : ctx(ctx) {}

  void visit(const copy_stmt* c) override {
    set_result(implement_copy(c));
  }
};  // namespace

}  // namespace

stmt implement_copies(const stmt& s, node_context& ctx) { return copy_implementer(ctx).mutate(s); }

}  // namespace slinky
