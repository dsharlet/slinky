#include "pipeline.h"

#include <cassert>
#include <iostream>

#include "print.h"

namespace slinky {

buffer_expr::buffer_expr(node_context& ctx, const std::string& name, std::size_t rank) {
  base = make_variable(ctx, name + ".base");
  dims.reserve(rank);
  for (std::size_t i = 0; i < rank; ++i) {
    std::string dim_name = name + "." + std::to_string(i);
    expr min = make_variable(ctx, dim_name + ".min");
    expr extent = make_variable(ctx, dim_name + ".extent");
    expr stride = make_variable(ctx, dim_name + ".stride");
    expr fold_factor = make_variable(ctx, dim_name + ".fold_factor");
    dims.emplace_back(min, extent, stride, fold_factor);
  }
}

index_t func::evaluate(eval_context& ctx) {
  return 0;
}

index_t pipeline::run_loop_level(eval_context& ctx, index_t loop_level, std::size_t stage_begin, std::size_t stage_end) {
  std::size_t min = 0;
  std::size_t extent = 1;
  for (index_t i = min; i < extent; ++i) {
    scoped_value<index_t> set_i(ctx, loop_level, i);
    for (std::size_t s = stage_begin; s < stage_end;) {
      if (stages_[s].loop_level == loop_level) {
        // This stage is at this loop level, run it.
        stages_[s].f.evaluate(ctx);
        ++s;
      } else {
        assert(stages_[s].loop_level > loop_level);
        // Find the end of the next loop level.
        // TODO: This search should be pre-computed.
        std::size_t se;
        for (se = s + 1; se < stage_end; ++se) {
          if (stages_[se].loop_level <= loop_level) break;
        }
        index_t result = run_loop_level(ctx, loop_level + 1, s, se);
        if (result != 0) {
          return result;
        }
      }
    }
  }
  return 0;
}

index_t pipeline::evaluate(eval_context& ctx) {
  return run_loop_level(ctx, 0, 0, stages_.size());
}

}  // namespace slinky