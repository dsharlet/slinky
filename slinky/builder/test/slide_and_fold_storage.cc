#include "slinky/builder/slide_and_fold_storage.h"

#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "slinky/builder/simplify.h"
#include "slinky/builder/substitute.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

namespace slinky {

namespace {

MATCHER_P(matches, expected, "") { return match(arg, expected); }

stmt dummy_call(std::vector<var> inputs, std::vector<var> outputs) {
  return call_stmt::make(nullptr, std::move(inputs), std::move(outputs), {},
                         {});
}

std::vector<dim_expr> make_foldable_dims(var sym,
                                         std::vector<interval_expr> bounds) {
  std::vector<dim_expr> dims;
  dims.reserve(bounds.size());
  for (int d = 0; d < static_cast<int>(bounds.size()); ++d) {
    dims.push_back({std::move(bounds[d]), expr(), buffer_fold_factor(sym, d)});
  }
  return dims;
}

struct crop_and_loop_collector : public recursive_node_visitor {
  std::vector<const loop*> loops;
  std::vector<const allocate*> allocs;
  std::vector<const crop_dim*> crop_dims;
  std::vector<const crop_buffer*> crop_buffers;

  void visit(const loop* op) override {
    loops.push_back(op);
    recursive_node_visitor::visit(op);
  }
  void visit(const allocate* op) override {
    allocs.push_back(op);
    recursive_node_visitor::visit(op);
  }
  void visit(const crop_dim* op) override {
    crop_dims.push_back(op);
    recursive_node_visitor::visit(op);
  }
  void visit(const crop_buffer* op) override {
    crop_buffers.push_back(op);
    recursive_node_visitor::visit(op);
  }
};

crop_and_loop_collector collect_ir(const stmt& s) {
  crop_and_loop_collector c;
  s.accept(&c);
  return c;
}

}  // namespace

TEST(slide_and_fold_storage, growing_other_dim_blocks_sliding) {
  node_context ctx;
  var buf(ctx, "buf");
  var ly(ctx, "y");

  // Dim 0 (`[0, y]`) overlaps monotonically while dim 1 (`[y, y]`) advances to
  // new rows each iteration. Dim 1 should fold to 1, but dim 0 must not slide.
  stmt s = allocate::make(
      buf, memory_type::heap, 4, make_foldable_dims(buf, {{0, 15}, {0, 15}}),
      loop::make(ly, loop::serial, {0, 15}, 1,
                 crop_buffer::make(buf, buf, {{0, ly}, {ly, ly}},
                                   dummy_call({}, {buf}))));

  stmt res = slide_and_fold_storage(s, ctx);
  auto info = collect_ir(res);
  ASSERT_EQ(info.loops.size(), 1);
  ASSERT_EQ(info.allocs.size(), 1);
  ASSERT_EQ(info.crop_buffers.size(), 1);

  EXPECT_THAT(info.loops[0]->bounds.min, matches(expr(0)));
  EXPECT_FALSE(info.allocs[0]->dims[0].fold_factor.defined());
  EXPECT_THAT(info.allocs[0]->dims[1].fold_factor, matches(expr(1)));
  EXPECT_THAT(info.crop_buffers[0]->bounds[0].min, matches(expr(0)));
  EXPECT_THAT(info.crop_buffers[0]->bounds[0].max, matches(min(ly, 15)));
}

TEST(slide_and_fold_storage,
     warmup_loop_min_shift_invalidates_other_dim_clamp) {
  node_context ctx;
  var buf(ctx, "buf");
  var ly(ctx, "y");

  // Originally `y in [10, 15]`, where `min(10, y)` simplifies to `10`. Sliding
  // dim 1 (`[y - 2, y]`) would shift `loop.bounds.min` back to `8`, where
  // `min(10, y)` shrinks to `8`/`9` on warmup iterations and leaves elements
  // uninitialized for `y == 10`. Sliding must be rejected.
  stmt s = allocate::make(
      buf, memory_type::heap, 4, make_foldable_dims(buf, {{0, 15}, {0, 15}}),
      loop::make(ly, loop::serial, {10, 15}, 1,
                 crop_buffer::make(buf, buf, {{0, min(10, ly)}, {ly - 2, ly}},
                                   dummy_call({}, {buf}))));

  stmt res = slide_and_fold_storage(s, ctx);
  auto info = collect_ir(res);
  ASSERT_EQ(info.loops.size(), 1);
  ASSERT_EQ(info.allocs.size(), 1);
  ASSERT_EQ(info.crop_buffers.size(), 1);

  EXPECT_THAT(info.loops[0]->bounds.min, matches(expr(10)));
  EXPECT_FALSE(info.allocs[0]->dims[1].fold_factor.defined());
  EXPECT_TRUE(
      prove_true(info.crop_buffers[0]->bounds[1].min == max(0, ly - 2)));
}

TEST(slide_and_fold_storage,
     multi_output_inner_allocated_sibling_blocks_sliding) {
  node_context ctx;
  var a(ctx, "a");
  var b(ctx, "b");
  var ly(ctx, "y");

  // Sibling `b` is allocated inside `crop_dim(a)`, so `buffer_bounds[b]` is
  // absent when evaluating `a`. Must fail closed and not slide `a`.
  stmt inner = allocate::make(b, memory_type::heap, 4,
                              make_foldable_dims(b, {{ly, ly}}),
                              dummy_call({}, {a, b}));
  stmt s = allocate::make(
      a, memory_type::heap, 4, make_foldable_dims(a, {{0, 15}}),
      loop::make(ly, loop::serial, {0, 15}, 1,
                 crop_dim::make(a, a, 0, {0, ly}, std::move(inner))));

  stmt res = slide_and_fold_storage(s, ctx);
  auto info = collect_ir(res);
  ASSERT_EQ(info.loops.size(), 1);
  ASSERT_EQ(info.crop_dims.size(), 1);

  EXPECT_THAT(info.loops[0]->bounds.min, matches(expr(0)));
  EXPECT_THAT(info.crop_dims[0]->bounds.min, matches(expr(0)));
  EXPECT_THAT(info.crop_dims[0]->bounds.max, matches(min(ly, 15)));
}

TEST(slide_and_fold_storage,
     multi_output_inner_cropped_sibling_blocks_sliding) {
  node_context ctx;
  var a(ctx, "a");
  var b(ctx, "b");
  var ly(ctx, "y");

  // Sibling `b` is allocated outside the loop but cropped inside `crop_dim(a)`
  // to `[y, y]`. When evaluating `a`, `buffer_bounds[b]` still has uncropped
  // `[0, 15]`; `a` must not slide while `b` can still fold to 1.
  stmt inner = crop_dim::make(b, b, 0, {ly, ly}, dummy_call({}, {a, b}));
  stmt s = allocate::make(
      a, memory_type::heap, 4, make_foldable_dims(a, {{0, 15}}),
      allocate::make(
          b, memory_type::heap, 4, make_foldable_dims(b, {{0, 15}}),
          loop::make(ly, loop::serial, {0, 15}, 1,
                     crop_dim::make(a, a, 0, {0, ly}, std::move(inner)))));

  stmt res = slide_and_fold_storage(s, ctx);
  auto info = collect_ir(res);
  ASSERT_EQ(info.loops.size(), 1);
  ASSERT_EQ(info.allocs.size(), 2);
  ASSERT_EQ(info.crop_dims.size(), 2);

  EXPECT_THAT(info.loops[0]->bounds.min, matches(expr(0)));
  EXPECT_FALSE(info.allocs[0]->dims[0].fold_factor.defined());
  EXPECT_THAT(info.allocs[1]->dims[0].fold_factor, matches(expr(1)));
  EXPECT_THAT(info.crop_dims[0]->bounds.min, matches(expr(0)));
  EXPECT_THAT(info.crop_dims[0]->bounds.max, matches(min(ly, 15)));
}

TEST(slide_and_fold_storage,
     multi_output_symmetric_stencil_keeps_consistent_bounds) {
  node_context ctx;
  var a(ctx, "a");
  var b(ctx, "b");
  var ly(ctx, "y");

  // Both outputs share identical overlapping bounds `[y - 1, y + 1]`. Sliding
  // only the outer sibling `a` while leaving `b` unslid would pass mismatched
  // bounds into the same call_stmt; both must remain consistent.
  stmt inner =
      crop_dim::make(b, b, 0, {ly - 1, ly + 1}, dummy_call({}, {a, b}));
  stmt s = allocate::make(
      a, memory_type::heap, 4, make_foldable_dims(a, {{0, 15}}),
      allocate::make(
          b, memory_type::heap, 4, make_foldable_dims(b, {{0, 15}}),
          loop::make(
              ly, loop::serial, {0, 15}, 1,
              crop_dim::make(a, a, 0, {ly - 1, ly + 1}, std::move(inner)))));

  stmt res = slide_and_fold_storage(s, ctx);
  auto info = collect_ir(res);
  ASSERT_EQ(info.loops.size(), 1);
  ASSERT_EQ(info.crop_dims.size(), 2);

  EXPECT_THAT(info.loops[0]->bounds.min, matches(expr(0)));
  EXPECT_THAT(info.crop_dims[0]->bounds.min,
              matches(info.crop_dims[1]->bounds.min));
  EXPECT_THAT(info.crop_dims[0]->bounds.max,
              matches(info.crop_dims[1]->bounds.max));
}

TEST(slide_and_fold_storage, zero_elem_size_buffer_not_folded_or_slid) {
  node_context ctx;
  var r_overlap(ctx, "r_overlap");
  var r_disjoint(ctx, "r_disjoint");
  var ly(ctx, "y");

  // Zero-elem-size metadata buffers must be marked unfolded and skip both
  // sliding (overlapping) and folding (disjoint).
  stmt body = block::make({
      crop_dim::make(r_overlap, r_overlap, 0, {0, ly},
                     dummy_call({}, {r_overlap})),
      crop_dim::make(r_disjoint, r_disjoint, 0, {ly, ly},
                     dummy_call({}, {r_disjoint})),
  });
  stmt s = allocate::make(
      r_overlap, memory_type::heap, 0, make_foldable_dims(r_overlap, {{0, 15}}),
      allocate::make(r_disjoint, memory_type::heap, 0,
                     make_foldable_dims(r_disjoint, {{0, 15}}),
                     loop::make(ly, loop::serial, {0, 15}, 1,
                                std::move(body))));

  stmt res = slide_and_fold_storage(s, ctx);
  auto info = collect_ir(res);
  ASSERT_EQ(info.loops.size(), 1);
  ASSERT_EQ(info.allocs.size(), 2);
  ASSERT_EQ(info.crop_dims.size(), 2);

  EXPECT_THAT(info.loops[0]->bounds.min, matches(expr(0)));
  EXPECT_THAT(info.allocs[0]->dims[0].fold_factor,
              matches(expr(dim::unfolded)));
  EXPECT_THAT(info.allocs[1]->dims[0].fold_factor,
              matches(expr(dim::unfolded)));
  EXPECT_THAT(info.crop_dims[0]->bounds.min, matches(expr(0)));
  EXPECT_THAT(info.crop_dims[0]->bounds.max, matches(min(ly, 15)));
}

TEST(slide_and_fold_storage, single_output_2d_stencil_still_slides_and_folds) {
  node_context ctx;
  var buf(ctx, "buf");
  var ly(ctx, "y");

  // Positive control: dim 0 is loop-invariant `[0, 15]`, so dim 1
  // (`[y - 1, y + 1]`) still slides and folds to 3.
  stmt s = allocate::make(
      buf, memory_type::heap, 4, make_foldable_dims(buf, {{0, 15}, {0, 15}}),
      loop::make(ly, loop::serial, {0, 15}, 1,
                 crop_buffer::make(buf, buf, {{0, 15}, {ly - 1, ly + 1}},
                                   dummy_call({}, {buf}))));

  stmt res = slide_and_fold_storage(s, ctx);
  auto info = collect_ir(res);
  ASSERT_EQ(info.loops.size(), 1);
  ASSERT_EQ(info.allocs.size(), 1);
  ASSERT_EQ(info.crop_buffers.size(), 1);

  EXPECT_THAT(info.loops[0]->bounds.min, matches(expr(-1)));
  EXPECT_THAT(info.allocs[0]->dims[1].fold_factor, matches(expr(3)));
  EXPECT_THAT(info.crop_buffers[0]->bounds[1].min, matches(ly + 1));
}

TEST(slide_and_fold_storage,
     uncropped_zero_elem_sibling_still_allows_sliding) {
  node_context ctx;
  var a(ctx, "a");
  var r(ctx, "r");
  var ly(ctx, "y");

  // Positive control: sibling `r` is loop-invariant and not modified inside
  // `crop_dim(a)`, so `a` still slides and folds while `r` stays unfolded.
  stmt crop_a =
      crop_dim::make(a, a, 0, {ly - 1, ly + 1}, dummy_call({}, {a, r}));
  stmt s = allocate::make(
      r, memory_type::heap, 0, make_foldable_dims(r, {{0, 15}}),
      allocate::make(a, memory_type::heap, 4, make_foldable_dims(a, {{0, 15}}),
                     loop::make(ly, loop::serial, {0, 15}, 1,
                                std::move(crop_a))));

  stmt res = slide_and_fold_storage(s, ctx);
  auto info = collect_ir(res);
  ASSERT_EQ(info.loops.size(), 1);
  ASSERT_EQ(info.allocs.size(), 2);
  ASSERT_EQ(info.crop_dims.size(), 1);

  EXPECT_THAT(info.loops[0]->bounds.min, matches(expr(-1)));
  EXPECT_THAT(info.allocs[0]->dims[0].fold_factor,
              matches(expr(dim::unfolded)));
  EXPECT_THAT(info.allocs[1]->dims[0].fold_factor, matches(expr(3)));
  EXPECT_THAT(info.crop_dims[0]->bounds.min, matches(ly + 1));
}

}  // namespace slinky