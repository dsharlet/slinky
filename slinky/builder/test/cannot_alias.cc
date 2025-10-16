#include <gtest/gtest.h>

#include "slinky/builder/pipeline.h"
#include "slinky/builder/test/context.h"
#include "slinky/builder/test/funcs.h"
#include "slinky/builder/test/util.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/pipeline.h"

namespace slinky {

// This set of tests ensures that we don't alias buffers when doing so would violate assumptions that the client code
// asked slinky to maintain.

class may_alias : public testing::TestWithParam<bool> {};

INSTANTIATE_TEST_SUITE_P(constrain, may_alias, testing::Bool());

TEST_P(may_alias, transpose_input) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  auto in_t = buffer_expr::make(ctx, "in_t", 2, sizeof(int));

  const bool may_alias = GetParam();

  if (!may_alias) {
    // Our callback requires the stride to be 1 element.
    in_t->dim(0).stride = sizeof(int);
  }

  // If we want to alias intermediate buffer to the output buffer,
  // we need to tell aliaser that output is unfolded and it's safe to alias.
  in->dim(0).fold_factor = dim::unfolded;
  in->dim(1).fold_factor = dim::unfolded;

  var x(ctx, "x");
  var y(ctx, "y");
  test_context eval_ctx;

  func transposed = func::make_copy({in, {point(y), point(x)}}, {in_t, {x, y}}, eval_ctx.copy);
  func add1 = func::make(
      [=](const buffer<const int>& a, const buffer<int>& b) -> index_t {
        if (!may_alias && a.dim(0).stride() != 4) return 1;
        return add_1<int>(a, b);
      },
      {{{in_t, {point(x), point(y)}}}}, {{{out, {x, y}}}}, call_stmt::attributes{.name = "add1"});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 20;
  const int H = 10;
  buffer<int, 2> in_buf({H, W});
  init_random(in_buf);

  buffer<int, 2> out_buf({W, H});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  ASSERT_EQ(0, p.evaluate(inputs, outputs, eval_ctx));

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y), in_buf(y, x) + 1);
    }
  }

  ASSERT_EQ(eval_ctx.heap.allocs.size(), may_alias ? 0 : 1);
}

TEST_P(may_alias, transpose_output) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  auto out_t = buffer_expr::make(ctx, "out_t", 2, sizeof(int));

  const bool may_alias = GetParam();

  if (!may_alias) {
    // Our callback requires the stride to be 1 element.
    out_t->dim(0).stride = sizeof(int);
  }

  // If we want to alias intermediate buffer to the output buffer,
  // we need to tell aliaser that output is unfolded and it's safe to alias.
  out->dim(0).fold_factor = dim::unfolded;
  out->dim(1).fold_factor = dim::unfolded;

  var x(ctx, "x");
  var y(ctx, "y");
  test_context eval_ctx;

  func add1 = func::make(
      [=](const buffer<const int>& a, const buffer<int>& b) -> index_t {
        if (!may_alias && b.dim(0).stride() != 4) return 1;
        return add_1<int>(a, b);
      },
      {{{in, {point(x), point(y)}}}}, {{{out_t, {x, y}}}}, call_stmt::attributes{.name = "add1"});
  func transposed = func::make_copy({out_t, {point(y), point(x)}}, {out, {x, y}}, eval_ctx.copy);

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 20;
  const int H = 10;
  buffer<int, 2> in_buf({H, W});
  init_random(in_buf);

  buffer<int, 2> out_buf({W, H});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  ASSERT_EQ(0, p.evaluate(inputs, outputs, eval_ctx));

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y), in_buf(y, x) + 1);
    }
  }

  ASSERT_EQ(eval_ctx.heap.allocs.size(), may_alias ? 0 : 1);
}

TEST_P(may_alias, aligned) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));

  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(short));

  var x(ctx, "x");
  var y(ctx, "y");
  test_context eval_ctx;

  intm->dim(0).bounds = align(intm->dim(0).bounds, 2);

  // In this pipeline, the result is copied to two outputs. We can only alias in this case if we know the two outputs
  // have the same bounds.
  const bool may_alias = GetParam();
  if (may_alias) {
    out->dim(0).bounds = align(out->dim(0).bounds, 2);
  }

  // If we want to alias intermediate buffer to the output buffer,
  // we need to tell aliaser that output is unfolded and it's safe to alias.
  out->dim(0).fold_factor = dim::unfolded;
  out->dim(1).fold_factor = dim::unfolded;

  func add = func::make(add_1<short>, {{in, {point(x), point(y)}}}, {{intm, {x, y}}});
  func copied = func::make_copy({intm, {point(x), point(y)}}, {out, {x, y}}, eval_ctx.copy);

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 20;
  const int H = 10;
  buffer<short, 2> in_buf({W, H});
  init_random(in_buf);

  buffer<short, 2> out_buf({W, H});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y), in_buf(x, y) + 1);
    }
  }

  ASSERT_EQ(eval_ctx.heap.allocs.size(), may_alias ? 0 : 1);
  ASSERT_EQ(eval_ctx.copy_calls, may_alias ? 0 : 1);
}

TEST_P(may_alias, same_bounds) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(short));
  auto out1 = buffer_expr::make(ctx, "out1", 2, sizeof(short));
  auto out2 = buffer_expr::make(ctx, "out2", 2, sizeof(short));

  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(short));

  var x(ctx, "x");
  var y(ctx, "y");
  test_context eval_ctx;

  // In this pipeline, the result is copied to two outputs. We can only alias in this case if we know the two outputs
  // have the same bounds.
  const bool may_alias = GetParam();
  if (may_alias) {
    out2->dim(0).bounds = out1->dim(0).bounds;
    out2->dim(1).bounds = out1->dim(1).bounds;
  }

  // If we want to alias intermediate buffer to the output buffer,
  // we need to tell aliaser that output is unfolded and it's safe to alias.
  out1->dim(0).fold_factor = dim::unfolded;
  out1->dim(1).fold_factor = dim::unfolded;
  out2->dim(0).fold_factor = dim::unfolded;
  out2->dim(1).fold_factor = dim::unfolded;

  func add = func::make(add_1<short>, {{in, {point(x), point(y)}}}, {{intm, {x, y}}});
  func copied1 = func::make_copy({intm, {point(x), point(y)}}, {out1, {x, y}}, eval_ctx.copy);
  func copied2 = func::make_copy({intm, {point(x), point(y)}}, {out2, {x, y}}, eval_ctx.copy);

  pipeline p = build_pipeline(ctx, {in}, {out1, out2});

  // Run the pipeline.
  const int W = 20;
  const int H = 10;
  buffer<short, 2> in_buf({W, H});
  init_random(in_buf);

  buffer<short, 2> out1_buf({W, H});
  buffer<short, 2> out2_buf({W, H});
  out1_buf.allocate();
  out2_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out1_buf, &out2_buf};
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out1_buf(x, y), in_buf(x, y) + 1);
      ASSERT_EQ(out2_buf(x, y), in_buf(x, y) + 1);
    }
  }

  // TODO: This requires the buffer_aliaser mutator to learn from checks to prove some predicates it needs to be true.
  // ASSERT_EQ(eval_ctx.heap.allocs.size(), may_alias ? 0 : 1);
  // ASSERT_EQ(eval_ctx.copy_calls, may_alias ? 0 : 2);
}

TEST_P(may_alias, unfolded) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));

  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(short));

  var x(ctx, "x");
  var y(ctx, "y");
  test_context eval_ctx;

  // In this pipeline, the result is copied to the output, and we want to fold the intermediate buffer and alias it to
  // the output. We can only alias it if we know the output is unfolded.
  const bool may_alias = GetParam();
  if (may_alias) {
    out->dim(1).fold_factor = dim::unfolded;
  }

  // If we want to alias intermediate buffer to the output buffer,
  // we need to tell aliaser that output is unfolded and it's safe to alias.
  out->dim(0).fold_factor = dim::unfolded;

  func add = func::make(add_1<short>, {{in, {point(x), point(y)}}}, {{intm, {x, y}}});
  func copied = func::make_copy({intm, {point(x), point(y)}}, {out, {x, y}}, eval_ctx.copy);

  // The fold factor must be > 1, so we can't assume that the intermediate fold factor divides the output fold factor.
  copied.loops({{y, 2}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 20;
  const int H = 10;
  buffer<short, 2> in_buf({W, H});
  init_random(in_buf);

  buffer<short, 2> out_buf({W, H});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y), in_buf(x, y) + 1);
    }
  }

  ASSERT_EQ(eval_ctx.heap.allocs.size(), may_alias ? 0 : 1);
}

TEST_P(may_alias, inside_loop) {
  const bool may_alias = GetParam();

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 1, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 1, sizeof(int));
  auto intm = buffer_expr::make(ctx, "intm", 1, sizeof(int));
  auto intm2 = buffer_expr::make(ctx, "intm2", 1, sizeof(int));

  var x(ctx, "x");

  func mul = func::make(
      multiply_2<int>, {{in, {point(x)}}}, {{intm, {x}}}, call_stmt::attributes{.allow_in_place = 0x1, .name = "mul"});
  func mul2 = func::make(multiply_2<int>, {{intm, {point(x)}}}, {{intm2, {x}}},
      call_stmt::attributes{.allow_in_place = 0x1, .name = "mul2"});
  func add = func::make(
      add_1<int>, {{intm2, {point(x)}}}, {{out, {x}}}, call_stmt::attributes{.allow_in_place = 0x1, .name = "add"});

  add.loops({{x, 1, 1}});

  if (may_alias) {
    intm->store_at({&add, x});
    intm2->store_at({&add, x});
  }

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline
  const int N = 10;

  buffer<int, 1> in_buf({N});
  in_buf.allocate();
  for (int i = 0; i < N; ++i) {
    in_buf(i) = i;
  }

  buffer<int, 1> out_buf({N});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(out_buf(i), 4 * i + 1);
  }

  ASSERT_EQ(eval_ctx.heap.allocs.size(), may_alias ? 0 : 2);
}

TEST(split_output, cannot_alias) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(short));
  auto out1 = buffer_expr::make(ctx, "out1", 2, sizeof(short));
  auto out2 = buffer_expr::make(ctx, "out2", 2, sizeof(short));

  auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(short));

  // If we want to alias intermediate buffer to the output buffer,
  // we need to tell aliaser that output is unfolded and it's safe to alias.
  for (auto i : {in, out1, out2}) {
    i->dim(0).fold_factor = dim::unfolded;
    i->dim(1).fold_factor = dim::unfolded;
  }

  var x(ctx, "x");
  var y(ctx, "y");
  test_context eval_ctx;

  // This pipeline is tempted to alias the intermediate to the output, but it isn't safe because we don't know it's big
  // enough.
  func add = func::make(add_1<short>, {{{in, {point(x), point(y)}}}}, {{{intm, {x, y}}}});
  func split1 = func::make_copy({intm, {point(x), point(y)}}, {out1, {x, y}}, eval_ctx.copy);
  func split2 = func::make_copy({intm, {point(x), point(y) + out1->dim(1).extent()}}, {out2, {x, y}}, eval_ctx.copy);
  pipeline p = build_pipeline(ctx, {in}, {out1, out2});

  // Run the pipeline.
  const int W = 20;
  const int H1 = 4;
  const int H2 = 7;
  buffer<short, 2> in_buf({W, H1 + H2});
  init_random(in_buf);

  buffer<short, 2> out1_buf({W, H1});
  buffer<short, 2> out2_buf({W, H2});
  out1_buf.allocate();
  out2_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out1_buf, &out2_buf};
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H1; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out1_buf(x, y), in_buf(x, y) + 1);
    }
  }

  for (int y = 0; y < H2; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out2_buf(x, y), in_buf(x, y + H1) + 1);
    }
  }

  ASSERT_EQ(eval_ctx.heap.allocs.size(), 1);
  ASSERT_EQ(eval_ctx.copy_calls, 2);
}

class multiple_uses : public testing::TestWithParam<std::tuple<int, bool>> {};

INSTANTIATE_TEST_SUITE_P(alias_split, multiple_uses, testing::Combine(testing::Values(0, 1), testing::Bool()),
    test_params_to_string<multiple_uses::ParamType>);

TEST_P(multiple_uses, cannot_alias) {
  const int in_place = std::get<0>(GetParam());
  const bool split = std::get<1>(GetParam());
  // Make the pipeline
  node_context ctx;

  // In the pipeline:
  // in -> a -> b -> out
  //       a -> c
  //
  // We could try to alias a to b or c, but it wouldn't be valid, because it is used more than once.

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));

  auto a = buffer_expr::make(ctx, "a", 2, sizeof(short));
  auto b = buffer_expr::make(ctx, "b", 2, sizeof(short));
  auto c = buffer_expr::make(ctx, "c", 2, sizeof(short));

  var x(ctx, "x");
  var y(ctx, "y");

  func in_a = func::make(add_1<short>, {{in, {point(x), point(y)}}}, {{a, {x, y}}});
  func a_b = func::make(add_1<short>, {{a, {point(x), point(y)}}}, {{b, {x, y}}},
      call_stmt::attributes{.allow_in_place = in_place == 0 ? 0x1 : 0, .name = "a_b"});
  func a_c = func::make(add_1<short>, {{a, {point(x), point(y)}}}, {{c, {x, y}}},
      call_stmt::attributes{.allow_in_place = in_place == 1 ? 0x1 : 0, .name = "a_c"});

  func sub = func::make(subtract<short>, {{b, {point(x), point(y)}}, {c, {point(x), point(y)}}}, {{out, {x, y}}});

  if (split) {
    sub.loops({{y}});
  }

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 20;
  const int H = 10;
  buffer<short, 2> in_buf({W, H});
  init_random(in_buf);

  buffer<short, 2> out_buf({W, H});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y), 0);
    }
  }
}

TEST_P(multiple_uses, cannot_alias_output) {
  const int in_place = std::get<0>(GetParam());
  const bool split = std::get<1>(GetParam());
  if (split) GTEST_SKIP();

  // Make the pipeline
  node_context ctx;

  // In the pipeline:
  // in -> a -> b
  //       a -> c
  // We could try to alias a to b or c, but it wouldn't be valid, because it is used more than once.

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(short));

  auto a = buffer_expr::make(ctx, "a", 2, sizeof(short));
  auto b = buffer_expr::make(ctx, "b", 2, sizeof(short));
  auto c = buffer_expr::make(ctx, "c", 2, sizeof(short));

  b->dim(0).fold_factor = dim::unfolded;
  b->dim(1).fold_factor = dim::unfolded;

  var x(ctx, "x");
  var y(ctx, "y");

  func in_a = func::make(add_1<short>, {{in, {point(x), point(y)}}}, {{a, {x, y}}});
  func a_b = func::make(add_1<short>, {{a, {point(x), point(y)}}}, {{b, {x, y}}},
      call_stmt::attributes{.allow_in_place = in_place == 0 ? 0x1 : 0, .name = "a_b"});
  func a_c = func::make(add_1<short>, {{a, {point(x), point(y)}}}, {{c, {x, y}}},
      call_stmt::attributes{.allow_in_place = in_place == 1 ? 0x1 : 0, .name = "a_c"});

  pipeline p = build_pipeline(ctx, {in}, {b, c});

  // Run the pipeline.
  const int W = 20;
  const int H = 10;
  buffer<short, 2> in_buf({W, H});
  init_random(in_buf);

  buffer<short, 2> b_buf({W, H});
  buffer<short, 2> c_buf({W, H});
  b_buf.allocate();
  c_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&b_buf, &c_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(b_buf(x, y), in_buf(x, y) + 2);
      ASSERT_EQ(c_buf(x, y), in_buf(x, y) + 2);
    }
  }
}

TEST_P(multiple_uses, cannot_alias_input_output) {
  const int in_place = std::get<0>(GetParam());
  const bool split = std::get<1>(GetParam());
  if (split) GTEST_SKIP();

  // Make the pipeline
  node_context ctx;

  // In the pipeline:
  //
  //   in -> a -> b -> c
  //
  // where a and c are both outputs, we can't compute b in place with a, because it corrupts the value of a, which is an output.

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(short));

  auto a = buffer_expr::make(ctx, "a", 2, sizeof(short));
  auto b = buffer_expr::make(ctx, "b", 2, sizeof(short));
  auto c = buffer_expr::make(ctx, "b", 2, sizeof(short));

  b->dim(0).fold_factor = dim::unfolded;
  b->dim(1).fold_factor = dim::unfolded;

  var x(ctx, "x");
  var y(ctx, "y");

  func in_a = func::make(add_1<short>, {{in, {point(x), point(y)}}}, {{a, {x, y}}});
  func a_b = func::make(add_1<short>, {{a, {point(x), point(y)}}}, {{b, {x, y}}},
      call_stmt::attributes{.allow_in_place = in_place == 0 ? 0x1 : 0, .name = "a_b"});
  func b_c = func::make(add_1<short>, {{b, {point(x), point(y)}}}, {{c, {x, y}}},
      call_stmt::attributes{.name = "b_c"});

  pipeline p = build_pipeline(ctx, {in}, {a, c});

  // Run the pipeline.
  const int W = 20;
  const int H = 10;
  buffer<short, 2> in_buf({W, H});
  init_random(in_buf);

  buffer<short, 2> a_buf({W, H});
  buffer<short, 2> c_buf({W, H});
  a_buf.allocate();
  c_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&a_buf, &c_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(a_buf(x, y), in_buf(x, y) + 1);
      ASSERT_EQ(c_buf(x, y), in_buf(x, y) + 3);
    }
  }
}

TEST(reused_in_loop, cannot_alias) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));

  auto sum_in = buffer_expr::make(ctx, "sum_in", 1, sizeof(int));
  auto sum_in_1 = buffer_expr::make(ctx, "sum_in_1", 1, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");

  func sum_y = func::make(
      [](const buffer<const int>& in, const buffer<int>& sum_in) -> index_t {
        for (index_t x = sum_in.dim(0).begin(); x < sum_in.dim(0).end(); ++x) {
          sum_in(x) = 0;
          for (index_t y = in.dim(1).begin(); y < in.dim(1).end(); ++y) {
            sum_in(x) += in(x, y);
          }
        }
        return 0;
      },
      {{in, {point(x), in->dim(1).bounds}}}, {{sum_in, {x}}});

  func add = func::make(add_1<int>, {{sum_in, {point(x)}}}, {{sum_in_1, {x}}},
      call_stmt::attributes{.allow_in_place = true, .name = "add_1"});

  func diff = func::make(
      [](const buffer<const int>& in, const buffer<const int>& sum_in_1, const buffer<int>& out) -> index_t {
        for (index_t x = out.dim(0).begin(); x < out.dim(0).end(); ++x) {
          for (index_t y = out.dim(1).begin(); y < out.dim(1).end(); ++y) {
            out(x, y) = in(x, y) - sum_in_1(x);
          }
        }
        return 0;
      },
      {{in, {point(x), point(y)}}, {sum_in_1, {point(x), point(y)}}}, {{out, {x, y}}},
      call_stmt::attributes{.allow_in_place = true, .name = "sub"});

  sum_y.compute_root();
  diff.loops({{x, 1}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 20;
  const int H = 10;
  buffer<int, 2> in_buf({W, H});
  init_random(in_buf);

  buffer<int, 2> out_buf({W, H});
  out_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int x = 0; x < W; ++x) {
    int sum_in_y_1 = 1;
    for (int y = 0; y < H; ++y) {
      sum_in_y_1 += in_buf(x, y);
    }

    for (int y = 0; y < H; ++y) {
      ASSERT_EQ(in_buf(x, y) - sum_in_y_1, out_buf(x, y));
    }
  }
}

TEST(multiple_producers, cannot_alias) {
  // Make the pipeline
  node_context ctx;

  // This pipeline contains a concatenate of two stages, followed by a consumer. It was tempting for the copy aliaser
  // to alias the concatenate buffer to one of the producers for it, but this is not correct without being able to alias
  // both of the producers.
  auto in1 = buffer_expr::make(ctx, "in1", 2, sizeof(short));
  auto in2 = buffer_expr::make(ctx, "in2", 2, sizeof(short));

  auto a = buffer_expr::make(ctx, "a", 2, sizeof(short));
  auto b = buffer_expr::make(ctx, "b", 2, sizeof(short));
  auto c = buffer_expr::make(ctx, "c", 2, sizeof(short));
  auto d = buffer_expr::make(ctx, "d", 2, sizeof(short));

  var x(ctx, "x");
  var y(ctx, "y");

  func in1_a = func::make(add_1<short>, {{in1, {point(x), point(y)}}}, {{a, {x, y}}});
  func in2_b = func::make(multiply_2<short>, {{in2, {point(x), point(y)}}}, {{b, {x, y}}});

  func concat =
      func::make_concat({a, b}, {c, {x, y}}, 1, {0, in1->dim(1).extent(), in1->dim(1).extent() + in2->dim(1).extent()});

  func c_d = func::make(add_1<short>, {{c, {point(x), point(y)}}}, {{d, {x, y}}});

  pipeline p = build_pipeline(ctx, {in1, in2}, {d});

  // Run the pipeline.
  const int W = 20;
  const int H = 5;
  buffer<short, 2> in1_buf({W, H});
  buffer<short, 2> in2_buf({W, H});
  init_random(in1_buf);
  init_random(in2_buf);

  buffer<short, 2> d_buf({W, H * 2});
  d_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in1_buf, &in2_buf};
  const raw_buffer* outputs[] = {&d_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(d_buf(x, y), in1_buf(x, y) + 2);
      ASSERT_EQ(d_buf(x, y + H), in2_buf(x, y) * 2 + 1);
    }
  }
}

TEST(cannot_alias, padded_constant) {
  const int padding_value = 3;

  // Make the pipeline
  node_context ctx;

  const int W = 10;
  const int H = 7;
  buffer<char, 2> in_buf({W - 2, H - 2});
  init_random(in_buf);

  auto in = buffer_expr::make_constant(ctx, "in", raw_buffer::make_copy(in_buf));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(char));
  auto padded_in = buffer_expr::make(ctx, "padded_intm", 2, sizeof(char));
  auto padding = buffer_expr::make_scalar<char>(ctx, "padding", padding_value);

  var x(ctx, "x");
  var y(ctx, "y");
  test_context eval_ctx;

  func crop = func::make_copy(
      {in, {point(x), point(y)}, in->bounds()}, {padded_in, {x, y}}, {padding, {point(x), point(y)}}, eval_ctx.copy);
  func copy_out = func::make(opaque_copy<char>, {{padded_in, {point(x), point(y)}}}, {{out, {x, y}}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline
  buffer<char, 2> out_buf({W, H});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      if (in_buf.contains(x, y)) {
        ASSERT_EQ(out_buf(x, y), in_buf(x, y));
      } else {
        ASSERT_EQ(out_buf(x, y), padding_value);
      }
    }
  }

  ASSERT_EQ(eval_ctx.heap.allocs.size(), 1);
  ASSERT_EQ(eval_ctx.copy_calls, 1);
}

class constrained_stencil : public testing::TestWithParam<int> {};

INSTANTIATE_TEST_SUITE_P(alias_split, constrained_stencil, testing::Values(1, 5));

TEST_P(constrained_stencil, may_alias) {
  const int S = GetParam();
  const int D = 1;
  const int K = 5;

  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", 1, sizeof(short));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(short));

  in->dim(0).fold_factor = dim::unfolded;

  auto in_copy = buffer_expr::make(ctx, "in_copy", 1, sizeof(short));
  auto stencil = buffer_expr::make(ctx, "stencil", 2, sizeof(short));

  stencil->dim(0).stride = sizeof(short);
  stencil->dim(1).stride = sizeof(short) * S;

  var x(ctx, "x");
  var dx(ctx, "dx");
  test_context eval_ctx;

  // This test computes the following stencil operation, in this case a convolution with a kernel of 1s:
  //
  //  for i in [0, N):
  //    for k in [0, K):
  //      out[i] += in[i * S + k * D]
  //
  // Using the following approach:
  // 1. Make a copy of the input such that stencil(x, dx) = in(x * S + dx * D)
  // 2. Compute a reduction of the dx dimension
  //
  // We expect slinky to alias the copy.
  func pre_copy = func::make(opaque_copy<short>, {{in, {point(x)}}}, {{in_copy, {x}}});
  func stencil_copy = func::make_copy({in_copy, {point(x * S + dx * D)}}, {stencil, {dx, x}}, eval_ctx.copy);
  func post_copy = func::make(opaque_copy<short>, {{stencil, {point(dx), point(x)}}}, {{out, {dx, x}}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.

  const int N = 10;

  buffer<short, 2> out_buf({K, N});
  out_buf.allocate();

  buffer<short, 1> in_buf({(N - 1) * S + (K - 1) * D + 1});
  init_random(in_buf);

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  p.evaluate(inputs, outputs, eval_ctx);

  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      ASSERT_EQ(out_buf(k, n), in_buf(n * S + k * D));
    }
  }

  if (S == 1) {
    // When S is 1, both stencil dimensions can be aliased without violating the stride constraint for either dimension.
    ASSERT_EQ(eval_ctx.heap.allocs.size(), 1);
    ASSERT_EQ(eval_ctx.copy_calls, 0);
  } else {
    ASSERT_EQ(eval_ctx.heap.allocs.size(), 2);
    ASSERT_EQ(eval_ctx.copy_calls, 1);
  }
}

}  // namespace slinky
