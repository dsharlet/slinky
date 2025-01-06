#include <gtest/gtest.h>

#include <random>
#include <string>

#include "builder/pipeline.h"
#include "builder/replica_pipeline.h"
#include "builder/test/funcs.h"

namespace slinky {

TEST(replica, matmuls) {
  // clang-format off
// BEGIN define_replica_pipeline() output
auto p = []() -> ::slinky::pipeline {
  using std::abs, std::min, std::max;
  node_context ctx;
  auto a = buffer_expr::make(ctx, "a", /*rank=*/2, /*elem_size=*/4);
  a->dim(1).stride = 4;
  auto b = buffer_expr::make(ctx, "b", /*rank=*/2, /*elem_size=*/4);
  b->dim(1).stride = 4;
  auto c = buffer_expr::make(ctx, "c", /*rank=*/2, /*elem_size=*/4);
  c->dim(1).stride = 4;
  auto abc = buffer_expr::make(ctx, "abc", /*rank=*/2, /*elem_size=*/4);
  abc->dim(1).stride = 4;
  auto i = var(ctx, "i");
  auto j = var(ctx, "j");
  auto ab = buffer_expr::make(ctx, "ab", /*rank=*/2, /*elem_size=*/4);
  ab->dim(1).stride = 4;
  auto _2 = a->sym();
  auto _replica_fn_3 = [=](const buffer<const void>& i0, const buffer<const void>& i1, const buffer<void>& o0) -> index_t {
    const buffer<const void>* input_buffers[] = {&i0, &i1};
    const buffer<void>* output_buffers[] = {&o0};
    const func::input inputs[] = {{a, {point(i), {(buffer_min(_2, 1)), (buffer_max(_2, 1))}}}, {b, {{(buffer_min(_2, 1)), (buffer_max(_2, 1))}, point(j)}}};
    const std::vector<var> outputs[] = {{i, j}};
    return ::slinky::internal::replica_pipeline_handler(input_buffers, output_buffers, inputs, outputs);
  };
  auto _fn_1 = func::make(std::move(_replica_fn_3), {{a, {point(i), {(buffer_min(_2, 1)), (buffer_max(_2, 1))}}}, {b, {{(buffer_min(_2, 1)), (buffer_max(_2, 1))}, point(j)}}}, {{ab, {i, j}}}, {});
  auto _4 = c->sym();
  auto _replica_fn_5 = [=](const buffer<const void>& i0, const buffer<const void>& i1, const buffer<void>& o0) -> index_t {
    const buffer<const void>* input_buffers[] = {&i0, &i1};
    const buffer<void>* output_buffers[] = {&o0};
    const func::input inputs[] = {{ab, {point(i), {(buffer_min(_4, 0)), (buffer_max(_4, 0))}}}, {c, {{(buffer_min(_4, 0)), (buffer_max(_4, 0))}, point(j)}}};
    const std::vector<var> outputs[] = {{i, j}};
    return ::slinky::internal::replica_pipeline_handler(input_buffers, output_buffers, inputs, outputs);
  };
  auto _fn_0 = func::make(std::move(_replica_fn_5), {{ab, {point(i), {(buffer_min(_4, 0)), (buffer_max(_4, 0))}}}, {c, {{(buffer_min(_4, 0)), (buffer_max(_4, 0))}, point(j)}}}, {{abc, {i, j}}}, {});
  _fn_0.loops({{i, 1, loop::serial}});
  auto p = build_pipeline(ctx, {}, {a, b, c}, {abc}, {}, {});
  return p;
};
// END define_replica_pipeline() output
  // clang-format on

  const int M = 10;
  const int N = 10;
  buffer<int, 2> a_buf({N, M});
  buffer<int, 2> b_buf({N, M});
  buffer<int, 2> c_buf({N, M});
  buffer<int, 2> abc_buf({N, M});
  std::swap(a_buf.dim(1), a_buf.dim(0));
  std::swap(b_buf.dim(1), b_buf.dim(0));
  std::swap(c_buf.dim(1), c_buf.dim(0));
  std::swap(abc_buf.dim(1), abc_buf.dim(0));

  init_random(a_buf);
  init_random(b_buf);
  init_random(c_buf);
  abc_buf.allocate();

  const raw_buffer* inputs[] = {&a_buf, &b_buf, &c_buf};
  const raw_buffer* outputs[] = {&abc_buf};

  eval_context eval_ctx;
  ASSERT_EQ(0, p().evaluate(inputs, outputs, eval_ctx));
}

TEST(replica, pyramid) {
  // clang-format off
// BEGIN define_replica_pipeline() output
auto p = []() -> ::slinky::pipeline {
  using std::abs, std::min, std::max;
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", /*rank=*/2, /*elem_size=*/4);
  auto out = buffer_expr::make(ctx, "out", /*rank=*/2, /*elem_size=*/4);
  auto x = var(ctx, "x");
  auto y = var(ctx, "y");
  auto intm = buffer_expr::make(ctx, "intm", /*rank=*/2, /*elem_size=*/4);
  auto _replica_fn_2 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* input_buffers[] = {&i0};
    const buffer<void>* output_buffers[] = {&o0};
    const func::input inputs[] = {{in, {{((((x * 2)) + 0)), ((((x * 2)) + 1))}, {((((y * 2)) + 0)), ((((y * 2)) + 1))}}}};
    const std::vector<var> outputs[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(input_buffers, output_buffers, inputs, outputs);
  };
  auto _fn_1 = func::make(std::move(_replica_fn_2), {{in, {{((((x * 2)) + 0)), ((((x * 2)) + 1))}, {((((y * 2)) + 0)), ((((y * 2)) + 1))}}}}, {{intm, {x, y}}}, {});
  auto _replica_fn_3 = [=](const buffer<const void>& i0, const buffer<const void>& i1, const buffer<void>& o0) -> index_t {
    const buffer<const void>* input_buffers[] = {&i0, &i1};
    const buffer<void>* output_buffers[] = {&o0};
    const func::input inputs[] = {{in, {point(x), point(y)}}, {intm, {{((x / 2)), ((((x + 1)) / 2))}, {((y / 2)), ((((y + 1)) / 2))}}}};
    const std::vector<var> outputs[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(input_buffers, output_buffers, inputs, outputs);
  };
  auto _fn_0 = func::make(std::move(_replica_fn_3), {{in, {point(x), point(y)}}, {intm, {{((x / 2)), ((((x + 1)) / 2))}, {((y / 2)), ((((y + 1)) / 2))}}}}, {{out, {x, y}}}, {});
  _fn_0.loops({{y, 1, loop::serial}});
  auto p = build_pipeline(ctx, {}, {in}, {out}, {}, {});
  return p;
};
// END define_replica_pipeline() output
  // clang-format on

  const int W = 10;
  const int H = 10;
  buffer<int, 2> in_buf({W + 4, H + 4});
  in_buf.translate(-2, -2);
  buffer<int, 2> out_buf({W, H});

  init_random(in_buf);
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};

  eval_context eval_ctx;
  ASSERT_EQ(0, p().evaluate(inputs, outputs, eval_ctx));
}

TEST(replica, multiple_outputs) {
  // clang-format off
// BEGIN define_replica_pipeline() output
auto p = []() -> ::slinky::pipeline {
  using std::abs, std::min, std::max;
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", /*rank=*/3, /*elem_size=*/4);
  auto sum_x = buffer_expr::make(ctx, "sum_x", /*rank=*/2, /*elem_size=*/4);
  auto y = var(ctx, "y");
  auto z = var(ctx, "z");
  auto _1 = in->sym();
  auto _replica_fn_2 = [=](const buffer<const void>& i0, const buffer<void>& o0, const buffer<void>& o1) -> index_t {
    const buffer<const void>* input_buffers[] = {&i0};
    const buffer<void>* output_buffers[] = {&o0, &o1};
    const func::input inputs[] = {{in, {{(buffer_min(_1, 0)), (buffer_max(_1, 0))}, {(buffer_min(_1, 1)), (buffer_max(_1, 1))}, point(z)}}};
    const std::vector<var> outputs[] = {{y, z}, {z}};
    return ::slinky::internal::replica_pipeline_handler(input_buffers, output_buffers, inputs, outputs);
  };
  auto sum_xy = buffer_expr::make(ctx, "sum_xy", /*rank=*/1, /*elem_size=*/4);
  auto _fn_0 = func::make(std::move(_replica_fn_2), {{in, {{(buffer_min(_1, 0)), (buffer_max(_1, 0))}, {(buffer_min(_1, 1)), (buffer_max(_1, 1))}, point(z)}}}, {{sum_x, {y, z}}, {sum_xy, {z}}}, {});
  _fn_0.loops({{z, 1, loop::serial}});
  auto p = build_pipeline(ctx, {}, {in}, {sum_x, sum_xy}, {}, {});
  return p;
};
// END define_replica_pipeline() output
  // clang-format on

  const int H = 20;
  const int W = 10;
  const int D = 5;
  buffer<int, 3> in_buf({W, H, D});
  init_random(in_buf);
  buffer<int, 2> sum_x_buf({H, D});
  buffer<int, 1> sum_xy_buf({D});
  sum_x_buf.allocate();
  sum_xy_buf.allocate();
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&sum_x_buf, &sum_xy_buf};

  eval_context eval_ctx;
  ASSERT_EQ(0, p().evaluate(inputs, outputs, eval_ctx));
}

TEST(replica, unrelated) {
  // clang-format off
// BEGIN define_replica_pipeline() output
auto p = []() -> ::slinky::pipeline {
  using std::abs, std::min, std::max;
  node_context ctx;
  auto in1 = buffer_expr::make(ctx, "in1", /*rank=*/2, /*elem_size=*/2);
  auto in2 = buffer_expr::make(ctx, "in2", /*rank=*/1, /*elem_size=*/4);
  auto out1 = buffer_expr::make(ctx, "out1", /*rank=*/2, /*elem_size=*/2);
  auto x = var(ctx, "x");
  auto y = var(ctx, "y");
  auto intm1 = buffer_expr::make(ctx, "intm1", /*rank=*/2, /*elem_size=*/2);
  auto _replica_fn_2 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* input_buffers[] = {&i0};
    const buffer<void>* output_buffers[] = {&o0};
    const func::input inputs[] = {{in1, {point(x), point(y)}}};
    const std::vector<var> outputs[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(input_buffers, output_buffers, inputs, outputs);
  };
  auto _fn_1 = func::make(std::move(_replica_fn_2), {{in1, {point(x), point(y)}}}, {{intm1, {x, y}}}, {.allow_in_place = true, .name = "add1"});
  auto _replica_fn_3 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* input_buffers[] = {&i0};
    const buffer<void>* output_buffers[] = {&o0};
    const func::input inputs[] = {{intm1, {{((x + -1)), ((x + 1))}, {((y + -1)), ((y + 1))}}}};
    const std::vector<var> outputs[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(input_buffers, output_buffers, inputs, outputs);
  };
  auto _fn_0 = func::make(std::move(_replica_fn_3), {{intm1, {{((x + -1)), ((x + 1))}, {((y + -1)), ((y + 1))}}}}, {{out1, {x, y}}}, {});
  _fn_0.loops({{y, 2, loop::serial}});
  auto out2 = buffer_expr::make(ctx, "out2", /*rank=*/1, /*elem_size=*/4);
  out2->dim(0).fold_factor = (index_t) 9223372036854775807;
  auto intm2 = buffer_expr::make(ctx, "intm2", /*rank=*/1, /*elem_size=*/4);
  auto _replica_fn_6 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* input_buffers[] = {&i0};
    const buffer<void>* output_buffers[] = {&o0};
    const func::input inputs[] = {{in2, {point(x)}}};
    const std::vector<var> outputs[] = {{x}};
    return ::slinky::internal::replica_pipeline_handler(input_buffers, output_buffers, inputs, outputs);
  };
  auto _fn_5 = func::make(std::move(_replica_fn_6), {{in2, {point(x)}}}, {{intm2, {x}}}, {.allow_in_place = true, .name = "mul2"});
  auto _replica_fn_7 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* input_buffers[] = {&i0};
    const buffer<void>* output_buffers[] = {&o0};
    const func::input inputs[] = {{intm2, {point(x)}}};
    const std::vector<var> outputs[] = {{x}};
    return ::slinky::internal::replica_pipeline_handler(input_buffers, output_buffers, inputs, outputs);
  };
  auto _fn_4 = func::make(std::move(_replica_fn_7), {{intm2, {point(x)}}}, {{out2, {x}}}, {.allow_in_place = true, .name = "add2"});
  auto p = build_pipeline(ctx, {}, {in1, in2}, {out1, out2}, {}, {});
  return p;
};
// END define_replica_pipeline() output
  // clang-format on

  const int W1 = 20;
  const int H1 = 10;
  buffer<short, 2> in1_buf({W1 + 2, H1 + 2});
  in1_buf.translate(-1, -1);
  buffer<short, 2> out1_buf({W1, H1});

  init_random(in1_buf);
  out1_buf.allocate();

  const int N2 = 30;
  buffer<int, 1> in2_buf({N2});
  in2_buf.allocate();
  for (int i = 0; i < N2; ++i) {
    in2_buf(i) = i;
  }

  buffer<int, 1> out2_buf({N2});
  out2_buf.allocate();

  const raw_buffer* inputs[] = {&in1_buf, &in2_buf};
  const raw_buffer* outputs[] = {&out1_buf, &out2_buf};

  eval_context eval_ctx;
  ASSERT_EQ(0, p().evaluate(inputs, outputs, eval_ctx));
}

TEST(replica, concatenated_result) {
  // clang-format off
// BEGIN define_replica_pipeline() output
auto p = []() -> ::slinky::pipeline {
  using std::abs, std::min, std::max;
  node_context ctx;
  auto in1 = buffer_expr::make(ctx, "in1", /*rank=*/2, /*elem_size=*/2);
  auto in2 = buffer_expr::make(ctx, "in2", /*rank=*/2, /*elem_size=*/2);
  auto out = buffer_expr::make(ctx, "out", /*rank=*/2, /*elem_size=*/2);
  out->dim(0).fold_factor = (index_t) 9223372036854775807;
  out->dim(1).fold_factor = (index_t) 9223372036854775807;
  auto x = var(ctx, "x");
  auto y = var(ctx, "y");
  auto intm1 = buffer_expr::make(ctx, "intm1", /*rank=*/2, /*elem_size=*/2);
  auto _replica_fn_2 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* input_buffers[] = {&i0};
    const buffer<void>* output_buffers[] = {&o0};
    const func::input inputs[] = {{in1, {point(x), point(y)}}};
    const std::vector<var> outputs[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(input_buffers, output_buffers, inputs, outputs);
  };
  auto _fn_1 = func::make(std::move(_replica_fn_2), {{in1, {point(x), point(y)}}}, {{intm1, {x, y}}}, {});
  auto _3 = in1->sym();
  auto intm2 = buffer_expr::make(ctx, "intm2", /*rank=*/2, /*elem_size=*/2);
  auto _replica_fn_5 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* input_buffers[] = {&i0};
    const buffer<void>* output_buffers[] = {&o0};
    const func::input inputs[] = {{in2, {point(x), point(y)}}};
    const std::vector<var> outputs[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(input_buffers, output_buffers, inputs, outputs);
  };
  auto _fn_4 = func::make(std::move(_replica_fn_5), {{in2, {point(x), point(y)}}}, {{intm2, {x, y}}}, {});
  auto _6 = out->sym();
  auto _fn_0 = func::make_copy({{intm1, {point(x), point(((y - 0)))}, {point(expr()), {0, (((((((((buffer_max(_3, 1)) - (buffer_min(_3, 1)))) + 1)) - 0)) - 1))}}, {point(expr()), {0, (((((((buffer_max(_3, 1)) - (buffer_min(_3, 1)))) + 1)) - 1))}}, {}}, {intm2, {point(x), point(((y - (((((buffer_max(_3, 1)) - (buffer_min(_3, 1)))) + 1)))))}, {point(expr()), {0, (((((((((buffer_max(_6, 1)) - (buffer_min(_6, 1)))) + 1)) - (((((buffer_max(_3, 1)) - (buffer_min(_3, 1)))) + 1)))) - 1))}}, {point(expr()), {(((((buffer_max(_3, 1)) - (buffer_min(_3, 1)))) + 1)), (((((((buffer_max(_6, 1)) - (buffer_min(_6, 1)))) + 1)) - 1))}}, {}}}, {out, {x, y}});
  auto p = build_pipeline(ctx, {}, {in1, in2}, {out}, {}, {.no_alias_buffers = true});
  return p;
};
// END define_replica_pipeline() output
  // clang-format on

  const int W = 20;
  const int H1 = 4;
  const int H2 = 7;
  buffer<short, 2> in1_buf({W, H1});
  buffer<short, 2> in2_buf({W, H2});
  init_random(in1_buf);
  init_random(in2_buf);

  buffer<short, 2> out_buf({W, H1 + H2});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in1_buf, &in2_buf};
  const raw_buffer* outputs[] = {&out_buf};

  eval_context eval_ctx;
  ASSERT_EQ(0, p().evaluate(inputs, outputs, eval_ctx));
}

TEST(replica, stacked_result) {
  // clang-format off
// BEGIN define_replica_pipeline() output
auto p = []() -> ::slinky::pipeline {
  using std::abs, std::min, std::max;
  node_context ctx;
  auto in1 = buffer_expr::make(ctx, "in1", /*rank=*/2, /*elem_size=*/2);
  auto in2 = buffer_expr::make(ctx, "in2", /*rank=*/2, /*elem_size=*/2);
  auto out = buffer_expr::make(ctx, "out", /*rank=*/3, /*elem_size=*/2);
  out->dim(0).fold_factor = (index_t) 9223372036854775807;
  out->dim(1).fold_factor = (index_t) 9223372036854775807;
  out->dim(2).fold_factor = (index_t) 9223372036854775807;
  auto x = var(ctx, "x");
  auto y = var(ctx, "y");
  auto intm1 = buffer_expr::make(ctx, "intm1", /*rank=*/2, /*elem_size=*/2);
  auto _replica_fn_2 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* input_buffers[] = {&i0};
    const buffer<void>* output_buffers[] = {&o0};
    const func::input inputs[] = {{in1, {point(x), point(y)}}};
    const std::vector<var> outputs[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(input_buffers, output_buffers, inputs, outputs);
  };
  auto _fn_1 = func::make(std::move(_replica_fn_2), {{in1, {point(x), point(y)}}}, {{intm1, {x, y}}}, {});
  auto intm2 = buffer_expr::make(ctx, "intm2", /*rank=*/2, /*elem_size=*/2);
  auto _replica_fn_4 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* input_buffers[] = {&i0};
    const buffer<void>* output_buffers[] = {&o0};
    const func::input inputs[] = {{in2, {point(x), point(y)}}};
    const std::vector<var> outputs[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(input_buffers, output_buffers, inputs, outputs);
  };
  auto _fn_3 = func::make(std::move(_replica_fn_4), {{in2, {point(x), point(y)}}}, {{intm2, {x, y}}}, {});
  auto _fn_0 = func::make_copy({{intm1, {point(x), point(y)}, {}, {}, {expr(), expr(), 0}}, {intm2, {point(x), point(y)}, {}, {}, {expr(), expr(), 1}}}, {out, {x, y}});
  auto p = build_pipeline(ctx, {}, {in1, in2}, {out}, {}, {});
  return p;
};
// END define_replica_pipeline() output
  // clang-format on

  const int W = 20;
  const int H = 8;
  buffer<short, 2> in1_buf({W, H});
  buffer<short, 2> in2_buf({W, H});
  init_random(in1_buf);
  init_random(in2_buf);

  buffer<short, 3> out_buf({W, H, 2});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in1_buf, &in2_buf};
  const raw_buffer* outputs[] = {&out_buf};

  eval_context eval_ctx;
  ASSERT_EQ(0, p().evaluate(inputs, outputs, eval_ctx));
}

TEST(replica, padded_stencil) {
  // clang-format off
// BEGIN define_replica_pipeline() output
auto p = []() -> ::slinky::pipeline {
  using std::abs, std::min, std::max;
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", /*rank=*/2, /*elem_size=*/2);
  auto out = buffer_expr::make(ctx, "out", /*rank=*/2, /*elem_size=*/2);
  auto x = var(ctx, "x");
  auto y = var(ctx, "y");
  auto padded_intm = buffer_expr::make(ctx, "padded_intm", /*rank=*/2, /*elem_size=*/2);
  auto intm = buffer_expr::make(ctx, "intm", /*rank=*/2, /*elem_size=*/2);
  auto _replica_fn_3 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* input_buffers[] = {&i0};
    const buffer<void>* output_buffers[] = {&o0};
    const func::input inputs[] = {{in, {point(x), point(y)}}};
    const std::vector<var> outputs[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(input_buffers, output_buffers, inputs, outputs);
  };
  auto _fn_2 = func::make(std::move(_replica_fn_3), {{in, {point(x), point(y)}}}, {{intm, {x, y}}}, {});
  auto _4 = in->sym();
  auto _fn_1 = func::make_copy({intm, {point(x), point(y)}, {{(buffer_min(_4, 0)), (buffer_max(_4, 0))}, {(buffer_min(_4, 1)), (buffer_max(_4, 1))}}, {}, {}}, {padded_intm, {x, y}}, {6, 0});
  _fn_1.compute_root();
  auto _replica_fn_5 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* input_buffers[] = {&i0};
    const buffer<void>* output_buffers[] = {&o0};
    const func::input inputs[] = {{padded_intm, {{((x + -1)), ((x + 1))}, {((y + -1)), ((y + 1))}}}};
    const std::vector<var> outputs[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(input_buffers, output_buffers, inputs, outputs);
  };
  auto _fn_0 = func::make(std::move(_replica_fn_5), {{padded_intm, {{((x + -1)), ((x + 1))}, {((y + -1)), ((y + 1))}}}}, {{out, {x, y}}}, {});
  _fn_0.loops({{y, 1, loop::serial}});
  auto p = build_pipeline(ctx, {}, {in}, {out}, {}, {});
  return p;
};
// END define_replica_pipeline() output
  // clang-format on

  const int W = 20;
  const int H = 30;
  buffer<short, 2> in_buf({W, H});
  buffer<short, 2> out_buf({W, H});

  init_random(in_buf);
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};

  eval_context eval_ctx;
  ASSERT_EQ(0, p().evaluate(inputs, outputs, eval_ctx));
}

TEST(replica, diamond_stencils) {
  // clang-format off
// BEGIN define_replica_pipeline() output
auto p = []() -> ::slinky::pipeline {
  using std::abs, std::min, std::max;
  node_context ctx;
  auto in1 = buffer_expr::make(ctx, "in1", /*rank=*/2, /*elem_size=*/2);
  auto out = buffer_expr::make(ctx, "out", /*rank=*/2, /*elem_size=*/2);
  auto x = var(ctx, "x");
  auto y = var(ctx, "y");
  auto intm3 = buffer_expr::make(ctx, "intm3", /*rank=*/2, /*elem_size=*/2);
  auto intm2 = buffer_expr::make(ctx, "intm2", /*rank=*/2, /*elem_size=*/2);
  auto _replica_fn_3 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* input_buffers[] = {&i0};
    const buffer<void>* output_buffers[] = {&o0};
    const func::input inputs[] = {{in1, {point(x), point(y)}}};
    const std::vector<var> outputs[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(input_buffers, output_buffers, inputs, outputs);
  };
  auto _fn_2 = func::make(std::move(_replica_fn_3), {{in1, {point(x), point(y)}}}, {{intm2, {x, y}}}, {});
  auto _replica_fn_4 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* input_buffers[] = {&i0};
    const buffer<void>* output_buffers[] = {&o0};
    const func::input inputs[] = {{intm2, {{((x + -1)), ((x + 1))}, {((y + -1)), ((y + 1))}}}};
    const std::vector<var> outputs[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(input_buffers, output_buffers, inputs, outputs);
  };
  auto _fn_1 = func::make(std::move(_replica_fn_4), {{intm2, {{((x + -1)), ((x + 1))}, {((y + -1)), ((y + 1))}}}}, {{intm3, {x, y}}}, {});
  auto intm4 = buffer_expr::make(ctx, "intm4", /*rank=*/2, /*elem_size=*/2);
  auto _replica_fn_6 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* input_buffers[] = {&i0};
    const buffer<void>* output_buffers[] = {&o0};
    const func::input inputs[] = {{intm2, {{((x + -2)), ((x + 2))}, {((y + -2)), ((y + 2))}}}};
    const std::vector<var> outputs[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(input_buffers, output_buffers, inputs, outputs);
  };
  auto _fn_5 = func::make(std::move(_replica_fn_6), {{intm2, {{((x + -2)), ((x + 2))}, {((y + -2)), ((y + 2))}}}}, {{intm4, {x, y}}}, {});
  auto _replica_fn_7 = [=](const buffer<const void>& i0, const buffer<const void>& i1, const buffer<void>& o0) -> index_t {
    const buffer<const void>* input_buffers[] = {&i0, &i1};
    const buffer<void>* output_buffers[] = {&o0};
    const func::input inputs[] = {{intm3, {point(x), point(y)}}, {intm4, {point(x), point(y)}}};
    const std::vector<var> outputs[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(input_buffers, output_buffers, inputs, outputs);
  };
  auto _fn_0 = func::make(std::move(_replica_fn_7), {{intm3, {point(x), point(y)}}, {intm4, {point(x), point(y)}}}, {{out, {x, y}}}, {});
  _fn_0.loops({{y, 1, loop::serial}});
  auto p = build_pipeline(ctx, {}, {in1}, {out}, {}, {});
  return p;
};
// END define_replica_pipeline() output
  // clang-format on

  const int W = 20;
  const int H = 10;
  buffer<short, 2> in_buf({W + 4, H + 4});
  in_buf.translate(-2, -2);
  buffer<short, 2> out_buf({W, H});

  init_random(in_buf);
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};

  eval_context eval_ctx;
  ASSERT_EQ(0, p().evaluate(inputs, outputs, eval_ctx));
}

TEST(replica, fork) {
  // clang-format off
// BEGIN define_replica_pipeline() output
auto p = []() -> ::slinky::pipeline {
  using std::abs, std::min, std::max;
  node_context ctx;
  auto in1 = buffer_expr::make(ctx, "in1", /*rank=*/2, /*elem_size=*/2);
  auto out1 = buffer_expr::make(ctx, "out1", /*rank=*/2, /*elem_size=*/2);
  auto x = var(ctx, "x");
  auto y = var(ctx, "y");
  auto intm2 = buffer_expr::make(ctx, "intm2", /*rank=*/2, /*elem_size=*/2);
  auto _replica_fn_2 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* input_buffers[] = {&i0};
    const buffer<void>* output_buffers[] = {&o0};
    const func::input inputs[] = {{in1, {point(x), point(y)}}};
    const std::vector<var> outputs[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(input_buffers, output_buffers, inputs, outputs);
  };
  auto _fn_1 = func::make(std::move(_replica_fn_2), {{in1, {point(x), point(y)}}}, {{intm2, {x, y}}}, {});
  auto _replica_fn_3 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* input_buffers[] = {&i0};
    const buffer<void>* output_buffers[] = {&o0};
    const func::input inputs[] = {{intm2, {point(x), point(y)}}};
    const std::vector<var> outputs[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(input_buffers, output_buffers, inputs, outputs);
  };
  auto _fn_0 = func::make(std::move(_replica_fn_3), {{intm2, {point(x), point(y)}}}, {{out1, {x, y}}}, {});
  auto out2 = buffer_expr::make(ctx, "out2", /*rank=*/2, /*elem_size=*/2);
  auto _replica_fn_5 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* input_buffers[] = {&i0};
    const buffer<void>* output_buffers[] = {&o0};
    const func::input inputs[] = {{intm2, {point(x), point(y)}}};
    const std::vector<var> outputs[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(input_buffers, output_buffers, inputs, outputs);
  };
  auto _fn_4 = func::make(std::move(_replica_fn_5), {{intm2, {point(x), point(y)}}}, {{out2, {x, y}}}, {});
  _fn_4.loops({{y, 1, loop::serial}});
  auto p = build_pipeline(ctx, {}, {in1}, {out1, out2}, {}, {});
  return p;
};
// END define_replica_pipeline() output
  // clang-format on

  const int W = 32;
  const int H = 32;
  buffer<short, 2> in_buf({W, H});
  buffer<short, 2> intm3_buf({W, H});
  buffer<short, 2> intm4_buf({W, H});

  init_random(in_buf);
  intm3_buf.allocate();
  intm4_buf.allocate();

  // Not having span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&intm3_buf, &intm4_buf};

  eval_context eval_ctx;
  ASSERT_EQ(0, p().evaluate(inputs, outputs, eval_ctx));
}

}  // namespace slinky
