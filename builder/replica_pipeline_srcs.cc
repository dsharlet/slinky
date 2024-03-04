#include <functional>

#include "builder/pipeline.h"
#include "builder/replica_pipeline.h"

namespace slinky {

// clang-format off
std::function<pipeline()> multiple_outputs_replica =
// BEGIN define_replica_pipeline() output
[]() -> ::slinky::pipeline {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", sizeof(uint32_t), 3);
  auto sum_x = buffer_expr::make(ctx, "sum_x", sizeof(uint32_t), 2);
  auto y = var(ctx, "y");
  auto z = var(ctx, "z");
  auto _1 = variable::make(in->sym());
  auto _replica_fn_2 = [=](const buffer<const void>& i0, const buffer<void>& o0, const buffer<void>& o1) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0, &o1};
    const func::input fins[] = {{in, {{(buffer_min(_1, 0)), (buffer_max(_1, 0))}, {(buffer_min(_1, 1)), (buffer_max(_1, 1))}, point(z)}}};
    const std::vector<var> fout_dims[] = {{y, z}, {z}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto sum_xy = buffer_expr::make(ctx, "sum_xy", sizeof(uint32_t), 1);
  auto _fn_0 = func::make(std::move(_replica_fn_2), {{in, {{(buffer_min(_1, 0)), (buffer_max(_1, 0))}, {(buffer_min(_1, 1)), (buffer_max(_1, 1))}, point(z)}}}, {{sum_x, {y, z}}, {sum_xy, {z}}});
  _fn_0.loops({{z, 1, loop_mode::serial}});
  auto p = build_pipeline(ctx, {}, {in}, {sum_x, sum_xy}, {});
  return p;
}
// END define_replica_pipeline() output
;
// clang-format on

// clang-format off
std::function<pipeline()> matmul_replica =
// BEGIN define_replica_pipeline() output
[]() -> ::slinky::pipeline {
  node_context ctx;
  auto a = buffer_expr::make(ctx, "a", sizeof(uint32_t), 2);
  a->dim(1).stride = 4;
  auto b = buffer_expr::make(ctx, "b", sizeof(uint32_t), 2);
  b->dim(1).stride = 4;
  auto c = buffer_expr::make(ctx, "c", sizeof(uint32_t), 2);
  c->dim(1).stride = 4;
  auto abc = buffer_expr::make(ctx, "abc", sizeof(uint32_t), 2);
  abc->dim(1).stride = 4;
  auto i = var(ctx, "i");
  auto j = var(ctx, "j");
  auto ab = buffer_expr::make(ctx, "ab", sizeof(uint32_t), 2);
  auto _1 = variable::make(ab->sym());
  ab->dim(0).stride = (((((((buffer_max(_1, 1)) - (buffer_min(_1, 1)))) + 1)) * 4));
  ab->dim(1).stride = 4;
  auto _3 = variable::make(a->sym());
  auto _replica_fn_4 = [=](const buffer<const void>& i0, const buffer<const void>& i1, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0, &i1};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{a, {point(i), {(buffer_min(_3, 1)), (buffer_max(_3, 1))}}}, {b, {{(buffer_min(_3, 1)), (buffer_max(_3, 1))}, point(j)}}};
    const std::vector<var> fout_dims[] = {{i, j}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_2 = func::make(std::move(_replica_fn_4), {{a, {point(i), {(buffer_min(_3, 1)), (buffer_max(_3, 1))}}}, {b, {{(buffer_min(_3, 1)), (buffer_max(_3, 1))}, point(j)}}}, {{ab, {i, j}}});
  auto _5 = variable::make(c->sym());
  auto _replica_fn_6 = [=](const buffer<const void>& i0, const buffer<const void>& i1, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0, &i1};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{ab, {point(i), {(buffer_min(_5, 0)), (buffer_max(_5, 0))}}}, {c, {{(buffer_min(_5, 0)), (buffer_max(_5, 0))}, point(j)}}};
    const std::vector<var> fout_dims[] = {{i, j}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_0 = func::make(std::move(_replica_fn_6), {{ab, {point(i), {(buffer_min(_5, 0)), (buffer_max(_5, 0))}}}, {c, {{(buffer_min(_5, 0)), (buffer_max(_5, 0))}, point(j)}}}, {{abc, {i, j}}});
  _fn_0.loops({{i, 1, loop_mode::serial}});
  auto p = build_pipeline(ctx, {}, {a, b, c}, {abc}, {});
  return p;
}
// END define_replica_pipeline() output
;
// clang-format on

// clang-format off
std::function<pipeline()> pyramid_replica =
// BEGIN define_replica_pipeline() output
[]() -> ::slinky::pipeline {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", sizeof(uint32_t), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(uint32_t), 2);
  auto x = var(ctx, "x");
  auto y = var(ctx, "y");
  auto intm = buffer_expr::make(ctx, "intm", sizeof(uint32_t), 2);
  auto _replica_fn_2 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{in, {{((((x * 2)) + 0)), ((((x * 2)) + 1))}, {((((y * 2)) + 0)), ((((y * 2)) + 1))}}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_1 = func::make(std::move(_replica_fn_2), {{in, {{((((x * 2)) + 0)), ((((x * 2)) + 1))}, {((((y * 2)) + 0)), ((((y * 2)) + 1))}}}}, {{intm, {x, y}}});
  auto _replica_fn_3 = [=](const buffer<const void>& i0, const buffer<const void>& i1, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0, &i1};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{in, {point(x), point(y)}}, {intm, {{((x / 2)), ((((x + 1)) / 2))}, {((y / 2)), ((((y + 1)) / 2))}}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_0 = func::make(std::move(_replica_fn_3), {{in, {point(x), point(y)}}, {intm, {{((x / 2)), ((((x + 1)) / 2))}, {((y / 2)), ((((y + 1)) / 2))}}}}, {{out, {x, y}}});
  _fn_0.loops({{y, 1, loop_mode::serial}});
  auto p = build_pipeline(ctx, {}, {in}, {out}, {});
  return p;
}
// END define_replica_pipeline() output
;
// clang-format on

// clang-format off
std::function<pipeline()> unrelated_replica =
// BEGIN define_replica_pipeline() output
[]() -> ::slinky::pipeline {
  node_context ctx;
  auto in1 = buffer_expr::make(ctx, "in1", sizeof(uint16_t), 2);
  auto in2 = buffer_expr::make(ctx, "in2", sizeof(uint32_t), 1);
  auto out1 = buffer_expr::make(ctx, "out1", sizeof(uint16_t), 2);
  auto x = var(ctx, "x");
  auto y = var(ctx, "y");
  auto intm1 = buffer_expr::make(ctx, "intm1", sizeof(uint16_t), 2);
  auto _replica_fn_2 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{in1, {point(x), point(y)}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_1 = func::make(std::move(_replica_fn_2), {{in1, {point(x), point(y)}}}, {{intm1, {x, y}}});
  auto _replica_fn_3 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{intm1, {{((x + -1)), ((x + 1))}, {((y + -1)), ((y + 1))}}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_0 = func::make(std::move(_replica_fn_3), {{intm1, {{((x + -1)), ((x + 1))}, {((y + -1)), ((y + 1))}}}}, {{out1, {x, y}}});
  _fn_0.loops({{y, 2, loop_mode::serial}});
  auto out2 = buffer_expr::make(ctx, "out2", sizeof(uint32_t), 1);
  auto intm2 = buffer_expr::make(ctx, "intm2", sizeof(uint32_t), 1);
  auto _replica_fn_6 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{in2, {point(x)}}};
    const std::vector<var> fout_dims[] = {{x}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_5 = func::make(std::move(_replica_fn_6), {{in2, {point(x)}}}, {{intm2, {x}}});
  auto _replica_fn_7 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{intm2, {point(x)}}};
    const std::vector<var> fout_dims[] = {{x}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_4 = func::make(std::move(_replica_fn_7), {{intm2, {point(x)}}}, {{out2, {x}}});
  auto p = build_pipeline(ctx, {}, {in1, in2}, {out1, out2}, {});
  return p;
}
// END define_replica_pipeline() output
;
// clang-format on

// clang-format off
std::function<pipeline()> concatenated_replica =
// BEGIN define_replica_pipeline() output
[]() -> ::slinky::pipeline {
  node_context ctx;
  auto in1 = buffer_expr::make(ctx, "in1", sizeof(uint16_t), 2);
  auto in2 = buffer_expr::make(ctx, "in2", sizeof(uint16_t), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(uint16_t), 2);
  auto intm1 = buffer_expr::make(ctx, "intm1", sizeof(uint16_t), 2);
  auto x = var(ctx, "x");
  auto y = var(ctx, "y");
  auto _replica_fn_2 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{in1, {point(x), point(y)}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_1 = func::make(std::move(_replica_fn_2), {{in1, {point(x), point(y)}}}, {{intm1, {x, y}}});
  auto _3 = variable::make(in1->sym());
  auto intm2 = buffer_expr::make(ctx, "intm2", sizeof(uint16_t), 2);
  auto _replica_fn_5 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{in2, {point(x), point(y)}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_4 = func::make(std::move(_replica_fn_5), {{in2, {point(x), point(y)}}}, {{intm2, {x, y}}});
  auto _6 = variable::make(out->sym());
  auto _fn_0 = func::make_copy({{intm1, {point(x), {((y - 0)), ((y - 0))}}, {point(expr()), {0, (((((((buffer_max(_3, 1)) - (buffer_min(_3, 1)))) + 1)) - 1))}}, {}}, {intm2, {point(x), {((y - (((((buffer_max(_3, 1)) - (buffer_min(_3, 1)))) + 1)))), ((y - (((((buffer_max(_3, 1)) - (buffer_min(_3, 1)))) + 1))))}}, {point(expr()), {(((((buffer_max(_3, 1)) - (buffer_min(_3, 1)))) + 1)), (((((((buffer_max(_6, 1)) - (buffer_min(_6, 1)))) + 1)) - 1))}}, {}}}, {out, {x, y}});
  auto p = build_pipeline(ctx, {}, {in1, in2}, {out}, {.no_alias_buffers = true});
  return p;
}
// END define_replica_pipeline() output
;
// clang-format on

// clang-format off
std::function<pipeline()> stacked_replica =
// BEGIN define_replica_pipeline() output
[]() -> ::slinky::pipeline {
  node_context ctx;
  auto in1 = buffer_expr::make(ctx, "in1", sizeof(uint16_t), 2);
  auto in2 = buffer_expr::make(ctx, "in2", sizeof(uint16_t), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(uint16_t), 3);
  auto intm1 = buffer_expr::make(ctx, "intm1", sizeof(uint16_t), 2);
  auto x = var(ctx, "x");
  auto y = var(ctx, "y");
  auto _replica_fn_2 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{in1, {point(x), point(y)}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_1 = func::make(std::move(_replica_fn_2), {{in1, {point(x), point(y)}}}, {{intm1, {x, y}}});
  auto intm2 = buffer_expr::make(ctx, "intm2", sizeof(uint16_t), 2);
  auto _replica_fn_4 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{in2, {point(x), point(y)}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_3 = func::make(std::move(_replica_fn_4), {{in2, {point(x), point(y)}}}, {{intm2, {x, y}}});
  auto _fn_0 = func::make_copy({{intm1, {point(x), point(y)}, {}, {expr(), expr(), 0}}, {intm2, {point(x), point(y)}, {}, {expr(), expr(), 1}}}, {out, {x, y}});
  auto p = build_pipeline(ctx, {}, {in1, in2}, {out}, {});
  return p;
}
// END define_replica_pipeline() output
;
// clang-format on

// clang-format off
std::function<pipeline()> diamond_stencils_replica =
// BEGIN define_replica_pipeline() output
[]() -> ::slinky::pipeline {
  node_context ctx;
  auto in1 = buffer_expr::make(ctx, "in1", sizeof(uint16_t), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(uint16_t), 2);
  auto x = var(ctx, "x");
  auto y = var(ctx, "y");
  auto intm3 = buffer_expr::make(ctx, "intm3", sizeof(uint16_t), 2);
  auto intm2 = buffer_expr::make(ctx, "intm2", sizeof(uint16_t), 2);
  auto _replica_fn_3 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{in1, {point(x), point(y)}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_2 = func::make(std::move(_replica_fn_3), {{in1, {point(x), point(y)}}}, {{intm2, {x, y}}});
  auto _replica_fn_4 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{intm2, {{((x + -1)), ((x + 1))}, {((y + -1)), ((y + 1))}}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_1 = func::make(std::move(_replica_fn_4), {{intm2, {{((x + -1)), ((x + 1))}, {((y + -1)), ((y + 1))}}}}, {{intm3, {x, y}}});
  auto intm4 = buffer_expr::make(ctx, "intm4", sizeof(uint16_t), 2);
  auto _replica_fn_6 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{intm2, {{((x + -2)), ((x + 2))}, {((y + -2)), ((y + 2))}}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_5 = func::make(std::move(_replica_fn_6), {{intm2, {{((x + -2)), ((x + 2))}, {((y + -2)), ((y + 2))}}}}, {{intm4, {x, y}}});
  auto _replica_fn_7 = [=](const buffer<const void>& i0, const buffer<const void>& i1, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0, &i1};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{intm3, {point(x), point(y)}}, {intm4, {point(x), point(y)}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_0 = func::make(std::move(_replica_fn_7), {{intm3, {point(x), point(y)}}, {intm4, {point(x), point(y)}}}, {{out, {x, y}}});
  _fn_0.loops({{y, 1, loop_mode::serial}});
  auto p = build_pipeline(ctx, {}, {in1}, {out}, {});
  return p;
}
// END define_replica_pipeline() output
;
// clang-format on

// clang-format off
std::function<pipeline()> padded_stencil_replica =
// BEGIN define_replica_pipeline() output
[]() -> ::slinky::pipeline {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", sizeof(uint16_t), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(uint16_t), 2);
  auto x = var(ctx, "x");
  auto y = var(ctx, "y");
  auto padded_intm = buffer_expr::make(ctx, "padded_intm", sizeof(uint16_t), 2);
  auto intm = buffer_expr::make(ctx, "intm", sizeof(uint16_t), 2);
  auto _replica_fn_3 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{in, {point(x), point(y)}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_2 = func::make(std::move(_replica_fn_3), {{in, {point(x), point(y)}}}, {{intm, {x, y}}});
  auto _4 = variable::make(in->sym());
  auto _fn_1 = func::make_copy({{intm, {point(x), point(y)}, {{(buffer_min(_4, 0)), (buffer_max(_4, 0))}, {(buffer_min(_4, 1)), (buffer_max(_4, 1))}}, {}}}, {padded_intm, {x, y}});
  _fn_1.compute_root();
  auto _replica_fn_5 = [=](const buffer<const void>& i0, const buffer<void>& o0) -> index_t {
    const buffer<const void>* ins[] = {&i0};
    const buffer<void>* outs[] = {&o0};
    const func::input fins[] = {{padded_intm, {{((x + -1)), ((x + 1))}, {((y + -1)), ((y + 1))}}}};
    const std::vector<var> fout_dims[] = {{x, y}};
    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);
  };
  auto _fn_0 = func::make(std::move(_replica_fn_5), {{padded_intm, {{((x + -1)), ((x + 1))}, {((y + -1)), ((y + 1))}}}}, {{out, {x, y}}});
  _fn_0.loops({{y, 1, loop_mode::serial}});
  auto p = build_pipeline(ctx, {}, {in}, {out}, {});
  return p;
}
// END define_replica_pipeline() output
;
// clang-format on

}  // namespace slinky
