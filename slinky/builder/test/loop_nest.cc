#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include "slinky/builder/pipeline.h"
#include "slinky/builder/test/context.h"
#include "slinky/builder/test/funcs.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/pipeline.h"
#include "slinky/runtime/stmt.h"

namespace slinky {
namespace {

TEST(loop_nest, tree_queries) {
  node_context ctx;
  auto a = buffer_expr::make(ctx, "a", 2, sizeof(int));
  auto b = buffer_expr::make(ctx, "b", 2, sizeof(int));
  var x(ctx, "x");
  var y(ctx, "y");

  loop_nest loops;
  EXPECT_EQ(loops.size(), 1u);
  EXPECT_TRUE(loops.root().is_root());

  loop_ref l1 = loops.add_loop({a, y}, 4);
  loop_ref l2 = loops.add_loop(l1, {a, x}, 8);
  loop_ref l3 = loops.add_loop(l1, {b, x}, 8);
  loop_ref l4 = loops.add_loop({b, y}, 2);

  EXPECT_EQ(loops.size(), 5u);
  EXPECT_TRUE(loops.is_ancestor(loops.root(), l2));
  EXPECT_TRUE(loops.is_ancestor(l1, l2));
  EXPECT_TRUE(loops.is_ancestor(l2, l2));
  EXPECT_FALSE(loops.is_ancestor(l2, l1));
  EXPECT_FALSE(loops.is_ancestor(l2, l3));

  EXPECT_EQ(loops.lca(l2, l3), l1);
  EXPECT_EQ(loops.lca(l1, l2), l1);
  EXPECT_EQ(loops.lca(l2, l4), loops.root());

  loops.place(a, l1);
  loops.store(b, loops.root());
  EXPECT_EQ(loops.placed_at(a), l1);
  EXPECT_EQ(loops.stored_at(b), loops.root());
  EXPECT_EQ(loops.placed_at(b), std::nullopt);
  EXPECT_EQ(loops.placed_at(a->sym()), l1);
  EXPECT_EQ(loops.stored_at(b->sym()), loops.root());

  loops[l1].step = 16;
  loops[l1].max_workers = loop::parallel;
  EXPECT_EQ(loops[l1].dims[0].buf, a);
  EXPECT_EQ(loops[l1].dims[0].dim, y);
  EXPECT_TRUE(is_constant(loops[l1].step, 16));
  EXPECT_TRUE(is_constant(loops[l1].max_workers, loop::parallel));
}

TEST(loop_nest, fused_chain_and_hoisted_place) {
  node_context ctx;
  auto x = buffer_expr::make(ctx, "x", 2, sizeof(int));
  auto w = buffer_expr::make(ctx, "w", 1, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  auto w_packed = buffer_expr::make(ctx, "w_packed", 1, sizeof(int));
  auto scaled = buffer_expr::make(ctx, "scaled", 2, sizeof(int));

  var d0(ctx, "d0");
  var d1(ctx, "d1");

  int pack_calls = 0;
  int scale_calls = 0;
  int bias_calls = 0;

  func pack_w = func::make(
      [&](const buffer<const int>& src, const buffer<int>& dst) -> index_t {
        ++pack_calls;
        return multiply_2<int>(src, dst);
      },
      {{w, {point(d0)}}}, {{w_packed, {d0}}});

  func scale = func::make(
      [&](const buffer<const int>& in_x, const buffer<const int>& in_w,
          const buffer<int>& dst) -> index_t {
        ++scale_calls;
        return multiply<int>(in_x, in_w, dst);
      },
      {{x, {point(d0), point(d1)}}, {w_packed, {point(d0)}}},
      {{scaled, {d0, d1}}});

  func add_one = func::make(
      [&](const buffer<const int>& src, const buffer<int>& dst) -> index_t {
        ++bias_calls;
        return add_1<int>(src, dst);
      },
      {{scaled, {point(d0), point(d1)}}}, {{out, {d0, d1}}});

  const int N = 12;
  const int M = 10;
  const int split_n = 4;
  const int split_m = 5;

  loop_nest loops;
  loop_ref ln = loops.add_loop({out, d0}, split_n);
  loops.add_loop(ln, {out, d1}, split_m);
  loops.place(w_packed, ln);

  pipeline p = build_pipeline(ctx, {x, w}, {out}, loops);

  buffer<int, 2> x_buf({N, M});
  buffer<int, 1> w_buf({N});
  buffer<int, 2> out_buf({N, M});
  init_random(x_buf);
  init_random(w_buf);
  out_buf.allocate();

  const raw_buffer* inputs[] = {&x_buf, &w_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  EXPECT_EQ(pack_calls, N / split_n);
  EXPECT_EQ(scale_calls, (N / split_n) * (M / split_m));
  EXPECT_EQ(bias_calls, (N / split_n) * (M / split_m));

  for (int j = 0; j < M; ++j) {
    for (int i = 0; i < N; ++i) {
      ASSERT_EQ(out_buf(i, j), x_buf(i, j) * (w_buf(i) * 2) + 1);
    }
  }

  std::vector<index_t> expected_allocs;
  for (int i = 0; i < N / split_n; ++i) {
    expected_allocs.push_back(split_n * sizeof(int));
    for (int j = 0; j < M / split_m; ++j) {
      expected_allocs.push_back(split_n * split_m * sizeof(int));
    }
  }
  ASSERT_EQ(eval_ctx.heap.allocs, expected_allocs);
}

TEST(loop_nest, multi_level_split_same_dim) {
  for (int outer_workers : {int{loop::serial}, int{loop::parallel}}) {
    node_context ctx;
    auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
    auto mid1 = buffer_expr::make(ctx, "mid1", 2, sizeof(int));
    auto mid2 = buffer_expr::make(ctx, "mid2", 2, sizeof(int));
    auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

    var x(ctx, "x");
    var y(ctx, "y");

    std::atomic<int> mid1_calls = 0;
    std::atomic<int> mid2_calls = 0;
    std::atomic<int> out_calls = 0;
    func f_mid1 = func::make(
        [&](const buffer<const int>& src, const buffer<int>& dst) -> index_t {
          ++mid1_calls;
          return multiply_2<int>(src, dst);
        },
        {{in, {point(x), point(y)}}}, {{mid1, {x, y}}});
    func f_mid2 = func::make(
        [&](const buffer<const int>& src, const buffer<int>& dst) -> index_t {
          ++mid2_calls;
          return add_1<int>(src, dst);
        },
        {{mid1, {point(x), point(y)}}}, {{mid2, {x, y}}});
    func f_out = func::make(
        [&](const buffer<const int>& src, const buffer<int>& dst) -> index_t {
          ++out_calls;
          return multiply_2<int>(src, dst);
        },
        {{mid2, {point(x), point(y)}}}, {{out, {x, y}}});

    const int outer_step = 12;
    const int inner_step = 3;

    loop_nest loops;
    loop_ref ly_outer = loops.add_loop({out, y}, outer_step, outer_workers);
    loops.add_loop(ly_outer, {out, y}, inner_step);
    loops.place(mid1, ly_outer);

    pipeline p = build_pipeline(ctx, {in}, {out}, loops);

    const int W = 16;
    const int H = 48;
    buffer<int, 2> in_buf({W, H});
    buffer<int, 2> out_buf({W, H});
    init_random(in_buf);
    out_buf.allocate();

    const raw_buffer* inputs[] = {&in_buf};
    const raw_buffer* outputs[] = {&out_buf};
    test_context eval_ctx;
    p.evaluate(inputs, outputs, eval_ctx);

    EXPECT_EQ(mid1_calls.load(), H / outer_step);
    EXPECT_EQ(mid2_calls.load(), H / inner_step);
    EXPECT_EQ(out_calls.load(), H / inner_step);
    for (int j = 0; j < H; ++j) {
      for (int i = 0; i < W; ++i) {
        ASSERT_EQ(out_buf(i, j), (in_buf(i, j) * 2 + 1) * 2);
      }
    }
    std::vector<index_t> expected_allocs;
    for (int o = 0; o < H / outer_step; ++o) {
      expected_allocs.push_back(W * outer_step * sizeof(int));
      for (int sub = 0; sub < outer_step / inner_step; ++sub) {
        expected_allocs.push_back(W * inner_step * sizeof(int));
      }
    }
    if (outer_workers == loop::serial) {
      ASSERT_EQ(eval_ctx.heap.allocs, expected_allocs);
    } else {
      ASSERT_THAT(eval_ctx.heap.allocs,
                  testing::UnorderedElementsAreArray(expected_allocs));
    }
  }
}

TEST(loop_nest, outer_split_k_reduction) {
  node_context ctx;
  auto a = buffer_expr::make(ctx, "a", 2, sizeof(int));
  auto b = buffer_expr::make(ctx, "b", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  const int M = 6;
  const int N = 8;
  const int K = 12;
  const int split_k = 4;
  const int split_n = 4;
  const int split_m = 3;

  auto b_scaled = buffer_expr::make(ctx, "b_scaled", 2, sizeof(int));
  auto dot_acc = buffer_expr::make(ctx, "dot_acc", 2, sizeof(int));
  auto k_buf = buffer_expr::make(ctx, "k_buf", 1, 0);
  k_buf->dim(0).bounds = min_extent(0, K);
  k_buf->dim(0).stride = 0;
  k_buf->dim(0).fold_factor = dim::unfolded;

  var n(ctx, "n");
  var m(ctx, "m");
  var k(ctx, "k");

  func scale_b = func::make(multiply_2<int>, {{b, {point(n), point(k)}}},
                            {{b_scaled, {n, k}}});

  func dot = func::make(
      [](const buffer<const int>& in_a, const buffer<const int>& in_b,
         const buffer<int>& acc, const buffer<void>& k_domain) -> index_t {
        const index_t k_begin = k_domain.dim(0).begin();
        const index_t k_end = k_domain.dim(0).end();
        for (index_t mi = acc.dim(1).begin(); mi < acc.dim(1).end(); ++mi) {
          for (index_t ni = acc.dim(0).begin(); ni < acc.dim(0).end(); ++ni) {
            int sum = (k_begin == 0) ? 0 : acc(ni, mi);
            for (index_t ki = k_begin; ki < k_end; ++ki) {
              sum += in_a(ki, mi) * in_b(ni, ki);
            }
            acc(ni, mi) = sum;
          }
        }
        return 0;
      },
      {{a, {point(k), point(m)}}, {b_scaled, {point(n), point(k)}}},
      {{dot_acc, {n, m}}, {k_buf, {k}}});

  func epilogue = func::make(add_1<int>, {{dot_acc, {point(n), point(m)}}},
                             {{out, {n, m}}});

  loop_nest loops;
  loop_ref lk = loops.add_loop({dot_acc, k}, split_k);
  loop_ref ln = loops.add_loop(lk, {dot_acc, n}, split_n);
  loops.add_loop(ln, {dot_acc, m}, split_m);
  loops.place(b_scaled, ln);

  pipeline p = build_pipeline(ctx, {a, b}, {out}, loops);

  buffer<int, 2> a_buf({K, M});
  buffer<int, 2> b_buf({N, K});
  buffer<int, 2> out_buf({N, M});
  init_random(a_buf);
  init_random(b_buf);
  out_buf.allocate();

  const raw_buffer* inputs[] = {&a_buf, &b_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int mi = 0; mi < M; ++mi) {
    for (int ni = 0; ni < N; ++ni) {
      int expected = 0;
      for (int ki = 0; ki < K; ++ki) {
        expected += a_buf(ki, mi) * (b_buf(ni, ki) * 2);
      }
      ASSERT_EQ(out_buf(ni, mi), expected + 1);
    }
  }

  // `dot_acc` allocates once at root (N * M) to accumulate across `lk`;
  // `b_scaled` allocates at `ln` (split_n * split_k).
  std::vector<index_t> expected_allocs = {N * M * sizeof(int)};
  expected_allocs.insert(expected_allocs.end(), (K / split_k) * (N / split_n),
                         split_n * split_k * sizeof(int));
  ASSERT_EQ(eval_ctx.heap.allocs, expected_allocs);
}

TEST(loop_nest, multi_output_co_cropping) {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  auto row_min = buffer_expr::make(ctx, "row_min", 1, sizeof(int));
  auto row_max = buffer_expr::make(ctx, "row_max", 1, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");
  interval_expr all_x = out->dim(0).bounds;

  func stats = func::make(
      [](const buffer<const int>& src, const buffer<int>& mins,
         const buffer<int>& maxs) -> index_t {
        EXPECT_EQ(mins.dim(0).min(), maxs.dim(0).min());
        EXPECT_EQ(mins.dim(0).max(), maxs.dim(0).max());
        for (index_t j = mins.dim(0).begin(); j < mins.dim(0).end(); ++j) {
          int mn = std::numeric_limits<int>::max();
          int mx = std::numeric_limits<int>::min();
          for (index_t i = src.dim(0).begin(); i < src.dim(0).end(); ++i) {
            mn = std::min(mn, src(i, j));
            mx = std::max(mx, src(i, j));
          }
          mins(j) = mn;
          maxs(j) = mx;
        }
        return 0;
      },
      {{in, {all_x, point(y)}}}, {{row_min, {y}}, {row_max, {y}}});

  func normalize_range = func::make(
      [](const buffer<const int>& src, const buffer<const int>& mins,
         const buffer<const int>& maxs, const buffer<int>& dst) -> index_t {
        for (index_t j = dst.dim(1).begin(); j < dst.dim(1).end(); ++j) {
          for (index_t i = dst.dim(0).begin(); i < dst.dim(0).end(); ++i) {
            dst(i, j) = (src(i, j) - mins(j)) + (maxs(j) - mins(j));
          }
        }
        return 0;
      },
      {{in, {point(x), point(y)}},
       {row_min, {point(y)}},
       {row_max, {point(y)}}},
      {{out, {x, y}}});

  // Tile `{row_min, y}` directly: `{row_max, y}` shares `y` and co-crops.
  loop_nest loops;
  loop_ref ly = loops.add_loop({out, y}, 4);
  loops.add_loop(ly, {row_min, y}, 2);
  loops.add_loop(ly, {out, x}, 5);

  pipeline p = build_pipeline(ctx, {in}, {out}, loops);

  const int W = 10;
  const int H = 8;
  buffer<int, 2> in_buf({W, H});
  buffer<int, 2> out_buf({W, H});
  init_random(in_buf);
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int j = 0; j < H; ++j) {
    int mn = std::numeric_limits<int>::max();
    int mx = std::numeric_limits<int>::min();
    for (int i = 0; i < W; ++i) {
      mn = std::min(mn, in_buf(i, j));
      mx = std::max(mx, in_buf(i, j));
    }
    for (int i = 0; i < W; ++i) {
      ASSERT_EQ(out_buf(i, j), (in_buf(i, j) - mn) + (mx - mn));
    }
  }
}

TEST(loop_nest, multi_output_nested_loops_across_siblings) {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
  auto mid = buffer_expr::make(ctx, "mid", 2, sizeof(int));
  auto a = buffer_expr::make(ctx, "a", 2, sizeof(int));
  auto b = buffer_expr::make(ctx, "b", 2, sizeof(int));

  var x(ctx, "x");
  var y(ctx, "y");

  const int W = 8;
  const int H = 6;
  const int split_x = 4;
  const int split_y = 2;

  func prep =
      func::make(add_1<int>, {{in, {point(x), point(y)}}}, {{mid, {x, y}}});

  int calls = 0;
  func both = func::make(
      [&](const buffer<const int>& src, const buffer<int>& out_a,
          const buffer<int>& out_b) -> index_t {
        ++calls;
        EXPECT_EQ(out_a.dim(0).min(), out_b.dim(0).min());
        EXPECT_EQ(out_a.dim(0).max(), out_b.dim(0).max());
        EXPECT_EQ(out_a.dim(1).min(), out_b.dim(1).min());
        EXPECT_EQ(out_a.dim(1).max(), out_b.dim(1).max());
        EXPECT_EQ(out_a.dim(0).extent(), split_x);
        EXPECT_EQ(out_a.dim(1).extent(), split_y);
        for (index_t j = out_a.dim(1).begin(); j < out_a.dim(1).end(); ++j) {
          for (index_t i = out_a.dim(0).begin(); i < out_a.dim(0).end(); ++i) {
            out_a(i, j) = src(i, j) + 1;
            out_b(i, j) = src(i, j) * 2;
          }
        }
        return 0;
      },
      {{mid, {point(x), point(y)}}}, {{a, {x, y}}, {b, {x, y}}});

  // Outer loop tiles `{a, x}` and nested inner loop tiles `{b, y}`: both loops
  // co-crop `a` and `b` and `both` executes at the inner loop.
  loop_nest loops;
  loop_ref lx = loops.add_loop({a, x}, split_x);
  loops.add_loop(lx, {b, y}, split_y);

  pipeline p = build_pipeline(ctx, {in}, {a, b}, loops);

  buffer<int, 2> in_buf({W, H});
  buffer<int, 2> a_buf({W, H});
  buffer<int, 2> b_buf({W, H});
  init_random(in_buf);
  a_buf.allocate();
  b_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&a_buf, &b_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  EXPECT_EQ(calls, (W / split_x) * (H / split_y));
  ASSERT_EQ(eval_ctx.heap.allocs,
            std::vector<index_t>((W / split_x) * (H / split_y),
                                 split_x * split_y * sizeof(int)));
  for (int j = 0; j < H; ++j) {
    for (int i = 0; i < W; ++i) {
      ASSERT_EQ(a_buf(i, j), in_buf(i, j) + 2);
      ASSERT_EQ(b_buf(i, j), (in_buf(i, j) + 1) * 2);
    }
  }
}

TEST(loop_nest, multi_pass_softmax_and_sibling_place) {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", 2, sizeof(float));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(float));

  auto biased_in = buffer_expr::make(ctx, "biased_in", 2, sizeof(float));
  auto softmax_in = buffer_expr::make(ctx, "softmax_in", 2, sizeof(float));
  auto max_in = buffer_expr::make(ctx, "max_in", 1, sizeof(float));
  auto exp_in = buffer_expr::make(ctx, "exp_in", 2, sizeof(float));
  auto sum_exp_in = buffer_expr::make(ctx, "sum_exp_in", 1, sizeof(float));

  var c(ctx, "c");
  var b(ctx, "b");
  interval_expr all_c = out->dim(0).bounds;

  func bias = func::make(add_1<float>, {{in, {point(c), point(b)}}},
                         {{biased_in, {c, b}}});
  func pass0 =
      func::make(multiply_2<float>, {{biased_in, {point(c), point(b)}}},
                 {{softmax_in, {c, b}}});
  func pass1 = func::make(
      [](const buffer<const float>& x, const buffer<float>& m) -> index_t {
        for (index_t bi = m.dim(0).begin(); bi < m.dim(0).end(); ++bi) {
          m(bi) = -std::numeric_limits<float>::infinity();
          for (index_t ci = x.dim(0).begin(); ci < x.dim(0).end(); ++ci) {
            m(bi) = std::max(m(bi), x(ci, bi));
          }
        }
        return 0;
      },
      {{softmax_in, {all_c, point(b)}}}, {{max_in, {b}}});
  func pass2 = func::make(
      [](const buffer<const float>& x, const buffer<const float>& m,
         const buffer<float>& e, const buffer<float>& s) -> index_t {
        for (index_t bi = e.dim(1).begin(); bi < e.dim(1).end(); ++bi) {
          s(bi) = 0.0f;
          for (index_t ci = e.dim(0).begin(); ci < e.dim(0).end(); ++ci) {
            e(ci, bi) = std::exp(x(ci, bi) - m(bi));
            s(bi) += e(ci, bi);
          }
        }
        return 0;
      },
      {{softmax_in, {all_c, point(b)}}, {max_in, {point(b)}}},
      {{exp_in, {c, b}}, {sum_exp_in, {b}}});
  func pass3 = func::make(
      [](const buffer<const float>& e, const buffer<const float>& b_in,
         const buffer<const float>& s, const buffer<float>& y) -> index_t {
        for (index_t bi = y.dim(1).begin(); bi < y.dim(1).end(); ++bi) {
          for (index_t ci = y.dim(0).begin(); ci < y.dim(0).end(); ++ci) {
            y(ci, bi) = e(ci, bi) / s(bi) + b_in(ci, bi);
          }
        }
        return 0;
      },
      {{exp_in, {point(c), point(b)}},
       {biased_in, {point(c), point(b)}},
       {sum_exp_in, {point(b)}}},
      {{out, {c, b}}});

  const int D = 16;
  const int B = 12;
  const int split_b = 4;
  const int split_c = 4;

  // `biased_in` is explicitly placed inside `lc_in` even though it is also
  // consumed in sibling `lc_out`: Slinky stores `biased_in` at `LCA(lc_in,
  // lc_out) == lb` and orders `lc_in -> pass1 -> pass2 -> lc_out`.
  loop_nest loops;
  loop_ref lb = loops.add_loop({out, b}, split_b);
  loop_ref lc_in = loops.add_loop(lb, {softmax_in, c}, split_c);
  loops.add_loop(lb, {out, c}, split_c);
  loops.place(biased_in, lc_in);
  loops.place(exp_in, lb);

  pipeline p = build_pipeline(ctx, {in}, {out}, loops);

  buffer<float, 2> in_buf({D, B});
  buffer<float, 2> out_buf({D, B});
  in_buf.allocate();
  int seed_idx = 0;
  for_each_element(
      [&seed_idx](float* v) {
        *v = static_cast<float>(((seed_idx++ * 5) % 9) - 4) * 0.25f;
      },
      in_buf);
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int bi = 0; bi < B; ++bi) {
    float mx = -std::numeric_limits<float>::infinity();
    for (int ci = 0; ci < D; ++ci) {
      mx = std::max(mx, (in_buf(ci, bi) + 1.0f) * 2.0f);
    }
    float sum = 0.0f;
    for (int ci = 0; ci < D; ++ci) {
      sum += std::exp((in_buf(ci, bi) + 1.0f) * 2.0f - mx);
    }
    for (int ci = 0; ci < D; ++ci) {
      float biased = in_buf(ci, bi) + 1.0f;
      float expected = std::exp(biased * 2.0f - mx) / sum + biased;
      ASSERT_NEAR(out_buf(ci, bi), expected, 1e-5f);
    }
  }

  ASSERT_EQ(eval_ctx.heap.allocs.size(), 5 * (B / split_b));
  EXPECT_EQ(std::count(eval_ctx.heap.allocs.begin(), eval_ctx.heap.allocs.end(),
                       D * split_b * sizeof(float)),
            3 * (B / split_b));
  EXPECT_EQ(std::count(eval_ctx.heap.allocs.begin(), eval_ctx.heap.allocs.end(),
                       split_b * sizeof(float)),
            2 * (B / split_b));
}

TEST(loop_nest, co_iterated_multi_buffer_loop) {
  node_context ctx;
  auto x = buffer_expr::make(ctx, "x", 2, sizeof(int));
  auto q_out = buffer_expr::make(ctx, "q_out", 2, sizeof(int));
  auto k_out = buffer_expr::make(ctx, "k_out", 2, sizeof(int));
  auto v_out = buffer_expr::make(ctx, "v_out", 2, sizeof(int));

  auto x_norm = buffer_expr::make(ctx, "x_norm", 2, sizeof(int));

  var d0(ctx, "d0");
  var d1(ctx, "d1");

  int norm_calls = 0;
  func norm = func::make(
      [&](const buffer<const int>& src, const buffer<int>& dst) -> index_t {
        ++norm_calls;
        return add_1<int>(src, dst);
      },
      {{x, {point(d0), point(d1)}}}, {{x_norm, {d0, d1}}});
  func fq = func::make(multiply_2<int>, {{x_norm, {point(d0), point(d1)}}},
                       {{q_out, {d0, d1}}});
  func fk = func::make(square<int>, {{x_norm, {point(d0), point(d1)}}},
                       {{k_out, {d0, d1}}});
  func fv = func::make(add_1<int>, {{k_out, {point(d0), point(d1)}}},
                       {{v_out, {d0, d1}}});

  const int N = 12;
  const int M = 9;
  const int split_m = 3;

  loop_nest loops;
  loop_ref lm = loops.add_loop({{q_out, d1}, {k_out, d1}}, split_m);
  loops[lm].dims.push_back({v_out, d1});
  loops.add_loop(lm, {q_out, d0}, 4);
  loops.add_loop(lm, {{k_out, d0}, {v_out, d0}}, 6);

  pipeline p = build_pipeline(ctx, {x}, {q_out, k_out, v_out}, loops);

  buffer<int, 2> x_buf({N, M});
  buffer<int, 2> q_buf({N, M});
  buffer<int, 2> k_buf({N, M});
  buffer<int, 2> v_buf({N, M});
  init_random(x_buf);
  q_buf.allocate();
  k_buf.allocate();
  v_buf.allocate();

  const raw_buffer* inputs[] = {&x_buf};
  const raw_buffer* outputs[] = {&q_buf, &k_buf, &v_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  // `x_norm` defaults to LCA(ln_q, ln_kv) == `lm`, so it runs once per `m` tile
  // and allocates `N * split_m`.
  EXPECT_EQ(norm_calls, M / split_m);
  ASSERT_EQ(eval_ctx.heap.allocs,
            std::vector<index_t>(M / split_m, N * split_m * sizeof(int)));

  for (int j = 0; j < M; ++j) {
    for (int i = 0; i < N; ++i) {
      const int xn = x_buf(i, j) + 1;
      ASSERT_EQ(q_buf(i, j), xn * 2);
      ASSERT_EQ(k_buf(i, j), xn * xn);
      ASSERT_EQ(v_buf(i, j), xn * xn + 1);
    }
  }
}

TEST(loop_nest, sliding_window_explicit_store_root) {
  for (int max_workers : {int{loop::serial}, int{loop::parallel}}) {
    node_context ctx;
    auto in = buffer_expr::make(ctx, "in", 2, sizeof(int));
    auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));
    auto intm = buffer_expr::make(ctx, "intm", 2, sizeof(int));

    var x(ctx, "x");
    var y(ctx, "y");

    func add =
        func::make(add_1<int>, {{in, {point(x), point(y)}}}, {{intm, {x, y}}});
    func stencil = func::make(
        sum3x3<int>, {{intm, {bounds(-1, 1) + x, bounds(-1, 1) + y}}},
        {{out, {x, y}}});

    loop_nest loops;
    loops.add_loop({out, y}, 1, max_workers);
    loops.store(intm, loops.root());

    pipeline p = build_pipeline(ctx, {in}, {out}, loops);

    const int W = 20;
    const int H = 30;
    buffer<int, 2> in_buf({W + 2, H + 2});
    in_buf.translate(-1, -1);
    buffer<int, 2> out_buf({W, H});
    init_random(in_buf);
    out_buf.allocate();

    const raw_buffer* inputs[] = {&in_buf};
    const raw_buffer* outputs[] = {&out_buf};
    test_context eval_ctx;
    p.evaluate(inputs, outputs, eval_ctx);

    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        int correct = 0;
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            correct += in_buf(x + dx, y + dy) + 1;
          }
        }
        ASSERT_EQ(correct, out_buf(x, y));
      }
    }

    const int parallel_extra = max_workers != loop::serial ? 1 : 0;
    ASSERT_THAT(eval_ctx.heap.allocs,
                testing::UnorderedElementsAre((W + 2) * (3 + parallel_extra) *
                                              sizeof(int)));
  }
}

TEST(loop_nest, external_output_consumed_in_graph_defaults_to_root) {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", 1, sizeof(int));
  auto mid = buffer_expr::make(ctx, "mid", 1, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 1, sizeof(int));

  var x(ctx, "x");
  int mid_calls = 0;
  func f_mid = func::make(
      [&](const buffer<const int>& src, const buffer<int>& dst) -> index_t {
        ++mid_calls;
        return add_1<int>(src, dst);
      },
      {{in, {point(x)}}}, {{mid, {x}}});
  func f_out = func::make(multiply_2<int>, {{mid, {point(x)}}}, {{out, {x}}});

  loop_nest loops;
  loops.add_loop({out, x}, 4);

  // `mid` is an external pipeline output as well as an input to `f_out`;
  // without co-tiling `mid`, `f_mid` must default to `root()` rather than
  // recomputing the full `mid` buffer on every tile of `out`.
  pipeline p = build_pipeline(ctx, {in}, {mid, out}, loops);

  const int N = 16;
  buffer<int, 1> in_buf({N});
  buffer<int, 1> mid_buf({N});
  buffer<int, 1> out_buf({N});
  init_random(in_buf);
  mid_buf.allocate();
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&mid_buf, &out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  EXPECT_EQ(mid_calls, 1);
  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(mid_buf(i), in_buf(i) + 1);
    ASSERT_EQ(out_buf(i), (in_buf(i) + 1) * 2);
  }
}

TEST(loop_nest, legacy_external_output_consumed_in_graph_with_tiled_producer) {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", 1, sizeof(int));
  auto pre = buffer_expr::make(ctx, "pre", 1, sizeof(int));
  auto mid = buffer_expr::make(ctx, "mid", 1, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 1, sizeof(int));

  var x(ctx, "x");
  int mid_calls = 0;
  func f_pre = func::make(multiply_2<int>, {{in, {point(x)}}}, {{pre, {x}}});
  func f_mid = func::make(
      [&](const buffer<const int>& src, const buffer<int>& dst) -> index_t {
        ++mid_calls;
        return add_1<int>(src, dst);
      },
      {{pre, {point(x)}}}, {{mid, {x}}});
  func f_out = func::make(multiply_2<int>, {{mid, {point(x)}}}, {{out, {x}}});

  f_pre.loops({{x, 2}});
  f_out.loops({{x, 4}});

  pipeline p = build_pipeline(ctx, {in}, {mid, out});

  const int N = 16;
  buffer<int, 1> in_buf({N});
  buffer<int, 1> mid_buf({N});
  buffer<int, 1> out_buf({N});
  init_random(in_buf);
  mid_buf.allocate();
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&mid_buf, &out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  EXPECT_EQ(mid_calls, 1);
  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(mid_buf(i), in_buf(i) * 2 + 1) << "at i=" << i;
    ASSERT_EQ(out_buf(i), (in_buf(i) * 2 + 1) * 2) << "at i=" << i;
  }
}

TEST(loop_nest, child_loop_with_independent_branch_runs_before_consumer) {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", 1, sizeof(int));
  auto mid = buffer_expr::make(ctx, "mid", 1, sizeof(int));
  auto out1 = buffer_expr::make(ctx, "out1", 1, sizeof(int));
  auto out2 = buffer_expr::make(ctx, "out2", 1, sizeof(int));

  var x(ctx, "x");
  interval_expr all_x = out2->dim(0).bounds;

  // Both `f_prod` and `f_extra` execute inside child loop `lx`, while `f_cons`
  // executes at `root()` and consumes `mid` from `f_prod`. With outputs ordered
  // `{out1, out2}`, `f_extra` appears after `f_cons` in topological order; the
  // contracted scope sort must still emit `lx` before `f_cons`.
  func f_prod = func::make(add_1<int>, {{in, {point(x)}}}, {{mid, {x}}});
  func f_cons = func::make(
      [](const buffer<const int>& src, const buffer<int>& dst) -> index_t {
        int sum = 0;
        for (index_t i = src.dim(0).begin(); i < src.dim(0).end(); ++i) {
          sum += src(i);
        }
        dst(0) = sum;
        return 0;
      },
      {{mid, {all_x}}}, {{out1, {x}}});
  func f_extra = func::make(multiply_2<int>, {{in, {point(x)}}}, {{out2, {x}}});

  loop_nest loops;
  loops.add_loop({{mid, x}, {out2, x}}, 4);

  pipeline p = build_pipeline(ctx, {in}, {out1, out2}, loops);

  const int N = 12;
  buffer<int, 1> in_buf({N});
  buffer<int, 1> out1_buf({1});
  buffer<int, 1> out2_buf({N});
  init_random(in_buf);
  out1_buf.allocate();
  out2_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out1_buf, &out2_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  int expected_sum = 0;
  for (int i = 0; i < N; ++i) {
    expected_sum += in_buf(i) + 1;
    ASSERT_EQ(out2_buf(i), in_buf(i) * 2) << "at i=" << i;
  }
  EXPECT_EQ(out1_buf(0), expected_sum);
}

TEST(loop_nest, stack_axis0_tiled_along_non_stacked_dim) {
  node_context ctx;
  auto in = buffer_expr::make(ctx, "in", 1, sizeof(int));
  auto a = buffer_expr::make(ctx, "a", 1, sizeof(int));
  auto b = buffer_expr::make(ctx, "b", 1, sizeof(int));
  auto out = buffer_expr::make(ctx, "out", 2, sizeof(int));

  var s(ctx, "s");
  var y(ctx, "y");

  func fa = func::make(add_1<int>, {{in, {point(y)}}}, {{a, {y}}});
  func fb = func::make(multiply_2<int>, {{in, {point(y)}}}, {{b, {y}}});
  func stacked = func::make_stack({a, b}, {out, {s, y}}, /*dim=*/0);

  const int H = 12;
  const int split_y = 3;

  loop_nest loops;
  loops.add_loop({out, y}, split_y);

  pipeline p = build_pipeline(ctx, {in}, {out}, loops);

  buffer<int, 1> in_buf({H});
  buffer<int, 2> out_buf({2, H});
  init_random(in_buf);
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  test_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int j = 0; j < H; ++j) {
    ASSERT_EQ(out_buf(0, j), in_buf(j) + 1) << "at j=" << j;
    ASSERT_EQ(out_buf(1, j), in_buf(j) * 2) << "at j=" << j;
  }
}

}  // namespace
}  // namespace slinky