#include <gtest/gtest.h>

#include <cassert>
#include <vector>

#include "builder/pipeline.h"
#include "runtime/expr.h"
#include "runtime/pipeline.h"

namespace slinky {

template <typename T, std::size_t N>
void init_random(buffer<T, N>& x) {
  x.allocate();
  for_each_index(x, [&](auto i) { x(i) = (rand() % 20) - 10; });
}

TEST(copy, trivial_1d) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 1);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 1);

  var x(ctx, "x");
  var dx(ctx, "dx");

  std::vector<char> padding(sizeof(int), 0);

  // This copy should be implemented as a single call to copy.
  func copy = func::make_copy({in, {point(x + dx)}}, {out, {x}}, padding);

  // TODO(https://github.com/dsharlet/slinky/issues/21): The checks on the input bounds are overzealous in this case. We
  // shouldn't need to disable checks.
  pipeline p = build_pipeline(ctx, {dx}, {in}, {out}, build_options{.no_checks = true});

  const int W = 10;
  buffer<int, 1> out_buf({W});
  out_buf.allocate();

  for (int offset : {0, 2, -2}) {
    for (int in_offset : {0, offset}) {
      // Run the pipeline.
      buffer<int, 1> in_buf({W});
      in_buf.translate(in_offset);
      init_random(in_buf);

      const index_t args[] = {offset};
      const raw_buffer* inputs[] = {&in_buf};
      const raw_buffer* outputs[] = {&out_buf};
      eval_context eval_ctx;
      p.evaluate(args, inputs, outputs, eval_ctx);

      for (int x = 0; x < W; ++x) {
        if (in_buf.contains(x + offset)) {
          ASSERT_EQ(out_buf(x), in_buf(x + offset));
        } else {
          ASSERT_EQ(out_buf(x), 0);
        }
      }
    }
  }
}

TEST(copy, trivial_2d) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 2);

  var x(ctx, "x");
  var y(ctx, "y");
  var dy(ctx, "dy");

  std::vector<char> padding(sizeof(int), 0);

  // This copy should be implemented as a single call to copy.
  func copy = func::make_copy({in, {point(x), point(y + dy)}}, {out, {x, y}}, padding);

  // TODO(https://github.com/dsharlet/slinky/issues/21): The checks on the input bounds are overzealous in this case. We
  // shouldn't need to disable checks.
  pipeline p = build_pipeline(ctx, {dy}, {in}, {out}, build_options{.no_checks = true});

  // Run the pipeline.
  const int H = 20;
  const int W = 10;
  buffer<int, 2> out_buf({W, H});
  out_buf.allocate();

  for (int offset : {0, -4, 3}) {
    for (int in_offset : {0, offset}) {
      buffer<int, 2> in_buf({W, H});
      in_buf.translate(0, in_offset);
      init_random(in_buf);

      const index_t args[] = {offset};
      const raw_buffer* inputs[] = {&in_buf};
      const raw_buffer* outputs[] = {&out_buf};
      eval_context eval_ctx;
      p.evaluate(args, inputs, outputs, eval_ctx);

      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          if (in_buf.contains(x, y + offset)) {
            ASSERT_EQ(out_buf(x, y), in_buf(x, y + offset));
          } else {
            ASSERT_EQ(out_buf(x, y), 0);
          }
        }
      }
    }
  }
}

TEST(copy, trivial_3d) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 3);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 3);

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");

  // This copy should be implemented as a single call to copy.
  func copy = func::make_copy({in, {point(x), point(y), point(z)}}, {out, {x, y, z}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int H = 20;
  const int W = 10;
  const int D = 5;
  buffer<int, 3> in_buf({W, H, D});
  init_random(in_buf);

  buffer<int, 3> out_buf({W, H, D});
  out_buf.allocate();
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int z = 0; z < D; ++z) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        ASSERT_EQ(out_buf(x, y, z), in_buf(x, y, z));
      }
    }
  }
}

TEST(copy, flip_x) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 1);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 1);

  var x(ctx, "x");

  func flip = func::make_copy({in, {point(-x)}}, {out, {x}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int W = 10;
  buffer<int, 1> in_buf({W});
  init_random(in_buf);

  buffer<int, 1> out_buf({W});
  out_buf.translate(-W + 1);
  out_buf.allocate();
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int x = 0; x < W; ++x) {
    ASSERT_EQ(out_buf(-x), in_buf(x));
  }
}

TEST(copy, flip_y) {
  for (int split : {-1, 1, 2, 3}) {
    // Make the pipeline
    node_context ctx;

    auto in = buffer_expr::make(ctx, "in", sizeof(int), 3);
    auto out = buffer_expr::make(ctx, "out", sizeof(int), 3);

    var x(ctx, "x");
    var y(ctx, "y");
    var z(ctx, "z");

    // This copy should be implemented with a loop over y, and a call to copy at each y.
    func flip = func::make_copy({in, {point(x), point(-y), point(z)}}, {out, {x, y, z}});

    if (split > 0) {
      flip.loops({{y, split}});
    }

    pipeline p = build_pipeline(ctx, {in}, {out});

    // Run the pipeline.
    const int H = 20;
    const int W = 10;
    const int D = 10;
    buffer<int, 3> in_buf({W, H, D});
    init_random(in_buf);

    buffer<int, 3> out_buf({W, H, D});
    out_buf.translate(0, -H + 1);
    out_buf.allocate();
    const raw_buffer* inputs[] = {&in_buf};
    const raw_buffer* outputs[] = {&out_buf};
    eval_context eval_ctx;
    p.evaluate(inputs, outputs, eval_ctx);

    for (int z = 0; z < D; ++z) {
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          ASSERT_EQ(out_buf(x, -y, z), in_buf(x, y, z));
        }
      }
    }
  }
}

TEST(copy, upsample_y) {
  for (int split : {-1, 1, 2, 4}) {
    // Make the pipeline
    node_context ctx;

    auto in = buffer_expr::make(ctx, "in", sizeof(int), 2);
    auto out = buffer_expr::make(ctx, "out", sizeof(int), 2);

    var x(ctx, "x");
    var y(ctx, "y");

    // This copy should be implemented with a loop over y, and a call to copy at each y.
    // TODO: It could be implemented as a copy for each two lines, with a broadcast in y!
    func upsample = func::make_copy({in, {point(x), point(y / 2)}}, {out, {x, y}});

    if (split > 0) {
      upsample.loops({{y, split}});
    }

    pipeline p = build_pipeline(ctx, {in}, {out});

    // Run the pipeline.
    const int H = 20;
    const int W = 10;
    buffer<int, 2> in_buf({W, H / 2});
    init_random(in_buf);

    buffer<int, 2> out_buf({W, H});
    out_buf.allocate();
    const raw_buffer* inputs[] = {&in_buf};
    const raw_buffer* outputs[] = {&out_buf};
    eval_context eval_ctx;
    p.evaluate(inputs, outputs, eval_ctx);

    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        ASSERT_EQ(out_buf(x, y), in_buf(x, y / 2));
      }
    }
  }
}

TEST(copy, transpose) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 3);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 3);

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");

  // Transpose the first two dimensions.
  // TODO: We should have a special case for transposing 2 dimensions, and this copy should
  // result in a loop plus a call to that.
  func flip = func::make_copy({in, {point(y), point(x), point(z)}}, {out, {x, y, z}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  // Run the pipeline.
  const int H = 20;
  const int W = 10;
  const int D = 10;
  buffer<int, 3> in_buf({H, W, D});
  init_random(in_buf);

  buffer<int, 3> out_buf({W, H, D});
  out_buf.allocate();
  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int z = 0; z < D; ++z) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        ASSERT_EQ(out_buf(x, y, z), in_buf(y, x, z));
      }
    }
  }
}

TEST(copy, broadcast) {
  for (int dim = 0; dim < 3; ++dim) {
    // Make the pipeline
    node_context ctx;

    auto in = buffer_expr::make(ctx, "in", sizeof(int), 3);
    auto out = buffer_expr::make(ctx, "out", sizeof(int), 3);

    var x(ctx, "x");
    var y(ctx, "y");
    var z(ctx, "z");

    box_expr bounds = {point(x), point(y), point(z)};
    bounds[dim] = point(0);
    func crop = func::make_copy({in, bounds}, {out, {x, y, z}});

    pipeline p = build_pipeline(ctx, {in}, {out});

    const int W = 8;
    const int H = 5;
    const int D = 3;

    // Run the pipeline.
    buffer<int, 3> in_buf({W, H, D});
    in_buf.dim(dim).set_extent(1);
    init_random(in_buf);

    // Ask for an output padded in every direction.
    buffer<int, 3> out_buf({W, H, D});
    out_buf.allocate();

    const raw_buffer* inputs[] = {&in_buf};
    const raw_buffer* outputs[] = {&out_buf};
    eval_context eval_ctx;
    p.evaluate(inputs, outputs, eval_ctx);

    for (int z = 0; z < D; ++z) {
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          index_t i[] = {x, y, z};
          i[dim] = 0;
          ASSERT_EQ(out_buf(x, y, z), in_buf(i));
        }
      }
    }
  }
}

TEST(copy, broadcast_sliced) {
  for (int dim = 0; dim < 3; ++dim) {
    // Make the pipeline
    node_context ctx;

    auto in = buffer_expr::make(ctx, "in", sizeof(int), 2);
    auto out = buffer_expr::make(ctx, "out", sizeof(int), 3);

    var x(ctx, "x");
    var y(ctx, "y");
    var z(ctx, "z");

    box_expr bounds = {point(x), point(y), point(z)};
    bounds.erase(bounds.begin() + dim);
    func crop = func::make_copy({in, bounds}, {out, {x, y, z}});

    pipeline p = build_pipeline(ctx, {in}, {out});

    const int W = 8;
    const int H = 5;
    const int D = 3;

    // Run the pipeline.
    std::vector<index_t> in_extents = {W, H, D};
    in_extents.erase(in_extents.begin() + dim);
    buffer<int, 2> in_buf({in_extents[0], in_extents[1]});
    init_random(in_buf);

    // Ask for an output padded in every direction.
    buffer<int, 3> out_buf({W, H, D});
    out_buf.allocate();

    const raw_buffer* inputs[] = {&in_buf};
    const raw_buffer* outputs[] = {&out_buf};
    eval_context eval_ctx;
    p.evaluate(inputs, outputs, eval_ctx);

    for (int z = 0; z < D; ++z) {
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          std::vector<index_t> i = {x, y, z};
          i.erase(i.begin() + dim);
          ASSERT_EQ(out_buf(x, y, z), in_buf(i));
        }
      }
    }
  }
}

TEST(copy, concatenate) {
  // Make the pipeline
  node_context ctx;

  auto in1 = buffer_expr::make(ctx, "in1", sizeof(int), 2);
  auto in2 = buffer_expr::make(ctx, "in2", sizeof(int), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 2);

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");

  func concat =
      func::make_copy({in1, {point(x), point(y)}}, {in2, {point(x), point(y - in1->dim(1).extent())}}, {out, {x, y}});

  // TODO(https://github.com/dsharlet/slinky/issues/21): The checks on the input bounds are overzealous in this case. We
  // shouldn't need to disable checks.
  pipeline p = build_pipeline(ctx, {in1, in2}, {out}, build_options{.no_checks = true});

  const int W = 8;
  const int H1 = 5;
  const int H2 = 4;

  // Run the pipeline.
  buffer<int, 2> in1_buf({W, H1});
  buffer<int, 2> in2_buf({W, H2});
  init_random(in1_buf);
  init_random(in2_buf);

  // Ask for an output padded in every direction.
  buffer<int, 2> out_buf({W, H1 + H2});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in1_buf, &in2_buf};
  const raw_buffer* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  for (int y = 0; y < H1 + H2; ++y) {
    for (int x = 0; x < W; ++x) {
      ASSERT_EQ(out_buf(x, y), y < H1 ? in1_buf(x, y) : in2_buf(x, y - H1));
    }
  }
}

TEST(copy, reshape) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 3);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 3);

  // To be a reshape that we can optimize, we need the buffers to be dense (strides equal to the product of extents of
  // prior dimensions).
  for (auto i : {in, out}) {
    i->dim(0).stride = static_cast<index_t>(sizeof(int));
    i->dim(1).stride = i->dim(0).stride * i->dim(0).extent();
    i->dim(2).stride = i->dim(1).stride * i->dim(1).extent();
  }

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");

  // Compute the "flat" index of the coordinates in the output.
  expr flat_out = x + y * out->dim(0).extent() + z * out->dim(0).extent() * out->dim(1).extent();

  // Unpack the coordinates in the input from the flat index of the output.
  box_expr bounds = {
      point(flat_out % in->dim(0).extent()),
      point((flat_out / in->dim(0).extent()) % in->dim(1).extent()),
      point(flat_out / (in->dim(0).extent() * in->dim(1).extent())),
  };
  func crop = func::make_copy({in, bounds}, {out, {x, y, z}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  const int W = 8;
  const int H = 5;
  const int D = 3;

  // Run the pipeline.
  buffer<int, 3> in_buf({W, H, D});
  init_random(in_buf);

  // The output should be the same size as the input, but with permuted dimensions.
  buffer<int, 3> out_buf({H, D, W});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  // This should have been a "flat" copy.
  for (int i = 0; i < W * H * D; ++i) {
    ASSERT_EQ(in_buf.base()[i], out_buf.base()[i]);
  }
}

TEST(copy, batch_reshape) {
  // Make the pipeline
  node_context ctx;

  auto in = buffer_expr::make(ctx, "in", sizeof(int), 4);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 4);

  // To be a reshape that we can optimize, we need the buffers to be dense (strides equal to the product of extents of
  // prior dimensions).
  for (auto i : {in, out}) {
    i->dim(0).stride = static_cast<index_t>(sizeof(int));
    i->dim(1).stride = i->dim(0).stride * i->dim(0).extent();
    i->dim(2).stride = i->dim(1).stride * i->dim(1).extent();
  }

  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");
  var w(ctx, "w");

  // Compute the "flat" index of the coordinates in the output.
  expr flat_out = x + y * out->dim(0).extent() + z * out->dim(0).extent() * out->dim(1).extent();

  // Unpack the coordinates in the input from the flat index of the output.
  box_expr bounds = {
      point(flat_out % in->dim(0).extent()),
      point((flat_out / in->dim(0).extent()) % in->dim(1).extent()),
      point(flat_out / (in->dim(0).extent() * in->dim(1).extent())),
      point(w),
  };
  func crop = func::make_copy({in, bounds}, {out, {x, y, z, w}});

  pipeline p = build_pipeline(ctx, {in}, {out});

  const int W = 8;
  const int H = 5;
  const int D = 3;
  const int N = 3;

  // Run the pipeline.
  buffer<int, 4> in_buf({W, H, D, N});
  init_random(in_buf);

  // The output should be the same size as the input, but with permuted dimensions.
  buffer<int, 4> out_buf({H, D, W, N});
  out_buf.allocate();

  const raw_buffer* inputs[] = {&in_buf};
  const raw_buffer* outputs[] = {&out_buf};
  eval_context eval_ctx;
  p.evaluate(inputs, outputs, eval_ctx);

  // This should have been a "flat" copy.
  for (int n = 0; n < N; ++n) {
    const int* in_base = &in_buf(0, 0, 0, n);
    const int* out_base = &out_buf(0, 0, 0, n);
    for (int i = 0; i < W * H * D; ++i) {
      ASSERT_EQ(in_base[i], out_base[i]);
    }
  }
}

}  // namespace slinky
