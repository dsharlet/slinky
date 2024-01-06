#include "expr.h"
#include "funcs.h"
#include "pipeline.h"
#include "print.h"
#include "test.h"

#include <cassert>
#include <map>

using namespace slinky;

// (Ab)use our expression mechanism to make an elementwise "calculator", for the purposes of testing.

template <typename T, std::size_t Rank>
class elementwise_pipeline_builder : public node_visitor {
  node_context& ctx;

  std::vector<interval_expr> bounds;
  std::vector<var> dims;

  std::string name(const buffer_expr_ptr& b) const { return ctx.name(b->sym()); }

  std::map<symbol_id, buffer_expr_ptr> vars;

public:
  elementwise_pipeline_builder(node_context& ctx) : ctx(ctx) {
    for (std::size_t d = 0; d < Rank; ++d) {
      dims.emplace_back(ctx, "d" + std::to_string(d));
      bounds.emplace_back(dims.back(), dims.back());
    }
  }

  std::vector<func> result_funcs;
  std::vector<buffer_expr_ptr> inputs;
  buffer_expr_ptr result;

  void visit(const variable* v) override {
    auto i = vars.find(v->sym);
    if (i != vars.end()) {
      result = i->second;
      return;
    }
    result = buffer_expr::make(v->sym, sizeof(T), Rank);
    inputs.push_back(result);
    vars[v->sym] = result;
  }

  void visit(const constant* c) override {
    buffer<T, Rank> value;
    for (std::size_t d = 0; d < Rank; ++d) {
      value.dims[d].set_stride(0);
    }
    value.allocate();
    memcpy(value.base(), &c->value, sizeof(T));
    result = buffer_expr::make(value);
  }

  buffer_expr_ptr visit_expr(const expr& e) {
    e.accept(this);
    return result;
  }

  template <typename Impl>
  void visit_binary(const buffer_expr_ptr& a, const buffer_expr_ptr& b, const Impl& impl) {
    result = buffer_expr::make(ctx, name(a) + "_" + name(b), sizeof(T), Rank);
    func r = func::make<const T, const T, T>(
        [=](const buffer<const T>& a, const buffer<const T>& b, const buffer<T>& c) {
          for_each_index(c, [&](auto i) { c(i) = impl(a(i), b(i)); });
          return 0;
        },
        {a, bounds}, {b, bounds}, {result, dims});
    result_funcs.push_back(std::move(r));
  }

  template <typename Impl>
  void visit_binary(const expr& a, const expr& b, const Impl& impl) {
    visit_binary(visit_expr(a), visit_expr(b), impl);
  }

  void visit(const class min* op) override {
    visit_binary(op->a, op->b, [](T a, T b) { return std::min(a, b); });
  }
  void visit(const class max* op) override {
    visit_binary(op->a, op->b, [](T a, T b) { return std::max(a, b); });
  }
  void visit(const class add* op) override { visit_binary(op->a, op->b, std::plus<T>()); }
  void visit(const sub* op) override { visit_binary(op->a, op->b, std::minus<T>()); }
  void visit(const mul* op) override { visit_binary(op->a, op->b, std::multiplies<T>()); }
  void visit(const slinky::div* op) override { visit_binary(op->a, op->b, std::divides<T>()); }
  void visit(const slinky::mod* op) override { visit_binary(op->a, op->b, std::modulus<T>()); }
  void visit(const less* op) override { visit_binary(op->a, op->b, std::less<T>()); }
  void visit(const less_equal* op) override { visit_binary(op->a, op->b, std::less_equal<T>()); }
  void visit(const equal* op) override { visit_binary(op->a, op->b, std::equal_to<T>()); }
  void visit(const not_equal* op) override { visit_binary(op->a, op->b, std::not_equal_to<T>()); }
  void visit(const logical_and* op) override { visit_binary(op->a, op->b, std::logical_and<T>()); }
  void visit(const logical_or* op) override { visit_binary(op->a, op->b, std::logical_or<T>()); }
  void visit(const class select* op) override {
    buffer_expr_ptr c = visit_expr(op->condition);
    buffer_expr_ptr t = visit_expr(op->true_value);
    buffer_expr_ptr f = visit_expr(op->false_value);
    result = buffer_expr::make(ctx, "select_" + name(c) + "_" + name(t) + "_" + name(f), sizeof(T), Rank);
    func r = func::make<const T, const T, const T, T>(
        [](const buffer<const T>& c, const buffer<const T>& t, const buffer<const T>& f,
            const buffer<T>& r) -> index_t {
          for_each_index(c, [&](auto i) { r(i) = c(i) != 0 ? t(i) : f(i); });
          return 0;
        },
        {c, bounds}, {t, bounds}, {f, bounds}, {result, dims});
    result_funcs.push_back(std::move(r));
  }

  void visit(const wildcard*) override { std::abort(); }
  void visit(const let*) override { std::abort(); }
  void visit(const call*) override { std::abort(); }
  void visit(const logical_not*) override { std::abort(); }

  void visit(const let_stmt*) override { std::abort(); }
  void visit(const block*) override { std::abort(); }
  void visit(const loop*) override { std::abort(); }
  void visit(const if_then_else*) override { std::abort(); }
  void visit(const call_stmt*) override { std::abort(); }
  void visit(const copy_stmt*) override { std::abort(); }
  void visit(const allocate*) override { std::abort(); }
  void visit(const make_buffer*) override { std::abort(); }
  void visit(const crop_buffer*) override { std::abort(); }
  void visit(const crop_dim*) override { std::abort(); }
  void visit(const slice_buffer*) override { std::abort(); }
  void visit(const slice_dim*) override { std::abort(); }
  void visit(const truncate_rank*) override { std::abort(); }
  void visit(const check*) override { std::abort(); }
};

template <typename T, std::size_t Rank>
pipeline make_expr_pipeline(node_context& ctx, const expr& e) {}

template <typename T, std::size_t Rank>
void test_expr_pipeline(node_context& ctx, const expr& e) {
  elementwise_pipeline_builder<T, Rank> builder(ctx);
  e.accept(&builder);

  pipeline p(ctx, builder.inputs, {builder.result});

  std::vector<index_t> extents;
  for (std::size_t i = 0; i < Rank; ++i) {
    extents.push_back((rand() % 100) + 1);
  }

  std::vector<const raw_buffer*> inputs;
  std::vector<raw_buffer_ptr> input_bufs;

  for (std::size_t i = 0; i < p.inputs().size(); ++i) {
    input_bufs.emplace_back(raw_buffer::make(sizeof(T), extents));
    inputs.push_back(&*input_bufs.back());
    init_random(input_bufs.back()->cast<T>());
  }

  raw_buffer_ptr output_buf = raw_buffer::make(sizeof(T), extents);
  output_buf->allocate();

  std::vector<const raw_buffer*> outputs;
  outputs.push_back(&*output_buf);

  p.evaluate(inputs, outputs);
}

// Compute max(a + b, 0) * c
void test_elementwise(const int W, const int H) {
  // Make the pipeline
  node_context ctx;

  auto a = buffer_expr::make(ctx, "a", sizeof(int), 2);
  auto b = buffer_expr::make(ctx, "b", sizeof(int), 2);
  auto c = buffer_expr::make(ctx, "c", sizeof(int), 2);

  auto ab = buffer_expr::make(ctx, "ab", sizeof(int), 2);
  auto maxab0 = buffer_expr::make(ctx, "maxab0", sizeof(int), 2);
  auto out = buffer_expr::make(ctx, "out", sizeof(int), 2);

  var x(ctx, "x");
  var y(ctx, "y");

  func f_ab = func::make<const int, const int, int>(
      add<int>, {a, {point(x), point(y)}}, {b, {point(x), point(y)}}, {ab, {x, y}});
  func f_maxab0 = func::make<const int, int>(max_0<int>, {ab, {point(x), point(y)}}, {maxab0, {x, y}});
  func f_maxab0c = func::make<const int, const int, int>(
      multiply<int>, {maxab0, {point(x), point(y)}}, {c, {point(x), point(y)}}, {out, {x, y}});

  pipeline p(ctx, {a, b, c}, {out});

  // Run the pipeline
  buffer<int, 2> a_buf({W, H});
  buffer<int, 2> b_buf({W, H});
  buffer<int, 2> c_buf({W, H});
  init_random(a_buf);
  init_random(b_buf);
  init_random(c_buf);

  buffer<int, 2> out_buf({W, H});
  out_buf.allocate();

  // Not having std::span(std::initializer_list<T>) is unfortunate.
  const raw_buffer* inputs[] = {&a_buf, &b_buf, &c_buf};
  const raw_buffer* outputs[] = {&out_buf};
  p.evaluate(inputs, outputs);

  for_each_index(
      out_buf, [&](std::span<index_t> i) { ASSERT_EQ(out_buf(i), std::max(a_buf(i) + b_buf(i), 0) * c_buf(i)); });
}

TEST(elementwise) { test_elementwise(40, 30); }

TEST(elementwise_add_xy) {
  node_context ctx;
  var x(ctx, "x");
  var y(ctx, "y");
  test_expr_pipeline<int, 1>(ctx, x + y);
}

TEST(elementwise_mul_add) {
  node_context ctx;
  var x(ctx, "x");
  var y(ctx, "y");
  var z(ctx, "z");
  test_expr_pipeline<int, 1>(ctx, x * y + z);
}
