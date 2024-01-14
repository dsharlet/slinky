#ifndef SLINKY_PIPELINE_H
#define SLINKY_PIPELINE_H

#include "evaluate.h"
#include "expr.h"
#include "ref_count.h"

namespace slinky {

class func;
class buffer_expr;

using buffer_expr_ptr = ref_count<buffer_expr>;

struct loop_id {
  const slinky::func* func = nullptr;
  slinky::var var;

  bool root() const { return !func; }
  symbol_id sym() const { return var.sym(); }
};

// Represents a symbolic buffer in a pipeline.
class buffer_expr : public ref_counted {
  symbol_id sym_;
  index_t elem_size_;
  std::vector<dim_expr> dims_;

  func* producer_;

  memory_type storage_ = memory_type::heap;
  loop_id store_at_;

  buffer_expr(symbol_id sym, index_t elem_size, std::size_t rank);
  buffer_expr(const raw_buffer& buffer);
  buffer_expr(const buffer_expr&) = delete;
  buffer_expr(buffer_expr&&) = delete;
  buffer_expr& operator=(const buffer_expr&) = delete;
  buffer_expr& operator=(buffer_expr&&) = delete;

  // Only func::func should add producers/consumers.
  friend class func;
  void set_producer(func* f);

public:
  static buffer_expr_ptr make(symbol_id sym, index_t elem_size, std::size_t rank);
  static buffer_expr_ptr make(node_context& ctx, const std::string& sym, index_t elem_size, std::size_t rank);
  static buffer_expr_ptr make(const raw_buffer& buffer);

  symbol_id sym() const { return sym_; }
  index_t elem_size() const { return elem_size_; }
  std::size_t rank() const { return dims_.size(); }
  const std::vector<dim_expr>& dims() const { return dims_; }
  dim_expr& dim(int i) { return dims_[i]; }
  const dim_expr& dim(int i) const { return dims_[i]; }

  buffer_expr& store_in(memory_type type) {
    storage_ = type;
    return *this;
  }
  memory_type storage() const { return storage_; }

  buffer_expr& store_at(loop_id at) {
    store_at_ = at;
    return *this;
  }
  const loop_id& store_at() const { return store_at_; }

  const func* producer() const { return producer_; }
};

// Represents a node of computation in a pipeline.
class func {
public:
  using callable = call_stmt::callable;
  template <typename... T>
  using callable_wrapper = std::function<index_t(const buffer<T>&...)>;

  // TODO(https://github.com/dsharlet/slinky/issues/7): There should be a separate descriptor
  // of a callable and the bounds/dims of inputs/outputs, which is constant over all the
  // instantiations of that descriptor. Then, that descriptor can be used for multiple
  // stages/pipelines, without redundantly indicating the bounds/dims.
  struct input {
    buffer_expr_ptr buffer;

    // These intervals should be a function of the expressions found in the output dims.
    std::vector<interval_expr> bounds;

    symbol_id sym() const { return buffer->sym(); }
  };

  struct output {
    buffer_expr_ptr buffer;

    std::vector<var> dims;

    symbol_id sym() const { return buffer->sym(); }
  };

  struct loop_info {
    slinky::var var;
    expr step;
    loop_mode mode;

    loop_info() {}
    loop_info(slinky::var var, expr step = 1, loop_mode mode = loop_mode::serial) : var(var), step(step), mode(mode) {}

    symbol_id sym() const { return var.sym(); }

    bool defined() const { return var.defined() && step.defined(); }
  };

private:
  callable impl_;
  std::vector<input> inputs_;
  std::vector<output> outputs_;

  std::vector<loop_info> loops_;
  loop_id compute_at_;

  std::vector<char> padding_;

  void add_this_to_buffers();
  void remove_this_from_buffers();

public:
  func() {}
  func(callable impl, std::vector<input> inputs, std::vector<output> outputs);
  func(std::vector<input> inputs, output out, std::vector<char> padding);
  func(func&&);
  func& operator=(func&&);
  ~func();
  func(const func&) = delete;
  func& operator=(const func&) = delete;

  bool defined() const { return impl_ != nullptr; }

  // Describes which loops should be explicit for this func, and the step size for that loop.
  func& loops(std::vector<loop_info> l) {
    loops_ = std::move(l);
    return *this;
  }
  const std::vector<loop_info>& loops() const { return loops_; }

  func& compute_at(const loop_id& at) {
    compute_at_ = at;
    return *this;
  }
  const loop_id& compute_at() const { return compute_at_; }

  // TODO(https://github.com/dsharlet/slinky/issues/8): Try to do this with a variadic template implementation.
  template <typename Out1>
  static func make(callable_wrapper<Out1> impl, output out1) {
    symbol_id out1_sym = out1.sym();
    return func(
        [=, impl = std::move(impl)](eval_context& ctx) -> index_t {
          const raw_buffer* out1_buf = ctx.lookup_buffer(out1_sym);
          return impl(out1_buf->cast<Out1>());
        },
        {}, {std::move(out1)});
  }

  template <typename In1, typename Out1>
  static func make(callable_wrapper<const In1, Out1> impl, input in1, output out1) {
    symbol_id in1_sym = in1.sym();
    symbol_id out1_sym = out1.sym();
    return func(
        [=, impl = std::move(impl)](eval_context& ctx) -> index_t {
          const raw_buffer* in1_buf = ctx.lookup_buffer(in1_sym);
          const raw_buffer* out1_buf = ctx.lookup_buffer(out1_sym);
          return impl(in1_buf->cast<const In1>(), out1_buf->cast<Out1>());
        },
        {std::move(in1)}, {std::move(out1)});
  }

  template <typename In1, typename In2, typename Out1>
  static func make(callable_wrapper<const In1, const In2, Out1> impl, input in1, input in2, output out1) {
    symbol_id in1_sym = in1.sym();
    symbol_id in2_sym = in2.sym();
    symbol_id out1_sym = out1.sym();
    return func(
        [=, impl = std::move(impl)](eval_context& ctx) -> index_t {
          const raw_buffer* in1_buf = ctx.lookup_buffer(in1_sym);
          const raw_buffer* in2_buf = ctx.lookup_buffer(in2_sym);
          const raw_buffer* out1_buf = ctx.lookup_buffer(out1_sym);
          return impl(in1_buf->cast<const In1>(), in2_buf->cast<const In2>(), out1_buf->cast<Out1>());
        },
        {std::move(in1), std::move(in2)}, {std::move(out1)});
  }

  template <typename In1, typename In2, typename In3, typename Out1>
  static func make(callable_wrapper<const In1, const In2, const In3, Out1> impl, input in1, input in2, input in3, output out1) {
    symbol_id in1_sym = in1.sym();
    symbol_id in2_sym = in2.sym();
    symbol_id in3_sym = in3.sym();
    symbol_id out1_sym = out1.sym();
    return func(
        [=, impl = std::move(impl)](eval_context& ctx) -> index_t {
          const raw_buffer* in1_buf = ctx.lookup_buffer(in1_sym);
          const raw_buffer* in2_buf = ctx.lookup_buffer(in2_sym);
          const raw_buffer* in3_buf = ctx.lookup_buffer(in3_sym);
          const raw_buffer* out1_buf = ctx.lookup_buffer(out1_sym);
          return impl(in1_buf->cast<const In1>(), in2_buf->cast<const In2>(), in3_buf->cast<const In3>(),
              out1_buf->cast<Out1>());
        },
        {std::move(in1), std::move(in2), std::move(in3)}, {std::move(out1)});
  }

  template <typename In1, typename Out1, typename Out2>
  static func make(callable_wrapper<const In1, Out1, Out2> impl, input in1, output out1, output out2) {
    symbol_id in1_sym = in1.sym();
    symbol_id out1_sym = out1.sym();
    symbol_id out2_sym = out2.sym();
    return func(
        [=, impl = std::move(impl)](
            eval_context& ctx) -> index_t {
          const raw_buffer* in1_buf = ctx.lookup_buffer(in1_sym);
          const raw_buffer* out1_buf = ctx.lookup_buffer(out1_sym);
          const raw_buffer* out2_buf = ctx.lookup_buffer(out2_sym);
          return impl(in1_buf->cast<const In1>(), out1_buf->cast<Out1>(), out2_buf->cast<Out2>());
        },
        {std::move(in1)}, {std::move(out1), std::move(out2)});
  }

  static func make_copy(std::vector<input> in, output out, std::vector<char> padding = {}) {
    return func(std::move(in), {std::move(out)}, std::move(padding));
  }
  static func make_copy(input in, output out, std::vector<char> padding = {}) {
    return func({std::move(in)}, {std::move(out)}, std::move(padding));
  }
  static func make_copy(input in1, input in2, output out, std::vector<char> padding = {}) {
    return func({std::move(in1), std::move(in2)}, {std::move(out)}, std::move(padding));
  }

  const call_stmt::callable& impl() const { return impl_; }
  const std::vector<input>& inputs() const { return inputs_; }
  const std::vector<output>& outputs() const { return outputs_; }
  const std::vector<char>& padding() const { return padding_; }

  stmt make_call() const;
};

// TODO: I wanted this to be pipeline::build_options, but I hit some tricky compiler error
struct build_options {
  // If true, removes bounds checks
  bool no_checks = false;
};

class pipeline {
  std::vector<symbol_id> args_;
  std::vector<buffer_expr_ptr> inputs_;
  std::vector<buffer_expr_ptr> outputs_;

  stmt body;

public:
  // TODO: The `args` should be limited to variables only, not arbitrary exprs (when we get type safe variable exprs
  // from https://github.com/dsharlet/slinky/issues/7, that could be used here).
  pipeline(node_context& ctx, std::vector<var> args, std::vector<buffer_expr_ptr> inputs,
      std::vector<buffer_expr_ptr> outputs, const build_options& options = build_options());
  pipeline(node_context& ctx, std::vector<buffer_expr_ptr> inputs, std::vector<buffer_expr_ptr> outputs,
      const build_options& options = build_options());

  using scalars = std::span<const index_t>;
  using buffers = std::span<const raw_buffer*>;

  index_t evaluate(scalars args, buffers inputs, buffers outputs, eval_context& ctx) const;
  index_t evaluate(buffers inputs, buffers outputs, eval_context& ctx) const;
  index_t evaluate(scalars args, buffers inputs, buffers outputs) const;
  index_t evaluate(buffers inputs, buffers outputs) const;

  const std::vector<buffer_expr_ptr>& inputs() const { return inputs_; }
  const std::vector<buffer_expr_ptr>& outputs() const { return outputs_; }
};

}  // namespace slinky

#endif  // SLINKY_PIPELINE_H
