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
  const func* f = nullptr;
  var loop;
};

// Represents a symbolic buffer in a pipeline.
class buffer_expr : public ref_counted {
  symbol_id name_;
  index_t elem_size_;
  std::vector<dim_expr> dims_;

  func* producer_;
  std::vector<func*> consumers_;

  memory_type storage_ = memory_type::heap;
  loop_id store_at_;

  buffer_expr(symbol_id name, index_t elem_size, std::size_t rank);
  buffer_expr(const buffer_expr&) = delete;
  buffer_expr(buffer_expr&&) = delete;
  buffer_expr& operator=(const buffer_expr&) = delete;
  buffer_expr& operator=(buffer_expr&&) = delete;

  // Only func::func should add producers/consumers.
  friend class func;
  void add_producer(func* f);
  void add_consumer(func* f);

public:
  static buffer_expr_ptr make(symbol_id name, index_t elem_size, std::size_t rank);
  static buffer_expr_ptr make(node_context& ctx, const std::string& name, index_t elem_size, std::size_t rank);

  symbol_id name() const { return name_; }
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

  // buffer_exprs can have many consumers, but only one producer.
  const func* producer() const { return producer_; }
  const std::vector<func*>& consumers() const { return consumers_; }
};

// Represents a node of computation in a pipeline.
class func {
public:
  using callable = call_func::callable;
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

    symbol_id name() const { return buffer->name(); }
  };

  struct output {
    buffer_expr_ptr buffer;

    std::vector<var> dims;

    // If this exists for a dimension, specifies the alignment required in that dimension.
    std::vector<index_t> alignment;

    symbol_id name() const { return buffer->name(); }
  };

private:
  callable impl_;
  std::vector<input> inputs_;
  std::vector<output> outputs_;

  std::vector<var> loops_;
  loop_id compute_at_;

  std::vector<char> padding_;

public:
  func() {}
  func(callable impl, std::vector<input> inputs, std::vector<output> outputs);
  func(std::vector<input> inputs, output out, std::vector<char> padding);
  func(const func&) = default;
  func(func&&) = default;
  func& operator=(const func&) = default;
  func& operator=(func&&) = default;

  bool defined() const { return impl_ != nullptr; }

  // Describes which loops should be explicit for this func.
  func& loops(std::vector<var> l) {
    loops_ = std::move(l);
    return *this;
  }
  const std::vector<var>& loops() const { return loops_; }

  func& compute_at(const loop_id& at) {
    compute_at_ = at;
    return *this;
  }
  const loop_id& compute_at() const { return compute_at_; }

  // TODO(https://github.com/dsharlet/slinky/issues/8): Try to do this with a variadic template implementation.
  template <typename Out1>
  static func make(callable_wrapper<Out1> impl, output out1) {
    return func(
        [impl = std::move(impl), out1 = out1.name()](eval_context& ctx) -> index_t {
          const raw_buffer* out1_buf = ctx.lookup_buffer(out1);
          return impl(out1_buf->cast<Out1>());
        },
        {}, {std::move(out1)});
  }

  template <typename In1, typename Out1>
  static func make(callable_wrapper<const In1, Out1> impl, input in1, output out1) {
    return func(
        [impl = std::move(impl), in1 = in1.name(), out1 = out1.name()](eval_context& ctx) -> index_t {
          const raw_buffer* in1_buf = ctx.lookup_buffer(in1);
          const raw_buffer* out1_buf = ctx.lookup_buffer(out1);
          return impl(in1_buf->cast<const In1>(), out1_buf->cast<Out1>());
        },
        {std::move(in1)}, {std::move(out1)});
  }

  template <typename In1, typename In2, typename Out1>
  static func make(callable_wrapper<const In1, const In2, Out1> impl, input in1, input in2, output out1) {
    return func(
        [impl = std::move(impl), in1 = in1.name(), in2 = in2.name(), out1 = out1.name()](
            eval_context& ctx) -> index_t {
          const raw_buffer* in1_buf = ctx.lookup_buffer(in1);
          const raw_buffer* in2_buf = ctx.lookup_buffer(in2);
          const raw_buffer* out1_buf = ctx.lookup_buffer(out1);
          return impl(in1_buf->cast<const In1>(), in2_buf->cast<const In2>(), out1_buf->cast<Out1>());
        },
        {std::move(in1), std::move(in2)}, {std::move(out1)});
  }

  template <typename In1, typename Out1, typename Out2>
  static func make(callable_wrapper<const In1, Out1, Out2> impl, input in1, output out1, output out2) {
    return func(
        [impl = std::move(impl), in1 = in1.name(), out1 = out1.name(), out2 = out2.name()](
            eval_context& ctx) -> index_t {
          const raw_buffer* in1_buf = ctx.lookup_buffer(in1);
          const raw_buffer* out1_buf = ctx.lookup_buffer(out1);
          const raw_buffer* out2_buf = ctx.lookup_buffer(out2);
          return impl(in1_buf->cast<const In1>(), out1_buf->cast<Out1>(), out2_buf->cast<Out2>());
        },
        {std::move(in1)}, {std::move(out1), std::move(out2)});
  }

  static func make_copy(input in, output out, std::vector<char> padding = {}) {
    return func({std::move(in)}, {std::move(out)}, std::move(padding));
  }

  const call_func::callable& impl() const { return impl_; }
  const std::vector<input>& inputs() const { return inputs_; }
  const std::vector<output>& outputs() const { return outputs_; }
  const std::vector<char>& padding() const { return padding_; }
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
};

}  // namespace slinky

#endif  // SLINKY_PIPELINE_H
