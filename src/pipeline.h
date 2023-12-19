#ifndef SLINKY_PIPELINE_H
#define SLINKY_PIPELINE_H

#include "interval.h"

namespace slinky {

class func;
class buffer_expr;

using buffer_expr_ptr = std::shared_ptr<buffer_expr>;

// Represents a symbolic buffer in a pipeline.
class buffer_expr : public std::enable_shared_from_this<buffer_expr> {
public:
  struct dim_expr {
    expr min;
    expr extent;
    expr stride_bytes;
    expr fold_factor;
  };

private:
  expr base_;
  std::vector<dim_expr> dims_;

  func* producer_;
  std::vector<func*> consumers_;

  buffer_expr(node_context& ctx, const std::string& name, std::size_t rank);
  buffer_expr(const buffer_expr&) = delete;
  buffer_expr(buffer_expr&&) = delete;
  buffer_expr& operator=(const buffer_expr&) = delete;
  buffer_expr& operator=(buffer_expr&&) = delete;

  // Only func::func should add producers/consumers.
  friend class func;
  void add_producer(func* f);
  void add_consumer(func* f);

public:
  static buffer_expr_ptr make(node_context& ctx, const std::string& name, std::size_t rank);

  const expr& base() const { return base_; }
  std::size_t rank() const { return dims_.size(); }
  dim_expr& dim(int i) { return dims_[i]; }
  const dim_expr& dim(int i) const { return dims_[i]; }

  // buffer_exprs can have many consumers, but only one producer.
  const func* producer() const { return producer_; }
  const std::vector<func*>& consumers() const { return consumers_; }
};

// Represents a node of computation in a pipeline.
class func {
public:
  using callable = std::function<index_t(std::span<buffer_base*>, std::span<buffer_base*>)>;

  template <typename... T>
  using callable_wrapper = std::function<index_t(const buffer<T>&...)>;

  struct input {
    buffer_expr_ptr buffer;

    // These intervals should be a function of the expressions found in the output dims.
    std::vector<interval> bounds;
  };

  struct output {
    buffer_expr_ptr buffer;

    // dims must be be variable nodes. It would be nice to enforce this via the type system.
    // TODO: Maybe they don't need to be variables?
    std::vector<expr> dims;

    // If this exists for a dimension, specifies the alignment required in that dimension.
    std::vector<index_t> alignment;
  };

private:
  callable impl_;
  std::vector<input> inputs_;
  std::vector<output> outputs_;

public:
  func() {}
  func(callable impl, std::vector<input> inputs, std::vector<output> outputs);
  func(const func&) = default;
  func(func&&) = default;
  func& operator=(const func&) = default;
  func& operator=(func&&) = default;

  bool defined() const { return impl_ != nullptr; }

  // TODO: Try to do this with a variadic template implementation.
  template <typename Out1>
  static func make(callable_wrapper<Out1> impl, output arg) {
    return func([impl = std::move(impl)](std::span<buffer_base*> inputs, std::span<buffer_base*> outputs) -> index_t {
      return impl(outputs[0]->cast<Out1>());
    }, {}, { std::move(arg) });
  }

  template <typename In1, typename Out1>
  static func make(callable_wrapper<const In1, Out1> impl, input in1, output out1) {
    return func([impl = std::move(impl)](std::span<buffer_base*> inputs, std::span<buffer_base*> outputs) -> index_t {
      return impl(inputs[0]->cast<const In1>(), outputs[0]->cast<Out1>());
      }, { std::move(in1) }, { std::move(out1) });
  }

  template <typename In1, typename In2, typename Out1>
  static func make(callable_wrapper<const In1, const In2, Out1> impl, input in1, input in2, output out1) {
    return func([impl = std::move(impl)](std::span<buffer_base*> inputs, std::span<buffer_base*> outputs) -> index_t {
      return impl(inputs[0]->cast<const In1>(), inputs[1]->cast<const In2>(), outputs[0]->cast<Out1>());
      }, { std::move(in1), std::move(in2) }, { std::move(out1) });
  }

  const callable& impl() const { return impl_; }
  const std::vector<input>& inputs() const { return inputs_; }
  const std::vector<output>& outputs() const { return outputs_; }
};

class pipeline {
  std::vector<buffer_expr_ptr> inputs_;
  std::vector<buffer_expr_ptr> outputs_;
  
  stmt body;

public:
  pipeline(std::vector<buffer_expr_ptr> inputs, std::vector<buffer_expr_ptr> outputs);

  index_t evaluate(std::span<buffer_base*> inputs, std::span<buffer_base*> outputs);
};

}  // namespace slinky

#endif  // SLINKY_PIPELINE_H
