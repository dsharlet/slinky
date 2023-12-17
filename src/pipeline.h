#ifndef LOCALITY_PIPELINE_H
#define LOCALITY_PIPELINE_H

#include "interval.h"
#include "evaluate.h"

namespace slinky {

// Represents a symbolic buffer in a pipeline.
struct buffer_expr {
  struct dim {
    expr min;
    expr extent;
    expr stride;
    expr fold_factor;
  };
  expr base;
  std::vector<dim> dims;

  buffer_expr(node_context& ctx, const std::string& name, std::size_t rank);

  buffer_expr(const buffer_expr&) = default;
  buffer_expr(buffer_expr&&) = default;
  buffer_expr& operator=(const buffer_expr&) = default;
  buffer_expr& operator=(buffer_expr&&) = default;
};

class func {
public:
  using callable = std::function<index_t(std::span<buffer<const void>*>, std::span<buffer<void>*>)>;

  template <typename... T>
  using callable_wrapper = std::function<index_t(const buffer<T>&...)>;

  struct input {
    symbol_id buffer;

    // These intervals should be a function of the expressions found in the output dims.
    std::vector<interval> bounds;
  };

  struct output {
    symbol_id buffer;

    // dims must be be variable nodes. It would be nice to enforce this via the type system.
    // TODO: Maybe they don't need to be variables?
    std::vector<expr> dims;
  };

private:
  callable impl;
  std::vector<input> inputs;
  std::vector<output> outputs;

public:
  func() {}
  func(callable impl, std::vector<input> inputs, std::vector<output> outputs)
    : impl(std::move(impl)), inputs(std::move(inputs)), outputs(std::move(outputs)) {}
  func(const func&) = default;
  func(func&&) = default;
  func& operator=(const func&) = default;
  func& operator=(func&&) = default;

  bool defined() const { return impl != nullptr; }

  // TODO: Try to do this with a variadic template implementation.
  template <typename Out1>
  static func make(callable_wrapper<Out1> impl, output arg) {
    return func([impl = std::move(impl)](std::span<buffer<const void>*> inputs, std::span<buffer<void>*> outputs) -> index_t {
      return impl(outputs[0]->cast<Out1>());
    }, {}, { arg });
  }

  template <typename In1, typename Out1>
  static func make(callable_wrapper<const In1, Out1> impl, input in1, output out1) {
    return func([impl = std::move(impl)](std::span<buffer<const void>*> inputs, std::span<buffer<void>*> outputs) -> index_t {
      return impl(inputs[0]->cast<const In1>(), outputs[0]->cast<Out1>());
      }, { in1 }, { out1 });
  }

  template <typename In1, typename In2, typename Out1>
  static func make(callable_wrapper<const In1, const In2, Out1> impl, input in1, input in2, output out1) {
    return func([impl = std::move(impl)](std::span<buffer<const void>*> inputs, std::span<buffer<void>*> outputs) -> index_t {
      return impl(inputs[0]->cast<const In1>(), inputs[1]->cast<const In2>(), outputs[0]->cast<Out1>());
      }, { in1, in2 }, { out1 });
  }

  index_t evaluate(eval_context& ctx);
};

class pipeline {
private:
  struct stage {
    index_t loop_level;
    func f;
  };
  std::vector<stage> stages_;

  std::vector<buffer_expr> buffers_;

  node_context context_;

  index_t run_loop_level(eval_context& ctx, index_t loop_level, std::size_t stage_begin, std::size_t stage_end);

public:
  pipeline() {}

  node_context& context() { return context_; }
  const node_context& context() const { return context_; }

  symbol_id add_buffer(const std::string& name, std::size_t rank) {
    symbol_id id = buffers_.size();
    buffers_.emplace_back(context_, name, rank);
    return id;
  }

  buffer_expr& get_buffer(symbol_id id) { return buffers_[id]; }
  const buffer_expr& get_buffer(symbol_id id) const { return buffers_[id]; }
    
  void add_stage(index_t loop_level, func f) {
    stages_.emplace_back(loop_level, std::move(f));
  }

  index_t evaluate(eval_context& ctx);
};

}  // namespace slinky

#endif