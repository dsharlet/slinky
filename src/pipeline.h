#ifndef SLINKY_PIPELINE_H
#define SLINKY_PIPELINE_H

#include "evaluate.h"
#include "expr.h"

namespace slinky {

class func;
class buffer_expr;

using buffer_expr_ptr = std::shared_ptr<buffer_expr>;

struct loop_id {
  const func* f = nullptr;
  expr loop;
};

// Represents a symbolic buffer in a pipeline.
class buffer_expr : public std::enable_shared_from_this<buffer_expr> {
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
  using callable = std::function<index_t(std::span<raw_buffer*>, std::span<raw_buffer*>)>;

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
  };

  struct output {
    buffer_expr_ptr buffer;

    // dims must be be variable nodes. It would be nice to enforce this via the type system.
    // TODO(https://github.com/dsharlet/slinky/issues/7): Maybe they don't need to be variables?
    std::vector<expr> dims;

    // If this exists for a dimension, specifies the alignment required in that dimension.
    std::vector<index_t> alignment;
  };

private:
  callable impl_;
  std::vector<input> inputs_;
  std::vector<output> outputs_;

  std::vector<expr> loops_;
  loop_id compute_at_;

public:
  func() {}
  func(callable impl, std::vector<input> inputs, std::vector<output> outputs);
  func(const func&) = default;
  func(func&&) = default;
  func& operator=(const func&) = default;
  func& operator=(func&&) = default;

  bool defined() const { return impl_ != nullptr; }

  // Describes which loops should be explicit for this func.
  func& loops(std::vector<expr> l) {
    loops_ = std::move(l);
    return *this;
  }
  const std::vector<expr>& loops() const { return loops_; }

  func& compute_at(const loop_id& at) {
    compute_at_ = at;
    return *this;
  }
  const loop_id& compute_at() const { return compute_at_; }

  // TODO(https://github.com/dsharlet/slinky/issues/8): Try to do this with a variadic template implementation.
  template <typename Out1>
  static func make(callable_wrapper<Out1> impl, output arg) {
    return func(
        [impl = std::move(impl)](std::span<raw_buffer*> inputs, std::span<raw_buffer*> outputs) -> index_t {
          assert(inputs.size() == 0);
          assert(outputs.size() == 1);
          assert(outputs[0] != nullptr);
          return impl(outputs[0]->cast<Out1>());
        },
        {}, {std::move(arg)});
  }

  template <typename In1, typename Out1>
  static func make(callable_wrapper<const In1, Out1> impl, input in1, output out1) {
    return func(
        [impl = std::move(impl)](std::span<raw_buffer*> inputs, std::span<raw_buffer*> outputs) -> index_t {
          assert(inputs.size() == 1);
          assert(outputs.size() == 1);
          assert(inputs[0] != nullptr);
          assert(outputs[0] != nullptr);
          return impl(inputs[0]->cast<const In1>(), outputs[0]->cast<Out1>());
        },
        {std::move(in1)}, {std::move(out1)});
  }

  template <typename In1, typename In2, typename Out1>
  static func make(callable_wrapper<const In1, const In2, Out1> impl, input in1, input in2, output out1) {
    return func(
        [impl = std::move(impl)](std::span<raw_buffer*> inputs, std::span<raw_buffer*> outputs) -> index_t {
          assert(inputs.size() == 2);
          assert(outputs.size() == 1);
          assert(inputs[0] != nullptr);
          assert(inputs[1] != nullptr);
          assert(outputs[0] != nullptr);
          return impl(inputs[0]->cast<const In1>(), inputs[1]->cast<const In2>(), outputs[0]->cast<Out1>());
        },
        {std::move(in1), std::move(in2)}, {std::move(out1)});
  }

  const callable& impl() const { return impl_; }
  const std::vector<input>& inputs() const { return inputs_; }
  const std::vector<output>& outputs() const { return outputs_; }
};

// TODO: I wanted this to be pipeline::build_options, but I hit some tricky compiler error
struct build_options {
  // If true, removes bounds checks
  bool no_checks = false;
};

class pipeline {
  std::vector<buffer_expr_ptr> inputs_;
  std::vector<buffer_expr_ptr> outputs_;

  stmt body;

public:
  pipeline(node_context& ctx, std::vector<buffer_expr_ptr> inputs, std::vector<buffer_expr_ptr> outputs,
      const build_options& options = build_options());

  index_t evaluate(
      std::span<const raw_buffer*> inputs, std::span<const raw_buffer*> outputs, eval_context& ctx) const;
  index_t evaluate(std::span<const raw_buffer*> inputs, std::span<const raw_buffer*> outputs) const;
};

}  // namespace slinky

#endif  // SLINKY_PIPELINE_H
