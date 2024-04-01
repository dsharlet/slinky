#ifndef SLINKY_BUILDER_PIPELINE_H
#define SLINKY_BUILDER_PIPELINE_H

#include "runtime/evaluate.h"
#include "runtime/expr.h"
#include "runtime/pipeline.h"
#include "runtime/util.h"

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
class buffer_expr : public ref_counted<buffer_expr> {
  symbol_id sym_;
  index_t elem_size_;
  std::vector<dim_expr> dims_;

  func* producer_;
  const_raw_buffer_ptr constant_;

  memory_type storage_ = memory_type::heap;
  std::optional<loop_id> store_at_;

  buffer_expr(symbol_id sym, std::size_t rank, index_t elem_size);
  buffer_expr(symbol_id sym, const_raw_buffer_ptr constant_buffer);
  buffer_expr(const buffer_expr&) = delete;
  buffer_expr(buffer_expr&&) = delete;
  buffer_expr& operator=(const buffer_expr&) = delete;
  buffer_expr& operator=(buffer_expr&&) = delete;

  // Only func::func should add producers/consumers.
  friend class func;
  void set_producer(func* f);

public:
  static buffer_expr_ptr make(symbol_id sym, std::size_t rank, index_t elem_size);
  static buffer_expr_ptr make(node_context& ctx, const std::string& sym, std::size_t rank, index_t elem_size);
  // Make a constant buffer_expr. It takes ownership of the buffer from the caller.
  static buffer_expr_ptr make(symbol_id sym, const_raw_buffer_ptr constant_buffer);
  static buffer_expr_ptr make(node_context& ctx, const std::string& sym, const_raw_buffer_ptr constant_buffer);

  symbol_id sym() const { return sym_; }
  index_t elem_size() const { return elem_size_; }
  std::size_t rank() const { return dims_.size(); }
  const std::vector<dim_expr>& dims() const { return dims_; }
  dim_expr& dim(int i) { return dims_[i]; }
  const dim_expr& dim(int i) const { return dims_[i]; }
  box_expr bounds() const;

  buffer_expr& store_in(memory_type type) {
    storage_ = type;
    return *this;
  }
  memory_type storage() const { return storage_; }

  buffer_expr& store_at(const loop_id& at) {
    store_at_ = at;
    return *this;
  }
  buffer_expr& store_at(std::optional<loop_id> at) {
    store_at_ = std::move(at);
    return *this;
  }
  buffer_expr& store_root() {
    store_at_ = loop_id();
    return *this;
  }
  const std::optional<loop_id>& store_at() const { return store_at_; }

  const func* producer() const { return producer_; }

  const_raw_buffer_ptr constant() const { return constant_; }

  static void destroy(buffer_expr* p) { delete p; }
};

// Represents a node of computation in a pipeline.
class func {
public:
  template <typename... T>
  using callable = std::function<index_t(const buffer<T>&...)>;

  // TODO(https://github.com/dsharlet/slinky/issues/7): There should be a separate descriptor
  // of a callable and the bounds/dims of inputs/outputs, which is constant over all the
  // instantiations of that descriptor. Then, that descriptor can be used for multiple
  // stages/pipelines, without redundantly indicating the bounds/dims.
  struct input {
    buffer_expr_ptr buffer;

    // These intervals should be a function of the expressions found in the output dims.
    box_expr bounds;

    // A region to crop the output to while consuming this input. Only used by copies.
    box_expr output_crop;

    // Slices to apply to the output while consuming this input. Only used by copies.
    std::vector<expr> output_slice;

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
    int max_workers;

    loop_info() = default;
    loop_info(slinky::var var, expr step = 1, int max_workers = loop::serial)
        : var(var), step(step), max_workers(max_workers) {}

    symbol_id sym() const { return var.sym(); }

    bool defined() const { return var.defined() && step.defined(); }
  };

private:
  call_stmt::callable impl_;
  call_stmt::callable_attrs attrs_;
  std::vector<input> inputs_;
  std::vector<output> outputs_;

  std::vector<loop_info> loops_;
  std::optional<loop_id> compute_at_;

  std::optional<std::vector<char>> padding_;

  void add_this_to_buffers();
  void remove_this_from_buffers();

public:
  func() = default;
  func(call_stmt::callable impl, std::vector<input> inputs, std::vector<output> outputs,
      call_stmt::callable_attrs attrs = {});
  func(std::vector<input> inputs, output out);
  func(input input, output out, std::optional<std::vector<char>> padding = std::nullopt);
  func(func&&) noexcept;
  func& operator=(func&&) noexcept;
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
  func& compute_at(std::optional<loop_id> at) {
    compute_at_ = std::move(at);
    return *this;
  }
  func& compute_root() {
    compute_at_ = loop_id();
    return *this;
  }

  const std::optional<loop_id>& compute_at() const { return compute_at_; }

private:
  template <typename... T, std::size_t... Indices>
  static inline index_t call_impl(
      const func::callable<T...>& impl, eval_context& ctx, const call_stmt* op, std::index_sequence<Indices...>) {
    return impl(
        ctx.lookup_buffer(Indices < op->inputs.size() ? op->inputs[Indices] : op->outputs[Indices - op->inputs.size()])
            ->template cast<T>()...);
  }

  template <typename Lambda>
  struct lambda_call_signature : lambda_call_signature<decltype(&Lambda::operator())> {};

  template <typename ReturnType, typename ClassType, typename... Args>
  struct lambda_call_signature<ReturnType (ClassType::*)(Args...) const> {
    using ret_type = ReturnType;
    using arg_types = std::tuple<Args...>;
    using std_function_type = std::function<ReturnType(Args...)>;
  };

public:
  // Version for std::function
  template <typename... T>
  static func make(callable<T...>&& fn, std::vector<input> inputs, std::vector<output> outputs,
      call_stmt::callable_attrs attrs = {}) {
    callable<T...> impl = std::move(fn);
    assert(sizeof...(T) == inputs.size() + outputs.size());

    auto wrapper = [impl = std::move(impl)](const call_stmt* op, eval_context& ctx) -> index_t {
      return call_impl<T...>(impl, ctx, op, std::make_index_sequence<sizeof...(T)>());
    };

    return func(std::move(wrapper), std::move(inputs), std::move(outputs), attrs);
  }

  // Version for lambdas
  template <typename Lambda>
  static func make(
      Lambda&& lambda, std::vector<input> inputs, std::vector<output> outputs, call_stmt::callable_attrs attrs = {}) {
    // Verify that the lambda returns an index_t; a different return type will fail to match
    // the std::function call and just call this same function in an endless death spiral.
    using sig = lambda_call_signature<Lambda>;
    static_assert(std::is_same_v<typename sig::ret_type, index_t>);

    using std_function_type = typename sig::std_function_type;
    std_function_type impl = std::move(lambda);
    return make(std::move(impl), std::move(inputs), std::move(outputs), attrs);
  }

  // Version for plain old function ptrs
  template <typename... T>
  static func make(index_t (*fn)(const buffer<T>&...), std::vector<input> inputs, std::vector<output> outputs,
      call_stmt::callable_attrs attrs = {}) {
    callable<T...> impl = fn;
    return make(std::move(impl), std::move(inputs), std::move(outputs), attrs);
  }

  // Make a copy from a single input to a single output.
  static func make_copy(input in, output out) { return func(std::move(in), {std::move(out)}); }
  // Make a copy from a single input to a single output, with padding outside the output crop.
  static func make_copy(input in, output out, std::vector<char> padding) {
    return func({std::move(in)}, std::move(out), std::move(padding));
  }
  // Make a copy from multiple inputs with undefined padding.
  static func make_copy(std::vector<input> in, output out) { return func(std::move(in), {std::move(out)}); }
  // Make a concatenation copy. This is a helper function for `make_copy`, where the crop for input i is a `crop_dim` in
  // dimension `dim` on the interval `[bounds[i], bounds[i + 1])`, and the input is translated by `-bounds[i]`.
  static func make_concat(std::vector<buffer_expr_ptr> in, output out, std::size_t dim, std::vector<expr> bounds);
  // Make a stack copy. This is a helper function for `make_copy`, where the crop for input i is a `slice_dim` of
  // dimension `dim` at i. If `dim` is greater than the rank of `out` (the default), the new stack dimension will be the
  // last dimension of the output.
  static func make_stack(std::vector<buffer_expr_ptr> in, output out, std::size_t dim = -1);

  const call_stmt::callable& impl() const { return impl_; }
  const std::vector<input>& inputs() const { return inputs_; }
  const std::vector<output>& outputs() const { return outputs_; }
  const std::optional<std::vector<char>>& padding() const { return padding_; }

  stmt make_call() const;
};

struct build_options {
  // If true, removes bounds checks
  bool no_checks = false;

  // Disable aliasing buffers.
  bool no_alias_buffers = false;
};

// Constructs a body and a pipeline object for a graph described by input and output buffers.
pipeline build_pipeline(node_context& ctx, std::vector<var> args, const std::vector<buffer_expr_ptr>& inputs,
    const std::vector<buffer_expr_ptr>& outputs, const build_options& options = build_options());
pipeline build_pipeline(node_context& ctx, const std::vector<buffer_expr_ptr>& inputs,
    const std::vector<buffer_expr_ptr>& outputs, const build_options& options = build_options());

}  // namespace slinky

#endif  // SLINKY_BUILDER_PIPELINE_H
