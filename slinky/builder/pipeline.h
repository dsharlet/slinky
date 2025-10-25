#ifndef SLINKY_BUILDER_PIPELINE_H
#define SLINKY_BUILDER_PIPELINE_H

#include <type_traits>

#include "slinky/base/ref_count.h"
#include "slinky/runtime/evaluate.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/pipeline.h"

namespace slinky {

class func;
class buffer_expr;

using buffer_expr_ptr = ref_count<buffer_expr>;

struct loop_id {
  const slinky::func* func = nullptr;
  slinky::var var;

  bool innermost(const slinky::func* other) const { return other == func && !var.defined(); }
  bool root() const { return !func; }
  // TODO: Deprecated
  slinky::var sym() const { return var; }
};

inline bool operator==(const loop_id& a, const loop_id& b) {
  if (!a.func) {
    return !b.func;
  } else if (a.func == b.func) {
    assert(a.var.defined());
    assert(b.var.defined());
    return a.var == b.var;
  } else {
    return false;
  }
}

// Represents a symbolic buffer in a pipeline.
class buffer_expr : public ref_counted<buffer_expr> {
  var sym_;
  expr elem_size_;
  std::vector<dim_expr> dims_;

  func* producer_;
  const_raw_buffer_ptr constant_;

  memory_type storage_ = memory_type::automatic;
  std::optional<loop_id> store_at_;

  buffer_expr(var sym, std::size_t rank, expr elem_size);
  buffer_expr(var sym, const_raw_buffer_ptr constant_buffer);
  buffer_expr(const buffer_expr&) = delete;
  buffer_expr(buffer_expr&&) = delete;
  buffer_expr& operator=(const buffer_expr&) = delete;
  buffer_expr& operator=(buffer_expr&&) = delete;

  // Only func::func should add producers/consumers.
  friend class func;
  void set_producer(func* f);

public:
  static buffer_expr_ptr make(var sym, std::size_t rank, expr elem_size);
  static buffer_expr_ptr make(node_context& ctx, const std::string& sym, std::size_t rank, expr elem_size);
  // Make a constant buffer_expr. It takes ownership of the buffer from the caller.
  static buffer_expr_ptr make_constant(var sym, const_raw_buffer_ptr constant_buffer);
  static buffer_expr_ptr make_constant(node_context& ctx, const std::string& sym, const_raw_buffer_ptr constant_buffer);
  template <typename T, typename = typename std::enable_if_t<std::is_trivial_v<T>>>
  static buffer_expr_ptr make_scalar(var sym, const T& value) {
    return make_constant(sym, raw_buffer::make_scalar<T>(value));
  }
  template <typename T, typename = typename std::enable_if_t<std::is_trivial_v<T>>>
  static buffer_expr_ptr make_scalar(node_context& ctx, const std::string& sym, const T& value) {
    return make_constant(ctx, sym, raw_buffer::make_scalar<T>(value));
  }

  var sym() const { return sym_; }
  expr& elem_size() { return elem_size_; }
  const expr& elem_size() const { return elem_size_; }
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

namespace internal {

template <typename T>
struct buffer_converter {
  static SLINKY_INLINE const auto& convert(const raw_buffer* buffer) {
    return buffer->cast<typename std::remove_cv<typename std::remove_reference<T>::type>::type::element>();
  }
};
template <>
struct buffer_converter<raw_buffer> {
  static SLINKY_INLINE const raw_buffer& convert(const raw_buffer* buffer) { return *buffer; }
};
template <>
struct buffer_converter<const raw_buffer&> {
  static SLINKY_INLINE const raw_buffer& convert(const raw_buffer* buffer) { return *buffer; }
};
template <>
struct buffer_converter<const raw_buffer*> {
  static SLINKY_INLINE const raw_buffer* convert(const raw_buffer* buffer) { return buffer; }
};

}  // namespace internal

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

    // A region to crop the input to while consuming this input. Only used by copies.
    box_expr input_crop;

    // A region to crop the output to while consuming this input. Only used by copies.
    box_expr output_crop;

    // Slices to apply to the output while consuming this input. Only used by copies.
    std::vector<expr> output_slice;

    var sym() const { return buffer->sym(); }
  };

  struct output {
    buffer_expr_ptr buffer;

    std::vector<var> dims;

    var sym() const { return buffer->sym(); }
  };

  struct loop_info {
    slinky::var var;
    expr step;
    expr max_workers;

    loop_info() = default;
    loop_info(slinky::var var, expr step = 1, expr max_workers = loop::serial)
        : var(var), step(step), max_workers(max_workers) {}

    slinky::var sym() const { return var; }

    bool defined() const { return var.defined() && step.defined(); }
  };

private:
  call_stmt::callable impl_;
  call_stmt::attributes attrs_;
  copy_stmt::callable copy_impl_;
  // A pointer to the optional user data.
  void* user_data_ = nullptr;

  std::vector<input> inputs_;
  std::vector<output> outputs_;
  std::vector<expr> scalars_;
  // If this is true, `inputs_` must have 2 elements, where the second input is the padding.
  bool is_padded_copy_ = false;

  std::vector<loop_info> loops_;
  std::optional<loop_id> compute_at_;

  void add_this_to_buffers();
  void remove_this_from_buffers();

public:
  func() = default;
  func(call_stmt::callable impl, std::vector<input> inputs, std::vector<output> outputs, std::vector<expr> scalars,
      call_stmt::attributes attrs = {});
  func(copy_stmt::callable impl, std::vector<input> inputs, output out);
  func(copy_stmt::callable impl, input src, output dst, input pad);
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
  static SLINKY_INLINE index_t call_impl(
      const func::callable<T...>& impl, eval_context& ctx, const call_stmt* op, std::index_sequence<Indices...>) {
    return impl(
        ctx.lookup_buffer(Indices < op->inputs.size() ? op->inputs[Indices] : op->outputs[Indices - op->inputs.size()])
            ->template cast<T>()...);
  }

  template <typename ArgTypes, typename Fn, std::size_t... Indices>
  static SLINKY_INLINE index_t call_impl_tuple(
      const Fn& impl, eval_context& ctx, const call_stmt* op, std::index_sequence<Indices...>) {
    return impl(
        internal::buffer_converter<typename std::tuple_element<Indices, ArgTypes>::type>::convert(ctx.lookup_buffer(
            Indices < op->inputs.size() ? op->inputs[Indices] : op->outputs[Indices - op->inputs.size()]))...);
  }

  template <typename Lambda>
  struct lambda_call_signature : lambda_call_signature<decltype(&Lambda::operator())> {};

  template <typename ReturnType, typename ClassType, typename... Args>
  struct lambda_call_signature<ReturnType (ClassType::*)(Args...) const> {
    using ret_type = ReturnType;
    using arg_types = std::tuple<Args...>;
    using std_function_type = std::function<ReturnType(Args...)>;
  };

  template <typename... T>
  static func make_impl(
      callable<T...>&& fn, std::vector<input> inputs, std::vector<output> outputs, call_stmt::attributes attrs = {}) {
    callable<T...> impl = std::move(fn);
    assert(sizeof...(T) == inputs.size() + outputs.size());

    auto wrapper = [impl = std::move(impl)](const call_stmt* op, eval_context& ctx) -> index_t {
      return call_impl<T...>(impl, ctx, op, std::make_index_sequence<sizeof...(T)>());
    };

    return func(std::move(wrapper), std::move(inputs), std::move(outputs), {}, std::move(attrs));
  }

public:
  // Version for std::function
  template <typename... T>
  static func make(
      callable<T...>&& fn, std::vector<input> inputs, std::vector<output> outputs, call_stmt::attributes attrs = {}) {
    return make_impl(std::move(fn), std::move(inputs), std::move(outputs), std::move(attrs));
  }

  // Version for lambdas
  template <typename Lambda>
  static func make(
      Lambda&& lambda, std::vector<input> inputs, std::vector<output> outputs, call_stmt::attributes attrs = {}) {
    using sig = lambda_call_signature<Lambda>;
    // Verify that the lambda returns an index_t; a different return type will fail to match
    // the std::function call and just call this same function in an endless death spiral.
    static_assert(std::is_same_v<typename sig::ret_type, index_t>);

    auto wrapper = [lambda = std::move(lambda)](const call_stmt* op, eval_context& ctx) -> index_t {
      return call_impl_tuple<typename sig::arg_types>(
          lambda, ctx, op, std::make_index_sequence<std::tuple_size<typename sig::arg_types>::value>());
    };

    return func(std::move(wrapper), std::move(inputs), std::move(outputs), {}, std::move(attrs));
  }

  // Version for plain old function ptrs
  template <typename... T>
  static func make(index_t (*fn)(const buffer<T>&...), std::vector<input> inputs, std::vector<output> outputs,
      call_stmt::attributes attrs = {}) {
    callable<T...> impl = fn;
    return make_impl(std::move(impl), std::move(inputs), std::move(outputs), std::move(attrs));
  }

  // The following functions make various forms of copy operations. `impl` is a function that can customize the
  // implementation of the copy. The function must be equivalent to `slinky::copy`. The `impl` function may not be
  // called if the copy is aliased.

  // Make a copy from a single input to a single output.
  static func make_copy(input src, output dst, copy_stmt::callable impl = slinky::copy) {
    return func(std::move(impl), {std::move(src)}, std::move(dst));
  }
  // Make a copy from a single input to a single output, with padding outside the output crop.
  static func make_copy(input src, output dst, input pad, copy_stmt::callable impl = slinky::copy) {
    return func(std::move(impl), std::move(src), std::move(dst), std::move(pad));
  }
  // Make a copy from multiple inputs with undefined padding.
  static func make_copy(std::vector<input> src, output dst, copy_stmt::callable impl = slinky::copy) {
    return func(std::move(impl), std::move(src), std::move(dst));
  }
  // Make a concatenation copy. This is a helper function for `make_copy`, where the crop for input i is a `crop_dim` in
  // dimension `dim` on the interval `[bounds[i], bounds[i + 1])`, and the input is translated by `-bounds[i]`.
  static func make_concat(std::vector<buffer_expr_ptr> src, output dst, std::size_t dim, std::vector<expr> bounds,
      copy_stmt::callable impl = slinky::copy);
  // Make a stack copy. This is a helper function for `make_copy`, where the crop for input i is a `slice_dim` of
  // dimension `dim` at i. If `dim` is greater than the rank of `out` (the default), the new stack dimension will be the
  // last dimension of the output.
  static func make_stack(
      std::vector<buffer_expr_ptr> src, output dst, std::size_t dim = -1, copy_stmt::callable impl = slinky::copy);

  const call_stmt::callable& impl() const { return impl_; }
  const std::vector<input>& inputs() const { return inputs_; }
  const std::vector<output>& outputs() const { return outputs_; }
  const call_stmt::attributes& attrs() const { return attrs_; }
  const void* user_data() const { return user_data_; }
  void*& user_data() { return user_data_; }
  bool is_padded_copy() const { return is_padded_copy_; }

  stmt make_call() const;
};

struct build_options {
  // If true, removes bounds checks
  bool no_checks = false;

  // Disable aliasing buffers.
  bool no_alias_buffers = false;

  // Generate trace_begin/trace_end calls to log the pipeline execution.
  bool trace = false;
};

// Constructs a body and a pipeline object for a graph described by input and output buffers.
pipeline build_pipeline(node_context& ctx, std::vector<var> args, const std::vector<buffer_expr_ptr>& inputs,
    const std::vector<buffer_expr_ptr>& outputs, std::vector<std::pair<var, expr>> lets = {},
    const build_options& options = build_options());
pipeline build_pipeline(node_context& ctx, const std::vector<buffer_expr_ptr>& inputs,
    const std::vector<buffer_expr_ptr>& outputs, const build_options& options = build_options());

}  // namespace slinky

#endif  // SLINKY_BUILDER_PIPELINE_H
