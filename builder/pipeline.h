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
  const raw_buffer* constant_;

  memory_type storage_ = memory_type::heap;
  std::optional<loop_id> store_at_;

  buffer_expr(symbol_id sym, index_t elem_size, std::size_t rank);
  buffer_expr(symbol_id sym, const raw_buffer* buffer);
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
  // Make a constant buffer_expr. This does not take ownership of the object, and it must be kept alive as long as this
  // buffer_expr is alive.
  // TODO: This should probably either be some kind of smart pointer, or maybe at least copy the raw_buffer object (but
  // not the underlying data).
  static buffer_expr_ptr make(symbol_id sym, const raw_buffer* buffer);
  static buffer_expr_ptr make(node_context& ctx, const std::string& sym, const raw_buffer* buffer);

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

  const raw_buffer* constant() const { return constant_; }

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
  call_stmt::callable impl_;
  std::vector<input> inputs_;
  std::vector<output> outputs_;

  std::vector<loop_info> loops_;
  std::optional<loop_id> compute_at_;

  std::vector<char> padding_;

  void add_this_to_buffers();
  void remove_this_from_buffers();

public:
  func() {}
  func(call_stmt::callable impl, std::vector<input> inputs, std::vector<output> outputs);
  func(std::vector<input> inputs, output out);
  func(input input, output out, std::vector<char> padding);
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
  template <typename First, typename... Rest>
  static auto build_tuple(eval_context& ctx, const symbol_id* symbols, std::size_t index = 0) {
    if constexpr (sizeof...(Rest) == 0) {
      // Don't use make_tuple() here; it will decay away the references, which we need
      const buffer<First>& b = ctx.lookup_buffer(symbols[index])->template cast<First>();
      return std::tuple<const buffer<First>&>(std::move(b));
    } else {
      return std::tuple_cat(build_tuple<First>(ctx, symbols, index), build_tuple<Rest...>(ctx, symbols, index + 1));
    }
  }

public:
  // Version for plain old function ptrs
  template <typename... T>
  static func make(index_t (*fn)(const buffer<T>&...), std::vector<input> inputs, std::vector<output> outputs) {
    callable<T...> impl = std::move(fn);
    assert(sizeof...(T) == inputs.size() + outputs.size());
    std::array<symbol_id, sizeof...(T)> symbols;
    std::size_t i = 0;
    for (const auto& in : inputs)
      symbols[i++] = in.sym();
    for (const auto& out : outputs)
      symbols[i++] = out.sym();

    const auto wrapper = [symbols = std::move(symbols), impl = std::move(impl)](eval_context& ctx) -> index_t {
      return std::apply(impl, build_tuple<T...>(ctx, symbols.data()));
    };

    return func(wrapper, {std::move(inputs)}, {std::move(outputs)});
  }

  // Version for std::function (usually )
  template <typename... T>
  static func make(callable<T...>&& fn, std::vector<input> inputs, std::vector<output> outputs) {
    callable<T...> impl = std::move(fn);
    assert(sizeof...(T) == inputs.size() + outputs.size());
    std::array<symbol_id, sizeof...(T)> symbols;
    std::size_t i = 0;
    for (const auto& in : inputs)
      symbols[i++] = in.sym();
    for (const auto& out : outputs)
      symbols[i++] = out.sym();

    const auto wrapper = [symbols = std::move(symbols), impl = std::move(impl)](eval_context& ctx) -> index_t {
      return std::apply(impl, build_tuple<T...>(ctx, symbols.data()));
    };

    return func(wrapper, {std::move(inputs)}, {std::move(outputs)});
  }

  static func make_copy(std::vector<input> in, output out) { return func(std::move(in), {std::move(out)}); }
  static func make_copy(input in, output out, std::vector<char> padding = {}) {
    return func(std::move(in), {std::move(out)}, std::move(padding));
  }
  static func make_copy(input in1, input in2, output out) {
    return func({std::move(in1), std::move(in2)}, {std::move(out)});
  }

  const call_stmt::callable& impl() const { return impl_; }
  const std::vector<input>& inputs() const { return inputs_; }
  const std::vector<output>& outputs() const { return outputs_; }
  const std::vector<char>& padding() const { return padding_; }

  stmt make_call() const;
};

struct build_options {
  // If true, removes bounds checks
  bool no_checks = false;
};

// Constructs a body and a pipeline object for a graph described by input and output buffers.
pipeline build_pipeline(node_context& ctx, std::vector<var> args, const std::vector<buffer_expr_ptr>& inputs,
    const std::vector<buffer_expr_ptr>& outputs, const build_options& options = build_options());
pipeline build_pipeline(node_context& ctx, const std::vector<buffer_expr_ptr>& inputs,
    const std::vector<buffer_expr_ptr>& outputs, const build_options& options = build_options());

}  // namespace slinky

#endif  // SLINKY_BUILDER_PIPELINE_H
