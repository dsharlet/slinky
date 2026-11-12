#include "slinky/builder/pipeline.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "slinky/base/chrome_trace.h"
#include "slinky/base/function_ref.h"
#include "slinky/builder/node_mutator.h"
#include "slinky/builder/optimizations.h"
#include "slinky/builder/simplify.h"
#include "slinky/builder/slide_and_fold_storage.h"
#include "slinky/builder/substitute.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/depends_on.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/pipeline.h"
#include "slinky/runtime/print.h"
#include "slinky/runtime/stmt.h"

namespace slinky {

// in print.cc, don't want to expose this.
const node_context* set_default_print_context(const node_context* ctx);

loop_ref loop_nest::add_loop(loop_ref parent, std::vector<output_dim> dims,
                             expr step, expr max_workers) {
  assert(parent.index_ >= 0 &&
         static_cast<std::size_t>(parent.index_) < loops_.size());
  assert(!dims.empty());
  assert(step.defined());
  for (const output_dim& d : dims) {
    (void)d;
    assert(d.buf != nullptr);
    assert(d.dim.defined());
  }
  const int index = static_cast<int>(loops_.size());
  loop l;
  l.parent = parent;
  l.dims = std::move(dims);
  l.step = std::move(step);
  l.max_workers = std::move(max_workers);
  loops_.push_back(std::move(l));
  return loop_ref(index);
}

void loop_nest::place(const buffer_expr_ptr& buf, loop_ref at) {
  assert(buf != nullptr);
  assert(at.index_ >= 0 && static_cast<std::size_t>(at.index_) < loops_.size());
  placements_[buf->sym()] = at;
}

void loop_nest::store(const buffer_expr_ptr& buf, loop_ref at) {
  assert(buf != nullptr);
  assert(at.index_ >= 0 && static_cast<std::size_t>(at.index_) < loops_.size());
  stores_[buf->sym()] = at;
}

std::optional<loop_ref> loop_nest::placed_at(var buf) const {
  auto it = placements_.find(buf);
  if (it == placements_.end()) return std::nullopt;
  return it->second;
}

std::optional<loop_ref> loop_nest::stored_at(var buf) const {
  auto it = stores_.find(buf);
  if (it == stores_.end()) return std::nullopt;
  return it->second;
}

bool loop_nest::is_ancestor(loop_ref ancestor, loop_ref descendant) const {
  assert(ancestor.index_ >= 0 &&
         static_cast<std::size_t>(ancestor.index_) < loops_.size());
  assert(descendant.index_ >= 0 &&
         static_cast<std::size_t>(descendant.index_) < loops_.size());
  for (loop_ref cur = descendant;; cur = loops_[cur.index_].parent) {
    if (cur == ancestor) return true;
    if (cur.is_root()) return false;
  }
}

loop_ref loop_nest::lca(loop_ref a, loop_ref b) const {
  assert(a.index_ >= 0 && static_cast<std::size_t>(a.index_) < loops_.size());
  assert(b.index_ >= 0 && static_cast<std::size_t>(b.index_) < loops_.size());
  auto depth_of = [this](loop_ref l) {
    int d = 0;
    while (!l.is_root()) {
      l = loops_[l.index_].parent;
      ++d;
    }
    return d;
  };
  int da = depth_of(a);
  int db = depth_of(b);
  while (da > db) {
    a = loops_[a.index_].parent;
    --da;
  }
  while (db > da) {
    b = loops_[b.index_].parent;
    --db;
  }
  while (a != b) {
    a = loops_[a.index_].parent;
    b = loops_[b.index_].parent;
  }
  return a;
}

buffer_expr::buffer_expr(var sym, std::size_t rank, expr elem_size)
    : sym_(sym), elem_size_(std::move(elem_size)), producer_(nullptr), constant_(nullptr) {
  dims_.reserve(rank);
  for (index_t i = 0; i < static_cast<index_t>(rank); ++i) {
    interval_expr bounds = buffer_bounds(sym, i);
    expr stride = buffer_stride(sym, i);
    expr fold_factor = buffer_fold_factor(sym, i);
    dims_.push_back({bounds, stride, fold_factor});
  }
}

buffer_expr::buffer_expr(var sym, const_raw_buffer_ptr constant_buffer)
    : sym_(sym), elem_size_(constant_buffer->elem_size), producer_(nullptr), constant_(std::move(constant_buffer)) {
  assert(constant_ != nullptr);
  dims_.reserve(constant_->rank);

  for (index_t d = 0; d < static_cast<index_t>(constant_->rank); ++d) {
    expr min = constant_->dims[d].min();
    expr max = constant_->dims[d].max();
    expr stride = constant_->dims[d].stride();
    expr fold_factor = constant_->dims[d].fold_factor();
    dims_.push_back({slinky::bounds(min, max), stride, fold_factor});
  }
}

buffer_expr_ptr buffer_expr::make(var sym, std::size_t rank, expr elem_size) {
  return buffer_expr_ptr(new buffer_expr(sym, rank, std::move(elem_size)));
}

buffer_expr_ptr buffer_expr::make(node_context& ctx, const std::string& sym, std::size_t rank, expr elem_size) {
  return buffer_expr_ptr(new buffer_expr(ctx.insert_unique(sym), rank, std::move(elem_size)));
}

buffer_expr_ptr buffer_expr::make_constant(var sym, const_raw_buffer_ptr constant_buffer) {
  return buffer_expr_ptr(new buffer_expr(sym, std::move(constant_buffer)));
}
buffer_expr_ptr buffer_expr::make_constant(
    node_context& ctx, const std::string& sym, const_raw_buffer_ptr constant_buffer) {
  return buffer_expr_ptr(new buffer_expr(ctx.insert_unique(sym), std::move(constant_buffer)));
}

void buffer_expr::set_producer(func* f) {
  assert(producer_ == nullptr || f == nullptr);
  producer_ = f;
}

box_expr buffer_expr::bounds() const {
  box_expr result(rank());
  for (std::size_t d = 0; d < rank(); ++d) {
    result[d] = dim(d).bounds;
  }
  return result;
}

func::func(call_stmt::callable impl, std::vector<input> inputs, std::vector<output> outputs, std::vector<expr> scalars,
    call_stmt::attributes attrs)
    : impl_(std::move(impl)), attrs_(std::move(attrs)), inputs_(std::move(inputs)), outputs_(std::move(outputs)),
      scalars_(std::move(scalars)) {
  add_this_to_buffers();
}

func::func(copy_stmt::callable impl, std::vector<input> src, output dst)
    : copy_impl_(std::move(impl)), inputs_(std::move(src)), outputs_({std::move(dst)}) {
  add_this_to_buffers();
}

func::func(copy_stmt::callable impl, input src, output dst, input pad)
    : func(std::move(impl), {std::move(src), std::move(pad)}, std::move(dst)) {
  is_padded_copy_ = true;
}

func::func(func&& m) noexcept { *this = std::move(m); }
func& func::operator=(func&& m) noexcept {
  if (this == &m) return *this;
  m.remove_this_from_buffers();
  impl_ = std::move(m.impl_);
  copy_impl_ = std::move(m.copy_impl_);
  inputs_ = std::move(m.inputs_);
  outputs_ = std::move(m.outputs_);
  scalars_ = std::move(m.scalars_);
  loops_ = std::move(m.loops_);
  compute_at_ = std::move(m.compute_at_);
  is_padded_copy_ = m.is_padded_copy_;
  attrs_ = std::move(m.attrs_);
  user_data_ = m.user_data_;
  add_this_to_buffers();
  return *this;
}

func::~func() { remove_this_from_buffers(); }

void func::add_this_to_buffers() {
  for (auto& i : outputs_) {
    i.buffer->set_producer(this);
  }
}
void func::remove_this_from_buffers() {
  for (auto& i : outputs_) {
    i.buffer->set_producer(nullptr);
  }
}

namespace {

stmt make_copy_func(copy_stmt::callable impl, const func::input& src, const func::output& dst, var padding = var()) {
  std::vector<expr> src_x;
  std::vector<var> dst_x;
  for (const interval_expr& i : src.bounds) {
    assert(match(i.min, i.max));
    src_x.push_back(i.min);
  }
  for (const var& i : dst.dims) {
    dst_x.push_back(i);
  }
  stmt copy = copy_stmt::make(impl, src.sym(), src_x, dst.sym(), dst_x, padding);
  if (!src.input_crop.empty()) {
    copy = crop_buffer::make(src.sym(), src.sym(), src.input_crop, copy);
  }
  if (!src.output_crop.empty()) {
    copy = crop_buffer::make(dst.sym(), dst.sym(), src.output_crop, copy);
  }
  if (!src.output_slice.empty()) {
    copy = slice_buffer::make(dst.sym(), dst.sym(), src.output_slice, copy);
  }
  return copy;
}

}  // namespace

stmt func::make_call() const {
  if (impl_) {
    call_stmt::symbol_list inputs;
    call_stmt::symbol_list outputs;
    for (const func::input& i : inputs_) {
      inputs.push_back(i.sym());
    }
    for (const func::output& i : outputs_) {
      outputs.push_back(i.sym());
    }
    return call_stmt::make(impl_, std::move(inputs), std::move(outputs), scalars_, attrs_);
  } else if (is_padded_copy_) {
    assert(inputs_.size() == 2);
    assert(outputs_.size() == 1);
    return make_copy_func(copy_impl_, inputs_[0], outputs_[0], inputs_[1].sym());
  } else {
    std::vector<stmt> copies;
    assert(outputs_.size() == 1);
    for (const func::input& input : inputs_) {
      copies.push_back(make_copy_func(copy_impl_, input, outputs_[0]));
    }
    return block::make(std::move(copies));
  }
}

func func::make_concat(
    std::vector<buffer_expr_ptr> src, output dst, std::size_t dim, std::vector<expr> bounds, copy_stmt::callable impl) {
  assert(src.size() + 1 == bounds.size());
  std::size_t rank = dst.buffer->rank();

  std::vector<func::input> inputs;
  for (std::size_t i = 0; i < src.size(); ++i) {
    // Prepare the input.
    assert(src[i]->rank() == rank);
    func::input input;

    input.buffer = src[i];
    input.bounds.resize(rank);
    for (std::size_t d = 0; d < rank; ++d) {
      input.bounds[d] = point(dst.dims[d]);
    }
    // We translate the input by the bounds to make concatenation a bit more natural (the concatenated buffers will
    // start at index 0 in the concatenated dimension).
    input.bounds[dim] -= bounds[i];

    // Translate the bounds into the crop needed by make_copy.
    // We leave the dimensions not concatenated undefined so infer_bounds will require each input to provide the full
    // output in those dimensions.
    input.input_crop.resize(dim + 1);
    input.input_crop[dim] = range(0, bounds[i + 1] - bounds[i]);
    input.output_crop.resize(dim + 1);
    input.output_crop[dim] = range(bounds[i], bounds[i + 1]);

    inputs.push_back(std::move(input));
  }
  return make_copy(std::move(inputs), std::move(dst), std::move(impl));
}

func func::make_stack(std::vector<buffer_expr_ptr> src, output dst, std::size_t dim, copy_stmt::callable impl) {
  std::size_t rank = dst.buffer->rank();
  assert(dst.dims.size() == rank);
  assert(rank > 0);
  dim = std::min(rank - 1, dim);

  std::vector<func::input> inputs;
  for (std::size_t i = 0; i < src.size(); ++i) {
    // Prepare the input.
    func::input input;
    input.buffer = src[i];
    input.bounds.resize(rank);
    for (std::size_t d = 0; d < rank; ++d) {
      input.bounds[d] = point(dst.dims[d]);
    }

    // Remove the stack dimension of the output from the input bounds, and slice the output at this point.
    input.bounds.erase(input.bounds.begin() + dim);
    input.output_slice.resize(dim + 1);
    input.output_slice[dim] = static_cast<index_t>(i);

    if (input.bounds.size() > src[i]->rank()) {
      input.bounds.resize(src[i]->rank());
    }

    inputs.push_back(std::move(input));
  }
  // Also apply the slice to the output dimensions.
  dst.dims.erase(dst.dims.begin() + dim);
  return make_copy(std::move(inputs), std::move(dst), std::move(impl));
}

namespace {

// This mutator replaces uses of buffer metadata with variables, and later defines those symbols.
// This is used to avoid a massive footgun for users of slinky: when defining bounds expressions, crops, and so on, it
// is very tempting to use buffer metadata. Doing this is problematic because the buffer metadata will be affected by
// crops and slices generated by slinky, which change the meaning of the expressions. This avoids this problem by
// replacing buffer metadata expressions with variables, and remembers the values to define these replacements outside
// of the relevant crops.
class sanitize_user_exprs : public node_mutator {
public:
  node_context& ctx;
  std::map<expr, var, node_less> replacements;
  std::vector<var> external;

  sanitize_user_exprs(node_context& ctx) : ctx(ctx) {}

  stmt define_replacements(stmt s) const {
    std::vector<std::pair<var, expr>> lets;
    lets.reserve(replacements.size());
    for (const auto& i : replacements) {
      lets.push_back({i.second, i.first});
    }
    return let_stmt::make(std::move(lets), std::move(s));
  }

  void visit(const variable* op) override {
    if (op->field == buffer_field::none) {
      node_mutator::visit(op);
      return;
    }

    // Don't lift internally allocated buffer metadata expressions.
    // TODO: This should be a proper API error.
    assert(std::binary_search(external.begin(), external.end(), op->sym));

    auto i = replacements.insert(std::pair<const expr, var>(op, 0));
    if (i.second) {
      i.first->second = ctx.insert_unique("g");
    }
    set_result(variable::make(i.first->second));
  }

  void visit(const call* op) override {
    if (op->intrinsic != intrinsic::buffer_at) {
      node_mutator::visit(op);
      return;
    }

    // Don't lift internally allocated buffer metadata expressions.
    assert(op->args.size() >= 1 && as_variable(op->args[0]));
    // TODO: This should be a proper API error.
    assert(std::binary_search(external.begin(), external.end(), *as_variable(op->args[0])));

    auto i = replacements.insert(std::pair<const expr, var>(op, 0));
    if (i.second) {
      i.first->second = ctx.insert_unique("g");
    }
    set_result(variable::make(i.first->second));
  }
};

int unsliced_dim(const std::vector<expr>& slices, int d) {
  int od = d;
  for (int slice_d = 0; slice_d <= od && slice_d < static_cast<int>(slices.size()); ++slice_d) {
    if (slices[slice_d].defined()) {
      ++od;
    }
  }
  return od;
}

bounds_map get_output_bounds(const std::vector<func::output>& outputs, const std::vector<expr>& slices) {
  bounds_map output_bounds;
  for (const func::output& o : outputs) {
    for (index_t d = 0; d < static_cast<index_t>(o.dims.size()); ++d) {
      // We need to skip the dimensions that we slice when indexing the buffer's dimensions.
      index_t od = unsliced_dim(slices, d);
      std::optional<interval_expr>& output_bounds_d = output_bounds[o.dims[d]];
      if (!output_bounds_d) {
        output_bounds_d = buffer_bounds(o.sym(), od);
      } else {
        *output_bounds_d |= buffer_bounds(o.sym(), od);
      }
    }
  }
  return output_bounds;
}

box_expr compute_input_bounds(
    const func* f, const func::input& i, const bounds_map& output_bounds, sanitize_user_exprs& sanitizer) {
  box_expr crop(i.buffer->rank());
  for (std::size_t d = 0; d < std::min(crop.size(), i.bounds.size()); ++d) {
    crop[d] = bounds_of(sanitizer.mutate(i.bounds[d]), output_bounds);

    if (d < i.input_crop.size()) {
      // We have an output crop for this input, intersect with the crop we have.
      // TODO: It would be nice if this were simply a crop_buffer inserted in the right place. However, that is
      // difficult to do because it could be used in several places, each with a different output crop to apply.
      crop[d] &= sanitizer.mutate(i.input_crop[d]);
    }
  }

  return crop;
}

void topological_sort_impl(const func* f, std::set<const func*>& processing, std::set<const func*>& visited,
    std::vector<const func*>& order, std::map<const func*, std::vector<const func*>>& deps) {
  if (visited.count(f) > 0) {
    return;
  }

  assert(processing.count(f) == 0);
  processing.insert(f);
  for (const auto& i : f->inputs()) {
    const auto& input = i.buffer;
    if (!input->producer()) continue;
    // Record that f is consumer of input->producer.
    deps[input->producer()].push_back(f);
    topological_sort_impl(input->producer(), processing, visited, order, deps);
  }
  processing.erase(f);
  visited.insert(f);
  order.push_back(f);
}

void topological_sort(const std::vector<buffer_expr_ptr>& outputs, std::vector<const func*>& order,
    std::map<const func*, std::vector<const func*>>& deps) {
  std::set<const func*> processing;
  std::set<const func*> visited;
  for (const auto& i : outputs) {
    if (!i->producer()) continue;
    topological_sort_impl(i->producer(), processing, visited, order, deps);
  }

  // Reverse the order, so outputs go first.
  std::reverse(order.begin(), order.end());
}

// Update dims vector by substittuting expression from the map.
std::vector<dim_expr> substitute_from_map(std::vector<dim_expr> dims, span<std::pair<expr, expr>> substitutions) {
  for (dim_expr& dim : dims) {
    dim_expr new_dim = dim;
    for (const std::pair<expr, expr>& j : substitutions) {
      new_dim.bounds.min = substitute(new_dim.bounds.min, j.first, j.second);
      new_dim.bounds.max = substitute(new_dim.bounds.max, j.first, j.second);
      new_dim.stride = substitute(new_dim.stride, j.first, j.second);
      new_dim.fold_factor = substitute(new_dim.fold_factor, j.first, j.second);
    }
    dim = new_dim;
  }
  return dims;
}

// Perform a substitute limited to call or copy inputs only.
stmt substitute_inputs(const stmt& s, const symbol_map<var>& subs) {
  class m : public stmt_mutator {
    const symbol_map<var>& subs;

  public:
    m(const symbol_map<var>& subs) : subs(subs) {}

    void visit(const call_stmt* op) override {
      bool changed = false;
      call_stmt::symbol_list inputs;
      for (const auto& i : op->inputs) {
        if (subs[i]) {
          changed = true;
          inputs.push_back(*subs[i]);
        } else {
          inputs.push_back(i);
        }
      }

      if (changed) {
        set_result(call_stmt::make(op->target, inputs, op->outputs, to_vector(op->scalars), *op->attrs));
      } else {
        set_result(op);
      }
    }

    void visit(const copy_stmt* op) override {
      var src = subs[op->src] ? *subs[op->src] : op->src;
      var pad = subs[op->pad] ? *subs[op->pad] : op->pad;
      if (src != op->src || pad != op->pad) {
        set_result(copy_stmt::make(op->impl, src, to_vector(op->src_x), op->dst, op->dst_x, pad));
      } else {
        set_result(op);
      }
    }
  };
  return m(subs).mutate(s);
}

// Finds if expr depends on any of the buffer_min/buffer_max from dependent_dims.
class dependent_dims_finder : public recursive_node_visitor {
  const std::map<var, std::vector<int>>& dependent_dims;

public:
  bool depends = false;
  dependent_dims_finder(const std::map<var, std::vector<int>>& dependent_dims) : dependent_dims(dependent_dims) {}

  void visit(const variable* v) override {
    switch (v->field) {
    case buffer_field::min:
    case buffer_field::max: {
      if (dependent_dims.count(v->sym) == 0) return;
      for (int d : dependent_dims.at(v->sym)) {
        if (v->dim == d) {
          depends = true;
          return;
        }
      }
      return;
    }
      [[fallthrough]];
    default: return;
    }
  }
};

// Finds which of the bounds depend on any of the buffer_min/buffer_max from dependent_dims.
void find_dependent_dims(
    var sym, const std::vector<interval_expr>& bounds, std::map<var, std::vector<int>>& dependent_dims) {
  for (size_t i = 0; i < bounds.size(); ++i) {
    dependent_dims_finder finder(dependent_dims);
    if (bounds[i].min.defined()) bounds[i].min.accept(&finder);
    if (bounds[i].max.defined()) bounds[i].max.accept(&finder);
    if (finder.depends) {
      dependent_dims[sym].push_back(i);
    }
  }
}

// Lowers a DAG of `func`s and a `loop_nest` schedule into a single Slinky IR
// `stmt` prior to storage folding, copy aliasing, and simplification passes.
//
// Lowering proceeds in four stages:
//
// 1. Schedule normalization and resolution (`build_loop_nest_from_funcs`,
//    `resolve_loop_nest`):
//    - When no external `loop_nest` is supplied, translates legacy per-`func`
//      directives (`func::loops()`, `func::compute_at()`, `store_at()`) into
//      `loops_` (defaulting unpinned buffers to `root()`).
//    - Expands each loop's tiled `(buffer, var)` dimensions to include sibling
//      outputs of multi-output `func`s sharing the same dimension `var`.
//    - Assigns each `func`'s execution scope `compute_at_[f]` to its explicit
//      `loops.place(...)`, its innermost tiling loop, or the LCA of its
//      consumers, and assigns each internal buffer's allocation scope
//      `store_at_[sym]` to the LCA of its definition scope (outside any loops
//      tiling `f`) and all of its consumers, or an explicit outer
//      `loops.store(...)`.
//
// 2. Bounds and lifetime setup (`compute_allocation_bounds`,
//    `substitute_buffer_dims`, `compute_deps_count`):
//    - Infers required bounds backwards from outputs to inputs, substitutes
//      them into internal buffer shapes, and counts buffer consumers and
//      copy-aliasing ordering constraints (`copy_inputs_`, `copy_deps_`).
//
// 3. Recursive loop-tree lowering and allocation layout (`make_loop`):
//    - At each scope `here` (starting from `loops_.root()`), contracts direct
//      `func`s and immediate child loops into items topologically ordered by
//      producer->consumer edges (breaking ties by `order_`), emitting direct
//      calls via `produce` (recording `[lifetime_start, lifetime_end]` ranges,
//      registering buffers at `store_at_`, and rewriting intermediate call/copy
//      inputs to `.uncropped` clones) and recursing into child loops.
//    - Lays out allocations owned by `here` via `lay_out_allocations`, building
//      a balanced (~O(log N)-depth) allocation tree ordered by lifetime length
//      while respecting copy-aliasing nesting constraints
//      (`place_constrained_buffers`) and sinking constants
//      (`place_constant_buffers`).
//
// 4. Per-iteration cropping and `loop` emission (`crop_buffers_for_loop`,
//    `emit_loop_stmt`):
//    - For non-root scopes, finds upstream buffers whose bounds transitively
//      depend on `here`'s tiled dimensions and emits per-iteration crops (or
//      metadata-only `make_buffer` nodes for inner allocations).
//    - Crops `here`'s tiled dimensions to `[loop_var, loop_var + step - 1]`
//      and wraps the body in a `loop` statement over their unioned bounds.
class pipeline_builder {
  node_context& ctx;

  struct allocation_candidate {
    buffer_expr_ptr buffer;
    int deps_count = 0;
    int consumers_produced = 0;
    int lifetime_start = -1;
    int lifetime_end = -1;

    explicit allocation_candidate(buffer_expr_ptr b) : buffer(b) {}
  };

  struct statement_with_range {
    stmt body;
    int start = 0;
    int end = 0;
    // Stores every allocation inside of the range.
    std::set<var> allocations;

    static statement_with_range merge(const statement_with_range& a, const statement_with_range& b) {
      assert(a.end + 1 == b.start);
      statement_with_range r;
      r.body = block::make({a.body, b.body});
      r.start = a.start;
      r.end = b.end;
      std::set_union(a.allocations.begin(), a.allocations.end(), b.allocations.begin(), b.allocations.end(),
          std::inserter(r.allocations, r.allocations.begin()));
      return r;
    }
  };

  // Candidate buffers scheduled for allocation at each loop scope, indexed by
  // `loop_ref::index()` (`0` for `loops_.root()`, `1..loops_.size()-1` for
  // child loops) and sized once in `resolve_loop_nest`.
  std::vector<std::set<var>> candidates_for_allocation_;
  // Information tracking the lifetimes of the buffers.
  symbol_map<allocation_candidate> allocation_info_;
  // `.uncropped` clone symbol for each intermediate buffer.
  symbol_map<var> uncropped_syms_;
  std::set<var> copy_inputs_;
  std::map<var, std::vector<var>> copy_deps_;
  // Direct buffer dependencies wrt to bounds for each buffer.
  std::map<var, std::set<var>> bounds_deps_;

  int functions_produced_ = 0;

  // Topologically sorted functions.
  std::vector<const func*> order_;
  std::map<const func*, std::vector<const func*>> deps_;

  loop_nest loops_;
  // Buffer dimensions `(buffer_sym, dim)` cropped by each `loop_ref::index()`.
  std::vector<std::vector<std::pair<var, int>>> loop_crops_;
  // Functions computed inside each `loop_ref::index()`'s subtree, in `order_`.
  std::vector<std::vector<const func*>> loop_funcs_;
  // Resolved execution scope of each `func` and allocation scope of each
  // intermediate buffer in `loops_`.
  std::map<const func*, loop_ref> compute_at_;
  std::map<var, loop_ref> store_at_;
  bool may_slide_or_fold_ = false;

  symbol_map<box_expr> allocation_bounds_;
  symbol_map<std::vector<dim_expr>> inferred_dims_;
  symbol_map<std::vector<dim_expr>> inferred_shapes_;
  symbol_map<std::vector<interval_expr>> inferred_bounds_;

  std::map<var, buffer_expr_ptr> input_syms_;
  std::map<var, buffer_expr_ptr> output_syms_;
  std::map<var, buffer_expr_ptr> constants_;

  sanitize_user_exprs sanitizer_;

  void substitute_buffer_dims() {
    for (auto i = order_.rbegin(); i != order_.rend(); ++i) {
      const func* f = *i;
      for (const func::output& o : f->outputs()) {
        const buffer_expr_ptr& b = o.buffer;
        if (output_syms_.count(b->sym())) continue;

        // First substitute the bounds.
        std::vector<std::pair<expr, expr>> substitutions;
        assert(allocation_bounds_[b->sym()]);
        const box_expr& bounds = *allocation_bounds_[b->sym()];
        for (index_t d = 0; d < static_cast<index_t>(bounds.size()); ++d) {
          const interval_expr& bounds_d = bounds[d];
          substitutions.emplace_back(buffer_min(b->sym(), d), bounds_d.min);
          substitutions.emplace_back(buffer_max(b->sym(), d), bounds_d.max);
        }
        std::vector<dim_expr> dims = substitute_from_map(b->dims(), substitutions);

        const bool can_fold_buffer =
            store_at_.at(b->sym()) != compute_at_.at(f);
        substitutions.clear();
        for (index_t d = 0; d < static_cast<index_t>(bounds.size()); ++d) {
          substitutions.emplace_back(buffer_stride(b->sym(), d), expr());
          if (!can_fold_buffer) {
            substitutions.emplace_back(buffer_fold_factor(b->sym(), d), expr());
          }
        }
        dims = substitute_from_map(dims, substitutions);
        if (!can_fold_buffer) {
          for (dim_expr& d : dims) {
            if (is_positive_infinity(d.fold_factor)) d.fold_factor = expr();
          }
        } else {
          for (const dim_expr& d : dims) {
            if (!is_finite(d.fold_factor)) {
              may_slide_or_fold_ = true;
              break;
            }
          }
        }

        std::vector<dim_expr> shape;
        std::vector<interval_expr> tmp;
        for (const auto& d : dims) {
          shape.push_back({d.bounds});
          tmp.push_back({d.bounds});
        }

        // Record the inferred dims.
        inferred_dims_[b->sym()] = dims;
        inferred_shapes_[b->sym()] = shape;
        inferred_bounds_[b->sym()] = tmp;
      }
    }

    for (const auto& i : input_syms_) {
      if (!allocation_bounds_[i.first]) continue;
      inferred_bounds_[i.first] = allocation_bounds_[i.first];
    }
  }

  void compute_allocation_bounds() {
    for (const func* f : order_) {
      for (const auto& i : f->inputs()) {
        bounds_map output_bounds = get_output_bounds(f->outputs(), i.output_slice);
        box_expr crop = compute_input_bounds(f, i, output_bounds, sanitizer_);
        auto& bound = allocation_bounds_[i.sym()];
        if (bound) {
          *bound = *bound | crop;
        } else {
          allocation_bounds_[i.sym()] = crop;
        }
        for (const auto& o : f->outputs()) {
          bounds_deps_[i.sym()].insert(o.sym());
        }
      }
    }

    // Check to see if there are any *intermediate* outputs that don't
    // have allocation bounds; if there are, create an empty allocation
    // bounds for them.
    for (const func* f : order_) {
      for (const auto& o : f->outputs()) {
        if (output_syms_.count(o.sym())) continue;
        if (allocation_bounds_[o.sym()]) continue;
        interval_expr empty{1, 0};
        box_expr crop(o.buffer->rank(), empty);
        allocation_bounds_[o.sym()] = crop;
      }
    }
  }

  // Returns generated statement for this function, as well as the
  // lifetime range covered by it.
  statement_with_range produce(const func* f) {
    stmt result =
        substitute_inputs(sanitizer_.mutate(f->make_call()), uncropped_syms_);

    for (const func::output& o : f->outputs()) {
      const buffer_expr_ptr& b = o.buffer;
      if (output_syms_.count(b->sym())) continue;

      uncropped_syms_[b->sym()] =
          ctx.insert_unique(ctx.name(b->sym()) + ".uncropped");
      const int store_idx = store_at_.at(b->sym()).index();
      candidates_for_allocation_[store_idx].insert(b->sym());
      std::optional<allocation_candidate>& info = allocation_info_[b->sym()];
      info->buffer = b;
      info->lifetime_start = functions_produced_;

      if (info->consumers_produced == info->deps_count) {
        info->lifetime_end = functions_produced_;
      }
    }

    for (const auto& i : f->inputs()) {
      const auto& input = i.buffer;
      if (input->constant()) {
        constants_[input->sym()] = input;
      }
      if (!input->producer()) {
        continue;
      }

      std::optional<allocation_candidate>& info = allocation_info_[input->sym()];
      if (!info) continue;
      info->consumers_produced++;

      if (info->consumers_produced == info->deps_count) {
        info->lifetime_end = functions_produced_;
      }
    }

    functions_produced_++;

    return {result, functions_produced_ - 1, functions_produced_ - 1};
  }

  // Wraps provided body statement with the allocation node for a given buffer.
  statement_with_range produce_allocation(const buffer_expr_ptr& b,
                                          statement_with_range s) {
    assert(uncropped_syms_[b->sym()]);
    s.body = clone_buffer::make(*uncropped_syms_[b->sym()], b->sym(), s.body);

    const std::vector<dim_expr>& dims = *inferred_dims_[b->sym()];
    assert(allocation_bounds_[b->sym()]);
    const box_expr& bounds = *allocation_bounds_[b->sym()];
    s.body = allocate::make(b->sym(), b->storage(), b->elem_size(), dims, s.body);

    std::vector<stmt> checks;
    for (std::size_t d = 0; d < std::min(dims.size(), bounds.size()); ++d) {
      checks.push_back(check::make(dims[d].is_broadcast() || dims[d].min() <= bounds[d].min));
      checks.push_back(check::make(dims[d].is_broadcast() || dims[d].max() >= bounds[d].max));
    }

    s.body = block::make(std::move(checks), s.body);
    s.allocations.insert(b->sym());
    return s;
  }

  // Computes number of consumers for each of the buffers.
  void compute_deps_count() {
    for (const func* f : order_) {
      for (const func::output& o : f->outputs()) {
        const buffer_expr_ptr& b = o.buffer;
        if (output_syms_.count(b->sym())) continue;

        if (!allocation_info_[b->sym()]) {
          allocation_info_[b->sym()].emplace(b);
        }

        if (!f->impl() && (f->is_padded_copy() || f->inputs().size() > 1)) {
          // For padded or multi-input copies (e.g. concatenate), the inputs
          // are aliased into slices of the output, so the output must be
          // allocated outside the inputs.
          copy_inputs_.insert(b->sym());
          for (const auto& i : f->inputs()) {
            const auto& input = i.buffer;

            if (!input->producer()) {
              continue;
            }

            copy_deps_[b->sym()].push_back(input->sym());
          }
        }
      }
    }

    for (const func* f : order_) {
      for (const auto& i : f->inputs()) {
        const auto& input = i.buffer;

        if ((!f->impl() && !f->is_padded_copy() && f->inputs().size() <= 1) || (input->constant())) {
          // For 1-input unpadded copies, the output is aliased into the input,
          // so the input must be allocated outside the output.
          copy_inputs_.insert(input->sym());
          for (const func::output& o : f->outputs()) {
            const buffer_expr_ptr& b = o.buffer;
            if (!input->constant() && output_syms_.count(b->sym())) continue;
            copy_deps_[input->sym()].push_back(b->sym());
          }
        }

        if (allocation_info_[input->sym()]) {
          allocation_info_[input->sym()]->deps_count++;
        }
      }
    }
  }

  bool has_all_allocations(buffer_expr_ptr candidate, std::set<var>& allocations) {
    for (const auto& b : copy_deps_[candidate->sym()]) {
      if (allocations.count(b) == 0) return false;
    }

    return true;
  }

  void place_constrained_buffers(
      std::vector<statement_with_range>& results, std::set<var>& candidates,
      const std::vector<allocation_candidate>& special,
      std::vector<allocation_candidate>& new_special) {
    new_special.reserve(special.size());
    for (std::size_t iy = 0; iy < special.size(); iy++) {
      bool found = false;
      for (std::size_t ix = 0; ix < results.size(); ix++) {
        const buffer_expr_ptr& candidate = special[iy].buffer;

        if (!has_all_allocations(candidate, results[ix].allocations)) continue;

        // The block range must fully cover the allocation range.
        if (results[ix].start <= special[iy].lifetime_start && special[iy].lifetime_end <= results[ix].end) {
          results[ix] = produce_allocation(candidate, results[ix]);
          candidates.erase(candidate->sym());
          found = true;
          break;
        }
      }

      if (found) continue;
      new_special.push_back(special[iy]);
    }
  }

  void place_constant_buffers(statement_with_range* results, std::size_t num_results) {
    for (std::size_t ix = 0; ix < num_results; ix++) {
      std::vector<var> constants_to_remove;
      constants_to_remove.reserve(constants_.size());

      for (const auto& i : constants_) {
        const buffer_expr_ptr& candidate = i.second;

        if (!has_all_allocations(candidate, results[ix].allocations)) continue;

        results[ix].body =
            let_stmt::make(candidate->sym(), constant_buffer::make(candidate->constant()), results[ix].body);
        constants_to_remove.push_back(i.first);
      }

      for (var b : constants_to_remove) {
        constants_.erase(b);
      }
    }
  }

  // This attempts to lay out allocation nodes such that the nesting
  // is minimized. The general idea is to iteratively build up a tree of
  // allocations starting from the allocations with the allocations with the
  // shortest life time as the lowest level of the tree. This is not always possible
  // in general to do and there are corner cases where nesting of allocations will
  // have depth N regardless of the approach, but in most practical situations this
  // will produce a structure close to the tree (for example, for the linear pipeline it
  // should build a perfect tree of depth ~log(N)). Similarly, the complexity of this
  // algorithm is O(N^2) for the worst case, but for the most practical pipelines it's
  // should be O(N*log(N)).
  statement_with_range lay_out_allocations(
      std::set<var>& candidates_at, std::vector<statement_with_range> results) {
    // The vector of allocations at this loop level.
    std::vector<allocation_candidate> lifetimes;
    // The same as above, but also has a set of special dependencies to
    // satisfy.
    std::vector<allocation_candidate> special;
    for (const auto& b : candidates_at) {
      if (copy_inputs_.count(b) > 0) {
        special.push_back(*allocation_info_[b]);
      } else {
        lifetimes.push_back(*allocation_info_[b]);
      }
    }

    // Sort vector by (end - start) and then start and then buffer sym.
    auto lifetime_less = [](allocation_candidate a, allocation_candidate b) {
      if ((a.lifetime_end - a.lifetime_start) == (b.lifetime_end - b.lifetime_start)) {
        if (a.lifetime_start == b.lifetime_start) {
          return a.buffer->sym() < b.buffer->sym();
        }
        return a.lifetime_start < b.lifetime_start;
      }

      return (a.lifetime_end - a.lifetime_start) < (b.lifetime_end - b.lifetime_start);
    };

    std::sort(lifetimes.begin(), lifetimes.end(), lifetime_less);
    std::sort(special.begin(), special.end(), lifetime_less);

    int iteration_count = 0;
    while (true) {
      std::vector<allocation_candidate> new_lifetimes;
      std::vector<statement_with_range> new_results;
      std::vector<allocation_candidate> new_special;

      new_lifetimes.reserve(lifetimes.size());
      new_results.reserve(results.size());

      std::size_t result_index = 0;
      for (std::size_t ix = 0; ix < lifetimes.size() && result_index < results.size();) {
        // Skip function bodies which go before the current buffer.
        while (result_index < results.size() && results[result_index].end < lifetimes[ix].lifetime_start) {
          new_results.push_back(results[result_index]);
          result_index++;
        }

        int new_min = std::numeric_limits<int>::max();
        int new_max = std::numeric_limits<int>::min();

        // Find which function bodies overlap with the lifetime of the buffer.
        std::vector<stmt> new_block;
        std::set<var> combined_allocs;
        while (result_index < results.size() && results[result_index].start <= lifetimes[ix].lifetime_end &&
               lifetimes[ix].lifetime_start <= results[result_index].end) {
          new_min = std::min(new_min, results[result_index].start);
          new_max = std::max(new_max, results[result_index].end);
          new_block.push_back(results[result_index].body);
          combined_allocs.insert(results[result_index].allocations.begin(), results[result_index].allocations.end());
          result_index++;
        }

        // Combine overlapping function bodies and wrap them into current buffer allocation.
        if (!new_block.empty()) {
          stmt new_body = block::make(new_block);

          buffer_expr_ptr b = lifetimes[ix].buffer;

          statement_with_range new_result = {new_body, new_min, new_max, std::move(combined_allocs)};
          new_result = produce_allocation(b, new_result);
          new_results.push_back(new_result);

          candidates_at.erase(b->sym());
        }

        // Move to the next buffer.
        ix++;

        // Skip buffers which go before the next statement range/.
        while (ix < lifetimes.size() && lifetimes[ix].lifetime_start <= new_max) {
          new_lifetimes.push_back(lifetimes[ix]);
          ix++;
        }
      }

      new_results.insert(new_results.end(), results.begin() + result_index, results.end());

      // See if any of the blocks can be wrapped in the allocations which are inputs to the copy.
      // This only can happen if all of it's dependencies are inside of the block.
      place_constrained_buffers(new_results, candidates_at, special,
                                new_special);

      // Attempt to place constant buffers as close to their usage location as possible.
      place_constant_buffers(new_results.data(), new_results.size());

      // No changes, so leave the loop.
      bool no_changes = (lifetimes.size() == new_lifetimes.size() && special.size() == new_special.size());

      lifetimes = std::move(new_lifetimes);
      special = std::move(new_special);
      results = std::move(new_results);

      if (no_changes) break;

      iteration_count++;
    }

    if (results.empty()) return {};

    statement_with_range result;
    result = results.front();
    for (std::size_t ix = 1; ix < results.size(); ix++) {
      result = statement_with_range::merge(result, results[ix]);
    }

    // Add all remaining allocations at this loop level. The allocations can be added in any order. This order enables
    // aliasing copy dsts to srcs, which is more flexible than aliasing srcs to dsts.
    for (const func* f : order_) {
      for (const func::output& o : f->outputs()) {
        const buffer_expr_ptr& b = o.buffer;
        if (candidates_at.count(b->sym()) == 0) continue;

        result = produce_allocation(b, result);
        candidates_at.erase(b->sym());
      }
    }

    // Try again for the combined statement.
    place_constant_buffers(&result, 1);

    return result;
  }

  // Returns true if v is connected to one of the terminal node.
  bool find_transitive_deps_impl(
      const var& v, const std::set<var>& terminal, std::set<var>& visited, std::set<var>& result) {
    // This is a terminal node, so return true.
    if (terminal.count(v)) {
      return true;
    }

    visited.insert(v);

    // If any of the connected nodes are connected to the terminal node, this one is connected too.
    bool can_reach_terminal = false;
    for (const var& p : bounds_deps_[v]) {
      // No point in visiting the node we've already seen.
      // NOTE(vksnk): this check is O(lg N) and we can make it O(1) by using
      // an additional `visited` array, but I don't think it's critical for
      // performance at all.
      if (visited.count(p) > 0) {
        // result set only contains nodes connected to the terminal nodes.
        can_reach_terminal = can_reach_terminal || (result.count(p) > 0);
        continue;
      }

      // This is not embedded into `or` expression below, because we still need to visit all of
      // the child nodes even if can_reach_terminal is already true.
      bool can_reach_terminal_from_p = find_transitive_deps_impl(p, terminal, visited, result);
      can_reach_terminal = can_reach_terminal || can_reach_terminal_from_p;
    }

    // We only keep nodes connected to the terminal nodes.
    if (can_reach_terminal) result.insert(v);

    return can_reach_terminal;
  }

  // Find transitive dependencies for a set of buffers wrt their bounds dependencies using DFS.
  void find_transitive_deps(const std::vector<var>& buffers, const std::set<var>& terminal, std::set<var>& result) {
    std::set<var> visited;
    for (const var& v : buffers) {
      if (constants_.count(v)) continue;
      find_transitive_deps_impl(v, terminal, visited, result);
    }
  }

  bool has_external_output(const func* f) const {
    return std::any_of(f->outputs().begin(), f->outputs().end(),
                       [this](const func::output& o) {
                         return output_syms_.count(o.sym()) > 0;
                       });
  }

  // Lowers legacy per-`func` scheduling directives (`func::loops()`,
  // `func::compute_at()`, and `buffer_expr::store_at()`) into an equivalent
  // `loop_nest` so `pipeline_builder` has a single scheduling representation.
  loop_nest build_loop_nest_from_funcs() const {
    loop_nest loops;
    std::map<std::pair<const func*, var>, loop_ref> id_to_ref;
    std::map<const func*, loop_ref> func_innermost;

    auto resolve_id = [&](const loop_id& id) {
      if (id.root()) return loops.root();
      auto it = id_to_ref.find({id.func, id.var});
      assert(it != id_to_ref.end());
      return it->second;
    };

    for (const func* f : order_) {
      loop_ref compute_at = loops.root();
      auto deps_it = deps_.find(f);
      if (!has_external_output(f) && deps_it != deps_.end() &&
          !deps_it->second.empty()) {
        compute_at = func_innermost.at(deps_it->second[0]);
        for (std::size_t k = 1; k < deps_it->second.size(); ++k) {
          compute_at =
              loops.lca(compute_at, func_innermost.at(deps_it->second[k]));
        }
      }
      if (f->compute_at()) {
        loop_ref explicit_at = resolve_id(*f->compute_at());
        assert(loops.is_ancestor(explicit_at, compute_at));
        compute_at = explicit_at;
      }

      // `func::loops()` lists loops from innermost to outermost.
      loop_ref parent = compute_at;
      for (auto l = f->loops().rbegin(); l != f->loops().rend(); ++l) {
        assert(!f->outputs().empty());
        parent = loops.add_loop(parent, {f->outputs()[0].buffer, l->var},
                                l->step, l->max_workers);
        id_to_ref[{f, l->var}] = parent;
      }
      func_innermost[f] = parent;
      if (f->compute_at() && !f->outputs().empty()) {
        loops.place(f->outputs()[0].buffer, parent);
      }

      // Preserve legacy default root storage when `store_at()` is unset so
      // `slide_and_fold_storage` folds unpinned buffers into root allocations.
      for (const func::output& o : f->outputs()) {
        if (output_syms_.count(o.sym())) continue;
        const buffer_expr_ptr& b = o.buffer;
        if (!b->store_at()) {
          loops.store(b, loops.root());
        } else if (b->store_at()->innermost(f)) {
          loops.store(b, compute_at);
        } else {
          loops.store(b, resolve_id(*b->store_at()));
        }
      }
    }

    return loops;
  }

  // Resolves the `loops_` schedule into concrete per-loop crops and
  // per-`func` / per-buffer scopes in two passes:
  // 1. For each loop `l`, maps its symbolic `loop_nest::output_dim{buf, dim}`
  //    entries to concrete `(buf_sym, dim_index)` pairs in `loop_crops_[idx]`
  //    (including sibling outputs of multi-output `func`s that share `dim`) and
  //    records `l` in `func_tiling_loops[f]`.
  // 2. Walks `order_` from outputs to inputs (consumers before producers) to
  //    assign each `func`'s execution scope `compute_at_[f]` (explicit
  //    `place`, innermost tiling loop, or consumer LCA) and each internal
  //    buffer's allocation scope `store_at_[buf]` (explicit `store` or LCA of
  //    its definition scope and all consumers).
  void resolve_loop_nest() {
    std::map<const func*, std::vector<loop_ref>> func_tiling_loops;
    candidates_for_allocation_.resize(loops_.size());
    loop_crops_.resize(loops_.size());
    loop_funcs_.resize(loops_.size());
    const std::set<const func*> reachable_funcs(order_.begin(), order_.end());
    // Pass 1: expand each loop's symbolic `output_dim`s into `(buf, dim)` pairs
    // in `loop_crops_` and collect `func_tiling_loops`.
    for (std::size_t idx = 1; idx < loops_.size(); ++idx) {
      loop_ref l = loops_.loop_at(idx);
      std::vector<std::pair<var, int>>& crops = loop_crops_[idx];
      for (const loop_nest::output_dim& out_dim : loops_[l].dims) {
        const func* f = out_dim.buf->producer();
        assert(f != nullptr);
        if (reachable_funcs.count(f) == 0) continue;
        std::vector<loop_ref>& tiling = func_tiling_loops[f];
        if (std::find(tiling.begin(), tiling.end(), l) == tiling.end()) {
          tiling.push_back(l);
        }
        const std::vector<expr> no_slices;
        const std::vector<expr>& slices =
            f->inputs().empty() ? no_slices : f->inputs()[0].output_slice;
        bool matched = false;
        for (const func::output& o : f->outputs()) {
          for (int d = 0; d < static_cast<int>(o.dims.size()); ++d) {
            if (o.dims[d] == out_dim.dim) {
              matched = true;
              const auto pair =
                  std::make_pair(o.sym(), unsliced_dim(slices, d));
              if (std::find(crops.begin(), crops.end(), pair) == crops.end()) {
                crops.push_back(pair);
              }
            }
          }
        }
        assert(matched);
      }
    }

    // Pass 2: walk `order_` (consumers before producers) to resolve
    // `compute_at_[f]` and `store_at_[sym]`, tracking the running LCA of
    // consumer execution scopes in `buf_consumer_lca`.
    std::map<var, loop_ref> buf_consumer_lca;
    for (const func* f : order_) {
      // Check if any output of `f` was explicitly pinned via `loops_.place()`.
      std::optional<loop_ref> explicit_place;
      for (const func::output& o : f->outputs()) {
        if (std::optional<loop_ref> p = loops_.placed_at(o.sym())) {
          assert(!explicit_place || *explicit_place == *p);
          explicit_place = *p;
        }
      }

      // `f` is emitted once and must execute inside every loop that tiles it.
      // Because each loop has a single parent, all loops tiling `f` must nest
      // inside one another along one chain between `outermost_tiling` and
      // `innermost_tiling` (matching `f->loops()` in the legacy API).
      loop_ref innermost_tiling = loops_.root();
      loop_ref outermost_tiling = loops_.root();
      auto tiling_it = func_tiling_loops.find(f);
      if (tiling_it != func_tiling_loops.end()) {
        innermost_tiling = outermost_tiling = tiling_it->second[0];
        for (std::size_t k = 1; k < tiling_it->second.size(); ++k) {
          loop_ref l = tiling_it->second[k];
          if (loops_.is_ancestor(innermost_tiling, l)) {
            innermost_tiling = l;
          } else {
            assert(loops_.is_ancestor(l, innermost_tiling));
            if (loops_.is_ancestor(l, outermost_tiling)) {
              outermost_tiling = l;
            }
          }
        }
      }

      const bool is_external = has_external_output(f);

      // Resolve `compute_at_[f]`: explicit `place` wins, then
      // `innermost_tiling`, then (for internal untiled `func`s) the LCA of all
      // consumers.
      loop_ref compute_at = loops_.root();
      if (explicit_place) {
        assert(loops_.is_ancestor(innermost_tiling, *explicit_place));
        compute_at = *explicit_place;
      } else if (!innermost_tiling.is_root()) {
        compute_at = innermost_tiling;
      } else if (!is_external) {
        bool first = true;
        for (const func::output& o : f->outputs()) {
          auto it = buf_consumer_lca.find(o.sym());
          if (it == buf_consumer_lca.end()) continue;
          compute_at = first ? it->second : loops_.lca(compute_at, it->second);
          first = false;
        }
      }
      // External outputs are not cropped by consumer loops, so every enclosing
      // loop of `f` must tile `f`.
      if (is_external) {
        for (loop_ref l = compute_at; !l.is_root(); l = loops_[l].parent) {
          assert(tiling_it != func_tiling_loops.end() &&
                 std::find(tiling_it->second.begin(), tiling_it->second.end(),
                           l) != tiling_it->second.end());
        }
      }
      compute_at_[f] = compute_at;
      for (loop_ref l = compute_at; !l.is_root(); l = loops_[l].parent) {
        loop_funcs_[l.index()].push_back(f);
      }

      // An output of `f` is defined once all loops tiling `f` finish
      // (`def_scope`), and must remain live across `def_scope` and all of its
      // consumers (`min_store`).
      const loop_ref def_scope = outermost_tiling.is_root()
                                     ? compute_at
                                     : loops_[outermost_tiling].parent;
      for (const func::output& o : f->outputs()) {
        var sym = o.sym();
        // If a multi-output `func` is tiled by loop `l` along a dimension not
        // present on output `sym`, no consumer of `sym` may execute inside `l`.
        if (tiling_it != func_tiling_loops.end() && f->outputs().size() > 1 &&
            deps_.count(f)) {
          for (loop_ref l : tiling_it->second) {
            const auto& l_crops = loop_crops_[l.index()];
            if (std::any_of(l_crops.begin(), l_crops.end(),
                            [sym](const auto& p) { return p.first == sym; })) {
              continue;
            }
            for (const func* c : deps_.at(f)) {
              for (const func::input& in : c->inputs()) {
                if (in.sym() == sym) {
                  assert(!loops_.is_ancestor(l, compute_at_.at(c)));
                }
              }
            }
          }
        }
        if (output_syms_.count(sym)) continue;
        auto cons_it = buf_consumer_lca.find(sym);
        const loop_ref min_store = (cons_it != buf_consumer_lca.end())
                                       ? loops_.lca(def_scope, cons_it->second)
                                       : def_scope;
        if (std::optional<loop_ref> explicit_store = loops_.stored_at(sym)) {
          assert(loops_.is_ancestor(*explicit_store, min_store));
          store_at_[sym] = *explicit_store;
        } else {
          store_at_[sym] = min_store;
        }
      }

      // Propagate `f`'s execution scope to its input buffers' consumer LCAs.
      for (const func::input& in : f->inputs()) {
        auto [it, inserted] =
            buf_consumer_lca.try_emplace(in.sym(), compute_at);
        if (!inserted) {
          it->second = loops_.lca(it->second, compute_at);
        }
      }
    }
  }

 public:
  pipeline_builder(node_context& ctx,
                   const std::vector<buffer_expr_ptr>& inputs,
                   const std::vector<buffer_expr_ptr>& outputs,
                   const loop_nest* loops = nullptr)
      : ctx(ctx), sanitizer_(ctx) {
    topological_sort(outputs, order_, deps_);

    sanitizer_.external.reserve(outputs.size() + inputs.size());
    for (auto& i : outputs) {
      output_syms_[i->sym()] = i;
      sanitizer_.external.push_back(i->sym());
    }
    for (const buffer_expr_ptr& i : inputs) {
      input_syms_[i->sym()] = i;
      sanitizer_.external.push_back(i->sym());
    }
    std::sort(sanitizer_.external.begin(), sanitizer_.external.end());

    if (loops != nullptr) {
      assert(std::all_of(order_.begin(), order_.end(), [](const func* f) {
        return f->loops().empty() && !f->compute_at() &&
               std::all_of(f->outputs().begin(), f->outputs().end(),
                           [](const func::output& o) {
                             return !o.buffer->store_at();
                           });
      }));
      loops_ = *loops;
    } else {
      loops_ = build_loop_nest_from_funcs();
    }
    resolve_loop_nest();

    // Compute allocation bounds.
    compute_allocation_bounds();

    // Substitute inferred bounds into user provided dims.
    substitute_buffer_dims();

    // Compute number of consumers for each of the buffers.
    compute_deps_count();
  }

  const std::vector<var>& external_symbols() const { return sanitizer_.external; }
  bool may_slide_or_fold() const { return may_slide_or_fold_; }

  stmt add_crop(stmt body, var sym, const std::vector<interval_expr>& bounds, const std::vector<int>& dependent_dim) {
    // In order to just produce crop without taking into account what dims can change use :
    // body = crop_buffer::make(sym, sym, bounds, body);
    if (dependent_dim.size() > 1) {
      std::vector<interval_expr> needed(bounds.size(), {expr(), expr()});
      for (int d : dependent_dim) {
        needed[d] = bounds[d];
      }
      body = crop_buffer::make(sym, sym, needed, body);
    } else if (dependent_dim.size() == 1) {
      int dim = dependent_dim[0];
      body = crop_dim::make(sym, sym, dim, bounds[dim], body);
    }

    return body;
  }

  // An executable unit directly inside a loop scope: either a single `func`
  // computed at that scope, or an immediate child loop (standing in for all
  // `func`s computed anywhere inside that child loop's subtree).
  struct scope_item {
    const func* direct_func = nullptr;
    loop_ref child_loop;
  };

  statement_with_range make_loop() { return make_loop(loops_.root()); }

  // Recursively lowers a `loop_nest` scope `here` (and its child loops) into
  // buffer allocations, per-iteration crops, and a `loop` statement.
  statement_with_range make_loop(loop_ref here) {
    // 1. Order the direct `func`s and immediate child loops of `here` so every
    //    producer runs before its consumers, then emit each item in sequence.
    //    Sequential emission matters because `produce()` stamps each `func`
    //    with a monotonic `functions_produced_` index used as the
    //    `[lifetime_start, lifetime_end]` interval during allocation layout.
    std::vector<scope_item> items = order_scope_items(here);
    if (items.empty()) return {};

    std::vector<statement_with_range> results;
    results.reserve(items.size());
    for (const scope_item& item : items) {
      results.push_back(item.direct_func ? produce(item.direct_func)
                                         : make_loop(item.child_loop));
    }

    // 2. Wrap the emitted statements in `allocate` (and `clone_buffer` for
    //    `.uncropped` inputs) for all intermediate buffers stored at `here`.
    statement_with_range body = lay_out_allocations(
        candidates_for_allocation_[here.index()], std::move(results));

    // The root scope has no enclosing loop or per-iteration cropping.
    if (here.is_root()) return body;

    // 3. Wrap the body with per-iteration crops (or metadata-only `make_buffer`
    //    nodes for buffers allocated inside `here`) for upstream buffers whose
    //    bounds depend on `here`'s tiled dimensions.
    const std::vector<std::pair<var, int>>& crops = loop_crops_[here.index()];
    body.body = crop_buffers_for_loop(here, std::move(body.body),
                                      body.allocations, crops);

    // 4. Crop `here`'s own tiled output dimensions to
    //    `[loop_var, loop_var + step - 1]` and wrap the body in a `loop`.
    body.body = emit_loop_stmt(here, crops, std::move(body.body));
    return body;
  }

  // Returns the direct `func`s and immediate child loops of `here` in a valid
  // execution order.
  //
  // A plain scan of `order_` is not enough when a child loop contains multiple
  // `func`s whose producers and consumers at `here` interleave: because a child
  // loop executes as a single statement at `here`, it must run after every item
  // at `here` that feeds *any* `func` in its subtree and before every item at
  // `here` that consumes from *any* `func` in its subtree. Contracting each
  // immediate child loop's subtree into one `scope_item` and topologically
  // sorting those items enforces both constraints.
  std::vector<scope_item> order_scope_items(loop_ref here) const {
    const std::vector<const func*>& funcs =
        here.is_root() ? order_ : loop_funcs_[here.index()];
    std::vector<scope_item> items;
    std::vector<std::vector<std::size_t>> preds;
    std::map<const func*, std::size_t> func_to_item;
    std::map<loop_ref, std::size_t> child_to_item;

    // `order_` (and `loop_funcs_`) lists outputs/consumers first and inputs/
    // producers last, so iterating in reverse visits every producer before all
    // of its consumers.
    for (auto it = funcs.rbegin(); it != funcs.rend(); ++it) {
      const func* f = *it;
      loop_ref loc = compute_at_.at(f);

      // Map `f` to its item at `here`: a new item if `f` runs directly at
      // `here`, or the shared item for the immediate child loop of `here`
      // containing `loc`.
      std::size_t item_idx = items.size();
      if (loc == here) {
        items.push_back({f, loop_ref()});
        preds.emplace_back();
      } else {
        while (loops_[loc].parent != here) {
          loc = loops_[loc].parent;
        }
        auto [child_it, inserted] = child_to_item.try_emplace(loc, item_idx);
        if (inserted) {
          items.push_back({nullptr, loc});
          preds.emplace_back();
        }
        item_idx = child_it->second;
      }
      func_to_item[f] = item_idx;

      // Record dependencies on earlier items in `here`'s subtree. Producers
      // computed outside `here` are not in `func_to_item` and already dominate
      // `here`; dependencies within the same child loop (`pit->second ==
      // item_idx`) are ordered when `make_loop` recurses into that child loop.
      for (const func::input& in : f->inputs()) {
        auto pit = func_to_item.find(in.buffer->producer());
        if (pit != func_to_item.end() && pit->second != item_idx) {
          preds[item_idx].push_back(pit->second);
        }
      }
    }
    // Post-order DFS emits every predecessor item before its dependents while
    // preserving `order_` tie-breaking among independent items.
    enum class visit_state { unvisited, visiting, visited };
    std::vector<visit_state> state(items.size(), visit_state::unvisited);
    std::vector<scope_item> ordered;
    ordered.reserve(items.size());

    auto visit = [&](auto& self, std::size_t item_idx) -> void {
      const visit_state cur_state = state[item_idx];
      if (cur_state == visit_state::visited) return;
      // Hitting an in-progress (`visiting`) item means two sibling items have a
      // circular producer/consumer dependency across `here` (e.g. a child loop
      // both consumes from and feeds a `func` or sibling loop at `here`).
      assert(cur_state == visit_state::unvisited);
      state[item_idx] = visit_state::visiting;
      for (std::size_t pred_idx : preds[item_idx]) {
        self(self, pred_idx);
      }
      state[item_idx] = visit_state::visited;
      ordered.push_back(items[item_idx]);
    };
    for (std::size_t i = 0; i < items.size(); ++i) {
      visit(visit, i);
    }
    return ordered;
  }

  // Wraps `body` with per-iteration crops (or metadata-only `make_buffer`
  // nodes for inner allocations) for upstream buffers varying with `crops`.
  stmt crop_buffers_for_loop(loop_ref here, stmt body,
                             const std::set<var>& allocations,
                             const std::vector<std::pair<var, int>>& crops) {
    // Copy inputs (`input_buffers_used`) don't need transitive bounds
    // expansion, whereas produced and allocated buffers (`buffers_used`) do.
    // Because `store_at_[sym]` is an ancestor of `compute_at_[f]` for every
    // output `sym` of `f`, every buffer allocated inside `here` (`allocations`)
    // is produced by some `f` in `loop_funcs_[here.index()]`.
    std::vector<var> input_buffers_used;
    std::vector<var> buffers_used;
    for (const func* f : loop_funcs_[here.index()]) {
      for (const func::output& o : f->outputs()) {
        buffers_used.push_back(o.sym());
      }
      if (!f->impl()) {
        for (const func::input& in : f->inputs()) {
          if (allocations.count(in.sym()) == 0) {
            input_buffers_used.push_back(in.sym());
          }
        }
      }
    }

    std::set<var> all_deps;
    std::set<var> terminal;
    std::map<var, std::vector<int>> dependent_dims;
    for (const auto& [sym, dim] : crops) {
      terminal.insert(sym);
      dependent_dims[sym].push_back(dim);
    }

    // We also need to include transitive dependencies of the used buffers in case metadata
    // needs to be updated.
    find_transitive_deps(buffers_used, terminal, all_deps);

    // Add buffers which are used as inputs into the set as well.
    for (var v : input_buffers_used) {
      all_deps.insert(v);
    }

    // Propagate dependent dims starting from the outer crop.
    for (auto i = order_.begin(); i != order_.end(); ++i) {
      const func* f = *i;
      for (const func::output& o : f->outputs()) {
        const buffer_expr_ptr& b = o.buffer;
        if (!inferred_bounds_[b->sym()]) continue;
        if (all_deps.count(b->sym()) == 0) continue;
        find_dependent_dims(b->sym(), *inferred_bounds_[b->sym()], dependent_dims);
      }
    }

    // Add crops for the buffers and their dependencies used inside of the body using previously inferred bounds.
    // Input syms should be the innermost.
    for (const auto& i : input_syms_) {
      var sym = i.first;
      if (!allocation_bounds_[sym]) continue;
      if (all_deps.count(sym) == 0) continue;
      find_dependent_dims(sym, *allocation_bounds_[sym], dependent_dims);
      body = add_crop(body, sym, *allocation_bounds_[sym], dependent_dims[sym]);
    }

    // Followed by intermediate buffers in the reverse topological order
    // (i.e. the outermost buffers are closer to the outputs of the pipeline).
    for (auto i = order_.rbegin(); i != order_.rend(); ++i) {
      const func* f = *i;
      for (const func::output& o : f->outputs()) {
        const buffer_expr_ptr& b = o.buffer;
        if (!inferred_bounds_[b->sym()]) continue;
        if (all_deps.count(b->sym()) == 0) continue;
        if (allocations.count(b->sym()) > 0) {
          // We can't produce the crop for the buffers which weren't allocated yet,
          // so produce make_buffer, because some other buffers might use metadata from them.
          const std::optional<std::vector<dim_expr>>& maybe_dims = inferred_shapes_[b->sym()];
          if (!maybe_dims) continue;
          body = make_buffer::make(b->sym(), expr(), expr(), *maybe_dims, body);
        } else {
          std::vector<int> dims_to_crop;
          for (int d : dependent_dims[b->sym()]) {
            const auto pair = std::make_pair(b->sym(), d);
            if (std::find(crops.begin(), crops.end(), pair) == crops.end()) {
              dims_to_crop.push_back(d);
            }
          }
          body = add_crop(body, b->sym(), *inferred_bounds_[b->sym()], dims_to_crop);
        }
      }
    }
    return body;
  }

  // Crops `crops` to a single `loop_var` step and wraps `body` in `loop`.
  stmt emit_loop_stmt(loop_ref here,
                      const std::vector<std::pair<var, int>>& crops,
                      stmt body) {
    assert(!crops.empty());
    const loop_nest::loop& l = loops_[here];
    // Create a unique loop variable (e.g. `buf.x`, or `x` for multi-output
    // funcs) from the first tiled dimension.
    const loop_nest::output_dim& primary = l.dims[0];
    const func* f = primary.buf->producer();
    std::string name;
    if (f->outputs().size() == 1) {
      name = ctx.name(primary.buf->sym()) + ".";
    }
    name += ctx.name(primary.dim);
    var loop_var = ctx.insert_unique(name);

    // Crop each dimension tiled by `here` to `[loop_var, loop_var + step - 1]`
    // and union their uncropped bounds for the loop range.
    expr step = sanitizer_.mutate(l.step);
    interval_expr tile =
        slinky::bounds(loop_var, simplify(loop_var + step - 1));
    interval_expr loop_bounds = interval_expr::union_identity();
    for (const auto& [buf, dim] : crops) {
      body = crop_dim::make(buf, buf, dim, tile, body);
      loop_bounds |= buffer_bounds(buf, dim);
    }
    return loop::make(loop_var, sanitizer_.mutate(l.max_workers),
                      simplify(loop_bounds), step, body);
  }

  stmt define_sanitized_replacements(const stmt& body) { return sanitizer_.define_replacements(body); }

  // Add checks that the inputs are sufficient based on inferred bounds.
  stmt add_input_checks(const stmt& body) {
    std::vector<stmt> checks;
    for (const auto& i : input_syms_) {
      const std::optional<box_expr>& bounds = allocation_bounds_[i.first];
      if (!bounds) {
        // This input must have been unused, ignore it.
        continue;
      }
      for (int d = 0; d < static_cast<int>(bounds->size()); ++d) {
        checks.push_back(check::make(i.second->dim(d).min() <= (*bounds)[d].min));
        checks.push_back(check::make(i.second->dim(d).max() >= (*bounds)[d].max));
      }
    }
    return block::make(std::move(checks), std::move(body));
  }

  // Wrap the statement into make_buffer-s to define the bounds of allocations.
  stmt make_buffers(stmt body) {
    // Place all remaining constant_buffer-s.
    for (std::pair<var, buffer_expr_ptr> i : constants_) {
      body = let_stmt::make(i.first, constant_buffer::make(i.second->constant()), std::move(body));
    }
    for (auto i = order_.rbegin(); i != order_.rend(); ++i) {
      const func* f = *i;
      for (const func::output& o : f->outputs()) {
        const buffer_expr_ptr& b = o.buffer;
        const std::optional<std::vector<dim_expr>>& maybe_dims = inferred_shapes_[b->sym()];
        if (!maybe_dims) continue;
        body = make_buffer::make(b->sym(), expr(), expr(), *maybe_dims, std::move(body));
      }
    }
    return body;
  }
};

void check_buffer(const buffer_expr_ptr& b, std::vector<stmt>& checks) {
  int rank = static_cast<int>(b->rank());
  checks.push_back(check::make(expr(b->sym()) != 0));
  checks.push_back(check::make(buffer_rank(b->sym()) == rank));
  checks.push_back(check::make(validate_buffer(b->sym())));
}

void check_buffer_constraints(const buffer_expr_ptr& b, bool output, std::vector<stmt>& checks) {
  int rank = static_cast<int>(b->rank());
  checks.push_back(check::make(buffer_elem_size(b->sym()) == b->elem_size()));
  for (int d = 0; d < rank; ++d) {
    expr fold_factor = buffer_fold_factor(b->sym(), d);
    checks.push_back(check::make(b->dim(d).min() == buffer_min(b->sym(), d)));
    checks.push_back(check::make(b->dim(d).max() == buffer_max(b->sym(), d)));
    checks.push_back(check::make(or_else(b->dim(d).extent() == 1, b->dim(d).stride == buffer_stride(b->sym(), d))));
    checks.push_back(check::make(or_else(b->dim(d).extent() == 1, b->dim(d).fold_factor == fold_factor)));
    if (output) {
      checks.push_back(check::make(or_else(fold_factor <= 0, b->dim(d).extent() <= fold_factor)));
    }
  }
}

bool is_verbose() {
  auto* s = std::getenv("SLINKY_VERBOSE");
  return (s && std::atoi(s) == 1);
}

// This function inserts calls to trace_begin and trace_end around calls and loops.
stmt inject_traces(const stmt& s, node_context& ctx) {
  class injector : public stmt_mutator {
  public:
    node_context& ctx;
    var token;
    var names_buf;
    std::vector<char> names;
    std::map<std::string, index_t> name_to_offset;

    injector(node_context& ctx) : ctx(ctx), token(ctx, "__trace_token"), names_buf(ctx, "__trace_names") {}

    // Returns a pointer to the names buffer for the argument.
    expr get_trace_arg(const std::string& arg) {
      // If we already have this name, use it. Otherwise, add it to the names buffer.
      auto i = name_to_offset.insert(std::pair<std::string, index_t>(arg, names.size()));
      if (i.second) {
        names.insert(names.end(), arg.begin(), arg.end());
        names.push_back(0);
      }
      expr args[] = {i.first->second};
      return buffer_at(names_buf, args);
    }

    expr get_trace_arg(const call_stmt* op) {
      if (!op->attrs->name.empty()) {
        return get_trace_arg(op->attrs->name);
      } else {
        return get_trace_arg("call");
      }
    }

    stmt add_trace(stmt s, expr trace_arg) {
      expr trace_begin = call::make(intrinsic::trace_begin, {trace_arg});
      expr trace_end = call::make(intrinsic::trace_end, {token});
      return let_stmt::make({{token, trace_begin}}, block::make({std::move(s), check::make(trace_end)}));
    }

    void visit(const call_stmt* op) override { set_result(add_trace(stmt(op), get_trace_arg(op))); }
    void visit(const loop* op) override {
      expr iter_name = get_trace_arg("loop " + ctx.name(op->sym) + " iteration");
      expr loop_name = get_trace_arg("loop " + ctx.name(op->sym));
      stmt body = add_trace(mutate(op->body), iter_name);
      stmt result = clone_with(op, std::move(body));
      set_result(add_trace(std::move(result), loop_name));
    }
    void visit(const async* op) override {
      expr task_name = get_trace_arg("task " + ctx.name(op->sym));
      stmt task = add_trace(mutate(op->task), task_name);
      set_result(async::make(op->sym, std::move(task), mutate(op->body)));
    }
  };

  injector m(ctx);
  stmt result = m.mutate(s);
  result = m.add_trace(result, m.get_trace_arg("pipeline"));
  buffer<char, 1> names_const_buf(m.names.data(), m.names.size());
  result = let_stmt::make(m.names_buf, constant_buffer::make(raw_buffer::make_copy(names_const_buf)), result);
  return result;
}

std::vector<var> vars(const std::vector<buffer_expr_ptr>& bufs) {
  std::vector<var> result;
  result.reserve(bufs.size());
  for (const buffer_expr_ptr& i : bufs) {
    result.push_back(i->sym());
  }
  return result;
}

pipeline build_pipeline_impl(node_context& ctx, std::vector<var> args,
                             const std::vector<buffer_expr_ptr>& inputs,
                             const std::vector<buffer_expr_ptr>& outputs,
                             std::vector<std::pair<var, expr>> lets,
                             const build_options& options,
                             const loop_nest* loops) {
  scoped_trace trace("build_pipeline");
  const node_context* old_context = set_default_print_context(&ctx);

  pipeline_builder builder(ctx, inputs, outputs, loops);

  stmt result = builder.make_loop().body;
  result = builder.add_input_checks(result);
  result = builder.make_buffers(result);
  result = builder.define_sanitized_replacements(result);

  std::vector<stmt> constraint_checks;
  for (const buffer_expr_ptr& i : inputs) {
    check_buffer_constraints(i, /*output=*/false, constraint_checks);
  }
  for (const buffer_expr_ptr& i : outputs) {
    check_buffer_constraints(i, /*output=*/true, constraint_checks);
  }
  result = block::make(std::move(constraint_checks), std::move(result));

  // Add user provided lets after checking buffers are non-null and of expected rank, but before checking constraints,
  // which may use these lets.
  result = let_stmt::make(std::move(lets), result);

  std::vector<stmt> buffer_checks;
  for (const buffer_expr_ptr& i : inputs) {
    check_buffer(i, buffer_checks);
  }
  for (const buffer_expr_ptr& i : outputs) {
    check_buffer(i, buffer_checks);
  }
  result = block::make(std::move(buffer_checks), std::move(result));

  if (builder.may_slide_or_fold()) {
    result = slide_and_fold_storage(result, ctx);
  }
  result = deshadow(result, builder.external_symbols(), ctx);
  result = lift_constants(result);
  result = simplify(result);

  // Try to reuse buffers and eliminate copies where possible.
  if (!options.no_alias_buffers) {
    result = alias_copies(result, ctx, inputs, outputs);
    result = alias_in_place(result, outputs);
  }

  // `evaluate` currently can't handle `copy_stmt`, so this is required.
  result = implement_copies(result, ctx);

  result = remove_pure_dims(result);

  // `implement_copies` adds shadowed declarations, remove them before simplifying.
  result = deshadow(result, builder.external_symbols(), ctx);

  result = cleanup_semaphores(result);

  result = simplify(result);

  result = fuse_siblings(result);

  if (options.no_checks) {
    result = recursive_mutate<check>(
        result, [](const check* op) { return has_side_effects(op->condition) ? stmt(op) : stmt(); });
  }

  result = lift_constants(result);
  result = simplify(result);

  result = insert_early_free(result);

  if (options.trace) {
    result = inject_traces(result, ctx);
  }

  // This pass adds closures around parallel loop bodies, any following passes need to maintain this closure.
  // The args, inputs, and outputs are bound by `pipeline::setup` even if they are unused, so we must not reuse them.
  std::vector<var> external_symbols(args.begin(), args.end());
  external_symbols.insert(external_symbols.end(), builder.external_symbols().begin(), builder.external_symbols().end());
  result = optimize_symbols(result, ctx, external_symbols);

  result = canonicalize_nodes(result);

  // This pass rewrites the nodes to be compact in memory, the stmt should not be modified after this.
  result = compact_nodes(result);

  if (is_verbose()) {
    std::cout << result << std::endl;
  }

  set_default_print_context(old_context);

  pipeline p;
  p.args = std::move(args);
  p.inputs = vars(inputs);
  p.outputs = vars(outputs);
  p.body = std::move(result);
  return p;
}

}  // namespace

pipeline build_pipeline(node_context& ctx, std::vector<var> args,
                        const std::vector<buffer_expr_ptr>& inputs,
                        const std::vector<buffer_expr_ptr>& outputs,
                        std::vector<std::pair<var, expr>> lets,
                        const build_options& options) {
  return build_pipeline_impl(ctx, std::move(args), inputs, outputs,
                             std::move(lets), options, nullptr);
}

pipeline build_pipeline(node_context& ctx,
                        const std::vector<buffer_expr_ptr>& inputs,
                        const std::vector<buffer_expr_ptr>& outputs,
                        const build_options& options) {
  return build_pipeline(ctx, {}, inputs, outputs, {}, options);
}

pipeline build_pipeline(node_context& ctx, std::vector<var> args,
                        const std::vector<buffer_expr_ptr>& inputs,
                        const std::vector<buffer_expr_ptr>& outputs,
                        const loop_nest& loops,
                        std::vector<std::pair<var, expr>> lets,
                        const build_options& options) {
  return build_pipeline_impl(ctx, std::move(args), inputs, outputs,
                             std::move(lets), options, &loops);
}

pipeline build_pipeline(node_context& ctx,
                        const std::vector<buffer_expr_ptr>& inputs,
                        const std::vector<buffer_expr_ptr>& outputs,
                        const loop_nest& loops, const build_options& options) {
  return build_pipeline(ctx, {}, inputs, outputs, loops, {}, options);
}

}  // namespace slinky
