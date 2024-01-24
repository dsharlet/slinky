#include "builder/pipeline.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <list>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "builder/infer_bounds.h"
#include "builder/node_mutator.h"
#include "builder/optimizations.h"
#include "builder/simplify.h"
#include "builder/substitute.h"
#include "runtime/evaluate.h"
#include "runtime/expr.h"
#include "runtime/pipeline.h"
#include "runtime/print.h"

namespace slinky {

loop_id loop_id::outermost() {
  // Never accessed, we just need a unique pointer.
  static loop_id id = {reinterpret_cast<const class func*>(&id)};
  return id;
}

loop_id loop_id::innermost() {
  static loop_id id = {reinterpret_cast<const class func*>(&id)};
  return id;
}

bool loop_id::is_outermost() const { return func == outermost().func; }
bool loop_id::is_innermost() const { return func == innermost().func; }

buffer_expr::buffer_expr(symbol_id sym, index_t elem_size, std::size_t rank)
    : sym_(sym), elem_size_(elem_size), producer_(nullptr), constant_(nullptr) {
  dims_.reserve(rank);
  auto var = variable::make(sym);
  for (index_t i = 0; i < static_cast<index_t>(rank); ++i) {
    interval_expr bounds = buffer_bounds(var, i);
    expr stride = buffer_stride(var, i);
    expr fold_factor = buffer_fold_factor(var, i);
    dims_.push_back({bounds, stride, fold_factor});
  }
}

buffer_expr::buffer_expr(symbol_id sym, const raw_buffer* buffer)
    : sym_(sym), elem_size_(buffer->elem_size), producer_(nullptr), constant_(buffer) {
  dims_.reserve(buffer->rank);

  for (index_t d = 0; d < static_cast<index_t>(buffer->rank); ++d) {
    expr min = buffer->dims[d].min();
    expr max = buffer->dims[d].max();
    expr stride = buffer->dims[d].stride();
    expr fold_factor = buffer->dims[d].fold_factor();
    dims_.push_back({bounds(min, max), stride, fold_factor});
  }
}

buffer_expr_ptr buffer_expr::make(symbol_id sym, index_t elem_size, std::size_t rank) {
  return buffer_expr_ptr(new buffer_expr(sym, elem_size, rank));
}

buffer_expr_ptr buffer_expr::make(node_context& ctx, const std::string& sym, index_t elem_size, std::size_t rank) {
  return buffer_expr_ptr(new buffer_expr(ctx.insert_unique(sym), elem_size, rank));
}

buffer_expr_ptr buffer_expr::make(symbol_id sym, const raw_buffer* buffer) {
  return buffer_expr_ptr(new buffer_expr(sym, buffer));
}
buffer_expr_ptr buffer_expr::make(node_context& ctx, const std::string& sym, const raw_buffer* buffer) {
  return buffer_expr_ptr(new buffer_expr(ctx.insert_unique(sym), buffer));
}

void buffer_expr::set_producer(func* f) {
  assert(producer_ == nullptr || f == nullptr);
  producer_ = f;
}

func::func(callable impl, std::vector<input> inputs, std::vector<output> outputs)
    : impl_(std::move(impl)), inputs_(std::move(inputs)), outputs_(std::move(outputs)) {
  add_this_to_buffers();
}

func::func(input input, output out, std::vector<char> padding) : func(nullptr, {std::move(input)}, {std::move(out)}) {
  padding_ = std::move(padding);
}

func::func(std::vector<input> inputs, output out) : func(nullptr, std::move(inputs), {std::move(out)}) {}

func::func(func&& m) { *this = std::move(m); }
func& func::operator=(func&& m) {
  if (this == &m) return *this;
  m.remove_this_from_buffers();
  impl_ = std::move(m.impl_);
  inputs_ = std::move(m.inputs_);
  outputs_ = std::move(m.outputs_);
  loops_ = std::move(m.loops_);
  compute_at_ = std::move(m.compute_at_);
  padding_ = std::move(m.padding_);
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

// Add crops to the inputs of f, using buffer intrinsics to get the bounds of the output.
stmt add_input_crops(stmt result, const func* f) {
  // Find the bounds of the outputs required in each dimension. This is the union of the all the intervals from each
  // output associated with a particular dimension.
  symbol_map<expr> output_mins, output_maxs;
  for (const func::output& o : f->outputs()) {
    for (std::size_t d = 0; d < o.dims.size(); ++d) {
      expr dim_min = o.buffer->dim(d).min();
      expr dim_max = o.buffer->dim(d).max();
      std::optional<expr>& min = output_mins[o.dims[d]];
      std::optional<expr>& max = output_maxs[o.dims[d]];
      min = min ? slinky::min(*min, dim_min) : dim_min;
      max = max ? slinky::max(*max, dim_max) : dim_max;
    }
  }
  // Use the output bounds, and the bounds expressions of the inputs, to determine the bounds required of the input.
  for (const func::input& i : f->inputs()) {
    box_expr crop(i.buffer->rank());
    for (int d = 0; d < static_cast<int>(crop.size()); ++d) {
      // TODO (https://github.com/dsharlet/slinky/issues/21): We may have been given bounds on the input that are
      // smaller than the bounds implied by the output, e.g. in the case of copy with padding.
      expr min = substitute(i.bounds[d].min, output_mins);
      expr max = substitute(i.bounds[d].max, output_maxs);
      // The bounds may have been negated.
      crop[d] = simplify(slinky::bounds(min, max) | slinky::bounds(max, min));
    }
    result = crop_buffer::make(i.sym(), crop, result);
  }
  return result;
}

stmt crop_for_loop(stmt body, const func* f, const func::loop_info& loop) {
  // Crop all the outputs of this buffer for this loop.
  for (const func::output& o : f->outputs()) {
    for (int d = 0; d < static_cast<int>(o.dims.size()); ++d) {
      if (o.dims[d].sym() == loop.sym()) {
        expr loop_max = buffer_max(var(o.sym()), d);
        interval_expr bounds = slinky::bounds(loop.var, min(simplify(loop.var + loop.step - 1), loop_max));
        body = crop_dim::make(o.sym(), d, bounds, body);
      }
    }
  }
  return body;
}

interval_expr get_loop_bounds(const func* f, const func::loop_info& loop) {
  interval_expr bounds = interval_expr::union_identity();
  for (const func::output& o : f->outputs()) {
    for (int d = 0; d < static_cast<int>(o.dims.size()); ++d) {
      if (o.dims[d].sym() == loop.sym()) {
        // This output uses this loop. Add it to the bounds.
        bounds |= o.buffer->dim(d).bounds;
      }
    }
  }
  return simplify(bounds);
}

stmt make_call(const func* f) {
  if (f->impl()) {
    // Make a call_stmt
    call_stmt::symbol_list inputs;
    call_stmt::symbol_list outputs;
    for (const func::input& input : f->inputs()) {
      inputs.push_back(input.sym());
    }
    for (const func::output& output : f->outputs()) {
      outputs.push_back(output.sym());
    }
    return call_stmt::make(f->impl(), std::move(inputs), std::move(outputs));
  } else {
    // Make copy_stmt(s).
    assert(f->padding().empty() || f->inputs().size() == 1);
    assert(f->outputs().size() == 1);
    const func::output& output = f->outputs().front();

    std::vector<stmt> copies;
    for (const func::input& input : f->inputs()) {
      std::vector<expr> src_x;
      std::vector<symbol_id> dst_x;
      for (const interval_expr& i : input.bounds) {
        // Copy bounds must be a single point.
        // TODO: Enforce this via stronger typing in the interface.
        assert(match(i.min, i.max));
        src_x.push_back(i.min);
      }
      for (const var& i : output.dims) {
        dst_x.push_back(i.sym());
      }
      copies.push_back(copy_stmt::make(input.sym(), src_x, output.sym(), dst_x, f->padding()));
    }
    return block::make(copies);
  }
}

// - Add crops for the inputs.
// - Make the loops requested.
stmt make_loops(const func* f, stmt result) {
  result = add_input_crops(result, f);

  // Generate the loops that we want to be explicit.
  for (const auto& loop : f->loops()) {
    // The loop body is done, and we have an actual loop to make here. Crop the body.
    result = crop_for_loop(result, f, loop);
    // And make the actual loop.
    result = loop::make(loop.sym(), loop.mode, get_loop_bounds(f, loop), loop.step, result);
    // Add the input crops here as well, in case something gets inserted here.
    result = add_input_crops(result, f);
  }

  return result;
}

bool operator==(const loop_id& a, const loop_id& b) {
  if (a.is_innermost()) return b.is_innermost();
  if (a.is_outermost()) return b.is_outermost();

  assert(a.var.defined());
  assert(b.var.defined());
  return a.func == b.func && a.var.sym() == b.var.sym();
}

std::set<symbol_id> find_productions(const stmt& s) {
  class finder : public recursive_node_visitor {
  public:
    std::set<symbol_id> result;

    void visit(const call_stmt* op) override { result.insert(op->outputs.begin(), op->outputs.end()); }
    void visit(const copy_stmt* op) override { result.insert(op->dst); }
  };
  finder f;
  s.accept(&f);
  return f.result;
}

struct inserter_loop_id {
  const base_stmt_node* op;
  symbol_id loop;
};

class producer_inserter : public node_mutator {
public:
  stmt to_insert;
  inserter_loop_id at;
  symbol_map<expr> loop_steps;
  // Tracks whether a buffer is in a scope that is redundantly consumed, e.g. the producer for a stencil while inside
  // the loop over the stencil dimension output.
  symbol_map<bool> redundantly_consumed;

  std::set<symbol_id> insert_produces;

  bool found = false;
  std::optional<symbol_id> in_loop;

  producer_inserter(stmt to_insert, inserter_loop_id at)
      : to_insert(to_insert), at(at), insert_produces(find_productions(to_insert)) {}

  stmt mutate(const stmt& s) override {
    if (s.get() == at.op) found = true;
    return node_mutator::mutate(s);
  }

  stmt maybe_insert(stmt before) {
    if (!found) {
      // We haven't found the target compute location yet.
      return before;
    }

    if (!to_insert.defined()) {
      // Already inserted?
      return before;
    }

    if (at.op != nullptr) {
      if (!in_loop || *in_loop != at.loop) {
        // We have a specific compute at location, and we're not there.
        return before;
      }
    }

    stmt result = block::make(to_insert, before);
    to_insert = stmt();
    return result;
  }

  void visit(const loop* op) override {
    auto set_loop_step = set_value_in_scope(loop_steps, op->sym, op->step);
    std::optional<symbol_id> loop_sym = op->sym;
    std::swap(loop_sym, in_loop);
    stmt body = mutate(op->body);

    if (found) {
      body = maybe_insert(body);
    }

    std::swap(loop_sym, in_loop);
    if (body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(clone_with_new_body(op, std::move(body)));
    }
  }

  void visit(const crop_dim* op) override {
    stmt body = mutate(op->body);
    if (found) {
      body = maybe_insert(body);
    }
    if (body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(clone_with_new_body(op, std::move(body)));
    }
  }

  void visit(const crop_buffer* op) override {
    stmt body = mutate(op->body);
    if (found) {
      body = maybe_insert(body);
    }
    if (body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(clone_with_new_body(op, std::move(body)));
    }
  }

  void visit(const call_stmt* op) override {
    for (const symbol_id i : op->inputs) {
      if (insert_produces.count(i)) {
        found = true;
      }
    }
    if (found) {
      set_result(maybe_insert(op));
    } else {
      set_result(op);
    }
  }

  void visit(const copy_stmt* op) override {
    if (insert_produces.count(op->src)) {
      found = true;
    }
    if (found) {
      set_result(maybe_insert(op));
    } else {
      set_result(op);
    }
  }

  void visit(const block* op) override { 
    stmt a = mutate(op->a);
    // If we find the insert point in a, we should not visit b. The insert point must be before a.
    stmt b = !found ? mutate(op->b) : op->b;
    if (a.same_as(op->a) && b.same_as(op->b)) {
      set_result(op);
    } else {
      set_result(block::make(a, b));
    }
  }
};

stmt insert_producer(stmt to_insert, inserter_loop_id at, stmt in) {
  producer_inserter m(to_insert, at);
  in = m.mutate(in);
  if (m.to_insert.defined()) {
    in = block::make(m.to_insert, in);
  }
  return in;
}

class allocate_inserter : public node_mutator {
public:
  buffer_expr_ptr buf;
  inserter_loop_id at;

  bool found = false;

  allocate_inserter(buffer_expr_ptr buf, inserter_loop_id at) : buf(buf), at(at) {}

  stmt make_allocate(stmt around) {
    if (buf) {
      stmt result = allocate::make(buf->sym(), buf->storage(), buf->elem_size(), buf->dims(), around);
      buf = nullptr;
      return result;
    } else {
      return around;
    }
  }

  void visit(const loop* op) {
    stmt body = mutate(op->body);

    if (found && op->sym == at.loop) {
      body = make_allocate(body);
    }

    if (body.same_as(op->body)) {
      set_result(op);
    } else {
      set_result(clone_with_new_body(op, std::move(body)));
    }
  }

  void visit(const call_stmt* op) {
    if (op == at.op) found = true;
    set_result(op);
  }

  void visit(const copy_stmt* op) {
    if (op == at.op) found = true;
    set_result(op);
  }
};

stmt insert_allocate(const buffer_expr_ptr buf, inserter_loop_id at, stmt in) {
  allocate_inserter m(buf, at);
  in = m.mutate(in);
  if (m.buf) {
    // If we failed to find the location, make it at root.
    in = m.make_allocate(in);
  }
  return in;
}

class pipeline_builder {
  // We're going to incrementally build the body, starting at the end of the pipeline and adding
  // producers as necessary.
  std::set<buffer_expr_ptr> to_produce;
  std::set<buffer_expr_ptr> produced;

  stmt result;

public:
  std::map<const func*, const base_stmt_node*> call_to_node;

  pipeline_builder(const std::vector<buffer_expr_ptr>& inputs, const std::vector<buffer_expr_ptr>& outputs,
      std::set<buffer_expr_ptr>& constants) {
    // To start with, we need to produce the outputs.
    for (auto& i : outputs) {
      to_produce.insert(i);
    }
    for (auto& i : inputs) {
      produced.insert(i);
    }

    // Find all the buffers we need to produce.
    while (true) {
      std::set<buffer_expr_ptr> produce_next;
      for (const buffer_expr_ptr& i : to_produce) {
        if (!i->producer()) {
          // Must be an input.
          continue;
        }

        for (const func::input& j : i->producer()->inputs()) {
          if (!to_produce.count(j.buffer)) {
            if (j.buffer->constant()) {
              constants.insert(j.buffer);
            } else if (!j.buffer->producer()) {
              // Must be an input.
              continue;
            } else {
              produce_next.insert(j.buffer);
            }
          }
        }
      }
      if (produce_next.empty()) break;

      to_produce.insert(produce_next.begin(), produce_next.end());
    }
  }

  // f can be called if it doesn't have an output that is consumed by a not yet produced buffer's producer.
  bool can_produce(const func* f) const {
    for (const buffer_expr_ptr& p : to_produce) {
      if (produced.count(p)) {
        // This buffer is already produced.
        continue;
      }
      if (!p->producer()) {
        // Must be an input.
        continue;
      }
      if (p->producer() == f) {
        // This is the producer we are considering now.
        continue;
      }
      for (const func::output& o : f->outputs()) {
        for (const func::input& i : p->producer()->inputs()) {
          if (i.buffer == o.buffer) {
            // f produces a buffer that one of the other yet to be produced buffers needs as an
            // input.
            return false;
          }
        }
      }
    }
    return true;
  }

  // Find the func f to run next. This is the func that produces a buffer we need that we have not
  // yet produced, and all the buffers produced by f are ready to be consumed.
  const func* find_next_producer(const loop_id& at = loop_id()) const {
    for (const buffer_expr_ptr& i : to_produce) {
      if (produced.count(i)) continue;

      if (!i->producer()) {
        // This is probably an input.
        continue;
      }

      if (!can_produce(i->producer())) {
        // This isn't ready to be produced yet.
        continue;
      }

      // We're in the right place, and the func is ready to be computed!
      return i->producer();
    }
    return nullptr;
  }

  bool complete() const { return to_produce.empty(); }

  stmt make_produce(const func* f) {
    for (const func::output& output : f->outputs()) {
      to_produce.erase(output.buffer);
      produced.insert(output.buffer);
    }
    stmt result = make_call(f);
    call_to_node[f] = result.get();
    return make_loops(f, result);
  }
};

void add_buffer_checks(const buffer_expr_ptr& b, bool output, std::vector<stmt>& checks) {
  int rank = static_cast<int>(b->rank());
  expr buf_var = variable::make(b->sym());
  checks.push_back(check::make(buf_var != 0));
  // TODO: Maybe this check is overzealous (https://github.com/dsharlet/slinky/issues/17).
  checks.push_back(check::make(buffer_rank(buf_var) == rank));
  checks.push_back(check::make(buffer_base(buf_var) != 0));
  checks.push_back(check::make(buffer_elem_size(buf_var) == b->elem_size()));
  for (int d = 0; d < rank; ++d) {
    expr fold_factor = buffer_fold_factor(buf_var, d);
    checks.push_back(check::make(b->dim(d).min() == buffer_min(buf_var, d)));
    checks.push_back(check::make(b->dim(d).max() == buffer_max(buf_var, d)));
    checks.push_back(check::make(b->dim(d).stride == buffer_stride(buf_var, d)));
    checks.push_back(check::make(b->dim(d).fold_factor == fold_factor));
    if (output) {
      checks.push_back(check::make(b->dim(d).extent() <= fold_factor));
    }
  }
}

stmt build_pipeline(node_context& ctx, const std::vector<buffer_expr_ptr>& inputs,
    const std::vector<buffer_expr_ptr>& outputs, std::set<buffer_expr_ptr>& constants, const build_options& options) {
  pipeline_builder builder(inputs, outputs, constants);

  stmt result;

  while (!builder.complete()) {
    // Find a buffer to produce.
    const func* f = builder.find_next_producer();

    // Call the producer.
    if (!f) {
      // TODO: Make a better error here.
      std::cerr << "Problem in dependency graph" << std::endl;
      std::abort();
    }

    stmt produce_f = builder.make_produce(f);
    if (!result.defined()) {
      result = produce_f;
    } else if (f->compute_at() && f->compute_at()->is_outermost()) {
      result = block::make(produce_f, result);
    } else {
      inserter_loop_id at = {builder.call_to_node[f->compute_at()->func], f->compute_at()->var.sym()};
      result = insert_producer(produce_f, at, result);
    }
    for (const func::output& output : f->outputs()) {
      const buffer_expr_ptr& buf = output.buffer;
      if (std::find(outputs.begin(), outputs.end(), buf) != outputs.end()) {
        // This is an output to the pipeline, don't need to allocate it.
        continue;
      }
      assert(std::find(inputs.begin(), inputs.end(), buf) == inputs.end());
      inserter_loop_id at = {builder.call_to_node[buf->store_at()->func], buf->store_at()->var.sym()};
      result = insert_allocate(buf, at, result);
    }
    std::cout << std::tie(result, ctx) << std::endl;
  }
  // Add checks that the buffer constraints the user set are satisfied.
  std::vector<stmt> checks;
  for (const buffer_expr_ptr& i : inputs) {
    add_buffer_checks(i, /*output=*/false, checks);
  }
  for (const buffer_expr_ptr& i : outputs) {
    add_buffer_checks(i, /*output=*/true, checks);
  }
  result = block::make(block::make(checks), result);

  std::vector<symbol_id> input_syms;
  input_syms.reserve(inputs.size());
  for (const buffer_expr_ptr& i : inputs) {
    input_syms.push_back(i->sym());
  }
  for (const buffer_expr_ptr& i : constants) {
    input_syms.push_back(i->sym());
  }
  result = infer_bounds(result, ctx, input_syms);

  result = fix_buffer_races(result);

  result = simplify(result);

  if (options.no_checks) {
    class remove_checks : public node_mutator {
    public:
      void visit(const check* op) override { set_result(stmt()); }
    };

    result = remove_checks().mutate(result);
  }

  std::cout << std::tie(result, ctx) << std::endl;

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

}  // namespace

pipeline build_pipeline(node_context& ctx, std::vector<var> args, const std::vector<buffer_expr_ptr>& inputs,
    const std::vector<buffer_expr_ptr>& outputs, const build_options& options) {
  std::set<buffer_expr_ptr> constants;
  stmt body = build_pipeline(ctx, inputs, outputs, constants, options);
  return pipeline(std::move(args), vars(inputs), vars(outputs), std::move(body));
}

pipeline build_pipeline(node_context& ctx, const std::vector<buffer_expr_ptr>& inputs,
    const std::vector<buffer_expr_ptr>& outputs, const build_options& options) {
  return build_pipeline(ctx, {}, inputs, outputs, options);
}

}  // namespace slinky