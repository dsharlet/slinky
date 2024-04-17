#include "runtime/expr.h"
#include "runtime/stmt.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "runtime/util.h"

namespace slinky {

var::var(node_context& ctx, const std::string& name) : var(ctx.insert_unique(name)) {}

expr var::operator-() const { return -expr(*this); }

std::string node_context::name(var v) const {
  if (v.id < sym_to_name.size()) {
    return sym_to_name[v.id];
  } else {
    return "<" + std::to_string(v.id) + ">";
  }
}

var node_context::insert(const std::string& name) {
  std::optional<var> sym = lookup(name);
  if (!sym) {
    sym = var(sym_to_name.size());
    sym_to_name.push_back(name);
  }
  return *sym;
}
var node_context::insert_unique(const std::string& prefix) {
  std::string name = prefix;
  for (std::size_t i = 0; i < sym_to_name.size(); ++i) {
    if (!lookup(name)) break;
    name = prefix + std::to_string(i);
  }
  return insert(name);
}
std::optional<var> node_context::lookup(const std::string& name) const {
  // TODO: At some point we might need a better data structure than doing this linear search.
  for (std::size_t i = 0; i < sym_to_name.size(); ++i) {
    if (sym_to_name[i] == name) {
      return var(i);
    }
  }
  return std::nullopt;
}

template <typename T>
const T* make_bin_op(expr a, expr b) {
  auto n = new T();
  if (T::commutative && should_commute(a, b)) {
    // Aggressively canonicalizing the order is a big speedup by avoiding unnecessary simplifier rewrites.
    std::swap(a, b);
  }
  n->a = std::move(a);
  n->b = std::move(b);
  return n;
}

template <typename T, typename Body>
const T* make_let(std::vector<std::pair<var, expr>> lets, Body body) {
  auto n = new T();
  n->lets = std::move(lets);
  if (const T* l = body.template as<T>()) {
    n->lets.insert(n->lets.end(), l->lets.begin(), l->lets.end());
    n->body = l->body;
  } else {
    n->body = std::move(body);
  }
  return n;
}

expr let::make(std::vector<std::pair<var, expr>> lets, expr body) {
  return make_let<let>(std::move(lets), std::move(body));
}

stmt let_stmt::make(std::vector<std::pair<var, expr>> lets, stmt body) {
  return make_let<let_stmt>(std::move(lets), std::move(body));
}

const variable* make_variable(var sym) {
  auto n = new variable();
  n->sym = sym;
  return n;
}

const constant* make_constant(index_t value) {
  auto n = new constant();
  n->value = value;
  return n;
}

expr::expr(index_t x) : expr(make_constant(x)) {}
expr::expr(var sym) : expr(make_variable(sym)) {}

expr variable::make(var sym) { return make_variable(sym); }

expr constant::make(index_t value) { return make_constant(value); }
expr constant::make(const void* value) { return make(reinterpret_cast<index_t>(value)); }

expr add::make(expr a, expr b) { return make_bin_op<add>(std::move(a), std::move(b)); }
expr sub::make(expr a, expr b) { return make_bin_op<sub>(std::move(a), std::move(b)); }
expr mul::make(expr a, expr b) { return make_bin_op<mul>(std::move(a), std::move(b)); }
expr div::make(expr a, expr b) { return make_bin_op<div>(std::move(a), std::move(b)); }
expr mod::make(expr a, expr b) { return make_bin_op<mod>(std::move(a), std::move(b)); }
expr min::make(expr a, expr b) { return make_bin_op<min>(std::move(a), std::move(b)); }
expr max::make(expr a, expr b) { return make_bin_op<max>(std::move(a), std::move(b)); }
expr equal::make(expr a, expr b) { return make_bin_op<equal>(std::move(a), std::move(b)); }
expr not_equal::make(expr a, expr b) { return make_bin_op<not_equal>(std::move(a), std::move(b)); }
expr less::make(expr a, expr b) { return make_bin_op<less>(std::move(a), std::move(b)); }
expr less_equal::make(expr a, expr b) { return make_bin_op<less_equal>(std::move(a), std::move(b)); }
expr logical_and::make(expr a, expr b) { return make_bin_op<logical_and>(std::move(a), std::move(b)); }
expr logical_or::make(expr a, expr b) { return make_bin_op<logical_or>(std::move(a), std::move(b)); }
expr logical_not::make(expr a) {
  logical_not* n = new logical_not();
  n->a = std::move(a);
  return n;
}

expr operator+(expr a, expr b) { return add::make(std::move(a), std::move(b)); }
expr operator-(expr a, expr b) { return sub::make(std::move(a), std::move(b)); }
expr operator*(expr a, expr b) { return mul::make(std::move(a), std::move(b)); }
expr operator/(expr a, expr b) { return div::make(std::move(a), std::move(b)); }
expr operator%(expr a, expr b) { return mod::make(std::move(a), std::move(b)); }
expr euclidean_div(expr a, expr b) { return div::make(std::move(a), std::move(b)); }
expr euclidean_mod(expr a, expr b) { return mod::make(std::move(a), std::move(b)); }
expr min(expr a, expr b) { return min::make(std::move(a), std::move(b)); }
expr max(expr a, expr b) { return max::make(std::move(a), std::move(b)); }
expr clamp(expr x, expr a, expr b) {
  if (a.defined()) x = max::make(std::move(x), std::move(a));
  if (b.defined()) x = min::make(std::move(x), std::move(b));
  return x;
}
expr select(expr c, expr t, expr f) { return select::make(std::move(c), std::move(t), std::move(f)); }
expr operator==(expr a, expr b) { return equal::make(std::move(a), std::move(b)); }
expr operator!=(expr a, expr b) { return not_equal::make(std::move(a), std::move(b)); }
expr operator<(expr a, expr b) { return less::make(std::move(a), std::move(b)); }
expr operator<=(expr a, expr b) { return less_equal::make(std::move(a), std::move(b)); }
expr operator>(expr a, expr b) { return less::make(std::move(b), std::move(a)); }
expr operator>=(expr a, expr b) { return less_equal::make(std::move(b), std::move(a)); }
expr operator&&(expr a, expr b) { return logical_and::make(std::move(a), std::move(b)); }
expr operator||(expr a, expr b) { return logical_or::make(std::move(a), std::move(b)); }
expr operator!(expr x) { return logical_not::make(std::move(x)); }

expr min(span<expr> x) {
  if (x.empty()) {
    return expr();
  } else if (x.size() == 1) {
    return x[0];
  } else {
    return min(x[0], min(x.subspan(1)));
  }
}

expr max(span<expr> x) {
  if (x.empty()) {
    return expr();
  } else if (x.size() == 1) {
    return x[0];
  } else {
    return max(x[0], max(x.subspan(1)));
  }
}

const interval_expr& interval_expr::all() {
  static interval_expr x = {negative_infinity(), positive_infinity()};
  return x;
}
const interval_expr& interval_expr::none() {
  static interval_expr x = {positive_infinity(), negative_infinity()};
  return x;
}
const interval_expr& interval_expr::union_identity() { return none(); }
const interval_expr& interval_expr::intersection_identity() { return all(); }

interval_expr& interval_expr::operator*=(const expr& scale) {
  if (is_non_negative(scale)) {
    if (min.defined()) min = mul::make(min, scale);
    if (max.defined()) max = mul::make(max, scale);
  } else if (is_negative(scale)) {
    std::swap(min, max);
    if (min.defined()) min = mul::make(min, scale);
    if (max.defined()) max = mul::make(max, scale);
  } else {
    if (min.defined()) min = mul::make(min, scale);
    if (max.defined()) max = mul::make(max, scale);
    *this |= bounds(max, min);
  }
  return *this;
}

interval_expr& interval_expr::operator/=(const expr& scale) {
  if (is_non_negative(scale)) {
    if (min.defined()) min = div::make(min, scale);
    if (max.defined()) max = div::make(max, scale);
  } else if (is_negative(scale)) {
    std::swap(min, max);
    if (min.defined()) min = div::make(min, scale);
    if (max.defined()) max = div::make(max, scale);
  } else {
    if (min.defined()) min = div::make(min, scale);
    if (max.defined()) max = div::make(max, scale);
    *this |= bounds(max, min);
  }
  return *this;
}

interval_expr& interval_expr::operator+=(const expr& offset) {
  if (min.defined()) min = add::make(min, offset);
  if (max.defined()) max = add::make(max, offset);
  return *this;
}

interval_expr& interval_expr::operator-=(const expr& offset) {
  if (min.defined()) min = sub::make(min, offset);
  if (max.defined()) max = sub::make(max, offset);
  return *this;
}

interval_expr interval_expr::operator*(const expr& scale) const {
  interval_expr result(*this);
  result *= scale;
  return result;
}

interval_expr interval_expr::operator/(const expr& scale) const {
  interval_expr result(*this);
  result /= scale;
  return result;
}

interval_expr interval_expr::operator+(const expr& offset) const {
  interval_expr result(*this);
  result += offset;
  return result;
}

interval_expr interval_expr::operator-(const expr& offset) const {
  interval_expr result(*this);
  result -= offset;
  return result;
}

interval_expr interval_expr::operator-() const {
  return {max.defined() ? -max : expr(), min.defined() ? -min : expr()};
}

interval_expr& interval_expr::operator|=(const interval_expr& r) {
  min = (min.defined() && r.min.defined()) ? slinky::min(min, r.min) : expr();
  max = (max.defined() && r.max.defined()) ? slinky::max(max, r.max) : expr();
  return *this;
}

interval_expr& interval_expr::operator&=(const interval_expr& r) {
  min = (min.defined() && r.min.defined()) ? slinky::max(min, r.min) : min.defined() ? min : r.min;
  max = (max.defined() && r.max.defined()) ? slinky::min(max, r.max) : max.defined() ? max : r.max;
  return *this;
}

interval_expr interval_expr::operator|(const interval_expr& r) const {
  interval_expr result(*this);
  result |= r;
  return result;
}

interval_expr interval_expr::operator&(const interval_expr& r) const {
  interval_expr result(*this);
  result &= r;
  return result;
}

box_expr operator|(box_expr a, const box_expr& b) {
  assert(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] |= b[i];
  }
  return a;
}

box_expr operator&(box_expr a, const box_expr& b) {
  assert(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] &= b[i];
  }
  return a;
}

expr select::make(expr condition, expr true_value, expr false_value) {
  auto n = new select();
  n->condition = std::move(condition);
  n->true_value = std::move(true_value);
  n->false_value = std::move(false_value);
  return n;
}

expr call::make(slinky::intrinsic i, std::vector<expr> args) {
  auto n = new call();
  n->intrinsic = i;
  n->args = std::move(args);
  return n;
}

stmt call_stmt::make(call_stmt::callable target, symbol_list inputs, symbol_list outputs, attributes attrs) {
  auto n = new call_stmt();
  n->target = std::move(target);
  n->inputs = std::move(inputs);
  n->outputs = std::move(outputs);
  n->attrs = attrs;
  return n;
}

stmt copy_stmt::make(
    var src, std::vector<expr> src_x, var dst, std::vector<var> dst_x, std::optional<std::vector<char>> padding) {
  auto n = new copy_stmt();
  n->src = src;
  n->src_x = std::move(src_x);
  n->dst = dst;
  n->dst_x = std::move(dst_x);
  n->padding = std::move(padding);
  return n;
}

namespace {

// Flatten any blocks into inlined statements.
//
// Note that we don't need to recurse because no fully-constructed block stmt
// should ever contain other block stmt (they should all already be flattened).
void flatten_blocks(std::vector<stmt>& v) {
  for (auto it = v.begin(); it != v.end();) {
    if (it->defined()) {
      if (const block* b = it->as<block>()) {
        const auto& stmts = b->stmts;
        it = v.insert(it, stmts.begin(), stmts.end()) + stmts.size();
        it = v.erase(it);
        continue;
      }
    }
    it++;
  }
}

// Remove all empty statements.
void erase_undefs(std::vector<stmt>& v) {
  for (auto it = v.begin(); it != v.end();) {
    if (!it->defined()) {
      it = v.erase(it);
    } else {
      it++;
    }
  }
}

}  // namespace

stmt block::make(std::vector<stmt> stmts) {
  flatten_blocks(stmts);
  erase_undefs(stmts);
  if (stmts.empty()) {
    return {};
  } else if (stmts.size() == 1) {
    return std::move(stmts[0]);
  } else {
    auto n = new block();
    n->stmts = std::move(stmts);
    return n;
  }
}

stmt block::make(std::vector<stmt> stmts, stmt tail_stmt) {
  stmts.push_back(std::move(tail_stmt));
  return make(std::move(stmts));
}

stmt loop::make(var sym, int max_workers, interval_expr bounds, expr step, stmt body) {
  auto l = new loop();
  l->sym = sym;
  l->max_workers = max_workers;
  l->bounds = std::move(bounds);
  l->step = std::move(step);
  l->body = std::move(body);
  return l;
}

stmt allocate::make(var sym, memory_type storage, expr elem_size, std::vector<dim_expr> dims, stmt body) {
  auto n = new allocate();
  n->sym = sym;
  n->storage = storage;
  n->elem_size = std::move(elem_size);
  n->dims = std::move(dims);
  n->body = std::move(body);
  return n;
}

stmt make_buffer::make(var sym, expr base, expr elem_size, std::vector<dim_expr> dims, stmt body) {
  auto n = new make_buffer();
  n->sym = sym;
  n->base = std::move(base);
  n->elem_size = std::move(elem_size);
  n->dims = std::move(dims);
  n->body = std::move(body);
  return n;
}

stmt clone_buffer::make(var sym, var src, stmt body) {
  auto n = new clone_buffer();
  n->sym = sym;
  n->src = src;
  n->body = std::move(body);
  return n;
}

stmt crop_buffer::make(var sym, std::vector<interval_expr> bounds, stmt body) {
  auto n = new crop_buffer();
  n->sym = sym;
  n->bounds = std::move(bounds);
  n->body = std::move(body);
  return n;
}

stmt crop_dim::make(var sym, int dim, interval_expr bounds, stmt body) {
  auto n = new crop_dim();
  n->sym = sym;
  n->dim = dim;
  n->bounds = std::move(bounds);
  n->body = std::move(body);
  return n;
}

stmt slice_buffer::make(var sym, std::vector<expr> at, stmt body) {
  auto n = new slice_buffer();
  n->sym = sym;
  n->at = std::move(at);
  n->body = std::move(body);
  return n;
}

stmt slice_dim::make(var sym, int dim, expr at, stmt body) {
  auto n = new slice_dim();
  n->sym = sym;
  n->dim = dim;
  n->at = std::move(at);
  n->body = std::move(body);
  return n;
}

stmt truncate_rank::make(var sym, int rank, stmt body) {
  auto n = new truncate_rank();
  n->sym = sym;
  n->rank = rank;
  n->body = std::move(body);
  return n;
}

stmt check::make(expr condition) {
  auto n = new check();
  n->condition = std::move(condition);
  return n;
}

const expr& positive_infinity() {
  static expr e = call::make(intrinsic::positive_infinity, {});
  return e;
}
const expr& negative_infinity() {
  static expr e = call::make(intrinsic::negative_infinity, {});
  return e;
}
const expr& infinity(int sign) {
  assert(sign != 0);
  return sign < 0 ? negative_infinity() : positive_infinity();
}
const expr& indeterminate() {
  static expr e = call::make(intrinsic::indeterminate, {});
  return e;
}

expr abs(expr x) { return call::make(intrinsic::abs, {std::move(x)}); }

expr buffer_rank(expr buf) { return call::make(intrinsic::buffer_rank, {std::move(buf)}); }
expr buffer_elem_size(expr buf) { return call::make(intrinsic::buffer_elem_size, {std::move(buf)}); }
expr buffer_min(expr buf, expr dim) { return call::make(intrinsic::buffer_min, {std::move(buf), std::move(dim)}); }
expr buffer_max(expr buf, expr dim) { return call::make(intrinsic::buffer_max, {std::move(buf), std::move(dim)}); }
expr buffer_extent(expr buf, expr dim) { return (buffer_max(buf, dim) - buffer_min(buf, dim)) + 1; }
expr buffer_stride(expr buf, expr dim) {
  return call::make(intrinsic::buffer_stride, {std::move(buf), std::move(dim)});
}
expr buffer_fold_factor(expr buf, expr dim) {
  return call::make(intrinsic::buffer_fold_factor, {std::move(buf), std::move(dim)});
}

expr buffer_at(expr buf, span<const expr> at) {
  std::vector<expr> args = {buf};
  args.insert(args.end(), at.begin(), at.end());
  return call::make(intrinsic::buffer_at, std::move(args));
}

expr buffer_at(expr buf, span<const var> at) {
  std::vector<expr> args = {buf};
  args.insert(args.end(), at.begin(), at.end());
  return call::make(intrinsic::buffer_at, std::move(args));
}

expr buffer_at(expr buf) { return call::make(intrinsic::buffer_at, {std::move(buf)}); }

interval_expr buffer_bounds(const expr& buf, const expr& dim) { return {buffer_min(buf, dim), buffer_max(buf, dim)}; }
dim_expr buffer_dim(const expr& buf, const expr& dim) {
  return {buffer_bounds(buf, dim), buffer_stride(buf, dim), buffer_fold_factor(buf, dim)};
}
std::vector<dim_expr> buffer_dims(const expr& buf, int rank) {
  std::vector<dim_expr> result;
  result.reserve(rank);
  for (int d = 0; d < rank; ++d) {
    result.push_back(buffer_dim(buf, d));
  }
  return result;
}

box_expr dims_bounds(span<const dim_expr> dims) {
  box_expr result(dims.size());
  for (std::size_t d = 0; d < dims.size(); ++d) {
    result[d] = dims[d].bounds;
  }
  return result;
}

bool is_buffer_intrinsic(intrinsic fn) {
  switch (fn) {
  case intrinsic::buffer_rank:
  case intrinsic::buffer_elem_size:
  case intrinsic::buffer_size_bytes:
  case intrinsic::buffer_min:
  case intrinsic::buffer_max:
  case intrinsic::buffer_stride:
  case intrinsic::buffer_fold_factor:
  case intrinsic::buffer_at: return true;
  default: return false;
  }
}

bool is_finite(const expr& x) {
  if (x.as<constant>()) return true;
  if (const call* c = x.as<call>()) {
    return is_buffer_intrinsic(c->intrinsic);
  }
  return false;
}

bool is_buffer_min(const expr& x, var sym, int dim) {
  const call* c = x.as<call>();
  if (!c || c->intrinsic != intrinsic::buffer_min) return false;

  assert(c->args.size() == 2);
  return is_variable(c->args[0], sym) && is_constant(c->args[1], dim);
}

bool is_buffer_max(const expr& x, var sym, int dim) {
  const call* c = x.as<call>();
  if (!c || c->intrinsic != intrinsic::buffer_max) return false;

  assert(c->args.size() == 2);
  return is_variable(c->args[0], sym) && is_constant(c->args[1], dim);
}

expr semaphore_init(expr sem, expr count) {
  return call::make(intrinsic::semaphore_init, {std::move(sem), std::move(count)});
}
expr semaphore_signal(expr sem, expr count) {
  return call::make(intrinsic::semaphore_signal, {std::move(sem), std::move(count)});
}
expr semaphore_wait(expr sem, expr count) {
  return call::make(intrinsic::semaphore_wait, {std::move(sem), std::move(count)});
}

namespace {

expr semaphore_helper(intrinsic fn, span<const expr> sems, span<const expr> counts) {
  std::vector<expr> args(sems.size() * 2);
  for (std::size_t i = 0; i < sems.size(); ++i) {
    args[i * 2 + 0] = sems[i];
    if (i < counts.size()) {
      args[i * 2 + 1] = counts[i];
    }
  }
  return call::make(fn, std::move(args));
}

}  // namespace

expr semaphore_signal(span<const expr> sems, span<const expr> counts) {
  return semaphore_helper(intrinsic::semaphore_signal, sems, counts);
}
expr semaphore_wait(span<const expr> sems, span<const expr> counts) {
  return semaphore_helper(intrinsic::semaphore_wait, sems, counts);
}

void recursive_node_visitor::visit(const variable*) {}
void recursive_node_visitor::visit(const constant*) {}

void recursive_node_visitor::visit(const let* op) {
  for (const auto& p : op->lets) {
    p.second.accept(this);
  }
  op->body.accept(this);
}

namespace {

template <typename T>
void visit_binary(recursive_node_visitor* _this, const T* op) {
  op->a.accept(_this);
  op->b.accept(_this);
}

}  // namespace

void recursive_node_visitor::visit(const add* op) { visit_binary(this, op); }
void recursive_node_visitor::visit(const sub* op) { visit_binary(this, op); }
void recursive_node_visitor::visit(const mul* op) { visit_binary(this, op); }
void recursive_node_visitor::visit(const div* op) { visit_binary(this, op); }
void recursive_node_visitor::visit(const mod* op) { visit_binary(this, op); }
void recursive_node_visitor::visit(const class min* op) { visit_binary(this, op); }
void recursive_node_visitor::visit(const class max* op) { visit_binary(this, op); }
void recursive_node_visitor::visit(const equal* op) { visit_binary(this, op); }
void recursive_node_visitor::visit(const not_equal* op) { visit_binary(this, op); }
void recursive_node_visitor::visit(const less* op) { visit_binary(this, op); }
void recursive_node_visitor::visit(const less_equal* op) { visit_binary(this, op); }
void recursive_node_visitor::visit(const logical_and* op) { visit_binary(this, op); }
void recursive_node_visitor::visit(const logical_or* op) { visit_binary(this, op); }
void recursive_node_visitor::visit(const logical_not* op) { op->a.accept(this); }
void recursive_node_visitor::visit(const class select* op) {
  op->condition.accept(this);
  op->true_value.accept(this);
  op->false_value.accept(this);
}
void recursive_node_visitor::visit(const call* op) {
  for (const expr& i : op->args) {
    if (i.defined()) i.accept(this);
  }
}

void recursive_node_visitor::visit(const let_stmt* op) {
  for (const auto& p : op->lets) {
    p.second.accept(this);
  }
  if (op->body.defined()) op->body.accept(this);
}
void recursive_node_visitor::visit(const block* op) {
  for (const auto& s : op->stmts) {
    s.accept(this);
  }
}
void recursive_node_visitor::visit(const loop* op) {
  op->bounds.min.accept(this);
  op->bounds.max.accept(this);
  if (op->step.defined()) op->step.accept(this);
  if (op->body.defined()) op->body.accept(this);
}
void recursive_node_visitor::visit(const call_stmt* op) {}
void recursive_node_visitor::visit(const copy_stmt* op) {
  for (const expr& i : op->src_x) {
    i.accept(this);
  }
}
void recursive_node_visitor::visit(const allocate* op) {
  op->elem_size.accept(this);
  for (const dim_expr& i : op->dims) {
    i.bounds.min.accept(this);
    i.bounds.max.accept(this);
    i.stride.accept(this);
    if (i.fold_factor.defined()) i.fold_factor.accept(this);
  }
  if (op->body.defined()) op->body.accept(this);
}
void recursive_node_visitor::visit(const make_buffer* op) {
  if (op->base.defined()) op->base.accept(this);
  if (op->elem_size.defined()) op->elem_size.accept(this);
  for (const dim_expr& i : op->dims) {
    i.bounds.min.accept(this);
    i.bounds.max.accept(this);
    i.stride.accept(this);
    if (i.fold_factor.defined()) i.fold_factor.accept(this);
  }
  if (op->body.defined()) op->body.accept(this);
}
void recursive_node_visitor::visit(const clone_buffer* op) {
  if (op->body.defined()) op->body.accept(this);
}
void recursive_node_visitor::visit(const crop_buffer* op) {
  for (const interval_expr& i : op->bounds) {
    if (i.min.defined()) i.min.accept(this);
    if (i.max.defined()) i.max.accept(this);
  }
  if (op->body.defined()) op->body.accept(this);
}
void recursive_node_visitor::visit(const crop_dim* op) {
  if (op->bounds.min.defined()) op->bounds.min.accept(this);
  if (op->bounds.max.defined()) op->bounds.max.accept(this);
  if (op->body.defined()) op->body.accept(this);
}
void recursive_node_visitor::visit(const slice_buffer* op) {
  for (const expr& i : op->at) {
    if (i.defined()) i.accept(this);
  }
  if (op->body.defined()) op->body.accept(this);
}
void recursive_node_visitor::visit(const slice_dim* op) {
  op->at.accept(this);
  if (op->body.defined()) op->body.accept(this);
}
void recursive_node_visitor::visit(const truncate_rank* op) {
  if (op->body.defined()) op->body.accept(this);
}
void recursive_node_visitor::visit(const check* op) {
  if (op->condition.defined()) op->condition.accept(this);
}

}  // namespace slinky
