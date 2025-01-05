#include "runtime/expr.h"
#include "runtime/stmt.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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
    name_to_sym[name] = *sym;
  }
  return *sym;
}
var node_context::insert_unique(const std::string& prefix) {
  std::string name = prefix;
  for (std::size_t i = 0; i < sym_to_name.size(); ++i) {
    if (name_to_sym.find(name) == name_to_sym.end()) break;
    name = prefix + "#" + std::to_string(i);
  }
  return insert(name);
}
std::optional<var> node_context::lookup(const std::string& name) const {
  auto i = name_to_sym.find(name);
  return i != name_to_sym.end() ? std::optional<var>(i->second) : std::nullopt;
}

template <typename T>
expr make_bin_op(expr a, expr b) {
  auto n = new T();
  if (T::commutative && should_commute(a, b)) {
    // Aggressively canonicalizing the order is a big speedup by avoiding unnecessary simplifier rewrites.
    std::swap(a, b);
  }
  n->a = std::move(a);
  n->b = std::move(b);
  return expr(n);
}

template <typename T, typename Body>
Body make_let(std::vector<std::pair<var, expr>> lets, Body body) {
  auto n = new T();
  n->lets = std::move(lets);
  if (const T* l = body.template as<T>()) {
    n->lets.insert(n->lets.end(), l->lets.begin(), l->lets.end());
    n->body = l->body;
  } else {
    n->body = std::move(body);
  }
  return Body(n);
}

expr let::make(std::vector<std::pair<var, expr>> lets, expr body) {
  return make_let<let>(std::move(lets), std::move(body));
}

expr let::make(var sym, expr value, expr body) { return make({{sym, std::move(value)}}, std::move(body)); }

stmt let_stmt::make(std::vector<std::pair<var, expr>> lets, stmt body) {
  return make_let<let_stmt>(std::move(lets), std::move(body));
}

stmt let_stmt::make(var sym, expr value, stmt body) { return make({{sym, std::move(value)}}, std::move(body)); }

namespace {

template <std::int64_t value>
const constant* make_static_constant() {
  static constant result;
  // Don't let the ref counting free this object.
  result.add_ref();
  result.value = value;
  return &result;
}

const variable* make_variable(var sym) {
  auto n = new variable();
  n->sym = sym;
  n->field = field_id::none;
  n->dim = -1;
  return n;
}

const constant* make_constant(std::int64_t value) {
  static const constant* zero = make_static_constant<0>();
  static const constant* one = make_static_constant<1>();
  if (value == 0) {
    return zero;
  } else if (value == 1) {
    return one;
  } else {
    assert(value <= std::numeric_limits<index_t>::max());
    assert(value >= std::numeric_limits<index_t>::min());
    auto n = new constant();
    n->value = value;
    return n;
  }
}

}  // namespace

expr::expr(std::int64_t x) : expr(make_constant(x)) {}
expr::expr(var sym) : expr(make_variable(sym)) {}

expr variable::make(var sym) { return expr(make_variable(sym)); }
expr variable::make(var sym, field_id field, int dim) { 
  variable* n = new variable();
  n->sym = sym;
  n->field = field;
  n->dim = dim;
  return expr(n);
}

expr constant::make(index_t value) { return expr(make_constant(value)); }
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
  return expr(n);
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
  if (b.defined()) x = min::make(std::move(x), std::move(b));
  if (a.defined()) x = max::make(std::move(x), std::move(a));
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

expr expr::operator-() const { return 0 - *this; }

expr& expr::operator+=(expr r) {
  *this = *this + std::move(r);
  return *this;
}
expr& expr::operator-=(expr r) {
  *this = *this - std::move(r);
  return *this;
}
expr& expr::operator*=(expr r) {
  *this = *this * std::move(r);
  return *this;
}
expr& expr::operator/=(expr r) {
  *this = *this / std::move(r);
  return *this;
}
expr& expr::operator%=(expr r) {
  *this = *this % std::move(r);
  return *this;
}

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

const expr& interval_expr::begin() const { return min; }
expr interval_expr::end() const { return max + 1; }
expr interval_expr::extent() const { return max - min + 1; }
expr interval_expr::empty() const { return min > max; }

interval_expr& interval_expr::operator*=(const expr& scale) {
  if (is_point()) {
    min = max = mul::make(min, scale);
  } else if (is_non_negative(scale)) {
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
  if (is_point()) {
    min = max = div::make(min, scale);
  } else if (is_non_negative(scale)) {
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
  if (is_point()) {
    min = max = add::make(min, offset);
  } else {
    if (min.defined()) min = add::make(min, offset);
    if (max.defined()) max = add::make(max, offset);
  }
  return *this;
}

interval_expr& interval_expr::operator-=(const expr& offset) {
  if (is_point()) {
    min = max = sub::make(min, offset);
  } else {
    if (min.defined()) min = sub::make(min, offset);
    if (max.defined()) max = sub::make(max, offset);
  }
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

interval_expr& interval_expr::operator|=(interval_expr r) {
  min = (min.defined() && r.min.defined()) ? slinky::min(std::move(min), std::move(r.min)) : expr();
  max = (max.defined() && r.max.defined()) ? slinky::max(std::move(max), std::move(r.max)) : expr();
  return *this;
}

interval_expr& interval_expr::operator&=(interval_expr r) {
  if (min.defined() && r.min.defined()) {
    min = slinky::max(std::move(min), std::move(r.min));
  } else if (!min.defined()) {
    min = std::move(r.min);
  }
  if (max.defined() && r.max.defined()) {
    max = slinky::min(std::move(max), std::move(r.max));
  } else if (!max.defined()) {
    max = std::move(r.max);
  }
  return *this;
}

interval_expr interval_expr::operator|(interval_expr r) const {
  interval_expr result(*this);
  result |= std::move(r);
  return result;
}

interval_expr interval_expr::operator&(interval_expr r) const {
  interval_expr result(*this);
  result &= std::move(r);
  return result;
}

interval_expr range(expr begin, expr end) { return {std::move(begin), std::move(end) - 1}; }
interval_expr bounds(expr min, expr max) { return {std::move(min), std::move(max)}; }
interval_expr min_extent(const expr& min, expr extent) { return {min, min + std::move(extent) - 1}; }

interval_expr operator*(const expr& a, const interval_expr& b) { return b * a; }
interval_expr operator+(const expr& a, const interval_expr& b) { return b + a; }

expr clamp(expr x, interval_expr bounds) { return clamp(std::move(x), std::move(bounds.min), std::move(bounds.max)); }
interval_expr select(const expr& c, interval_expr t, interval_expr f) {
  if (t.is_point() && f.is_point()) {
    return point(select(c, std::move(t.min), std::move(f.min)));
  } else {
    return {
        select(c, std::move(t.min), std::move(f.min)),
        select(c, std::move(t.max), std::move(f.max)),
    };
  }
}

box_expr operator|(box_expr a, const box_expr& b) {
  a.resize(std::max(a.size(), b.size()));
  for (std::size_t i = 0; i < b.size(); ++i) {
    a[i] |= b[i];
  }
  return a;
}

box_expr operator&(box_expr a, const box_expr& b) {
  a.resize(std::max(a.size(), b.size()));
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    a[i] &= b[i];
  }
  for (std::size_t i = b.size(); i < a.size(); ++i) {
    a[i] = b[i];
  }
  return a;
}

expr select::make(expr condition, expr true_value, expr false_value) {
  auto n = new select();
  n->condition = std::move(condition);
  n->true_value = std::move(true_value);
  n->false_value = std::move(false_value);
  return expr(n);
}

expr call::make(slinky::intrinsic i, std::vector<expr> args) {
  auto n = new call();
  n->intrinsic = i;
  n->args = std::move(args);
  return expr(n);
}

stmt call_stmt::make(call_stmt::callable target, symbol_list inputs, symbol_list outputs, attributes attrs) {
  auto n = new call_stmt();
  n->target = std::move(target);
  n->inputs = std::move(inputs);
  n->outputs = std::move(outputs);
  n->attrs = std::move(attrs);
  return stmt(n);
}

stmt copy_stmt::make(
    var src, std::vector<expr> src_x, var dst, std::vector<var> dst_x, std::optional<std::vector<char>> padding) {
  auto n = new copy_stmt();
  n->src = src;
  n->src_x = std::move(src_x);
  n->dst = dst;
  n->dst_x = std::move(dst_x);
  n->padding = std::move(padding);
  return stmt(n);
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
    return stmt(n);
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
  return stmt(l);
}

stmt allocate::make(var sym, memory_type storage, expr elem_size, std::vector<dim_expr> dims, stmt body) {
  auto n = new allocate();
  n->sym = sym;
  n->storage = storage;
  n->elem_size = std::move(elem_size);
  n->dims = std::move(dims);
  n->body = std::move(body);
  return stmt(n);
}

stmt make_buffer::make(var sym, expr base, expr elem_size, std::vector<dim_expr> dims, stmt body) {
  auto n = new make_buffer();
  n->sym = sym;
  n->base = std::move(base);
  n->elem_size = std::move(elem_size);
  n->dims = std::move(dims);
  n->body = std::move(body);
  return stmt(n);
}

stmt clone_buffer::make(var sym, var src, stmt body) {
  auto n = new clone_buffer();
  n->sym = sym;
  n->src = src;
  n->body = std::move(body);
  return stmt(n);
}

stmt crop_buffer::make(var sym, var src, std::vector<interval_expr> bounds, stmt body) {
  auto n = new crop_buffer();
  n->sym = sym;
  n->src = src;
  n->bounds = std::move(bounds);
  n->body = std::move(body);
  return stmt(n);
}

stmt crop_dim::make(var sym, var src, int dim, interval_expr bounds, stmt body) {
  auto n = new crop_dim();
  n->sym = sym;
  n->src = src;
  n->dim = dim;
  n->bounds = std::move(bounds);
  n->body = std::move(body);
  return stmt(n);
}

stmt slice_buffer::make(var sym, var src, std::vector<expr> at, stmt body) {
  auto n = new slice_buffer();
  n->sym = sym;
  n->src = src;
  n->at = std::move(at);
  n->body = std::move(body);
  return stmt(n);
}

stmt slice_dim::make(var sym, var src, int dim, expr at, stmt body) {
  auto n = new slice_dim();
  n->sym = sym;
  n->src = src;
  n->dim = dim;
  n->at = std::move(at);
  n->body = std::move(body);
  return stmt(n);
}

stmt transpose::make(var sym, var src, std::vector<int> dims, stmt body) {
  auto n = new transpose();
  n->sym = sym;
  n->src = src;
  n->dims = dims;
  n->body = std::move(body);
  return stmt(n);
}

stmt transpose::make_truncate(var sym, var src, int rank, stmt body) {
  std::vector<int> dims(rank);
  std::iota(dims.begin(), dims.end(), 0);
  return make(sym, src, std::move(dims), std::move(body));
}

bool transpose::is_truncate(span<const int> dims) {
  for (std::size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] != static_cast<int>(i)) return false;
  }
  return true;
}
bool transpose::is_truncate() const { return is_truncate(dims); }

stmt check::make(expr condition) {
  auto n = new check();
  n->condition = std::move(condition);
  return stmt(n);
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

bool is_positive(expr_ref x) {
  if (is_positive_infinity(x)) return true;
  if (const call* c = as_intrinsic(x, intrinsic::abs)) {
    assert(c->args.size() == 1);
    return is_positive(c->args[0]);
  }
  auto c = as_constant(x);
  return c ? *c > 0 : false;
}

bool is_non_negative(expr_ref x) {
  if (is_positive_infinity(x)) return true;
  if (as_intrinsic(x, intrinsic::abs)) return true;
  auto c = as_constant(x);
  return c ? *c >= 0 : false;
}

bool is_negative(expr_ref x) {
  if (is_negative_infinity(x)) return true;
  auto c = as_constant(x);
  return c ? *c < 0 : false;
}

bool is_non_positive(expr_ref x) {
  if (is_negative_infinity(x)) return true;
  auto c = as_constant(x);
  return c ? *c <= 0 : false;
}

expr abs(expr x) { return call::make(intrinsic::abs, {std::move(x)}); }
expr align_down(expr x, const expr& a) { return (std::move(x) / a) * a; }
expr align_up(expr x, const expr& a) { return ((std::move(x) + a - 1) / a) * a; }
interval_expr align(interval_expr x, const expr& a) { return {align_down(std::move(x.min), a), align_up(std::move(x.max) + 1, a) - 1}; }

expr and_then(std::vector<expr> args) { return call::make(intrinsic::and_then, std::move(args)); }
expr or_else(std::vector<expr> args) { return call::make(intrinsic::or_else, std::move(args)); }

expr buffer_rank(expr buf) { return call::make(intrinsic::buffer_rank, {std::move(buf)}); }
expr buffer_elem_size(expr buf) { return call::make(intrinsic::buffer_elem_size, {std::move(buf)}); }
expr buffer_min(expr buf, expr dim) { return call::make(intrinsic::buffer_min, {std::move(buf), std::move(dim)}); }
expr buffer_max(expr buf, expr dim) { return call::make(intrinsic::buffer_max, {std::move(buf), std::move(dim)}); }
expr buffer_extent(const expr& buf, const expr& dim) { return (buffer_max(buf, dim) - buffer_min(buf, dim)) + 1; }
expr buffer_stride(expr buf, expr dim) {
  return call::make(intrinsic::buffer_stride, {std::move(buf), std::move(dim)});
}
expr buffer_fold_factor(expr buf, expr dim) {
  return call::make(intrinsic::buffer_fold_factor, {std::move(buf), std::move(dim)});
}

expr buffer_at(expr buf, span<const expr> at) {
  std::vector<expr> args;
  args.reserve(at.size() + 1);
  args.push_back(std::move(buf));
  args.insert(args.end(), at.begin(), at.end());
  return call::make(intrinsic::buffer_at, std::move(args));
}

expr buffer_at(expr buf, span<const var> at) {
  std::vector<expr> args;
  args.reserve(at.size() + 1);
  args.push_back(std::move(buf));
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

bool is_buffer_dim_intrinsic(intrinsic fn) {
  switch (fn) {
  case intrinsic::buffer_min:
  case intrinsic::buffer_max:
  case intrinsic::buffer_stride:
  case intrinsic::buffer_fold_factor: return true;
  default: return false;
  }
}

bool is_positive_infinity(expr_ref x) { return as_intrinsic(x, intrinsic::positive_infinity); }
bool is_negative_infinity(expr_ref x) { return as_intrinsic(x, intrinsic::negative_infinity); }
bool is_indeterminate(expr_ref x) { return as_intrinsic(x, intrinsic::indeterminate); }
int is_infinity(expr_ref x) {
  if (is_positive_infinity(x)) return 1;
  if (is_negative_infinity(x)) return -1;
  return 0;
}

bool is_finite(expr_ref x) {
  if (x.as<constant>()) return true;
  if (const call* c = x.as<call>()) {
    return is_buffer_intrinsic(c->intrinsic);
  }
  return false;
}

expr boolean(const expr& x) {
  if (!x.defined() || is_boolean(x)) {
    return x;
  } else if (auto c = as_constant(x)) {
    return *c != 0;
  } else {
    return not_equal::make(x, 0);
  }
}
bool is_boolean(expr_ref x) { return is_boolean_node(x.type()) || is_one(x) || is_zero(x); }

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

void visit_binary(recursive_node_visitor* _this, const expr& a, const expr& b) {
  if (a.defined()) a.accept(_this);
  if (b.defined()) b.accept(_this);
}

}  // namespace

void recursive_node_visitor::visit(const add* op) { visit_binary(this, op->a, op->b); }
void recursive_node_visitor::visit(const sub* op) { visit_binary(this, op->a, op->b); }
void recursive_node_visitor::visit(const mul* op) { visit_binary(this, op->a, op->b); }
void recursive_node_visitor::visit(const div* op) { visit_binary(this, op->a, op->b); }
void recursive_node_visitor::visit(const mod* op) { visit_binary(this, op->a, op->b); }
void recursive_node_visitor::visit(const class min* op) { visit_binary(this, op->a, op->b); }
void recursive_node_visitor::visit(const class max* op) { visit_binary(this, op->a, op->b); }
void recursive_node_visitor::visit(const equal* op) { visit_binary(this, op->a, op->b); }
void recursive_node_visitor::visit(const not_equal* op) { visit_binary(this, op->a, op->b); }
void recursive_node_visitor::visit(const less* op) { visit_binary(this, op->a, op->b); }
void recursive_node_visitor::visit(const less_equal* op) { visit_binary(this, op->a, op->b); }
void recursive_node_visitor::visit(const logical_and* op) { visit_binary(this, op->a, op->b); }
void recursive_node_visitor::visit(const logical_or* op) { visit_binary(this, op->a, op->b); }
void recursive_node_visitor::visit(const logical_not* op) {
  if (op->a.defined()) op->a.accept(this);
}
void recursive_node_visitor::visit(const class select* op) {
  if (op->condition.defined()) op->condition.accept(this);
  if (op->true_value.defined()) op->true_value.accept(this);
  if (op->false_value.defined()) op->false_value.accept(this);
}
void recursive_node_visitor::visit(const call* op) {
  for (const expr& i : op->args) {
    if (i.defined()) i.accept(this);
  }
}

void recursive_node_visitor::visit(const let_stmt* op) {
  for (const auto& p : op->lets) {
    if (p.second.defined()) p.second.accept(this);
  }
  if (op->body.defined()) op->body.accept(this);
}
void recursive_node_visitor::visit(const block* op) {
  for (const auto& s : op->stmts) {
    if (s.defined()) s.accept(this);
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
    if (i.stride.defined()) i.stride.accept(this);
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
void recursive_node_visitor::visit(const transpose* op) {
  if (op->body.defined()) op->body.accept(this);
}
void recursive_node_visitor::visit(const check* op) {
  if (op->condition.defined()) op->condition.accept(this);
}

}  // namespace slinky
