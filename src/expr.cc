#include "expr.h"

#include "simplify.h"

namespace slinky {

std::string node_context::name(symbol_id i) const {
  if (i < sym_to_name.size()) {
    return sym_to_name[i];
  } else {
    return "<" + std::to_string(i) + ">";
  }
}

symbol_id node_context::insert(const std::string& name) {
  std::optional<symbol_id> sym = lookup(name);
  if (!sym) {
    sym = sym_to_name.size();
    sym_to_name.push_back(name);
  }
  return *sym;
}
symbol_id node_context::insert_unique(const std::string& prefix) {
  std::string name = prefix;
  for (std::size_t i = 0; i < sym_to_name.size(); ++i) {
    if (!lookup(name)) break;
    name = prefix + std::to_string(i);
  }
  return insert(name);
}
std::optional<symbol_id> node_context::lookup(const std::string& name) const {
  // TODO: At some point we might need a better data structure than doing this linear search.
  for (symbol_id i = 0; i < sym_to_name.size(); ++i) {
    if (sym_to_name[i] == name) {
      return i;
    }
  }
  return {};
}

template <typename T>
const T* make_bin_op(expr a, expr b) {
  auto n = new T();
  n->a = std::move(a);
  n->b = std::move(b);
  return n;
}

template <typename T, typename Body>
const T* make_let(symbol_id sym, expr value, Body body) {
  auto n = new T();
  n->sym = sym;
  n->value = std::move(value);
  n->body = std::move(body);
  return n;
}

expr let::make(symbol_id sym, expr value, expr body) { return make_let<let>(sym, std::move(value), std::move(body)); }

stmt let_stmt::make(symbol_id sym, expr value, stmt body) {
  return make_let<let_stmt>(sym, std::move(value), std::move(body));
}

// TODO(https://github.com/dsharlet/slinky/issues/4): At this time, the top CPU user
// of simplify_fuzz is malloc/free. Perhaps caching common values of variables (yes
// we can cache variables!) would be worth doing.
const variable* make_variable(symbol_id sym) {
  auto n = new variable();
  n->sym = sym;
  return n;
}

const constant* make_constant(index_t value) {
  auto n = new constant();
  n->value = value;
  return n;
}

expr::expr(index_t value) : expr(make_constant(value)) {}

expr variable::make(symbol_id sym) { return make_variable(sym); }

expr wildcard::make(symbol_id sym, std::function<bool(const expr&)> matches) {
  auto n = new wildcard();
  n->sym = sym;
  n->matches = std::move(matches);
  return n;
}

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

expr min(std::span<expr> x) {
  if (x.empty()) {
    return expr();
  } else if (x.size() == 1) {
    return x[0];
  } else {
    return min(x[0], min(x.subspan(1)));
  }
}

expr max(std::span<expr> x) {
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
    min = simplify(static_cast<mul*>(nullptr), min, scale);
    max = simplify(static_cast<mul*>(nullptr), max, scale);
  } else if (is_negative(scale)) {
    std::swap(min, max);
    min = simplify(static_cast<mul*>(nullptr), min, scale);
    max = simplify(static_cast<mul*>(nullptr), max, scale);
  } else {
    min = simplify(static_cast<mul*>(nullptr), min, scale);
    max = simplify(static_cast<mul*>(nullptr), max, scale);
    *this |= bounds(max, min);
  }
  return *this;
}

interval_expr& interval_expr::operator/=(const expr& scale) {
  if (is_non_negative(scale)) {
    min = simplify(static_cast<div*>(nullptr), min, scale);
    max = simplify(static_cast<div*>(nullptr), max, scale);
  } else if (is_negative(scale)) {
    std::swap(min, max);
    min = simplify(static_cast<div*>(nullptr), min, scale);
    max = simplify(static_cast<div*>(nullptr), max, scale);
  } else {
    min = simplify(static_cast<div*>(nullptr), min, scale);
    max = simplify(static_cast<div*>(nullptr), max, scale);
    *this |= bounds(max, min);
  }
  return *this;
}

interval_expr& interval_expr::operator+=(const expr& offset) {
  min = simplify(static_cast<add*>(nullptr), min, offset);
  max = simplify(static_cast<add*>(nullptr), max, offset);
  return *this;
}

interval_expr& interval_expr::operator-=(const expr& offset) {
  min = simplify(static_cast<sub*>(nullptr), min, offset);
  max = simplify(static_cast<sub*>(nullptr), max, offset);
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

interval_expr interval_expr::operator-() const { return {-max, -min}; }

interval_expr& interval_expr::operator|=(const interval_expr& r) {
  min = simplify(static_cast<class min*>(nullptr), min, r.min);
  max = simplify(static_cast<class max*>(nullptr), max, r.max);
  return *this;
}

interval_expr& interval_expr::operator&=(const interval_expr& r) {
  min = simplify(static_cast<class max*>(nullptr), min, r.min);
  max = simplify(static_cast<class min*>(nullptr), max, r.max);
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

stmt call_stmt::make(call_stmt::callable target, symbol_list inputs, symbol_list outputs) {
  auto n = new call_stmt();
  n->target = std::move(target);
  n->inputs = std::move(inputs);
  n->outputs = std::move(outputs);
  return n;
}

stmt copy_stmt::make(
    symbol_id src, std::vector<expr> src_x, symbol_id dst, std::vector<symbol_id> dst_x, std::vector<char> padding) {
  auto n = new copy_stmt();
  n->src = src;
  n->src_x = std::move(src_x);
  n->dst = dst;
  n->dst_x = std::move(dst_x);
  n->padding = std::move(padding);
  return n;
}

stmt block::make(stmt a, stmt b) {
  auto n = new block();
  n->a = std::move(a);
  n->b = std::move(b);
  return n;
}

stmt loop::make(symbol_id sym, interval_expr bounds, expr step, stmt body) {
  auto l = new loop();
  l->sym = sym;
  l->bounds = std::move(bounds);
  l->step = std::move(step);
  l->body = std::move(body);
  return l;
}

stmt if_then_else::make(expr condition, stmt true_body, stmt false_body) {
  auto n = new if_then_else();
  n->condition = std::move(condition);
  n->true_body = std::move(true_body);
  n->false_body = std::move(false_body);
  return n;
}

stmt allocate::make(memory_type storage, symbol_id sym, std::size_t elem_size, std::vector<dim_expr> dims, stmt body) {
  auto n = new allocate();
  n->storage = storage;
  n->sym = sym;
  n->elem_size = elem_size;
  n->dims = std::move(dims);
  n->body = std::move(body);
  return n;
}

stmt make_buffer::make(symbol_id sym, expr base, expr elem_size, std::vector<dim_expr> dims, stmt body) {
  auto n = new make_buffer();
  n->sym = sym;
  n->base = std::move(base);
  n->elem_size = std::move(elem_size);
  n->dims = std::move(dims);
  n->body = std::move(body);
  return n;
}

stmt crop_buffer::make(symbol_id sym, std::vector<interval_expr> bounds, stmt body) {
  auto n = new crop_buffer();
  n->sym = sym;
  n->bounds = std::move(bounds);
  n->body = std::move(body);
  return n;
}

stmt crop_dim::make(symbol_id sym, int dim, interval_expr bounds, stmt body) {
  auto n = new crop_dim();
  n->sym = sym;
  n->dim = dim;
  n->bounds = std::move(bounds);
  n->body = std::move(body);
  return n;
}

stmt slice_buffer::make(symbol_id sym, std::vector<expr> at, stmt body) {
  auto n = new slice_buffer();
  n->sym = sym;
  n->at = std::move(at);
  n->body = std::move(body);
  return n;
}

stmt slice_dim::make(symbol_id sym, int dim, expr at, stmt body) {
  auto n = new slice_dim();
  n->sym = sym;
  n->dim = dim;
  n->at = std::move(at);
  n->body = std::move(body);
  return n;
}

stmt truncate_rank::make(symbol_id sym, int rank, stmt body) {
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

const expr& indeterminate() {
  static expr e = call::make(intrinsic::indeterminate, {});
  return e;
}

expr abs(expr x) { return call::make(intrinsic::abs, {std::move(x)}); }

expr buffer_rank(expr buf) { return call::make(intrinsic::buffer_rank, {std::move(buf)}); }
expr buffer_base(expr buf) { return call::make(intrinsic::buffer_base, {std::move(buf)}); }
expr buffer_elem_size(expr buf) { return call::make(intrinsic::buffer_elem_size, {std::move(buf)}); }
expr buffer_min(expr buf, expr dim) { return call::make(intrinsic::buffer_min, {std::move(buf), std::move(dim)}); }
expr buffer_max(expr buf, expr dim) { return call::make(intrinsic::buffer_max, {std::move(buf), std::move(dim)}); }
expr buffer_extent(expr buf, expr dim) {
  return call::make(intrinsic::buffer_extent, {std::move(buf), std::move(dim)});
}
expr buffer_stride(expr buf, expr dim) {
  return call::make(intrinsic::buffer_stride, {std::move(buf), std::move(dim)});
}
expr buffer_fold_factor(expr buf, expr dim) {
  return call::make(intrinsic::buffer_fold_factor, {std::move(buf), std::move(dim)});
}
expr buffer_at(expr buf, const std::vector<expr>& at) {
  std::vector<expr> args = {buf};
  args.insert(args.end(), at.begin(), at.end());
  return call::make(intrinsic::buffer_at, std::move(args));
}
expr buffer_at(expr buf, const std::vector<var>& at) {
  std::vector<expr> args = {buf};
  args.reserve(at.size() + 1);
  for (const var& i : at) {
    args.push_back(i);
  }
  return call::make(intrinsic::buffer_at, std::move(args));
}

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

box_expr dims_bounds(std::span<const dim_expr> dims) {
  box_expr result(dims.size());
  for (std::size_t d = 0; d < dims.size(); ++d) {
    result[d] = dims[d].bounds;
  }
  return result;
}

bool is_buffer_intrinsic(intrinsic i) {
  switch (i) {
  case intrinsic::buffer_rank:
  case intrinsic::buffer_base:
  case intrinsic::buffer_elem_size:
  case intrinsic::buffer_size_bytes:
  case intrinsic::buffer_min:
  case intrinsic::buffer_max:
  case intrinsic::buffer_stride:
  case intrinsic::buffer_fold_factor:
  case intrinsic::buffer_extent:
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

var::var() : sym_(-1) {}
var::var(symbol_id sym) : sym_(sym) {}
var::var(node_context& ctx, const std::string& sym) : sym_(ctx.insert(sym)) {}

var::operator expr() const {
  assert(sym_ != -1);
  return expr(variable::make(sym_));
}

}  // namespace slinky
