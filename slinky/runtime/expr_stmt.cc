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

#include "slinky/runtime/evaluate.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/print.h"
#include "slinky/runtime/stmt.h"

namespace slinky {

var::var(node_context& ctx, const std::string& name) : var(ctx.insert_unique(name)) {}

expr var::operator-() const { return -expr(*this); }

std::string node_context::name(var v) const {
  if (v.id < sym_to_name.size() && !sym_to_name[v.id].empty()) {
    return sym_to_name[v.id];
  } else if (v.defined()) {
    return "." + std::to_string(v.id);
  } else {
    return ".";
  }
}

void node_context::clear_name(var i) {
  if (i.id >= sym_to_name.size()) return;
  auto j = name_to_sym.find(sym_to_name[i.id]);
  if (j != name_to_sym.end() && j->second == i) name_to_sym.erase(j);
  sym_to_name[i.id].clear();
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

bool can_evaluate(intrinsic fn) {
  switch (fn) {
  case intrinsic::negative_infinity:
  case intrinsic::positive_infinity:
  case intrinsic::indeterminate: return false;
  default: return true;
  }
}

// A node's arrays are constructed in its own allocation, immediately following the node. This layout is private to
// the nodes we make here; nothing but the `span` of each array escapes.
// Every element type we put in a node array has at most this alignment, so every array can be aligned the same way,
// and the storage a node needs can be computed before allocating it.
constexpr std::size_t array_alignment = alignof(void*);

// The number of bytes the elements of `x` occupy in the storage of the node that owns them.
template <typename Array>
std::size_t size_of(const Array& x) {
  using T = typename Array::value_type;
  static_assert(alignof(T) <= array_alignment, "node array elements must not be overaligned");
  return align_up(x.size() * sizeof(T), array_alignment);
}

// Construct an array in `storage` by moving the elements of `src` into it, and advance `storage` past it.
template <typename T>
span<T> make_span(void*& storage, std::vector<T>& src) {
  T* result = static_cast<T*>(storage);
  for (std::size_t i = 0; i < src.size(); ++i) {
    new (result + i) T(std::move(src[i]));
  }
  storage = static_cast<char*>(storage) + size_of(src);
  return span<T>(result, src.size());
}

// Construct an array in `storage` by copying `src`, which we don't own, and advance `storage` past it.
template <typename T>
span<T> make_span(void*& storage, span<T> src) {
  T* result = static_cast<T*>(storage);
  for (std::size_t i = 0; i < src.size(); ++i) {
    new (result + i) T(src[i]);
  }
  storage = static_cast<char*>(storage) + size_of(src);
  return span<T>(result, src.size());
}

template <typename T>
void destroy_span(span<T> x) {
  for (const T& i : x) {
    const_cast<T&>(i).~T();
  }
}

// Allocate a node with `array_bytes` of storage for its arrays following it.
template <typename T>
T* make_node(std::size_t array_bytes = 0) {
  static_assert(sizeof(T) % array_alignment == 0, "node arrays would be misaligned");
  return new (::operator new(sizeof(T) + array_bytes)) T();
}

template <typename T>
expr make_bin_op(expr a, expr b) {
  // Here we eagerly constant fold arithmetic.
  if (!a.defined()) return a;
  if (!b.defined()) return b;
  const constant* ac = a.as<constant>();
  const constant* bc = b.as<constant>();
  if (ac && bc && !binary_overflows<T>(ac->value, bc->value)) {
    return make_binary<T>(ac->value, bc->value);
  }

  auto n = new T();
  if (T::commutative && should_commute(a, b)) {
    // Aggressively canonicalizing the order is a big speedup by avoiding unnecessary simplifier rewrites.
    std::swap(a, b);
  }
  n->a = std::move(a);
  n->b = std::move(b);
  return expr(n);
}

template <typename T, typename Lets, typename Body>
T* make_let(Lets&& lets, Body body) {
  auto n = make_node<T>(size_of(lets));
  void* arrays = n + 1;
  n->lets = make_span(arrays, lets);
  n->body = std::move(body);
  return n;
}

expr let::make(std::vector<std::pair<var, expr>> lets, expr body) { return expr(make_let<let>(lets, std::move(body))); }
expr let::make(span<std::pair<var, expr>> lets, expr body) { return expr(make_let<let>(lets, std::move(body))); }

expr let::make(var sym, expr value, expr body) { return make({{sym, std::move(value)}}, std::move(body)); }

namespace {

int max_decl_id(span<std::pair<var, expr>> lets) {
  int result = -1;
  for (const std::pair<var, expr>& i : lets) {
    result = std::max<int>(i.first.id, result);
  }
  return result;
}

bool is_self_assignment(const std::pair<var, expr>& let) { return is_variable(let.second, let.first); }

}  // namespace

stmt let_stmt::make(std::vector<std::pair<var, expr>> lets, stmt body, bool is_closure, int max_symbol_id) {
  let_stmt* n = make_let<let_stmt>(lets, std::move(body));
  n->is_closure = is_closure;
  n->max_symbol_id = std::max(max_symbol_id, max_decl_id(n->lets));
  assert(!is_closure || std::all_of(n->lets.begin(), n->lets.end(), is_self_assignment));
  return stmt(n);
}

stmt let_stmt::make(span<std::pair<var, expr>> lets, stmt body, bool is_closure, int max_symbol_id) {
  let_stmt* n = make_let<let_stmt>(lets, std::move(body));
  n->is_closure = is_closure;
  n->max_symbol_id = std::max(max_symbol_id, max_decl_id(n->lets));
  assert(!is_closure || std::all_of(n->lets.begin(), n->lets.end(), is_self_assignment));
  return stmt(n);
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
  n->field = buffer_field::none;
  n->dim = -1;
  return n;
}

const constant* get_constant(std::int64_t value) {
  if (value == 0) {
    static const constant* zero = make_static_constant<0>();
    return zero;
  } else if (value == 1) {
    static const constant* one = make_static_constant<1>();
    return one;
  } else {
    return nullptr;
  }
}

const constant* make_constant(std::int64_t value) {
  if (const constant* n = get_constant(value)) return n;

  assert(value <= std::numeric_limits<index_t>::max());
  assert(value >= std::numeric_limits<index_t>::min());
  auto n = new constant();
  n->value = value;
  return n;
}

}  // namespace

let::~let() { destroy_span(lets); }
call::~call() { destroy_span(args); }

let_stmt::~let_stmt() { destroy_span(lets); }
block::~block() { destroy_span(stmts); }
allocate::~allocate() { destroy_span(dims); }
make_buffer::~make_buffer() { destroy_span(dims); }
crop_buffer::~crop_buffer() { destroy_span(bounds); }
slice_buffer::~slice_buffer() { destroy_span(at); }
transpose::~transpose() { destroy_span(dims); }

call_stmt::call_stmt(const call_stmt& other)
    : stmt_node<call_stmt>(other), target(other.target), inputs(other.inputs), outputs(other.outputs),
      scalars(other.scalars) {
  if (other.attrs) attrs = std::make_unique<attributes>(*other.attrs);
}

call_stmt::~call_stmt() {
  destroy_span(inputs);
  destroy_span(outputs);
  destroy_span(scalars);
}

copy_stmt::~copy_stmt() {
  destroy_span(src_x);
  destroy_span(dst_x);
}

expr::expr(std::int64_t x) : expr(make_constant(x)) {}
expr::expr(var sym) : expr(make_variable(sym)) {}

expr variable::make(var sym) { return expr(make_variable(sym)); }
expr variable::make(var sym, buffer_field field, int dim) {
  assert(dim >= std::numeric_limits<std::int16_t>::min());
  assert(dim <= std::numeric_limits<std::int16_t>::max());
  variable* n = new variable();
  n->sym = sym;
  n->field = field;
  n->dim = dim;
  return expr(n);
}

expr_ref constant::get(index_t value) { return get_constant(value); }
expr constant::make(index_t value) { return expr(make_constant(value)); }
expr constant::make(const void* value) { return make(reinterpret_cast<index_t>(value)); }

expr constant_buffer::make(const_raw_buffer_ptr value) {
  auto n = new constant_buffer();
  n->value = std::move(value);
  return expr(n);
}

expr add::make(expr a, expr b) {
  if (is_zero(a)) return b;
  if (is_zero(b)) return a;
  return make_bin_op<add>(std::move(a), std::move(b));
}
expr sub::make(expr a, expr b) {
  if (is_zero(b)) return a;
  if (auto cb = as_constant(b)) {
    if (*cb == 0) return a;
    if (!sub_overflows<index_t>(0, *cb)) {
      // Canonicalize to addition with constants.
      return add::make(std::move(a), -*cb);
    }
  }
  return make_bin_op<sub>(std::move(a), std::move(b));
}
expr mul::make(expr a, expr b) {
  if (is_zero(a) || is_one(b)) return a;
  if (is_zero(b) || is_one(a)) return b;
  return make_bin_op<mul>(std::move(a), std::move(b));
}
expr div::make(expr a, expr b) {
  // Division by zero is defined to be zero.
  if (is_zero(a) || is_one(b)) return a;
  if (is_zero(b)) return b;
  return make_bin_op<div>(std::move(a), std::move(b));
}
expr mod::make(expr a, expr b) {
  // `0%x`, `x%0` and `x%1` are all zero.
  if (is_zero(a) || is_zero(b) || is_one(b)) return expr(0);
  return make_bin_op<mod>(std::move(a), std::move(b));
}
expr min::make(expr a, expr b) { return make_bin_op<min>(std::move(a), std::move(b)); }
expr max::make(expr a, expr b) { return make_bin_op<max>(std::move(a), std::move(b)); }
expr equal::make(expr a, expr b) { return make_bin_op<equal>(std::move(a), std::move(b)); }
expr not_equal::make(expr a, expr b) { return make_bin_op<not_equal>(std::move(a), std::move(b)); }
expr less::make(expr a, expr b) { return make_bin_op<less>(std::move(a), std::move(b)); }
expr less_equal::make(expr a, expr b) { return make_bin_op<less_equal>(std::move(a), std::move(b)); }
expr logical_and::make(expr a, expr b) {
  if (is_true(a) || is_false(b)) return boolean(b);
  if (is_true(b) || is_false(a)) return boolean(a);
  return make_bin_op<logical_and>(std::move(a), std::move(b));
}
expr logical_or::make(expr a, expr b) {
  if (is_true(a) || is_false(b)) return boolean(a);
  if (is_true(b) || is_false(a)) return boolean(b);
  return make_bin_op<logical_or>(std::move(a), std::move(b));
}
expr logical_not::make(expr a) {
  if (const constant* c = a.as<constant>()) {
    return expr(make_constant(c->value == 0 ? 1 : 0));
  }
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

expr interval_expr::contains(expr_ref x) const {
  if (min.defined() && max.defined()) {
    return min <= x && x <= max;
  } else if (min.defined()) {
    return min <= x;
  } else if (max.defined()) {
    return x <= max;
  } else {
    return expr{1};
  }
}

interval_expr& interval_expr::operator*=(const expr& scale) {
  if (is_one(scale)) {
  } else if (is_point()) {
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
  if (is_one(scale)) {
  } else if (is_point()) {
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
  if (is_zero(offset)) {
  } else if (is_point()) {
    min = max = add::make(min, offset);
  } else {
    if (min.defined()) min = add::make(min, offset);
    if (max.defined()) max = add::make(max, offset);
  }
  return *this;
}

interval_expr& interval_expr::operator-=(const expr& offset) {
  if (is_zero(offset)) {
  } else if (is_point()) {
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
interval_expr operator-(const expr& a, const interval_expr& b) { return -b + a; }

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
  if (!condition.defined()) return condition;
  if (!true_value.defined() && !false_value.defined()) {
    // We need both sides of a select to be undefined to unconditionally be undefined.
    return true_value;
  }
  if (is_true(condition)) {
    return true_value;
  } else if (is_false(condition)) {
    return false_value;
  }

  auto n = new select();
  n->condition = std::move(condition);
  n->true_value = std::move(true_value);
  n->false_value = std::move(false_value);
  return expr(n);
}

expr call::make(slinky::intrinsic i, callable target, std::vector<expr> args) {
  auto n = make_node<call>(size_of(args));
  n->intrinsic = i;
  n->target = std::move(target);
  void* arrays = n + 1;
  n->args = make_span(arrays, args);
  expr result(n);

  if (n->target || can_evaluate(i)) {
    if (std::all_of(n->args.begin(), n->args.end(), [](const expr& i) { return as_constant(i); })) {
      return evaluate(result);
    }
  }
  return result;
}

expr call::make(slinky::intrinsic i, std::vector<expr> args) { return call::make(i, nullptr, std::move(args)); }

expr call::make(callable target, std::vector<expr> args) {
  return call::make(intrinsic::none, std::move(target), std::move(args));
}

stmt call_stmt::make(
    call_stmt::callable target, span<var> inputs, span<var> outputs, std::vector<expr> scalars, attributes attrs) {
  auto n = make_node<call_stmt>(size_of(inputs) + size_of(outputs) + size_of(scalars));
  void* arrays = n + 1;
  n->target = std::move(target);
  n->inputs = make_span(arrays, inputs);
  n->outputs = make_span(arrays, outputs);
  n->scalars = make_span(arrays, scalars);
  n->attrs = std::make_unique<attributes>(std::move(attrs));
  return stmt(n);
}

template <typename DstX>
stmt make_copy_stmt(copy_stmt::callable impl, var src, std::vector<expr> src_x, var dst, DstX&& dst_x, var pad) {
  auto n = make_node<copy_stmt>(size_of(src_x) + size_of(dst_x));
  void* arrays = n + 1;
  n->src = src;
  n->src_x = make_span(arrays, src_x);
  n->dst = dst;
  n->dst_x = make_span(arrays, dst_x);
  n->pad = pad;
  n->impl = impl;
  return stmt(n);
}

stmt copy_stmt::make(
    copy_stmt::callable impl, var src, std::vector<expr> src_x, var dst, std::vector<var> dst_x, var pad) {
  return make_copy_stmt(impl, src, std::move(src_x), dst, std::move(dst_x), pad);
}

stmt copy_stmt::make(copy_stmt::callable impl, var src, std::vector<expr> src_x, var dst, span<var> dst_x, var pad) {
  return make_copy_stmt(impl, src, std::move(src_x), dst, dst_x, pad);
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
    auto n = make_node<block>(size_of(stmts));
    void* arrays = n + 1;
    n->stmts = make_span(arrays, stmts);
    return stmt(n);
  }
}

stmt block::make(stmt a, stmt b) {
  if (!a.defined()) return b;
  if (!b.defined()) return a;
  return block::make({std::move(a), std::move(b)});
}

stmt block::make(std::vector<stmt> stmts, stmt tail_stmt) {
  stmts.push_back(std::move(tail_stmt));
  return make(std::move(stmts));
}

stmt loop::make(var sym, expr max_workers, interval_expr bounds, expr step, stmt body) {
  auto l = new loop();
  l->sym = sym;
  l->max_workers = std::move(max_workers);
  l->bounds = std::move(bounds);
  l->step = std::move(step);
  l->body = std::move(body);
  return stmt(l);
}

template <typename Dims>
stmt make_allocate(var sym, memory_type storage, expr elem_size, Dims&& dims, stmt body) {
  auto n = make_node<allocate>(size_of(dims));
  n->sym = sym;
  n->storage = storage;
  n->elem_size = std::move(elem_size);
  void* arrays = n + 1;
  n->dims = make_span(arrays, dims);
  n->body = std::move(body);
  return stmt(n);
}

stmt allocate::make(var sym, memory_type storage, expr elem_size, std::vector<dim_expr> dims, stmt body) {
  return make_allocate(sym, storage, std::move(elem_size), dims, std::move(body));
}

stmt allocate::make(var sym, memory_type storage, expr elem_size, span<dim_expr> dims, stmt body) {
  return make_allocate(sym, storage, std::move(elem_size), dims, std::move(body));
}

template <typename Dims>
stmt make_make_buffer(var sym, expr base, expr elem_size, Dims&& dims, stmt body) {
  auto n = make_node<make_buffer>(size_of(dims));
  n->sym = sym;
  n->base = std::move(base);
  n->elem_size = std::move(elem_size);
  void* arrays = n + 1;
  n->dims = make_span(arrays, dims);
  n->body = std::move(body);
  return stmt(n);
}

stmt make_buffer::make(var sym, expr base, expr elem_size, std::vector<dim_expr> dims, stmt body) {
  return make_make_buffer(sym, std::move(base), std::move(elem_size), dims, std::move(body));
}

stmt make_buffer::make(var sym, expr base, expr elem_size, span<dim_expr> dims, stmt body) {
  return make_make_buffer(sym, std::move(base), std::move(elem_size), dims, std::move(body));
}

stmt clone_buffer::make(var sym, var src, stmt body) {
  auto n = new clone_buffer();
  n->sym = sym;
  n->src = src;
  n->body = std::move(body);
  return stmt(n);
}

template <typename Bounds>
stmt make_crop_buffer(var sym, var src, Bounds&& bounds, stmt body) {
  auto n = make_node<crop_buffer>(size_of(bounds));
  n->sym = sym;
  n->src = src;
  void* arrays = n + 1;
  n->bounds = make_span(arrays, bounds);
  n->body = std::move(body);
  return stmt(n);
}

stmt crop_buffer::make(var sym, var src, std::vector<interval_expr> bounds, stmt body) {
  return make_crop_buffer(sym, src, bounds, std::move(body));
}

stmt crop_buffer::make(var sym, var src, span<interval_expr> bounds, stmt body) {
  return make_crop_buffer(sym, src, bounds, std::move(body));
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

template <typename At>
stmt make_slice_buffer(var sym, var src, At&& at, stmt body) {
  auto n = make_node<slice_buffer>(size_of(at));
  n->sym = sym;
  n->src = src;
  void* arrays = n + 1;
  n->at = make_span(arrays, at);
  n->body = std::move(body);
  return stmt(n);
}

stmt slice_buffer::make(var sym, var src, std::vector<expr> at, stmt body) {
  return make_slice_buffer(sym, src, at, std::move(body));
}

stmt slice_buffer::make(var sym, var src, span<expr> at, stmt body) {
  return make_slice_buffer(sym, src, at, std::move(body));
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

template <typename Dims>
stmt make_transpose(var sym, var src, Dims&& dims, stmt body) {
  auto n = make_node<transpose>(size_of(dims));
  n->sym = sym;
  n->src = src;
  void* arrays = n + 1;
  n->dims = make_span(arrays, dims);
  n->body = std::move(body);
  return stmt(n);
}

stmt transpose::make(var sym, var src, std::vector<int> dims, stmt body) {
  return make_transpose(sym, src, dims, std::move(body));
}

stmt transpose::make(var sym, var src, span<int> dims, stmt body) {
  return make_transpose(sym, src, dims, std::move(body));
}

stmt transpose::make_truncate(var sym, var src, int rank, stmt body) {
  std::vector<int> dims(rank);
  std::iota(dims.begin(), dims.end(), 0);
  return make(sym, src, std::move(dims), std::move(body));
}

bool transpose::is_truncate(span<int> dims) {
  for (std::size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] != static_cast<int>(i)) return false;
  }
  return true;
}
bool transpose::is_truncate() const { return is_truncate(dims); }

stmt async::make(var sym, stmt task, stmt body) {
  auto n = new async();
  n->sym = sym;
  n->task = std::move(task);
  n->body = std::move(body);
  return stmt(n);
}

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

bool is_variable(expr_ref x, var b, buffer_field field, int dim) {
  if (const variable* v = x.as<variable>()) {
    return v->sym == b && v->field == field && v->dim == dim;
  } else {
    return false;
  }
}

expr abs(expr x) { return call::make(intrinsic::abs, {std::move(x)}); }
expr align_down(expr x, const expr& a) { return is_one(a) ? x : (std::move(x) / a) * a; }
expr align_up(expr x, const expr& a) { return is_one(a) ? x : ((std::move(x) + (a - 1)) / a) * a; }
interval_expr align(interval_expr x, const expr& a) {
  if (is_one(a)) {
    return x;
  } else {
    return {align_down(std::move(x.min), a), ((std::move(x.max) + a) / a) * a - 1};
  }
}

expr and_then(expr a, expr b) { return call::make(intrinsic::and_then, {std::move(a), std::move(b)}); }
expr or_else(expr a, expr b) { return call::make(intrinsic::or_else, {std::move(a), std::move(b)}); }

expr buffer_rank(var buf) { return variable::make(buf, buffer_field::rank); }
expr buffer_elem_size(var buf) { return variable::make(buf, buffer_field::elem_size); }
expr buffer_min(var buf, int dim) { return variable::make(buf, buffer_field::min, dim); }
expr buffer_max(var buf, int dim) { return variable::make(buf, buffer_field::max, dim); }
expr buffer_extent(var buf, int dim) { return (buffer_max(buf, dim) - buffer_min(buf, dim)) + 1; }
expr buffer_stride(var buf, int dim) { return variable::make(buf, buffer_field::stride, dim); }
expr buffer_fold_factor(var buf, int dim) { return variable::make(buf, buffer_field::fold_factor, dim); }
expr validate_buffer(var buf) { return call::make(intrinsic::validate_buffer, {buf}); }
expr buffer_size_bytes(expr buf) { return call::make(intrinsic::buffer_size_bytes, {std::move(buf)}); }

interval_expr buffer_bounds(var buf, int dim) { return {buffer_min(buf, dim), buffer_max(buf, dim)}; }
dim_expr buffer_dim(var buf, int dim) {
  return {buffer_bounds(buf, dim), buffer_stride(buf, dim), buffer_fold_factor(buf, dim)};
}
std::vector<dim_expr> buffer_dims(var buf, int rank) {
  std::vector<dim_expr> result;
  result.reserve(rank);
  for (int d = 0; d < rank; ++d) {
    result.push_back(buffer_dim(buf, d));
  }
  return result;
}
std::vector<dim_expr> buffer_dims(const raw_buffer& buf) {
  std::vector<dim_expr> result;
  result.reserve(buf.rank);
  for (std::size_t d = 0; d < buf.rank; ++d) {
    result.push_back(buf.dim(d));
  }
  return result;
}

expr buffer_at(expr buf, span<expr> at) {
  std::vector<expr> args;
  args.reserve(at.size() + 1);
  args.push_back(std::move(buf));
  args.insert(args.end(), at.begin(), at.end());
  return call::make(intrinsic::buffer_at, std::move(args));
}

expr buffer_at(expr buf, span<var> at) {
  std::vector<expr> args;
  args.reserve(at.size() + 1);
  args.push_back(std::move(buf));
  args.insert(args.end(), at.begin(), at.end());
  return call::make(intrinsic::buffer_at, std::move(args));
}

expr buffer_at(expr buf) { return call::make(intrinsic::buffer_at, {std::move(buf)}); }

box_expr dims_bounds(span<dim_expr> dims) {
  box_expr result(dims.size());
  for (std::size_t d = 0; d < dims.size(); ++d) {
    result[d] = dims[d].bounds;
  }
  return result;
}

const expr& dim_expr::get_field(buffer_field field) const {
  switch (field) {
  case buffer_field::min: return bounds.min;
  case buffer_field::max: return bounds.max;
  case buffer_field::stride: return stride;
  case buffer_field::fold_factor: return fold_factor;
  default: break;
  }
  SLINKY_UNREACHABLE << "buffer_field " << to_string(field) << " is not a dim field";
  return fold_factor;
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
    return c->intrinsic == intrinsic::buffer_at;
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

expr semaphore_helper(intrinsic fn, span<expr> sems, span<expr> counts) {
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

expr semaphore_signal(span<expr> sems, span<expr> counts) {
  return semaphore_helper(intrinsic::semaphore_signal, sems, counts);
}
expr semaphore_wait(span<expr> sems, span<expr> counts) {
  return semaphore_helper(intrinsic::semaphore_wait, sems, counts);
}

expr wait_for(expr task) { return wait_for(std::vector<expr>{std::move(task)}); }
expr wait_for(std::vector<expr> tasks) { return call::make(intrinsic::wait_for, std::move(tasks)); }

void recursive_node_visitor::visit(const variable*) {}
void recursive_node_visitor::visit(const constant*) {}
void recursive_node_visitor::visit(const constant_buffer*) {}

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
  if (op->max_workers.defined()) op->max_workers.accept(this);
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
    if (i.stride.defined()) i.stride.accept(this);
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
void recursive_node_visitor::visit(const async* op) {
  if (op->task.defined()) op->task.accept(this);
  if (op->body.defined()) op->body.accept(this);
}
void recursive_node_visitor::visit(const check* op) {
  if (op->condition.defined()) op->condition.accept(this);
}

}  // namespace slinky
