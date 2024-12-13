#ifndef SLINKY_BUILDER_REWRITE_H
#define SLINKY_BUILDER_REWRITE_H

#include <iostream>

#include "builder/substitute.h"
#include "runtime/evaluate.h"
#include "runtime/expr.h"
#include "runtime/print.h"

// This pattern matching engine is heavily inspired by https://github.com/halide/Halide/blob/main/src/IRMatch.h.

namespace slinky {
namespace rewrite {

// The maximum number of values pattern_wildcard::idx and pattern_constant::idx can have, starting from 0.
constexpr int symbol_count = 6;
constexpr int constant_count = 5;

template <int N>
class pattern_wildcard;
template <int N>
class pattern_constant;

struct match_context {
  std::array<const base_expr_node*, symbol_count> vars;
  std::array<index_t, constant_count> constants;
  int variant;
  int variant_bits;

  template <int N>
  expr_ref matched(const pattern_wildcard<N>&) const {
    return vars[N];
  }
  template <int N>
  index_t matched(const pattern_constant<N>& p) const {
    return constants[N];
  }
};

template <int matched>
SLINKY_UNIQUE bool match(index_t p, expr_ref x, match_context&) {
  return is_constant(x, p);
}
SLINKY_UNIQUE index_t substitute(index_t p, const match_context&, bool&) { return p; }

template <typename T>
struct pattern_info {
  static constexpr expr_node_type type = T::type;
  static constexpr bool is_boolean = T::is_boolean;
  static constexpr bool is_canonical = T::is_canonical;
  static constexpr int matched = T::matched;
};
template <>
struct pattern_info<std::int32_t> {
  static constexpr expr_node_type type = expr_node_type::constant;
  static constexpr bool is_boolean = false;
  static constexpr bool is_canonical = true;
  static constexpr int matched = 0;
};
template <>
struct pattern_info<std::int64_t> {
  static constexpr expr_node_type type = expr_node_type::constant;
  static constexpr bool is_boolean = false;
  static constexpr bool is_canonical = true;
  static constexpr int matched = 0;
};
template <>
struct pattern_info<bool> {
  static constexpr expr_node_type type = expr_node_type::constant;
  static constexpr bool is_boolean = true;
  static constexpr bool is_canonical = true;
  static constexpr int matched = 0;
};
#ifdef __EMSCRIPTEN__
template <>
struct pattern_info<long> {
  static constexpr expr_node_type type = expr_node_type::constant;
  static constexpr bool is_boolean = false;
  static constexpr bool is_canonical = true;
  static constexpr int matched = 0;
};
#endif

class pattern_expr {
public:
  expr_ref e;
};

template <>
struct pattern_info<pattern_expr> {
  static constexpr expr_node_type type = expr_node_type::none;
  static constexpr bool is_boolean = false;
  static constexpr bool is_canonical = true;
  static constexpr int matched = 0;
};

SLINKY_UNIQUE std::ostream& operator<<(std::ostream& os, const pattern_expr& e) { return os << e.e; }

template <int N>
class pattern_wildcard {
public:
  static constexpr expr_node_type type = expr_node_type::none;
  static constexpr bool is_boolean = false;
  static constexpr bool is_canonical = true;
  static constexpr int matched = 1 << N;
};

template <int matched, int N>
SLINKY_UNIQUE bool match(const pattern_wildcard<N>& p, expr_ref x, match_context& ctx) {
  if (matched & (1 << N)) {
    return slinky::match(x.get(), ctx.vars[N]);
  } else {
    ctx.vars[N] = x.get();
    return true;
  }
}

template <int N>
SLINKY_UNIQUE expr_ref substitute(const pattern_wildcard<N>&, const match_context& ctx, bool&) {
  return ctx.vars[N];
}

template <int N>
SLINKY_UNIQUE std::ostream& operator<<(std::ostream& os, const pattern_wildcard<N>&) {
  static constexpr char names[] = "xyzwuv";
  return os << names[N];
}

template <int N>
class pattern_constant {
public:
  static constexpr expr_node_type type = expr_node_type::constant;
  static constexpr bool is_boolean = false;
  static constexpr bool is_canonical = true;
  static constexpr int matched = 1 << (symbol_count + N);
};

template <int matched, int N>
SLINKY_UNIQUE bool match(const pattern_constant<N>& p, expr_ref x, match_context& ctx) {
  if (const constant* c = x.as<constant>()) {
    if (matched & (1 << (symbol_count + N))) {
      return ctx.constants[N] == c->value;
    } else {
      ctx.constants[N] = c->value;
      return true;
    }
  }
  return false;
}

template <int N>
SLINKY_UNIQUE index_t substitute(const pattern_constant<N>&, const match_context& ctx, bool&) {
  return ctx.constants[N];
}

template <int N>
SLINKY_UNIQUE std::ostream& operator<<(std::ostream& os, const pattern_constant<N>&) {
  return os << 'c' << N;
}

template <typename T, typename A, typename B>
class pattern_binary {
public:
  A a;
  B b;
};

template <typename T, typename A, typename B>
struct pattern_info<pattern_binary<T, A, B>> {
  static constexpr expr_node_type type = T::static_type;
  static constexpr bool is_boolean = is_boolean_node(T::static_type);
  static constexpr bool is_canonical =
      pattern_info<A>::is_canonical && pattern_info<B>::is_canonical &&
      (!T::commutative || !should_commute(pattern_info<A>::type, pattern_info<B>::type));
  static constexpr int matched = pattern_info<A>::matched | pattern_info<B>::matched;

  static constexpr expr_node_type A_type = pattern_info<A>::type;
  static constexpr expr_node_type B_type = pattern_info<B>::type;

  static constexpr bool could_commute =
      T::commutative && (A_type == expr_node_type::none || B_type == expr_node_type::none || A_type == B_type);
};

template <int matched, typename T, typename A, typename B>
SLINKY_UNIQUE bool match_binary(const pattern_binary<T, A, B>& p, expr_ref a, expr_ref b, match_context& ctx) {
  if (pattern_info<pattern_binary<T, A, B>>::could_commute) {
    // This is a commutative operation and we can't canonicalize the ordering.
    // Remember which bit in the variant index is ours, and increment the bit for the next commutative node.
    const int this_bit = ctx.variant_bits++;
    if ((ctx.variant & (1 << this_bit)) != 0) {
      // We should commute in this variant.
      std::swap(a, b);
    }
  }
  return match<matched>(p.a, a, ctx) && match<matched | pattern_info<A>::matched>(p.b, b, ctx);
}

template <int matched, typename T, typename A, typename B>
SLINKY_UNIQUE bool match(const pattern_binary<T, A, B>& p, expr_ref x, match_context& ctx) {
  if (const T* t = x.as<T>()) {
    return match_binary<matched>(p, t->a, t->b, ctx);
  } else {
    return false;
  }
}

template <int matched, typename T, typename A, typename B>
SLINKY_UNIQUE bool match(
    const pattern_binary<T, A, B>& p, const pattern_binary<T, pattern_expr, pattern_expr>& x, match_context& ctx) {
  return match_binary<matched>(p, x.a.e, x.b.e, ctx);
}

template <typename T>
SLINKY_UNIQUE expr substitute_binary(expr a, expr b, bool&) {
  return make_binary<T>(std::move(a), std::move(b));
}

template <typename T>
SLINKY_UNIQUE index_t substitute_binary(index_t a, index_t b, bool& overflowed) {
  if (binary_overflows<T>(a, b)) {
    overflowed = true;
    return 0;
  } else {
    return make_binary<T>(a, b);
  }
}

template <typename T, typename A, typename B>
SLINKY_UNIQUE auto substitute(const pattern_binary<T, A, B>& p, const match_context& ctx, bool& overflowed) {
  return substitute_binary<T>(substitute(p.a, ctx, overflowed), substitute(p.b, ctx, overflowed), overflowed);
}

template <typename T, typename A, typename B>
std::ostream& operator<<(std::ostream& os, const pattern_binary<T, A, B>& p) {
  switch (T::static_type) {
  case add::static_type: return os << '(' << p.a << " + " << p.b << ')';
  case sub::static_type: return os << '(' << p.a << " - " << p.b << ')';
  case mul::static_type: return os << '(' << p.a << " * " << p.b << ')';
  case div::static_type: return os << '(' << p.a << " / " << p.b << ')';
  case mod::static_type: return os << '(' << p.a << " % " << p.b << ')';
  case min::static_type: return os << "min(" << p.a << ", " << p.b << ')';
  case max::static_type: return os << "max(" << p.a << ", " << p.b << ')';
  case less::static_type: return os << '(' << p.a << " < " << p.b << ')';
  case less_equal::static_type: return os << '(' << p.a << " <= " << p.b << ')';
  case equal::static_type: return os << '(' << p.a << " == " << p.b << ')';
  case not_equal::static_type: return os << '(' << p.a << " != " << p.b << ')';
  case logical_and::static_type: return os << '(' << p.a << " && " << p.b << ')';
  case logical_or::static_type: return os << '(' << p.a << " || " << p.b << ')';
  default: std::abort();
  }
}

template <typename T, typename A>
class pattern_unary {
public:
  static constexpr expr_node_type type = T::static_type;
  static constexpr bool is_boolean = is_boolean_node(T::static_type);
  static constexpr bool is_canonical = pattern_info<A>::is_canonical;
  static constexpr int matched = pattern_info<A>::matched;
  A a;
};

template <int matched, typename T, typename A>
SLINKY_UNIQUE bool match(const pattern_unary<T, A>& p, expr_ref x, match_context& ctx) {
  if (const T* t = x.as<T>()) {
    return match<matched>(p.a, t->a, ctx);
  } else {
    return false;
  }
}

template <int matched, typename T, typename A>
SLINKY_UNIQUE bool match(const pattern_unary<T, A>& p, const pattern_unary<T, pattern_expr>& x, match_context& ctx) {
  return match<matched>(p.a, x.a.e, ctx);
}

template <typename T, typename A>
SLINKY_UNIQUE std::ostream& operator<<(std::ostream& os, const pattern_unary<T, A>& p) {
  switch (T::static_type) {
  case logical_not::static_type: return os << '!' << p.a;
  default: std::abort();
  }
}

template <typename T>
SLINKY_UNIQUE expr make_unary(expr a) {
  return T::make(std::move(a));
}
// clang-format off
template <typename T> SLINKY_UNIQUE index_t make_unary(index_t a);
template <> inline index_t make_unary<logical_not>(index_t a) { return a == 0 ? 1 : 0; }
// clang-format on

template <typename T, typename A>
SLINKY_UNIQUE auto substitute(const pattern_unary<T, A>& p, const match_context& ctx, bool& overflowed) {
  return make_unary<T>(substitute(p.a, ctx, overflowed));
}

template <typename C, typename T, typename F>
class pattern_select {
public:
  static constexpr expr_node_type type = expr_node_type::select;
  static constexpr bool is_boolean = pattern_info<T>::is_boolean && pattern_info<F>::is_boolean;
  static constexpr bool is_canonical =
      pattern_info<C>::is_canonical && pattern_info<T>::is_canonical && pattern_info<F>::is_canonical;
  static constexpr int matched = pattern_info<C>::matched | pattern_info<T>::matched | pattern_info<F>::matched;
  C c;
  T t;
  F f;
};

template <int matched, typename C, typename T, typename F>
SLINKY_UNIQUE bool match(const pattern_select<C, T, F>& p, expr_ref x, match_context& ctx) {
  if (const class select* s = x.as<class select>()) {
    return match<matched>(p.c, s->condition, ctx) &&
           match<matched | pattern_info<C>::matched>(p.t, s->true_value, ctx) &&
           match<matched | pattern_info<C>::matched | pattern_info<T>::matched>(p.f, s->false_value, ctx);
  } else {
    return false;
  }
}

template <int matched, typename C, typename T, typename F>
SLINKY_UNIQUE bool match(const pattern_select<C, T, F>& p,
    const pattern_select<pattern_expr, pattern_expr, pattern_expr>& x, match_context& ctx) {
  return match<matched>(p.c, x.c.e, ctx) && match<matched | pattern_info<C>::matched>(p.t, x.t.e, ctx) &&
         match<matched | pattern_info<C>::matched | pattern_info<T>::matched>(p.f, x.f.e, ctx);
}

template <typename C, typename T, typename F>
SLINKY_UNIQUE expr substitute(const pattern_select<C, T, F>& p, const match_context& ctx, bool& overflowed) {
  return select::make(
      substitute(p.c, ctx, overflowed), substitute(p.t, ctx, overflowed), substitute(p.f, ctx, overflowed));
}

template <typename C, typename T, typename F>
SLINKY_UNIQUE std::ostream& operator<<(std::ostream& os, const pattern_select<C, T, F>& p) {
  return os << "select(" << p.c << ", " << p.t << ", " << p.f << ")";
}

template <typename... Args>
class pattern_call {
public:
  static constexpr expr_node_type type = expr_node_type::call;
  static constexpr bool is_boolean = false;
  static constexpr bool is_canonical = (... && pattern_info<Args>::is_canonical);
  static constexpr int matched = (0 | ... | pattern_info<Args>::matched);
  slinky::intrinsic fn;
  std::tuple<Args...> args;
};

template <int matched>
SLINKY_UNIQUE bool match_tuple(const std::tuple<>& t, const std::vector<expr>& x, match_context& ctx) {
  return true;
}
template <int matched, typename A>
SLINKY_UNIQUE bool match_tuple(const std::tuple<A>& t, const std::vector<expr>& x, match_context& ctx) {
  return match<matched>(std::get<0>(t), x[0], ctx);
}
template <int matched, typename A, typename B>
SLINKY_UNIQUE bool match_tuple(const std::tuple<A, B>& t, const std::vector<expr>& x, match_context& ctx) {
  return match<matched>(std::get<0>(t), x[0], ctx) &&
         match<matched | pattern_info<A>::matched>(std::get<1>(t), x[1], ctx);
}

template <int matched, typename... Args>
SLINKY_UNIQUE bool match(const pattern_call<Args...>& p, expr_ref x, match_context& ctx) {
  if (const call* c = x.as<call>()) {
    if (c->intrinsic == p.fn) {
      assert(c->args.size() == sizeof...(Args));
      return match_tuple<matched>(p.args, c->args, ctx);
    }
  }
  return false;
}

SLINKY_UNIQUE expr substitute(const pattern_call<>& p, const match_context& ctx, bool&) { return call::make(p.fn, {}); }
template <typename A>
SLINKY_UNIQUE expr substitute(const pattern_call<A>& p, const match_context& ctx, bool& overflowed) {
  return call::make(p.fn, {substitute(std::get<0>(p.args), ctx, overflowed)});
}
template <typename A, typename B>
SLINKY_UNIQUE expr substitute(const pattern_call<A, B>& p, const match_context& ctx, bool& overflowed) {
  return call::make(
      p.fn, {substitute(std::get<0>(p.args), ctx, overflowed), substitute(std::get<1>(p.args), ctx, overflowed)});
}

SLINKY_UNIQUE std::ostream& operator<<(std::ostream& os, const pattern_call<>& p) { return os << p.fn << "()"; }
template <typename A>
SLINKY_UNIQUE std::ostream& operator<<(std::ostream& os, const pattern_call<A>& p) {
  return os << p.fn << "(" << std::get<0>(p.args) << ")";
}
template <typename A, typename B>
SLINKY_UNIQUE std::ostream& operator<<(std::ostream& os, const pattern_call<A, B>& p) {
  return os << p.fn << "(" << std::get<0>(p.args) << ", " << std::get<1>(p.args) << ")";
}

template <typename T, typename Fn>
class replacement_predicate {
public:
  static constexpr expr_node_type type = expr_node_type::constant;
  static constexpr bool is_canonical = true;
  T a;
  Fn fn;
};

template <typename T, typename Fn>
SLINKY_UNIQUE bool substitute(const replacement_predicate<T, Fn>& r, const match_context& ctx, bool& overflowed) {
  return r.fn(substitute(r.a, ctx, overflowed));
}

template <typename T, typename Fn>
SLINKY_UNIQUE replacement_predicate<T, Fn> make_predicate(T t, Fn fn) {
  return {t, fn};
}

template <typename T, typename Fn>
SLINKY_UNIQUE std::ostream& operator<<(std::ostream& os, const replacement_predicate<T, Fn>&) {
  return os << "<unknown predicate>";
}

template <typename T>
class replacement_eval {
public:
  static constexpr expr_node_type type = expr_node_type::constant;
  static constexpr bool is_boolean = pattern_info<T>::is_boolean;
  static constexpr bool is_canonical = true;
  T a;
};

template <typename T>
SLINKY_UNIQUE index_t substitute(const replacement_eval<T>& r, const match_context& ctx, bool& overflowed) {
  return substitute(r.a, ctx, overflowed);
}

template <typename T>
SLINKY_UNIQUE std::ostream& operator<<(std::ostream& os, const replacement_eval<T>& r) {
  return os << r.a;
}

template <typename T>
class replacement_boolean {
public:
  static constexpr expr_node_type type = expr_node_type::none;
  static constexpr bool is_boolean = true;
  static constexpr bool is_canonical = pattern_info<T>::is_canonical;
  T a;
};

template <typename T>
SLINKY_UNIQUE auto substitute(const replacement_boolean<T>& r, const match_context& ctx, bool& overflowed) {
  return boolean(substitute(r.a, ctx, overflowed));
}

template <typename T>
SLINKY_UNIQUE std::ostream& operator<<(std::ostream& os, const replacement_boolean<T>& r) {
  return os << "boolean(" << r.a << ")";
}

// We need a thing that lets us do SFINAE to disable overloads when none of the operand types are pattern expressions.
template <typename... Ts>
struct enable_pattern_ops {
  using type = void;
};

template <typename T, typename... Ts>
struct enable_pattern_ops<T, Ts...> {
  using type = typename enable_pattern_ops<Ts...>::type;
};

// clang-format off
template <typename... Ts>
struct enable_pattern_ops<pattern_expr, Ts...> { using type = std::true_type; };
template <int N, typename... Ts>
struct enable_pattern_ops<pattern_wildcard<N>, Ts...> { using type = std::true_type; };
template <int N, typename... Ts>
struct enable_pattern_ops<pattern_constant<N>, Ts...> { using type = std::true_type; };
template <typename T, typename A, typename B, typename... Ts>
struct enable_pattern_ops<pattern_binary<T, A, B>, Ts...> { using type = std::true_type; };
template <typename T, typename A, typename... Ts>
struct enable_pattern_ops<pattern_unary<T, A>, Ts...> { using type = std::true_type; };
template <typename C, typename T, typename F, typename... Ts>
struct enable_pattern_ops<pattern_select<C, T, F>, Ts...> { using type = std::true_type; };
template <typename... Args, typename... Ts>
struct enable_pattern_ops<pattern_call<Args...>, Ts...> { using type = std::true_type; };
template <typename T, typename... Ts>
struct enable_pattern_ops<replacement_eval<T>, Ts...> { using type = std::true_type; };
template <typename T, typename Fn, typename... Ts>
struct enable_pattern_ops<replacement_predicate<T, Fn>, Ts...> { using type = std::true_type; };

template <typename A, bool = typename enable_pattern_ops<A>::type()>
SLINKY_UNIQUE auto operator!(const A& a) { return pattern_unary<logical_not, A>{a}; }
template <typename A, bool = typename enable_pattern_ops<A>::type()>
SLINKY_UNIQUE auto operator-(const A& a) { return pattern_binary<sub, index_t, A>{0, a}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
SLINKY_UNIQUE auto operator+(const A& a, const B& b) { return pattern_binary<add, A, B>{a, b}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
SLINKY_UNIQUE auto operator-(const A& a, const B& b) { return pattern_binary<sub, A, B>{a, b}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
SLINKY_UNIQUE auto operator*(const A& a, const B& b) { return pattern_binary<mul, A, B>{a, b}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
SLINKY_UNIQUE auto operator/(const A& a, const B& b) { return pattern_binary<div, A, B>{a, b}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
SLINKY_UNIQUE auto operator%(const A& a, const B& b) { return pattern_binary<mod, A, B>{a, b}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
SLINKY_UNIQUE auto operator==(const A& a, const B& b) { return pattern_binary<equal, A, B>{a, b}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
SLINKY_UNIQUE auto operator!=(const A& a, const B& b) { return pattern_binary<not_equal, A, B>{a, b}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
SLINKY_UNIQUE auto operator<(const A& a, const B& b) { return pattern_binary<less, A, B>{a, b}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
SLINKY_UNIQUE auto operator<=(const A& a, const B& b) { return pattern_binary<less_equal, A, B>{a, b}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
SLINKY_UNIQUE auto operator>(const A& a, const B& b) { return pattern_binary<less, B, A>{b, a}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
SLINKY_UNIQUE auto operator>=(const A& a, const B& b) { return pattern_binary<less_equal, B, A>{b, a}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
SLINKY_UNIQUE auto operator&&(const A& a, const B& b) { return pattern_binary<logical_and, A, B>{a, b}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
SLINKY_UNIQUE auto operator||(const A& a, const B& b) { return pattern_binary<logical_or, A, B>{a, b}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
SLINKY_UNIQUE auto min(const A& a, const B& b) { return pattern_binary<class min, A, B>{a, b}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
SLINKY_UNIQUE auto max(const A& a, const B& b) { return pattern_binary<class max, A, B>{a, b}; }
template <typename C, typename T, typename F, bool = typename enable_pattern_ops<C, T, F>::type()>
SLINKY_UNIQUE auto select(const C& c, const T& t, const F& f) { return pattern_select<C, T, F>{c, t, f}; }
template <typename T, bool = typename enable_pattern_ops<T>::type()>
SLINKY_UNIQUE auto abs(const T& x) { return pattern_call<T>{intrinsic::abs, {x}}; }
template <typename T, bool = typename enable_pattern_ops<T>::type()>
SLINKY_UNIQUE auto boolean(const T& x) { return replacement_boolean<T>{x}; }
SLINKY_UNIQUE auto positive_infinity() { return pattern_call<>{intrinsic::positive_infinity, {}}; }
SLINKY_UNIQUE auto negative_infinity() { return pattern_call<>{intrinsic::negative_infinity, {}}; }
SLINKY_UNIQUE auto indeterminate() { return pattern_call<>{intrinsic::indeterminate, {}}; }

template <typename T, bool = typename enable_pattern_ops<T>::type()>
SLINKY_UNIQUE auto is_finite(const T& x) { return make_predicate(x, slinky::is_finite); }
template <typename T, bool = typename enable_pattern_ops<T>::type()>
SLINKY_UNIQUE auto is_constant(const T& x) { return make_predicate(x, [](expr_ref x) { return x.as<constant>() != nullptr; }); }
template <typename T, bool = typename enable_pattern_ops<T>::type()>
SLINKY_UNIQUE auto is_zero(const T& x) { return make_predicate(x, slinky::is_zero); }
template <typename T, bool = typename enable_pattern_ops<T>::type()>
SLINKY_UNIQUE auto is_boolean(const T& x) { return make_predicate(x, slinky::is_boolean); }
// clang-format on

template <int N1, int N2>
using buffer_dim_meta = pattern_call<pattern_wildcard<N1>, pattern_wildcard<N2>>;

template <int N1, int N2>
SLINKY_UNIQUE auto buffer_min(pattern_wildcard<N1> buf, pattern_wildcard<N2> dim) {
  return buffer_dim_meta<N1, N2>{intrinsic::buffer_min, {buf, dim}};
}
template <int N1, int N2>
SLINKY_UNIQUE auto buffer_max(pattern_wildcard<N1> buf, pattern_wildcard<N2> dim) {
  return buffer_dim_meta<N1, N2>{intrinsic::buffer_max, {buf, dim}};
}

template <typename T>
SLINKY_UNIQUE auto eval(const T& x) {
  return replacement_eval<T>{x};
}

template <typename Pattern, typename Target>
SLINKY_UNIQUE bool match_any_variant(Pattern p, const Target& x, match_context& ctx) {
  static_assert(pattern_info<Pattern>::is_canonical);

  // We'll find out how many variants we have when we try to match.
  // This can grow if we fail early due to a commutative variant that doesn't match near the root
  // of the expression, so we track the max we've seen.
  int variant_count = 1;
  for (ctx.variant = 0; ctx.variant < variant_count; ++ctx.variant) {
    ctx.variant_bits = 0;
    if (match<0>(p, x, ctx)) {
      return true;
    }
    variant_count = std::max(variant_count, 1 << ctx.variant_bits);
  }
  return false;
}

template <typename Pattern, typename Target, typename Predicate>
SLINKY_UNIQUE bool match(match_context& ctx, Pattern p, const Target& x, Predicate pr) {
  if (!match_any_variant(p, x, ctx)) return false;
  bool overflowed = false;
  return substitute(pr, ctx, overflowed) && !overflowed;
}

template <typename Pattern, typename Target>
SLINKY_UNIQUE bool match(match_context& ctx, Pattern p, const Target& x) {
  return match_any_variant(p, x, ctx);
}

template <typename Target>
class base_rewriter {
  Target x;

  template <typename Pattern>
  SLINKY_ALWAYS_INLINE static bool find_replacement(const match_context& ctx) {
    return false;
  }

  template <typename Pattern, typename Replacement>
  SLINKY_ALWAYS_INLINE bool find_replacement(const match_context& ctx, Replacement r) {
    static_assert(pattern_info<Replacement>::is_canonical);
    static_assert(!pattern_info<Pattern>::is_boolean || pattern_info<Replacement>::is_boolean);
    bool overflowed = false;
    result = substitute(r, ctx, overflowed);
    return !overflowed;
  }

  template <typename Pattern, typename Replacement, typename Predicate, typename... ReplacementPredicates>
  SLINKY_ALWAYS_INLINE bool find_replacement(
      const match_context& ctx, Replacement r, Predicate pr, ReplacementPredicates... r_pr) {
    static_assert(pattern_info<Replacement>::is_canonical);
    static_assert(!pattern_info<Pattern>::is_boolean || pattern_info<Replacement>::is_boolean);

    bool overflowed = false;
    if (substitute(pr, ctx, overflowed) && !overflowed) {
      result = substitute(r, ctx, overflowed);
      if (!overflowed) return true;
    }
    // Try the next replacement
    return find_replacement<Pattern>(ctx, r_pr...);
  }

public:
  expr result;

  base_rewriter(Target x) : x(std::move(x)) {}
  base_rewriter(const base_rewriter&) = delete;

  // If the pattern p matches the target, substitute with the replacement r if the predicate pr is true.
  // If the predicate is false, consider the next replacement and predicate.
  // The last predicate is optional and defaults to true.
  template <typename Pattern, typename... ReplacementPredicate>
  SLINKY_ALWAYS_INLINE bool operator()(Pattern p, ReplacementPredicate... r_pr) {
    static_assert(pattern_info<Pattern>::is_canonical);

    match_context ctx;
    if (!match_any_variant(p, x, ctx)) return false;

    return find_replacement<Pattern>(ctx, r_pr...);
  }
};

class rewriter : public base_rewriter<expr_ref> {
public:
  rewriter(const expr& x) : base_rewriter(x) {}
  rewriter(expr_ref x) : base_rewriter(x) {}
  using base_rewriter::operator();
};

template <typename Target>
SLINKY_UNIQUE base_rewriter<Target> make_rewriter(Target x) {
  return base_rewriter<Target>(std::move(x));
}

}  // namespace rewrite
}  // namespace slinky

#endif  // SLINKY_BUILDER_REWRITE_H