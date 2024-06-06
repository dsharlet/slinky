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
  std::array<const index_t*, constant_count> constants;
  // A bitset indicating which vars have been matched.
  int vars_mask;
  int variant;
  int variant_bits;

  template <int N>
  const base_expr_node* matched(const pattern_wildcard<N>&) const;
  template <int N>
  const index_t* matched(const pattern_constant<N>&) const;
  template <int N>
  index_t matched(const pattern_constant<N>& p, index_t def) const;
};

SLINKY_ALWAYS_INLINE inline bool match(index_t p, const expr& x, match_context&) { return is_constant(x, p); }
SLINKY_ALWAYS_INLINE inline index_t substitute(index_t p, const match_context&) { return p; }

template <typename T>
struct pattern_info {
  static constexpr expr_node_type type = T::type;
  static constexpr bool is_boolean = T::is_boolean;
  static constexpr bool is_canonical = T::is_canonical;
};
template <>
struct pattern_info<std::int32_t> {
  static constexpr expr_node_type type = expr_node_type::constant;
  static constexpr bool is_boolean = false;
  static constexpr bool is_canonical = true;
};
template <>
struct pattern_info<std::int64_t> {
  static constexpr expr_node_type type = expr_node_type::constant;
  static constexpr bool is_boolean = false;
  static constexpr bool is_canonical = true;
};
template <>
struct pattern_info<bool> {
  static constexpr expr_node_type type = expr_node_type::constant;
  static constexpr bool is_boolean = true;
  static constexpr bool is_canonical = true;
};

class pattern_expr {
public:
  const expr& e;
};

template <>
struct pattern_info<pattern_expr> {
  static constexpr expr_node_type type = expr_node_type::none;
  static constexpr bool is_boolean = false;
  static constexpr bool is_canonical = true;
};

inline std::ostream& operator<<(std::ostream& os, const pattern_expr& e) { return os << e.e; }

SLINKY_ALWAYS_INLINE inline const expr& substitute(const pattern_expr& p, const match_context& ctx) { return p.e; }

template <int N>
class pattern_wildcard {
public:
  static constexpr expr_node_type type = expr_node_type::none;
  static constexpr bool is_boolean = false;
  static constexpr bool is_canonical = true;
};

template <int N>
inline bool match(const pattern_wildcard<N>& p, const expr& x, match_context& ctx) {
  if (ctx.vars_mask & (1 << N)) {
    // Try pointer comparison first to short circuit the full match.
    return x.get() == ctx.vars[N] || slinky::compare(x.get(), ctx.vars[N]) == 0;
  } else {
    ctx.vars_mask |= (1 << N);
    ctx.vars[N] = x.get();
    return true;
  }
}

template <int N>
inline const base_expr_node* substitute(const pattern_wildcard<N>& p, const match_context& ctx) {
  assert(ctx.vars_mask & (1 << N));
  return ctx.vars[N];
}

template <int N>
std::ostream& operator<<(std::ostream& os, const pattern_wildcard<N>&) {
  static constexpr char names[] = "xyzwuv";
  return os << names[N];
}

template <int N>
class pattern_constant {
public:
  static constexpr expr_node_type type = expr_node_type::constant;
  static constexpr bool is_boolean = false;
  static constexpr bool is_canonical = true;
};

template <int N>
inline bool match(const pattern_constant<N>& p, const expr& x, match_context& ctx) {
  if (const constant* c = x.as<constant>()) {
    if (ctx.constants[N]) {
      return *ctx.constants[N] == c->value;
    } else {
      ctx.constants[N] = &c->value;
      return true;
    }
  }
  return false;
}

template <int N>
inline index_t substitute(const pattern_constant<N>& p, const match_context& ctx) {
  assert(ctx.constants[N]);
  return *ctx.constants[N];
}

template <int N>
std::ostream& operator<<(std::ostream& os, const pattern_constant<N>&) {
  return os << 'c' << N;
}

template <int N>
inline const base_expr_node* match_context::matched(const pattern_wildcard<N>& p) const {
  return vars[N];
}
template <int N>
inline const index_t* match_context::matched(const pattern_constant<N>& p) const {
  return constants[N];
}
template <int N>
inline index_t match_context::matched(const pattern_constant<N>& p, index_t def) const {
  return constants[N] ? *constants[N] : def;
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
};

template <typename T, typename A, typename B>
bool match_binary(const pattern_binary<T, A, B>& p, const expr& a, const expr& b, match_context& ctx) {
  int this_bit = -1;
  if (T::commutative) {
    constexpr expr_node_type ta = pattern_info<A>::type;
    constexpr expr_node_type tb = pattern_info<B>::type;
    if (ta == expr_node_type::none || tb == expr_node_type::none || ta == tb) {
      // This is a commutative operation and we can't canonicalize the ordering.
      // Remember which bit in the variant index is ours, and increment the bit for the next commutative node.
      this_bit = ctx.variant_bits++;
    }
  }
  if (this_bit >= 0 && (ctx.variant & (1 << this_bit)) != 0) {
    // We should commute in this variant.
    return match(p.a, b, ctx) && match(p.b, a, ctx);
  } else {
    return match(p.a, a, ctx) && match(p.b, b, ctx);
  }
}

template <typename T, typename A, typename B>
bool match(const pattern_binary<T, A, B>& p, const expr& x, match_context& ctx) {
  if (const T* t = x.as<T>()) {
    return match_binary(p, t->a, t->b, ctx);
  } else {
    return false;
  }
}

template <typename T, typename A, typename B>
bool match(
    const pattern_binary<T, A, B>& p, const pattern_binary<T, pattern_expr, pattern_expr>& x, match_context& ctx) {
  return match_binary(p, x.a.e, x.b.e, ctx);
}

template <typename T, typename A, typename B>
auto substitute(const pattern_binary<T, A, B>& p, const match_context& ctx) {
  return make_binary<T>(substitute(p.a, ctx), substitute(p.b, ctx));
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
  A a;
};

template <typename T, typename A>
bool match(const pattern_unary<T, A>& p, const expr& x, match_context& ctx) {
  if (const T* t = x.as<T>()) {
    return match(p.a, t->a, ctx);
  } else {
    return false;
  }
}

template <typename T, typename A>
bool match(const pattern_unary<T, A>& p, const pattern_unary<T, pattern_expr>& x, match_context& ctx) {
  return match(p.a, x.a.e, ctx);
}

template <typename T, typename A>
std::ostream& operator<<(std::ostream& os, const pattern_unary<T, A>& p) {
  switch (T::static_type) {
  case logical_not::static_type: return os << '!' << p.a;
  default: std::abort();
  }
}

template <typename T>
expr make_unary(expr a) {
  return T::make(std::move(a));
}
// clang-format off
template <typename T> index_t make_unary(index_t a);
template <> inline index_t make_unary<logical_not>(index_t a) { return a == 0 ? 1 : 0; }
// clang-format on

template <typename T, typename A>
auto substitute(const pattern_unary<T, A>& p, const match_context& ctx) {
  return make_unary<T>(substitute(p.a, ctx));
}

template <typename C, typename T, typename F>
class pattern_select {
public:
  static constexpr expr_node_type type = expr_node_type::select;
  static constexpr bool is_boolean = pattern_info<T>::is_boolean && pattern_info<F>::is_boolean;
  static constexpr bool is_canonical =
      pattern_info<C>::is_canonical && pattern_info<T>::is_canonical && pattern_info<F>::is_canonical;
  C c;
  T t;
  F f;
};

template <typename C, typename T, typename F>
bool match(const pattern_select<C, T, F>& p, const expr& x, match_context& ctx) {
  if (const class select* s = x.as<class select>()) {
    return match(p.c, s->condition, ctx) && match(p.t, s->true_value, ctx) && match(p.f, s->false_value, ctx);
  } else {
    return false;
  }
}

template <typename C, typename T, typename F>
bool match(const pattern_select<C, T, F>& p, const pattern_select<pattern_expr, pattern_expr, pattern_expr>& x,
    match_context& ctx) {
  return match(p.c, x.c.e, ctx) && match(p.t, x.t.e, ctx) && match(p.f, x.f.e, ctx);
}

template <typename C, typename T, typename F>
expr substitute(const pattern_select<C, T, F>& p, const match_context& ctx) {
  return select::make(substitute(p.c, ctx), substitute(p.t, ctx), substitute(p.f, ctx));
}

template <typename C, typename T, typename F>
std::ostream& operator<<(std::ostream& os, const pattern_select<C, T, F>& p) {
  return os << "select(" << p.c << ", " << p.t << ", " << p.f << ")";
}

template <typename... Args>
class pattern_call {
public:
  static constexpr expr_node_type type = expr_node_type::call;
  static constexpr bool is_boolean = false;
  static constexpr bool is_canonical = (... && pattern_info<Args>::is_canonical);
  slinky::intrinsic fn;
  std::tuple<Args...> args;
};

template <typename T, std::size_t... Is>
bool match_tuple(const T& t, const std::vector<expr>& x, match_context& ctx, std::index_sequence<Is...>) {
  return (... && match(std::get<Is>(t), x[Is], ctx));
}

template <typename T, std::size_t... Is>
std::vector<expr> substitute_tuple(const T& t, const match_context& ctx, std::index_sequence<Is...>) {
  return {substitute(std::get<Is>(t), ctx)...};
}

template <typename... Args>
bool match(const pattern_call<Args...>& p, const expr& x, match_context& ctx) {
  if (const call* c = x.as<call>()) {
    if (c->intrinsic == p.fn) {
      constexpr std::size_t ArgsSize = sizeof...(Args);
      assert(c->args.size() == ArgsSize);
      return match_tuple(p.args, c->args, ctx, std::make_index_sequence<ArgsSize>());
    }
  }
  return false;
}

template <typename... Args>
expr substitute(const pattern_call<Args...>& p, const match_context& ctx) {
  constexpr std::size_t ArgsSize = sizeof...(Args);
  return call::make(p.fn, substitute_tuple(p.args, ctx, std::make_index_sequence<ArgsSize>()));
}

inline std::ostream& operator<<(std::ostream& os, const pattern_call<>& p) { return os << p.fn << "()"; }
template <typename A>
std::ostream& operator<<(std::ostream& os, const pattern_call<A>& p) {
  return os << p.fn << "(" << std::get<0>(p.args) << ")";
}
template <typename A, typename B>
std::ostream& operator<<(std::ostream& os, const pattern_call<A, B>& p) {
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
bool substitute(const replacement_predicate<T, Fn>& r, const match_context& ctx) {
  return r.fn(substitute(r.a, ctx));
}

template <typename T, typename Fn>
replacement_predicate<T, Fn> make_predicate(T t, Fn fn) {
  return {t, fn};
}

template <typename T, typename Fn>
std::ostream& operator<<(std::ostream& os, const replacement_predicate<T, Fn>&) {
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
index_t substitute(const replacement_eval<T>& r, const match_context& ctx) {
  return substitute(r.a, ctx);
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const replacement_eval<T>& r) {
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
expr substitute(const replacement_boolean<T>& r, const match_context& ctx) {
  expr result = substitute(r.a, ctx);
  if (!is_boolean(result)) result = result != 0;
  return result;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const replacement_boolean<T>& r) {
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
template <typename Args, typename... Ts>
struct enable_pattern_ops<pattern_call<Args>, Ts...> { using type = std::true_type; };
template <typename T, typename... Ts>
struct enable_pattern_ops<replacement_eval<T>, Ts...> { using type = std::true_type; };
template <typename T, typename Fn, typename... Ts>
struct enable_pattern_ops<replacement_predicate<T, Fn>, Ts...> { using type = std::true_type; };

template <typename A, bool = typename enable_pattern_ops<A>::type()>
auto operator!(const A& a) { return pattern_unary<logical_not, A>{a}; }
template <typename A, bool = typename enable_pattern_ops<A>::type()>
auto operator-(const A& a) { return pattern_binary<sub, index_t, A>{0, a}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
auto operator+(const A& a, const B& b) { return pattern_binary<add, A, B>{a, b}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
auto operator-(const A& a, const B& b) { return pattern_binary<sub, A, B>{a, b}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
auto operator*(const A& a, const B& b) { return pattern_binary<mul, A, B>{a, b}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
auto operator/(const A& a, const B& b) { return pattern_binary<div, A, B>{a, b}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
auto operator%(const A& a, const B& b) { return pattern_binary<mod, A, B>{a, b}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
auto operator==(const A& a, const B& b) { return pattern_binary<equal, A, B>{a, b}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
auto operator!=(const A& a, const B& b) { return pattern_binary<not_equal, A, B>{a, b}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
auto operator<(const A& a, const B& b) { return pattern_binary<less, A, B>{a, b}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
auto operator<=(const A& a, const B& b) { return pattern_binary<less_equal, A, B>{a, b}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
auto operator>(const A& a, const B& b) { return pattern_binary<less, B, A>{b, a}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
auto operator>=(const A& a, const B& b) { return pattern_binary<less_equal, B, A>{b, a}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
auto operator&&(const A& a, const B& b) { return pattern_binary<logical_and, A, B>{a, b}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
auto operator||(const A& a, const B& b) { return pattern_binary<logical_or, A, B>{a, b}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
auto min(const A& a, const B& b) { return pattern_binary<class min, A, B>{a, b}; }
template <typename A, typename B, bool = typename enable_pattern_ops<A, B>::type()>
auto max(const A& a, const B& b) { return pattern_binary<class max, A, B>{a, b}; }
template <typename C, typename T, typename F, bool = typename enable_pattern_ops<C, T, F>::type()>
auto select(const C& c, const T& t, const F& f) { return pattern_select<C, T, F>{c, t, f}; }
template <typename T, bool = typename enable_pattern_ops<T>::type()>
auto abs(const T& x) { return pattern_call<T>{intrinsic::abs, {x}}; }
template <typename T, bool = typename enable_pattern_ops<T>::type()>
auto boolean(const T& x) { return replacement_boolean<T>{x}; }
inline auto positive_infinity() { return pattern_call<>{intrinsic::positive_infinity, {}}; }
inline auto negative_infinity() { return pattern_call<>{intrinsic::negative_infinity, {}}; }
inline auto indeterminate() { return pattern_call<>{intrinsic::indeterminate, {}}; }

template <typename T>
auto is_finite(const T& x) { return make_predicate(x, slinky::is_finite); }
template <typename T>
auto is_constant(const T& x) { return make_predicate(x, slinky::as_constant); }
template <typename T>
auto is_zero(const T& x) { return make_predicate(x, slinky::is_zero); }
template <typename T>
auto is_boolean(const T& x) { return make_predicate(x, slinky::is_boolean); }
// clang-format on

template <int N1, int N2>
using buffer_dim_meta = pattern_call<pattern_wildcard<N1>, pattern_wildcard<N2>>;

template <int N1, int N2>
inline auto buffer_min(const pattern_wildcard<N1>& buf, const pattern_wildcard<N2>& dim) {
  return buffer_dim_meta<N1, N2>{intrinsic::buffer_min, {buf, dim}};
}
template <int N1, int N2>
inline auto buffer_max(const pattern_wildcard<N1>& buf, const pattern_wildcard<N2>& dim) {
  return buffer_dim_meta<N1, N2>{intrinsic::buffer_max, {buf, dim}};
}

template <typename T>
auto eval(const T& x) {
  return replacement_eval<T>{x};
}

template <typename Pattern, typename T>
bool match_any_variant(const Pattern& p, const T& x, match_context& ctx) {
  static_assert(pattern_info<Pattern>::is_canonical);

  // We'll find out how many variant bits we have when we try to match.
  // This can grow if we fail early due to a commutative variant that doesn't match near the root
  // of the expression, so we track the max we've seen.
  int max_variant_bits = 0;
  for (int variant = 0; variant < (1 << max_variant_bits); ++variant) {
    memset(&ctx, 0, sizeof(ctx));
    ctx.variant = variant;
    if (match(p, x, ctx)) {
      return true;
    }
    max_variant_bits = std::max(max_variant_bits, ctx.variant_bits);
  }
  return false;
}

template <typename Pattern, typename T, typename Predicate>
bool match(match_context& ctx, const Pattern& p, const T& x, const Predicate& pr) {
  return match_any_variant(p, x, ctx) && substitute(pr, ctx);
}

template <typename Pattern, typename T>
bool match(match_context& ctx, const Pattern& p, const T& x) {
  return match_any_variant(p, x, ctx);
}

template <typename T>
class base_rewriter {
  T x;

public:
  expr result;

  base_rewriter(T x) : x(std::move(x)) {}
  base_rewriter(const base_rewriter&) = delete;

  template <typename Pattern, typename Replacement>
  bool operator()(const Pattern& p, const Replacement& r) {
    static_assert(pattern_info<Pattern>::is_canonical);
    static_assert(pattern_info<Replacement>::is_canonical);
    static_assert(!pattern_info<Pattern>::is_boolean || pattern_info<Replacement>::is_boolean);

    match_context ctx;
    if (!match_any_variant(p, x, ctx)) return false;

    result = substitute(r, ctx);
    return true;
  }

  template <typename Pattern, typename Replacement, typename Predicate>
  bool operator()(const Pattern& p, const Replacement& r, const Predicate& pr) {
    static_assert(pattern_info<Pattern>::is_canonical);
    static_assert(pattern_info<Replacement>::is_canonical);
    static_assert(!pattern_info<Pattern>::is_boolean || pattern_info<Replacement>::is_boolean);

    match_context ctx;
    if (!match_any_variant(p, x, ctx)) return false;

    if (!substitute(pr, ctx)) return false;

    result = substitute(r, ctx);
    return true;
  }
};

class rewriter : public base_rewriter<const expr&> {
public:
  rewriter(const expr& x) : base_rewriter(x) {}
  using base_rewriter::operator();
};

template <typename T>
base_rewriter<T> make_rewriter(T x) {
  return base_rewriter<T>(std::move(x));
}

}  // namespace rewrite
}  // namespace slinky

#endif  // SLINKY_BUILDER_REWRITE_H