#ifndef SLINKY_BUILDER_REWRITE_H
#define SLINKY_BUILDER_REWRITE_H

#include "builder/substitute.h"
#include "runtime/evaluate.h"
#include "runtime/expr.h"

// This pattern matching engine is heavily inspired by https://github.com/halide/Halide/blob/main/src/IRMatch.h.

namespace slinky {
namespace rewrite {

// The maximum number of values pattern_wildcard::idx and pattern_constant::idx can have, starting from 0.
constexpr int symbol_count = 4;
constexpr int constant_count = 3;

struct match_context {
  const base_expr_node* vars[symbol_count];
  const index_t* constants[constant_count];
  int variant;
  int variant_bits;
};

SLINKY_ALWAYS_INLINE inline bool match(index_t p, const expr& x, match_context&) { return is_constant(x, p); }
SLINKY_ALWAYS_INLINE inline index_t substitute(index_t p, const match_context&) { return p; }
SLINKY_ALWAYS_INLINE inline expr_node_type pattern_type(index_t) { return expr_node_type::constant; }

class pattern_expr {
public:
  const expr& e;
};

SLINKY_ALWAYS_INLINE inline bool match(const pattern_expr& p, const expr& x, match_context& ctx) {
  // Assume that exprs in patterns are canonical constants.
  return p.e.same_as(x);
}
SLINKY_ALWAYS_INLINE inline const expr& substitute(const pattern_expr& p, const match_context& ctx) { return p.e; }
SLINKY_ALWAYS_INLINE inline expr_node_type pattern_type(const pattern_expr& p) { return p.e.type(); }

const pattern_expr& positive_infinity();
const pattern_expr& negative_infinity();
const pattern_expr& indeterminate();

class pattern_wildcard {
public:
  int idx;
};

SLINKY_ALWAYS_INLINE inline expr_node_type pattern_type(const pattern_wildcard&) { return expr_node_type::none; }

inline bool match(const pattern_wildcard& p, const expr& x, match_context& ctx) {
  if (ctx.vars[p.idx]) {
    // Try pointer comparison first to short circuit the full match.
    return x.get() == ctx.vars[p.idx] || slinky::compare(x.get(), ctx.vars[p.idx]) == 0;
  } else if (x.get()) {
    ctx.vars[p.idx] = x.get();
    return true;
  } else {
    return false;
  }
}

inline const base_expr_node* substitute(const pattern_wildcard& p, const match_context& ctx) {
  assert(ctx.vars[p.idx]);
  return ctx.vars[p.idx];
}

class pattern_constant {
public:
  int idx;
};

SLINKY_ALWAYS_INLINE inline expr_node_type pattern_type(const pattern_constant&) { return expr_node_type::constant; }

inline bool match(const pattern_constant& p, const expr& x, match_context& ctx) {
  if (const constant* c = x.as<constant>()) {
    if (ctx.constants[p.idx]) {
      return *ctx.constants[p.idx] == c->value;
    } else {
      ctx.constants[p.idx] = &c->value;
      return true;
    }
  }
  return false;
}

inline index_t substitute(const pattern_constant& p, const match_context& ctx) {
  assert(ctx.constants[p.idx]);
  return *ctx.constants[p.idx];
}

template <typename T, typename A, typename B>
class pattern_binary {
public:
  A a;
  B b;

  pattern_binary(A a, B b) : a(a), b(b) {
    assert(!T::commutative || !should_commute(pattern_type(this->a), pattern_type(this->b)));
  }
};

template <typename T, typename A, typename B>
SLINKY_ALWAYS_INLINE inline expr_node_type pattern_type(const pattern_binary<T, A, B>&) {
  return T::static_type;
}

template <typename T, typename A, typename B>
bool match_binary(const pattern_binary<T, A, B>& p, const expr& a, const expr& b, match_context& ctx) {
  int this_bit = -1;
  if (T::commutative) {
    expr_node_type ta = pattern_type(p.a);
    expr_node_type tb = pattern_type(p.b);
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

template <typename T, typename A>
class pattern_unary {
public:
  A a;
};

template <typename T, typename A>
SLINKY_ALWAYS_INLINE inline expr_node_type pattern_type(const pattern_unary<T, A>&) {
  return T::static_type;
}

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
  C c;
  T t;
  F f;
};

template <typename C, typename T, typename F>
SLINKY_ALWAYS_INLINE inline expr_node_type pattern_type(const pattern_select<C, T, F>&) {
  return expr_node_type::select;
}

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

template <typename... Args>
class pattern_call {
public:
  slinky::intrinsic fn;
  std::tuple<Args...> args;
};

template <typename... Args>
SLINKY_ALWAYS_INLINE inline expr_node_type pattern_type(const pattern_call<Args...>&) {
  return expr_node_type::call;
}

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
      assert(c->args.size() == sizeof...(Args));
      return match_tuple(p.args, c->args, ctx, std::make_index_sequence<sizeof...(Args)>());
    }
  }
  return false;
}

template <typename... Args>
expr substitute(const pattern_call<Args...>& p, const match_context& ctx) {
  return call::make(p.fn, substitute_tuple(p.args, ctx, std::make_index_sequence<sizeof...(Args)>()));
}

template <typename T, typename Fn>
class replacement_predicate {
public:
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

template <typename T>
class replacement_eval {
public:
  T a;
};

template <typename T>
SLINKY_ALWAYS_INLINE inline expr_node_type pattern_type(const replacement_eval<T>&) {
  return expr_node_type::call;
}

template <typename T>
index_t substitute(const replacement_eval<T>& r, const match_context& ctx) {
  return substitute(r.a, ctx);
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
template <typename... Ts>
struct enable_pattern_ops<pattern_wildcard, Ts...> { using type = std::true_type; };
template <typename... Ts>
struct enable_pattern_ops<pattern_constant, Ts...> { using type = std::true_type; };
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
template <typename T>
auto is_finite(const T& x) { return make_predicate(x, slinky::is_finite); }
template <typename T>
auto is_constant(const T& x) { return make_predicate(x, slinky::as_constant); }
// clang-format on

using buffer_dim_meta = pattern_call<pattern_wildcard, pattern_wildcard>;

inline auto buffer_min(const pattern_wildcard& buf, const pattern_wildcard& dim) {
  return buffer_dim_meta{intrinsic::buffer_min, {buf, dim}};
}
inline auto buffer_max(const pattern_wildcard& buf, const pattern_wildcard& dim) {
  return buffer_dim_meta{intrinsic::buffer_max, {buf, dim}};
}

template <typename T>
auto eval(const T& x) {
  return replacement_eval<T>{x};
}

template <typename T>
class base_rewriter {
  T x;

  template <typename Pattern>
  bool variant_match(const Pattern& p, match_context& ctx) {
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

public:
  expr result;

  base_rewriter(T x) : x(std::move(x)) {}

  template <typename Pattern, typename Replacement>
  bool rewrite(const Pattern& p, const Replacement& r) {
    match_context ctx;
    if (!variant_match(p, ctx)) return false;

    result = substitute(r, ctx);
    return true;
  }

  template <typename Pattern, typename Replacement, typename Predicate>
  bool rewrite(const Pattern& p, const Replacement& r, const Predicate& pr) {
    match_context ctx;
    if (!variant_match(p, ctx)) return false;

    if (!substitute(pr, ctx)) return false;

    result = substitute(r, ctx);
    return true;
  }
};

class rewriter : public base_rewriter<const expr&> {
public:
  rewriter(const expr& x) : base_rewriter(x) {}
  using base_rewriter::result;
  using base_rewriter::rewrite;
};

template <typename T>
base_rewriter<T> make_rewriter(T x) {
  return base_rewriter<T>(std::move(x));
}

}  // namespace rewrite
}  // namespace slinky

#endif  // SLINKY_BUILDER_REWRITE_H