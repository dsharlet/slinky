#ifndef SLINKY_BUILDER_REWRITE_H
#define SLINKY_BUILDER_REWRITE_H

#include "builder/substitute.h"
#include "runtime/evaluate.h"
#include "runtime/expr.h"

// This pattern matching engine is heavily inspired by https://github.com/halide/Halide/blob/main/src/IRMatch.h.

namespace slinky {
namespace rewrite {

constexpr int symbol_count = 4;
constexpr int constant_count = 3;

struct match_context {
  const base_expr_node* vars[symbol_count];
  const index_t* constants[constant_count];
  int variant;
  int variant_bits;

  void clear() {
    // Memset initializing these makes debug builds significantly faster.
    memset(vars, 0, sizeof(vars));
    memset(constants, 0, sizeof(constants));
  }
};

SLINKY_ALWAYS_INLINE inline bool match(index_t p, const expr& x, match_context& ctx) { return is_constant(x, p); }
SLINKY_ALWAYS_INLINE inline bool match(const expr& p, const expr& x, match_context& ctx) { 
  // We can use same_as here because expressions used in patterns should be canonical constants.
  return p.same_as(x); 
}
SLINKY_ALWAYS_INLINE inline expr substitute(index_t p, const match_context& ctx) { return p; }
SLINKY_ALWAYS_INLINE inline expr substitute(const expr& p, const match_context& ctx) { return p; }

SLINKY_ALWAYS_INLINE inline node_type pattern_type(index_t) { return node_type::constant; }
SLINKY_ALWAYS_INLINE inline node_type pattern_type(const expr& e) { return e.type(); }

class pattern_wildcard {
public:
  using is_pattern = std::true_type;
  int idx;
};

SLINKY_ALWAYS_INLINE inline node_type pattern_type(const pattern_wildcard&) { return node_type::none; }

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

inline expr substitute(const pattern_wildcard& p, const match_context& ctx) {
  assert(ctx.vars[p.idx]);
  return ctx.vars[p.idx]; 
}

class pattern_constant {
public:
  using is_pattern = std::true_type;
  int idx;
};

SLINKY_ALWAYS_INLINE inline node_type pattern_type(const pattern_constant&) { return node_type::constant; }

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

expr substitute(const pattern_constant& p, const match_context& ctx) {
  assert(ctx.constants[p.idx]);
  return *ctx.constants[p.idx];
}

template <typename T, typename A, typename B>
class pattern_binary {
public:
  using is_pattern = std::true_type;
  A a;
  B b;

  pattern_binary(A a, B b) : a(a), b(b) {
    assert(!T::commutative || !should_commute(pattern_type(this->a), pattern_type(this->b)));
  }
};

template <typename T, typename A, typename B>
SLINKY_ALWAYS_INLINE inline node_type pattern_type(const pattern_binary<T, A, B>&) {
  return T::static_type;
}

template <typename T, typename A, typename B>
bool match(const pattern_binary<T, A, B>& p, const expr& x, match_context& ctx) {
  int this_bit = -1;
  if (T::commutative) {
    node_type ta = pattern_type(p.a);
    node_type tb = pattern_type(p.b);
    if (ta == node_type::none || tb == node_type::none || ta == tb) {
      // This is a commutative operation and we can't canonicalize the ordering.
      // Remember which bit in the variant index is ours, and increment the bit for the next commutative node.
      this_bit = ctx.variant_bits++;
    }
  }

  if (const T* t = x.as<T>()) {
    if (this_bit >= 0 && (ctx.variant & (1 << this_bit)) != 0) {
      // We should commute in this variant.
      return match(p.a, t->b, ctx) && match(p.b, t->a, ctx);
    }
    return match(p.a, t->a, ctx) && match(p.b, t->b, ctx);
  } else {
    return false;
  }
}

template <typename T, typename A, typename B>
expr substitute(const pattern_binary<T, A, B>& p, const match_context& ctx) {
  return T::make(substitute(p.a, ctx), substitute(p.b, ctx));
}

template <typename T, typename A>
class pattern_unary {
public:
  using is_pattern = std::true_type;
  A a;
};

template <typename T, typename A>
SLINKY_ALWAYS_INLINE inline node_type pattern_type(const pattern_unary<T, A>&) {
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
expr substitute(const pattern_unary<T, A>& p, const match_context& ctx) {
  return T::make(substitute(p.a, ctx));
}

template <typename C, typename T, typename F>
class pattern_select {
public:
  using is_pattern = std::true_type;
  C c;
  T t;
  F f;
};

template <typename C, typename T, typename F>
SLINKY_ALWAYS_INLINE inline node_type pattern_type(const pattern_select<C, T, F>&) {
  return node_type::select;
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
expr substitute(const pattern_select<C, T, F>& p, const match_context& ctx) {
  return select::make(substitute(p.c, ctx), substitute(p.t, ctx), substitute(p.f, ctx));
}

template <typename... Args>
class pattern_call {
public:
  using is_pattern = std::true_type;
  slinky::intrinsic fn;
  std::tuple<Args...> args;
};

template <typename... Args>
SLINKY_ALWAYS_INLINE inline node_type pattern_type(const pattern_call<Args...>&) {
  return node_type::call;
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
SLINKY_ALWAYS_INLINE inline node_type pattern_type(const replacement_predicate<T, Fn>&) {
  return node_type::none;
}

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
SLINKY_ALWAYS_INLINE inline node_type pattern_type(const replacement_eval<T>&) {
  return node_type::call;
}

template <typename T>
index_t substitute(const replacement_eval<T>& r, const match_context& ctx) {
  return evaluate(substitute(r.a, ctx));
}

template <typename A>
auto operator!(const A& a) {
  return pattern_unary<logical_not, A>{a};
}
template <typename A>
auto operator-(const A& a) {
  return pattern_binary<sub, index_t, A>{0, a};
}
template <typename A, typename B>
auto operator+(const A& a, const B& b) {
  return pattern_binary<add, A, B>{a, b};
}
template <typename A, typename B>
auto operator-(const A& a, const B& b) {
  return pattern_binary<sub, A, B>{a, b};
}
template <typename A, typename B>
auto operator*(const A& a, const B& b) {
  return pattern_binary<mul, A, B>{a, b};
}
template <typename A, typename B>
auto operator/(const A& a, const B& b) {
  return pattern_binary<div, A, B>{a, b};
}
template <typename A, typename B>
auto operator%(const A& a, const B& b) {
  return pattern_binary<mod, A, B>{a, b};
}
template <typename A, typename B, typename = typename A::is_pattern>
auto operator==(const A& a, const B& b) {
  return pattern_binary<equal, A, B>{a, b};
}
template <typename A, typename B, typename = typename A::is_pattern>
auto operator!=(const A& a, const B& b) {
  return pattern_binary<not_equal, A, B>{a, b};
}
template <typename A, typename B>
auto operator<(const A& a, const B& b) {
  return pattern_binary<less, A, B>{a, b};
}
template <typename A, typename B>
auto operator<=(const A& a, const B& b) {
  return pattern_binary<less_equal, A, B>{a, b};
}
template <typename A, typename B>
auto operator>(const A& a, const B& b) {
  return pattern_binary<less, B, A>{b, a};
}
template <typename A, typename B>
auto operator>=(const A& a, const B& b) {
  return pattern_binary<less_equal, B, A>{b, a};
}
template <typename A, typename B, typename = typename A::is_pattern>
auto operator&&(const A& a, const B& b) {
  return pattern_binary<logical_and, A, B>{a, b};
}
template <typename A, typename B, typename = typename A::is_pattern>
auto operator||(const A& a, const B& b) {
  return pattern_binary<logical_or, A, B>{a, b};
}
template <typename A, typename B>
auto min(const A& a, const B& b) {
  return pattern_binary<class min, A, B>{a, b};
}
template <typename A, typename B>
auto max(const A& a, const B& b) {
  return pattern_binary<class max, A, B>{a, b};
}
template <typename C, typename T, typename F>
auto select(const C& c, const T& t, const F& f) {
  return pattern_select<C, T, F>{c, t, f};
}
template <typename T>
auto abs(const T& x) {
  return pattern_call<T>{intrinsic::abs, {x}};
}
template <typename T>
auto is_finite(const T& x) {
  return make_predicate(x, slinky::is_finite);
}
template <typename T>
auto is_constant(const T& x) {
  return make_predicate(x, slinky::as_constant);
}

using buffer_dim_meta = pattern_call<pattern_wildcard, pattern_wildcard>;

inline auto buffer_min(const pattern_wildcard& buf, const pattern_wildcard& dim) {
  return buffer_dim_meta{intrinsic::buffer_min, {buf, dim}};
}
inline auto buffer_max(const pattern_wildcard& buf, const pattern_wildcard& dim) {
  return buffer_dim_meta{intrinsic::buffer_max, {buf, dim}};
}
inline auto buffer_extent(const pattern_wildcard& buf, const pattern_wildcard& dim) {
  return buffer_dim_meta{intrinsic::buffer_extent, {buf, dim}};
}

template <typename T>
auto eval(const T& x) {
  return replacement_eval<T>{x};
}

class rewriter {
  const expr& x;

  template <typename Pattern>
  bool variant_match(const Pattern& p, match_context& ctx) {
    for (int variant = 0;; ++variant) {
      ctx.variant = variant;
      ctx.variant_bits = 0;
      ctx.clear();
      if (match(p, x, ctx)) {
        return true;
      }
      // variant_bits *should* be constant across all variants. We're done when
      // there are no more bits in the variant index to flip.
      if (variant >= (1 << ctx.variant_bits)) {
        break;
      }
    }
    return false;
  }

public:
  expr result;

  rewriter(const expr& x) : x(x) {}

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

static pattern_wildcard x{0};
static pattern_wildcard y{1};
static pattern_wildcard z{2};
static pattern_wildcard w{3};

static pattern_constant c0{0};
static pattern_constant c1{1};
static pattern_constant c2{2};

}  // namespace rewrite
}  // namespace slinky

#endif  // SLINKY_BUILDER_REWRITE_H