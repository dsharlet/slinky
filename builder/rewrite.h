#ifndef SLINKY_BUILDER_REWRITE_H
#define SLINKY_BUILDER_REWRITE_H

#include "builder/substitute.h"
#include "runtime/evaluate.h"
#include "runtime/expr.h"

#include "runtime/print.h"

namespace slinky {
namespace rewrite {

constexpr int max_symbol = 4;

struct match_context {
  std::array<expr, max_symbol> vars;
  std::array<std::optional<index_t>, max_symbol> constants;
  int variant;
  int variant_bit;

  void clear() {
    for (int i = 0; i < max_symbol; ++i) {
      vars[i] = expr();
      constants[i] = {};
    }
  }
};

inline bool match(index_t p, const expr& x, match_context& m) { return is_constant(x, p); }
inline bool match(const expr& p, const expr& x, match_context& m) { return p.same_as(x); }
inline expr substitute(int p, const match_context& m) { return p; }
inline expr substitute(const expr& p, const match_context& m) { return p; }

inline node_type static_type(index_t) { return node_type::constant; }
inline node_type static_type(const expr& e) { return e.type(); }

class pattern_variable {
public:
  using is_pattern = std::true_type;
  int sym;
};

inline node_type static_type(const pattern_variable&) { return node_type::variable; }

inline bool match(const pattern_variable& p, const expr& x, match_context& m) {
  if (m.vars[p.sym].defined()) {
    return slinky::match(x, m.vars[p.sym]);
  } else {
    m.vars[p.sym] = x;
    return true;
  }
}

inline expr substitute(const pattern_variable& p, const match_context& m) { return m.vars[p.sym]; }

class pattern_constant {
public:
  using is_pattern = std::true_type;
  int sym;
};

inline node_type static_type(const pattern_constant&) { return node_type::constant; }

inline bool match(const pattern_constant& p, const expr& x, match_context& m) {
  if (const constant* c = x.as<constant>()) {
    if (m.constants[p.sym]) {
      return *m.constants[p.sym] == c->value;
    } else {
      m.constants[p.sym] = c->value;
      return true;
    }
  }
  return false;
}

expr substitute(const pattern_constant& p, const match_context& m) {
  assert(m.constants[p.sym]);
  return *m.constants[p.sym];
}

template <typename T, typename A, typename B>
class pattern_binary {
public:
  using is_pattern = std::true_type;
  A a;
  B b;

  pattern_binary(A a, B b) : a(std::move(a)), b(std::move(b)) {
    if (typename T::commutative()) {
      assert(static_type(this->a) <= static_type(this->b));
    }
  }
};

template <typename T, typename A, typename B>
inline node_type static_type(const pattern_binary<T, A, B>&) {
  return T::static_type;
}

template <typename T, typename A, typename B>
bool match(const pattern_binary<T, A, B>& p, const expr& x, match_context& m) {
  if (const T* t = x.as<T>()) {
    if (typename T::commutative()) {
      int this_bit = m.variant_bit++;
      if ((m.variant & (1 << this_bit)) != 0) {
        // We should commute in this variant.
        return match(p.a, t->b, m) && match(p.b, t->a, m);
      }
    }
    return match(p.a, t->a, m) && match(p.b, t->b, m);
  } else {
    return false;
  }
}

template <typename T, typename A, typename B>
expr substitute(const pattern_binary<T, A, B>& p, const match_context& m) {
  return T::make(substitute(p.a, m), substitute(p.b, m));
}

template <typename T, typename A>
class pattern_unary {
public:
  using is_pattern = std::true_type;
  A a;
};

template <typename T, typename A>
inline node_type static_type(const pattern_unary<T, A>&) {
  return T::static_type;
}

template <typename T, typename A>
bool match(const pattern_unary<T, A>& p, const expr& x, match_context& m) {
  if (const T* t = x.as<T>()) {
    return match(p.a, t->a, m);
  } else {
    return false;
  }
}

template <typename T, typename A>
expr substitute(const pattern_unary<T, A>& p, const match_context& m) {
  return T::make(substitute(p.a, m));
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
inline node_type static_type(const pattern_select<C, T, F>&) {
  return node_type::select;
}

template <typename C, typename T, typename F>
bool match(const pattern_select<C, T, F>& p, const expr& x, match_context& m) {
  if (const class select* s = x.as<class select>()) {
    return match(p.c, s->condition, m) && match(p.t, s->true_value, m) && match(p.f, s->false_value, m);
  } else {
    return false;
  }
}

template <typename C, typename T, typename F>
expr substitute(const pattern_select<C, T, F>& p, const match_context& m) {
  return select::make(substitute(p.c, m), substitute(p.t, m), substitute(p.f, m));
}

template <typename... Args>
class pattern_call {
public:
  using is_pattern = std::true_type;
  slinky::intrinsic fn;
  std::tuple<Args...> args;
};

template <typename... Args>
inline node_type static_type(const pattern_call<Args...>&) {
  return node_type::call;
}

template <typename T, std::size_t... Is>
bool match_tuple(const T& t, const std::vector<expr>& x, match_context& m, std::index_sequence<Is...>) {
  return (... && match(std::get<Is>(t), x[Is], m));
}

template <typename T, std::size_t... Is>
std::vector<expr> substitute_tuple(const T& t, const match_context& m, std::index_sequence<Is...>) {
  return {substitute(std::get<Is>(t), m)...};
}

template <typename... Args>
bool match(const pattern_call<Args...>& p, const expr& x, match_context& m) {
  if (const call* c = x.as<call>()) {
    if (c->intrinsic == p.fn) {
      assert(c->args.size() == sizeof...(Args));
      return match_tuple(p.args, c->args, m, std::make_index_sequence<sizeof...(Args)>());
    }
  }
  return false;
}

template <typename... Args>
expr substitute(const pattern_call<Args...>& p, const match_context& m) {
  return call::make(p.fn, substitute_tuple(p.args, m, std::make_index_sequence<sizeof...(Args)>()));
}

template <typename T>
class replacement_is_finite {
public:
  T a;
};

template <typename T>
inline node_type static_type(const replacement_is_finite<T>&) {
  return node_type::call;
}

template <typename T>
bool substitute(const replacement_is_finite<T>& r, const match_context& m) {
  return is_finite(substitute(r.a, m));
}

template <typename T>
class replacement_eval {
public:
  T a;
};

template <typename T>
inline node_type static_type(const replacement_eval<T>&) {
  return node_type::call;
}

template <typename T>
index_t substitute(const replacement_eval<T>& r, const match_context& m) {
  return evaluate(substitute(r.a, m));
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
  return replacement_is_finite<T>{x};
}

using buffer_dim_meta = pattern_call<pattern_variable, pattern_variable>;

inline auto buffer_min(const pattern_variable& buf, const pattern_variable& dim) {
  return buffer_dim_meta{intrinsic::buffer_min, {buf, dim}};
}
inline auto buffer_max(const pattern_variable& buf, const pattern_variable& dim) {
  return buffer_dim_meta{intrinsic::buffer_max, {buf, dim}};
}
inline auto buffer_extent(const pattern_variable& buf, const pattern_variable& dim) {
  return buffer_dim_meta{intrinsic::buffer_extent, {buf, dim}};
}

template <typename T>
auto eval(const T& x) {
  return replacement_eval<T>{x};
}

class rewriter {
  const expr& x;

  template <typename Pattern>
  bool variant_match(const Pattern& p, const expr& x, match_context& m) {
    for (int variant = 0; ; ++variant) {
      m.variant = variant;
      m.variant_bit = 0;
      if (match(p, x, m)) {
        return true;
      }
      if (variant >= (1 << m.variant_bit)) {
        break;
      }
      m.clear();
    }
    return false;
  }

public:
  expr result;

  rewriter(const expr& x) : x(x) {}

  template <typename Pattern, typename Replacement>
  bool rewrite(const Pattern& p, const Replacement& r) {
    match_context m;
    if (!variant_match(p, x, m)) return false;

    result = substitute(r, m);
    return true;
  }

  template <typename Pattern, typename Replacement, typename Predicate>
  bool rewrite(const Pattern& p, const Replacement& r, const Predicate& pr) {
    match_context m;
    if (!variant_match(p, x, m)) return false;

    if (!substitute(eval(pr), m)) return false;

    result = substitute(r, m);
    return true;
  }
};

static pattern_variable x{0};
static pattern_variable y{1};
static pattern_variable z{2};
static pattern_variable w{3};

static pattern_constant c0{0};
static pattern_constant c1{1};
static pattern_constant c2{2};

}  // namespace rewrite
}  // namespace slinky

#endif  // SLINKY_BUILDER_REWRITE_H