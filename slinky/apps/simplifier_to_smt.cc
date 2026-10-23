// This tool is a second interpreter of the rewrite rule DSL in simplify_rules.h (the first being rule_tester.h).
// Instead of fuzzing a rule, it writes one SMT-LIB2 query per (rule, replacement). Each query is `unsat` iff the
// replacement is correct.
//
// All values are SMT Ints, with booleans as 0/1. Wildcards and constants are limited to the int64 range, and
// subexpressions of the pattern are assumed not to overflow. Subterms that rewrite.h constant folds (those for which
// substitute returns index_t) get the same overflow checks that rewrite.h does.
//
// Rules that mention positive_infinity(), negative_infinity() or indeterminate() (and, with --extended-all, every
// rule) use extended semantics instead, in which wildcards range over Z + {+inf, -inf}. Plain Ints are still used (not
// datatypes, which yices does not support): a value is a pair (tag, v), where tag 0 is a finite value v, tag 1 is
// +inf, tag -1 is -inf and tag 2 is indeterminate, and v is normalized to 0 whenever tag != 0. Each non-trivial node
// becomes a pair of nullary define-funs (nK_t, nK_v) built from the x* helpers in the extended prelude, which follow
// simplify() in simplify_exprs.cc: indeterminate propagates through everything, inf + -inf is indeterminate,
// inf * 0, inf / inf and inf / 0 are indeterminate, finite / inf is 0, mod with an infinity is indeterminate,
// min/max/comparisons use the total order -inf < Z < +inf, and infinities are truthy. Wildcards are never
// indeterminate (simplify() returns early on indeterminate operands) and constants are always finite. The in64
// assumptions apply only to finite values. A rewrite is correct if the pattern is indeterminate (anything goes), or if
// it has the same tag and value as the replacement. Wildcard predicates use the same may/must approximations (e.g.
// is_non_negative(x) may hold for tag = 1 or (tag = 0 and v >= 0)).

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "slinky/base/util.h"
#include "slinky/builder/rewrite.h"
#include "slinky/builder/simplify_rules.h"
#include "slinky/runtime/print.h"

namespace slinky {

namespace {

constexpr const char* int64_min = "(- 9223372036854775808)";

// True if rewrite.h evaluates this node entirely on index_t (i.e. it is constant folded).
template <typename P>
constexpr bool is_folded = std::is_same_v<decltype(rewrite::substitute(std::declval<const P&>(),
                                              std::declval<const rewrite::match_context&>(), std::declval<bool&>())),
    index_t>;

std::string lit(index_t v) {
  if (v < 0) return "(- " + std::to_string(std::uint64_t(0) - std::uint64_t(v)) + ")";
  return std::to_string(v);
}

std::string str(const char* op, const std::string& a) { return "(" + std::string(op) + " " + a + ")"; }
std::string str(const char* op, const std::string& a, const std::string& b) {
  return "(" + std::string(op) + " " + a + " " + b + ")";
}
std::string str(const char* op, const std::string& a, const std::string& b, const std::string& c) {
  return "(" + std::string(op) + " " + a + " " + b + " " + c + ")";
}
std::string str(const char* op, const std::string& a, const std::string& b, const std::string& c, const std::string& d) {
  return "(" + std::string(op) + " " + a + " " + b + " " + c + " " + d + ")";
}
std::string truthy(const std::string& a) { return str("distinct", a, "0"); }

std::string conj(const std::vector<std::string>& terms) {
  if (terms.empty()) return "true";
  if (terms.size() == 1) return terms[0];
  std::string result = "(and";
  for (const std::string& i : terms) result += " " + i;
  return result + ")";
}

std::string disj(const std::vector<std::string>& terms) {
  if (terms.empty()) return "false";
  if (terms.size() == 1) return terms[0];
  std::string result = "(or";
  for (const std::string& i : terms) result += " " + i;
  return result + ")";
}

// Sound approximations of a predicate P: P => may, and must => P.
struct pred_result {
  std::string may, must;
};

// An SMT term for a DSL node. In extended mode, t is the tag and v is the value. Otherwise t is empty.
struct term {
  std::string t, v;
};

// Translates DSL nodes to SMT terms. This mirrors the substitute overloads in rule_tester.h (for patterns) and
// rewrite.h (for replacements and predicates).
class translator {
public:
  static constexpr char wildcard_names[] = "xyzwuvt";

  // True for extended semantics (infinities).
  bool ext;
  explicit translator(bool ext) : ext(ext) {}

  std::array<bool, rewrite::symbol_count> vars = {};
  std::array<bool, rewrite::constant_count> consts = {};
  // True if the rule can't be translated.
  bool unsupported = false;
  // True if the rule mentions an infinity or indeterminate.
  bool saw_infinity = false;
  // Translating the pattern assumes all of its non-constant subterms are in range.
  bool in_pattern = false;
  std::vector<std::string> assumes;

  // Per-piece state, cleared by take_*.
  std::vector<std::string> ovf;
  // Boolean conditions that must hold for the piece to not overflow (or otherwise give up).
  std::vector<std::string> conds;
  std::vector<std::string> hyps;
  // Definitions (extended mode only) of the nodes translated since the last take_defs.
  std::vector<std::string> defs;

  std::vector<std::string> take_defs() { return std::move(std::exchange(defs, {})); }
  std::vector<std::string> take_ovf() { return std::move(std::exchange(ovf, {})); }
  std::vector<std::string> take_conds() { return std::move(std::exchange(conds, {})); }
  std::vector<std::string> take_hyps() { return std::move(std::exchange(hyps, {})); }

private:
  int wildcard_uses = 0;
  int staircase_bounds = 0;
  int node_count = 0;

  // The tag of a term that is known to be finite.
  std::string ft() const { return ext ? "0" : ""; }
  bool fin(const term& a) const { return a.t == ft(); }
  term fin_term(std::string v) const { return {ft(), std::move(v)}; }

  // Names a node in extended mode.
  term def(const std::string& t, const std::string& v) {
    const std::string n = "n" + std::to_string(node_count++);
    defs.push_back("(define-fun " + n + "_t () Int " + t + ")\n(define-fun " + n + "_v () Int " + v + ")\n");
    return {n + "_t", n + "_v"};
  }
  static std::string ext4(const char* op, const term& a, const term& b) {
    return "(" + std::string(op) + " " + a.t + " " + a.v + " " + b.t + " " + b.v + ")";
  }
  term ext_unary(const char* name, const term& a) {
    return def(str((std::string(name) + "_t").c_str(), a.t, a.v), str((std::string(name) + "_v").c_str(), a.t, a.v));
  }

  term note(int wildcards_before, term s) {
    if (in_pattern && wildcard_uses != wildcards_before) {
      if (fin(s)) {
        assumes.push_back(str("in64", s.v));
      } else {
        assumes.push_back(str("=>", str("=", s.t, "0"), str("in64", s.v)));
      }
    }
    return s;
  }

  // Binary arithmetic and logic. This is used by both pattern_binary and staircase.
  term binop(expr_node_type t, const term& a, const term& b) {
    if (fin(a) && fin(b)) {
      std::string r;
      // clang-format off
      if (t == expr_node_type::add) r = str("+", a.v, b.v);
      else if (t == expr_node_type::sub) r = str("-", a.v, b.v);
      else if (t == expr_node_type::mul) r = str("*", a.v, b.v);
      else if (t == expr_node_type::div) r = str("ediv", a.v, b.v);
      else if (t == expr_node_type::mod) r = str("emod", a.v, b.v);
      else if (t == expr_node_type::min) r = str("ite", str("<=", a.v, b.v), a.v, b.v);
      else if (t == expr_node_type::max) r = str("ite", str(">=", a.v, b.v), a.v, b.v);
      else if (t == expr_node_type::less) r = str("b2i", str("<", a.v, b.v));
      else if (t == expr_node_type::less_equal) r = str("b2i", str("<=", a.v, b.v));
      else if (t == expr_node_type::equal) r = str("b2i", str("=", a.v, b.v));
      else if (t == expr_node_type::not_equal) r = str("b2i", str("distinct", a.v, b.v));
      else if (t == expr_node_type::logical_and) r = str("b2i", str("and", truthy(a.v), truthy(b.v)));
      else if (t == expr_node_type::logical_or) r = str("b2i", str("or", truthy(a.v), truthy(b.v)));
      else assert(false);
      // clang-format on
      return fin_term(r);
    }
    const char* tn = nullptr;
    const char* vn = nullptr;
    // clang-format off
    if (t == expr_node_type::add) { tn = "xadd_t"; vn = "xadd_v"; }
    else if (t == expr_node_type::sub) { tn = "xsub_t"; vn = "xsub_v"; }
    else if (t == expr_node_type::mul) { tn = "xmul_t"; vn = "xmul_v"; }
    else if (t == expr_node_type::div) { tn = "xdiv_t"; vn = "xdiv_v"; }
    else if (t == expr_node_type::mod) { tn = "xmod_t"; vn = "xmod_v"; }
    else if (t == expr_node_type::min) { tn = "xmin_t"; vn = "xmin_v"; }
    else if (t == expr_node_type::max) { tn = "xmax_t"; vn = "xmax_v"; }
    else if (t == expr_node_type::less) { tn = "xcmp_t"; vn = "xlt_v"; }
    else if (t == expr_node_type::less_equal) { tn = "xcmp_t"; vn = "xle_v"; }
    else if (t == expr_node_type::equal) { tn = "xcmp_t"; vn = "xeq_v"; }
    else if (t == expr_node_type::not_equal) { tn = "xcmp_t"; vn = "xne_v"; }
    else if (t == expr_node_type::logical_and) { tn = "xcmp_t"; vn = "xand_v"; }
    else if (t == expr_node_type::logical_or) { tn = "xcmp_t"; vn = "xor_v"; }
    else assert(false);
    // clang-format on
    return def(ext4(tn, a, b), ext4(vn, a, b));
  }

public:
  template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
  term to(T v) {
    return fin_term(lit(static_cast<index_t>(v)));
  }

  template <int N>
  term to(const rewrite::pattern_wildcard<N>&) {
    vars[N] = true;
    ++wildcard_uses;
    const std::string n(1, wildcard_names[N]);
    if (ext) return {n + "_t", n + "_v"};
    return {"", n};
  }

  template <int N>
  term to(const rewrite::pattern_constant<N>&) {
    consts[N] = true;
    return fin_term("c" + std::to_string(N));
  }

  // The free variable already covers the default value.
  template <typename A, index_t Default>
  term to(const rewrite::pattern_optional<A, Default>& p) {
    return to(p.a);
  }

  template <typename T, typename A, typename B>
  term to(const rewrite::pattern_binary<T, A, B>& p) {
    using P = rewrite::pattern_binary<T, A, B>;
    constexpr expr_node_type t = T::static_type;
    static_assert(t != expr_node_type::none, "unknown binary operator");
    const int w = wildcard_uses;
    term a = to(p.a);
    term b = to(p.b);
    term r = binop(t, a, b);
    if constexpr (is_folded<P>) {
      // Folded mod never overflows: INT64_MIN % -1 is 0.
      if constexpr (t == expr_node_type::add || t == expr_node_type::sub || t == expr_node_type::mul ||
                    t == expr_node_type::div) {
        ovf.push_back(r.v);
      }
    }
    return note(w, r);
  }

  template <typename T, typename A>
  term to(const rewrite::pattern_unary<T, A>& p) {
    static_assert(T::static_type == expr_node_type::logical_not);
    const int w = wildcard_uses;
    term a = to(p.a);
    if (fin(a)) return note(w, fin_term(str("b2i", str("=", a.v, "0"))));
    return note(w, ext_unary("xnot", a));
  }

  template <typename C, typename T, typename F>
  term to(const rewrite::pattern_select<C, T, F>& p) {
    const int w = wildcard_uses;
    term c = to(p.c);
    term t = to(p.t);
    term f = to(p.f);
    if (fin(c) && fin(t) && fin(f)) return note(w, fin_term(str("ite", truthy(c.v), t.v, f.v)));
    return note(w, def(str("xsel_t", c.t, c.v, t.t, f.t), str("xsel_v", c.t, c.v, t.v, f.v)));
  }

  term call_term(intrinsic fn, const std::vector<term>& args, int w) {
    switch (fn) {
    case intrinsic::abs:
      if (fin(args[0])) {
        const std::string& a = args[0].v;
        return note(w, fin_term(str("ite", str("<", a, "0"), str("-", a), a)));
      }
      return note(w, ext_unary("xabs", args[0]));
    case intrinsic::and_then: return note(w, binop(expr_node_type::logical_and, args[0], args[1]));
    case intrinsic::or_else: return note(w, binop(expr_node_type::logical_or, args[0], args[1]));
    case intrinsic::positive_infinity:
      saw_infinity = true;
      if (ext) return {"1", "0"};
      break;
    case intrinsic::negative_infinity:
      saw_infinity = true;
      if (ext) return {"(- 1)", "0"};
      break;
    case intrinsic::indeterminate:
      saw_infinity = true;
      if (ext) return {"2", "0"};
      break;
    default: break;
    }
    // Anything else can't be modeled.
    unsupported = true;
    return fin_term("0");
  }
  term to(const rewrite::pattern_call<>& p) { return call_term(p.fn, {}, wildcard_uses); }
  template <typename A>
  term to(const rewrite::pattern_call<A>& p) {
    const int w = wildcard_uses;
    return call_term(p.fn, {to(std::get<0>(p.args))}, w);
  }
  template <typename A, typename B>
  term to(const rewrite::pattern_call<A, B>& p) {
    const int w = wildcard_uses;
    term a = to(std::get<0>(p.args));
    term b = to(std::get<1>(p.args));
    return call_term(p.fn, {a, b}, w);
  }

  // ((x + a)/b)*c
  template <typename X, typename A, typename B, typename C>
  term to(const rewrite::pattern_staircase<X, A, B, C>& p) {
    const int w = wildcard_uses;
    term x = to(p.x);
    term a = to(p.a);
    term b = to(p.b);
    term c = to(p.c);
    return note(w, binop(expr_node_type::mul, binop(expr_node_type::div, binop(expr_node_type::add, x, a), b), c));
  }

  template <typename T>
  term to(const rewrite::replacement_eval<T>& r) {
    return to(r.a);
  }

  template <typename T>
  term to(const rewrite::replacement_boolean<T>& r) {
    term a = to(r.a);
    if (fin(a)) return fin_term(str("b2i", truthy(a.v)));
    return ext_unary("xbool", a);
  }

  // staircase_sum_bounds returns either unbounded or the exact range of g(k) = ((k + a1)/b1)*c1 + ((k + a2)/b2)*c2.
  // When it is unbounded, rewrite.h reports overflow and the rule doesn't fire. `pre` is the part of the C++'s
  // conditions for giving a bound that doesn't depend on int64 overflow: |b| <= 1024 (which limits the period search)
  // and the both-zero or periodic condition. Whenever the C++ returns a bound, `pre` holds and the bound is correct for
  // g over the unbounded integers. So the premise here is a superset of the cases where the rule fires, and an unsat
  // result still proves those cases. When the C++ returns unbounded because of int64 overflow, the rule just doesn't
  // fire, which only removes cases. Given pre, a fresh constant m is a bound on g. We only need the bound at the
  // pattern's own wildcard x, so rather than quantifying over k we instantiate it at k = x. That is a weaker
  // hypothesis, so it remains sound, and it keeps the queries quantifier-free. Every rule using this has a single
  // wildcard, which is the x of its staircase patterns.
  template <typename A1, typename B1, typename C1, typename A2, typename B2, typename C2>
  term to(const rewrite::replacement_staircase_sum_bound<A1, B1, C1, A2, B2, C2>& r) {
    // These are all constants, so they are finite.
    std::string a1 = to(r.a1).v;
    std::string b1 = to(r.b1).v;
    std::string c1 = to(r.c1).v;
    std::string a2 = to(r.a2).v;
    std::string b2 = to(r.b2).v;
    std::string c2 = to(r.c2).v;
    std::string x;
    for (std::size_t i = 0; i < vars.size(); ++i) {
      if (!vars[i]) continue;
      if (!x.empty()) unsupported = true;  // More than one wildcard: we don't know which one is the staircase's x.
      x = std::string(1, wildcard_names[i]);
    }
    // In extended mode, the bound only applies when x is finite.
    const std::string x_finite = ext ? str("=", x + "_t", "0") : "true";
    if (ext && !x.empty()) x += "_v";
    if (x.empty()) {
      unsupported = true;
      x = "0";
    }
    const std::string m = "m" + std::to_string(staircase_bounds++);
    auto term = [](const std::string& k, const std::string& a, const std::string& b, const std::string& c) {
      return str("*", str("ediv", str("+", k, a), b), c);
    };
    auto abs_le = [](const std::string& a, index_t limit) {
      return str("<=", str("ite", str("<", a, "0"), str("-", a), a), lit(limit));
    };
    // This must match the non-overflow conditions in staircase_sum_bounds in arithmetic.cc.
    std::vector<std::string> pre;
    for (const std::string& b : {b1, b2}) pre.push_back(abs_le(b, 1024));
    const std::string both_zero = str("and", str("=", b1, "0"), str("=", b2, "0"));
    const std::string periodic = conj({str("=", str("*", c1, b2), str("-", str("*", c2, b1))),
        str("distinct", b1, "0"), str("distinct", b2, "0")});
    pre.push_back(str("or", both_zero, periodic));
    const std::string pre_all = conj(pre);
    conds.push_back(pre_all);
    const std::string g = str("+", term(x, a1, b1, c1), term(x, a2, b2, c2));
    const bool is_min = r.bound_sign < 0;
    hyps.push_back("(declare-const " + m + " Int)");
    const std::string hyp = ext ? conj({x_finite, pre_all}) : pre_all;
    hyps.push_back(str("assert", str("=>", hyp, str(is_min ? ">=" : "<=", g, m))));
    return fin_term(m);
  }

  // Predicates. Wildcard predicates have no exact encoding, so we use may/must approximations.
  template <typename T, typename Fn>
  pred_result pred(const rewrite::replacement_predicate<T, Fn>& p) {
    term t = to(p.a);
    const std::string& a = t.v;
    const char* name = p.name;
    std::string may;
    if (!fin(t)) {
      // Extended semantics.
      const std::string finite = str("=", t.t, "0");
      if (!std::strcmp(name, "is_zero")) {
        may = str("and", finite, str("=", a, "0"));
      } else if (!std::strcmp(name, "is_non_negative")) {
        may = str("or", str("=", t.t, "1"), str("and", finite, str(">=", a, "0")));
      } else if (!std::strcmp(name, "is_non_positive")) {
        may = str("or", str("=", t.t, "(- 1)"), str("and", finite, str("<=", a, "0")));
      } else if (!std::strcmp(name, "is_boolean")) {
        may = str("and", finite, str("or", str("=", a, "0"), str("=", a, "1")));
      } else if (!std::strcmp(name, "is_constant") || !std::strcmp(name, "is_finite")) {
        may = finite;
      } else {
        unsupported = true;
        may = "true";
      }
    } else if (!std::strcmp(name, "is_zero")) {
      may = str("=", a, "0");
    } else if (!std::strcmp(name, "is_non_negative")) {
      may = str(">=", a, "0");
    } else if (!std::strcmp(name, "is_non_positive")) {
      may = str("<=", a, "0");
    } else if (!std::strcmp(name, "is_boolean")) {
      may = str("or", str("=", a, "0"), str("=", a, "1"));
    } else if (!std::strcmp(name, "is_constant") || !std::strcmp(name, "is_finite")) {
      may = "true";
    } else {
      unsupported = true;
      may = "true";
    }
    if constexpr (is_folded<T>) {
      // The argument is a constant, so the predicate is exact.
      return {may, may};
    } else {
      return {may, "false"};
    }
  }

  template <typename A, typename B>
  pred_result pred(const rewrite::pattern_binary<logical_and, A, B>& p) {
    pred_result a = pred(p.a);
    pred_result b = pred(p.b);
    return {str("and", a.may, b.may), str("and", a.must, b.must)};
  }

  template <typename A, typename B>
  pred_result pred(const rewrite::pattern_binary<logical_or, A, B>& p) {
    pred_result a = pred(p.a);
    pred_result b = pred(p.b);
    return {str("or", a.may, b.may), str("or", a.must, b.must)};
  }

  template <typename A>
  pred_result pred(const rewrite::pattern_unary<logical_not, A>& p) {
    pred_result a = pred(p.a);
    return {str("not", a.must), str("not", a.may)};
  }

  // A constant expression.
  template <typename P>
  pred_result pred(const P& p) {
    static_assert(is_folded<P>, "predicate must be a constant expression");
    std::string t = truthy(to(p).v);
    return {t, t};
  }
};

std::string sanitize(std::string s) {
  for (char& c : s) {
    if (c == '\t' || c == '\n' || c == '\r') c = ' ';
  }
  return s;
}

std::string comment_lines(const std::string& s) {
  std::string result;
  std::istringstream ss(s);
  std::string line;
  while (std::getline(ss, line)) result += "; " + line + "\n";
  return result;
}

struct repl_data {
  term r;
  pred_result pr;
  std::vector<std::string> ovf_pr, ovf_r;
  std::vector<std::string> conds_pr, conds_r;
  std::vector<std::string> hyps;
  std::vector<std::string> defs;
  std::string text;
};

// Everything translated from a rule.
struct translated_rule {
  explicit translated_rule(bool ext) : tr(ext) {}
  translator tr;
  term pattern;
  std::vector<std::string> pattern_defs;
  std::vector<repl_data> repls;
};

const char* const prelude =
    "(set-logic ALL)\n"
    "(define-fun b2i ((b Bool)) Int (ite b 1 0))\n"
    "(define-fun ediv ((a Int) (b Int)) Int (ite (= b 0) 0 (div a b)))\n"
    "(define-fun emod ((a Int) (b Int)) Int (ite (= b 0) 0 (mod a b)))\n"
    "(define-fun in64 ((a Int)) Bool (and (<= (- 9223372036854775808) a) (<= a 9223372036854775807)))\n";

// Helpers for extended semantics. See the header comment. Values are (tag, v) pairs, and every binary helper takes
// (at av bt bv), even when it ignores some of them.
const char* const ext_prelude =
    "(define-fun xtruthy ((t Int) (v Int)) Bool (or (distinct t 0) (distinct v 0)))\n"
    "(define-fun xsgn ((t Int) (v Int)) Int (ite (distinct t 0) t (ite (> v 0) 1 (ite (< v 0) (- 1) 0))))\n"
    "(define-fun xind ((at Int) (bt Int)) Bool (or (= at 2) (= bt 2)))\n"
    "(define-fun xfin ((at Int) (bt Int)) Bool (and (= at 0) (= bt 0)))\n"
    "(define-fun xlt ((at Int) (av Int) (bt Int) (bv Int)) Bool (or (< at bt) (and (= at bt) (< av bv))))\n"
    "(define-fun xneg_t ((t Int)) Int (ite (= t 2) 2 (- t)))\n"
    "(define-fun xadd_t ((at Int) (av Int) (bt Int) (bv Int)) Int\n"
    "  (ite (xind at bt) 2 (ite (= at 0) bt (ite (= bt 0) at (ite (= at bt) at 2)))))\n"
    "(define-fun xadd_v ((at Int) (av Int) (bt Int) (bv Int)) Int (ite (xfin at bt) (+ av bv) 0))\n"
    "(define-fun xsub_t ((at Int) (av Int) (bt Int) (bv Int)) Int (xadd_t at av (xneg_t bt) (- bv)))\n"
    "(define-fun xsub_v ((at Int) (av Int) (bt Int) (bv Int)) Int (xadd_v at av (xneg_t bt) (- bv)))\n"
    "(define-fun xsinf ((at Int) (av Int) (bt Int) (bv Int)) Int\n"
    "  (let ((s (* (xsgn at av) (xsgn bt bv)))) (ite (= s 0) 2 s)))\n"
    "(define-fun xmul_t ((at Int) (av Int) (bt Int) (bv Int)) Int\n"
    "  (ite (xind at bt) 2 (ite (xfin at bt) 0 (xsinf at av bt bv))))\n"
    "(define-fun xmul_v ((at Int) (av Int) (bt Int) (bv Int)) Int (ite (xfin at bt) (* av bv) 0))\n"
    "(define-fun xdiv_t ((at Int) (av Int) (bt Int) (bv Int)) Int\n"
    "  (ite (xind at bt) 2 (ite (= at 0) 0 (ite (= bt 0) (xsinf at av bt bv) 2))))\n"
    "(define-fun xdiv_v ((at Int) (av Int) (bt Int) (bv Int)) Int (ite (xfin at bt) (ediv av bv) 0))\n"
    "(define-fun xmod_t ((at Int) (av Int) (bt Int) (bv Int)) Int (ite (xfin at bt) 0 2))\n"
    "(define-fun xmod_v ((at Int) (av Int) (bt Int) (bv Int)) Int (ite (xfin at bt) (emod av bv) 0))\n"
    "(define-fun xmin_t ((at Int) (av Int) (bt Int) (bv Int)) Int\n"
    "  (ite (xind at bt) 2 (ite (xlt bt bv at av) bt at)))\n"
    "(define-fun xmin_v ((at Int) (av Int) (bt Int) (bv Int)) Int\n"
    "  (ite (xind at bt) 0 (ite (xlt bt bv at av) bv av)))\n"
    "(define-fun xmax_t ((at Int) (av Int) (bt Int) (bv Int)) Int\n"
    "  (ite (xind at bt) 2 (ite (xlt at av bt bv) bt at)))\n"
    "(define-fun xmax_v ((at Int) (av Int) (bt Int) (bv Int)) Int\n"
    "  (ite (xind at bt) 0 (ite (xlt at av bt bv) bv av)))\n"
    "(define-fun xcmp_t ((at Int) (av Int) (bt Int) (bv Int)) Int (ite (xind at bt) 2 0))\n"
    "(define-fun xlt_v ((at Int) (av Int) (bt Int) (bv Int)) Int (ite (xind at bt) 0 (b2i (xlt at av bt bv))))\n"
    "(define-fun xle_v ((at Int) (av Int) (bt Int) (bv Int)) Int (ite (xind at bt) 0 (b2i (not (xlt bt bv at av)))))\n"
    "(define-fun xeq_v ((at Int) (av Int) (bt Int) (bv Int)) Int\n"
    "  (ite (xind at bt) 0 (b2i (and (= at bt) (= av bv)))))\n"
    "(define-fun xne_v ((at Int) (av Int) (bt Int) (bv Int)) Int\n"
    "  (ite (xind at bt) 0 (b2i (not (and (= at bt) (= av bv))))))\n"
    "(define-fun xand_v ((at Int) (av Int) (bt Int) (bv Int)) Int\n"
    "  (ite (xind at bt) 0 (b2i (and (xtruthy at av) (xtruthy bt bv)))))\n"
    "(define-fun xor_v ((at Int) (av Int) (bt Int) (bv Int)) Int\n"
    "  (ite (xind at bt) 0 (b2i (or (xtruthy at av) (xtruthy bt bv)))))\n"
    "(define-fun xnot_t ((t Int) (v Int)) Int (ite (= t 2) 2 0))\n"
    "(define-fun xnot_v ((t Int) (v Int)) Int (ite (= t 2) 0 (b2i (not (xtruthy t v)))))\n"
    "(define-fun xbool_t ((t Int) (v Int)) Int (ite (= t 2) 2 0))\n"
    "(define-fun xbool_v ((t Int) (v Int)) Int (ite (= t 2) 0 (b2i (xtruthy t v))))\n"
    "(define-fun xabs_t ((t Int) (v Int)) Int (ite (= t 2) 2 (ite (= t 0) 0 1)))\n"
    "(define-fun xabs_v ((t Int) (v Int)) Int (ite (= t 0) (ite (< v 0) (- v) v) 0))\n"
    "(define-fun xsel_t ((ct Int) (cv Int) (pt Int) (qt Int)) Int (ite (= ct 2) 2 (ite (xtruthy ct cv) pt qt)))\n"
    "(define-fun xsel_v ((ct Int) (cv Int) (pv Int) (qv Int)) Int (ite (= ct 2) 0 (ite (xtruthy ct cv) pv qv)))\n";

class smt_rule_emitter {
  std::string dir_;
  std::string ruleset_;
  std::ofstream& manifest_;
  bool extended_all_;
  int index_ = 0;

  template <typename P>
  void add_repls(translator&, std::vector<repl_data>&, const P&) {}
  template <typename P, typename R>
  void add_repls(translator& tr, std::vector<repl_data>& repls, const P& p, const R& r) {
    add_repl(tr, repls, p, r, 1);
  }
  template <typename P, typename R, typename Pr, typename... Rest>
  void add_repls(
      translator& tr, std::vector<repl_data>& repls, const P& p, const R& r, const Pr& pr, const Rest&... rest) {
    add_repl(tr, repls, p, r, pr);
    add_repls(tr, repls, p, rest...);
  }

  template <typename P, typename R, typename Pr>
  void add_repl(translator& tr, std::vector<repl_data>& repls, const P& p, const R& r, const Pr& pr) {
    repl_data d;
    d.pr = tr.pred(pr);
    d.ovf_pr = tr.take_ovf();
    d.conds_pr = tr.take_conds();
    d.r = tr.to(r);
    d.ovf_r = tr.take_ovf();
    d.conds_r = tr.take_conds();
    d.hyps = tr.take_hyps();
    d.defs = tr.take_defs();
    std::stringstream ss;
    ss << p << " -> " << r << " if " << pr;
    d.text = sanitize(ss.str());
    repls.push_back(std::move(d));
  }

  static std::string rule_id(const std::string& ruleset, int index) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "_%03d", index);
    return ruleset + buf;
  }

  void write_row(const std::string& id, int index, const std::string& repl, const char* kind, const std::string& text) {
    manifest_ << id << '\t' << ruleset_ << '\t' << index << '\t' << repl << '\t' << kind << '\t' << text << '\n';
  }

  void write_file(const std::string& id, const std::string& contents) {
    std::ofstream f(dir_ + "/" + id + ".smt2");
    f << contents;
    if (!f) {
      std::cerr << "failed to write " << id << ".smt2\n";
      std::exit(1);
    }
  }

public:
  smt_rule_emitter(std::string dir, std::string ruleset, std::ofstream& manifest, bool extended_all)
      : dir_(std::move(dir)), ruleset_(std::move(ruleset)), manifest_(manifest), extended_all_(extended_all) {}

  template <typename P, typename R, typename... Rest>
  translated_rule translate(bool ext, const P& p, const R& r, const Rest&... rest) {
    translated_rule result(ext);
    translator& tr = result.tr;
    tr.in_pattern = true;
    result.pattern = tr.to(p);
    tr.in_pattern = false;
    tr.take_ovf();
    tr.take_conds();
    tr.take_hyps();
    result.pattern_defs = tr.take_defs();
    add_repls(tr, result.repls, p, r, rest...);
    return result;
  }

  template <typename P, typename R>
  bool operator()(const P& p, const R& r) {
    return operator()(p, r, 1);
  }

  template <typename P, typename R, typename Pr, typename... Rest>
  bool operator()(const P& p, const R& r, const Pr& pr, const Rest&... rest) {
    const int index = index_++;
    const std::string rule = rule_id(ruleset_, index);

    // Rules that mention an infinity use extended semantics. Everything else only does if it is requested.
    translated_rule rule_data = translate(extended_all_, p, r, pr, rest...);
    if (!extended_all_ && rule_data.tr.saw_infinity) rule_data = translate(true, p, r, pr, rest...);

    std::stringstream ss;
    ss << p;
    emit_rule(index, rule, rule_data, sanitize(ss.str()));
    return false;
  }

private:
  // Everything here is independent of the rule's types, so keep it out of the templated operator().
  SLINKY_NO_INLINE void emit_rule(
      int index, const std::string& rule, translated_rule& rule_data, const std::string& pattern_text) {
    translator& tr = rule_data.tr;
    const bool ext = tr.ext;
    std::vector<repl_data>& repls = rule_data.repls;

    if (tr.unsupported) {
      write_row(rule, index, "-", "skipped", pattern_text + " -> ...");
      return;
    }

    std::string decls;
    for (std::size_t i = 0; i < tr.vars.size(); ++i) {
      if (!tr.vars[i]) continue;
      std::string v(1, translator::wildcard_names[i]);
      if (ext) {
        decls += "(declare-const " + v + "_t Int)\n(declare-const " + v + "_v Int)\n";
        decls += "(assert (and (<= (- 1) " + v + "_t) (<= " + v + "_t 1) (=> (distinct " + v + "_t 0) (= " + v +
                 "_v 0))))\n";
        decls += "(assert (=> (= " + v + "_t 0) (in64 " + v + "_v)))\n";
        continue;
      }
      decls += "(declare-const " + v + " Int)\n(assert (in64 " + v + "))\n";
    }
    for (std::size_t i = 0; i < tr.consts.size(); ++i) {
      if (!tr.consts[i]) continue;
      std::string v = "c" + std::to_string(i);
      decls += "(declare-const " + v + " Int)\n(assert (in64 " + v + "))\n";
    }
    for (const std::string& i : rule_data.pattern_defs) decls += i;
    for (const std::string& i : tr.assumes) decls += "(assert " + i + ")\n";

    std::vector<std::string> fell_through;  // not (replacement j fired), for j < i.
    std::string hyps;
    for (std::size_t i = 0; i < repls.size(); ++i) {
      const repl_data& d = repls[i];
      const std::string id = rule + "_r" + std::to_string(i);
      for (const std::string& h : d.hyps) hyps += h + "\n";
      for (const std::string& def : d.defs) hyps += def;

      std::string header = comment_lines("ruleset " + ruleset_ + ", rule " + std::to_string(index) +
                                         ", replacement " + std::to_string(i)) +
                           comment_lines(d.text);

      std::vector<std::string> pre = fell_through;
      std::vector<std::string> noovf_pr, noovf_r;
      for (const std::string& o : d.ovf_pr) noovf_pr.push_back(str("in64", o));
      for (const std::string& o : d.ovf_r) noovf_r.push_back(str("in64", o));
      noovf_pr.insert(noovf_pr.end(), d.conds_pr.begin(), d.conds_pr.end());
      noovf_r.insert(noovf_r.end(), d.conds_r.begin(), d.conds_r.end());

      // Replacement i is used when its predicate is true and nothing overflows, otherwise we fall through.
      std::vector<std::string> all = pre;
      all.push_back(d.pr.may);
      all.insert(all.end(), noovf_pr.begin(), noovf_pr.end());
      all.insert(all.end(), noovf_r.begin(), noovf_r.end());

      std::string body = header;
      if (ext) body += comment_lines("extended semantics: values are (tag, v) pairs");
      body += prelude;
      if (ext) body += ext_prelude;
      body += decls + hyps;
      body += "(assert " + conj(all) + ")\n";
      if (ext) {
        // Correct if the pattern is indeterminate, or the tag and value match.
        body += "(assert " + str("not", str("or", str("=", rule_data.pattern.t, "2"),
                                             str("and", str("=", rule_data.pattern.t, d.r.t),
                                                 str("=", rule_data.pattern.v, d.r.v)))) +
                ")\n";
      } else {
        body += "(assert " + str("distinct", rule_data.pattern.v, d.r.v) + ")\n";
      }
      body += "(check-sat)\n(get-model)\n";
      write_file(id, body);
      write_row(id, index, std::to_string(i), "query", d.text);

      // This replacement is used (and later ones are not) when its predicate holds and nothing overflowed.
      std::vector<std::string> fire = {d.pr.must};
      fire.insert(fire.end(), noovf_pr.begin(), noovf_pr.end());
      fire.insert(fire.end(), noovf_r.begin(), noovf_r.end());
      fell_through.push_back(str("not", conj(fire)));
    }
  }
};

// Deliberately wrong rules. Every query generated from these must be sat.
template <typename Fn>
bool apply_selftest_rules(Fn&& apply) {
  return
      // Wrong when c0 != 1.
      apply((x / c0) * c0, x) ||
      apply(min(x, y), x) ||
      apply(x + c0 < y, x < y + c0) ||
      // Both replacements are wrong (the second only when c0 < 0, i.e. when we fell through).
      apply(min(x, x + may_be<0>(c0)), x + c0, c0 >= 1, x - c0) ||
      // Wildcard predicates.
      apply(x * c0, x, is_boolean(x)) ||
      // Infinities (extended semantics). Wrong for x < 0 (the result is -inf).
      apply(x * rewrite::positive_infinity(), rewrite::positive_infinity()) ||
      // Wrong for c0 < 0 (the result is -inf).
      apply(rewrite::positive_infinity() * c0, rewrite::positive_infinity()) ||
      // Wrong for finite x.
      apply(max(x, rewrite::positive_infinity()), x) ||
      // Wrong for x = 0 and for infinite x, where the pattern is indeterminate or an infinity.
      apply(x + y, rewrite::positive_infinity(), is_finite(x)) ||
      // Wrong when x is -inf (the pattern is -inf).
      apply(abs(x), x) ||
      // Wrong for every truthy non-negative x, including +inf (the result is -inf).
      apply(select(x, rewrite::negative_infinity(), rewrite::positive_infinity()), rewrite::positive_infinity(),
          is_non_negative(x)) ||
      false;
}

struct ruleset {
  const char* name;
  void (*emit)(smt_rule_emitter&);
};

#define RULESET(name) {#name, [](smt_rule_emitter& e) { apply_##name##_rules(e); }}

const ruleset rulesets[] = {
    RULESET(min),
    RULESET(max),
    RULESET(add),
    RULESET(sub),
    RULESET(mul),
    RULESET(div),
    RULESET(mod),
    RULESET(less),
    RULESET(equal),
    RULESET(logical_and),
    RULESET(logical_or),
    RULESET(logical_not),
    RULESET(select),
    RULESET(call),
};

const ruleset selftest = RULESET(selftest);

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> result;
  std::istringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) result.push_back(item);
  }
  return result;
}

}  // namespace

}  // namespace slinky

int main(int argc, char** argv) {
  using namespace slinky;

  std::string out_dir;
  std::vector<std::string> selected;
  bool self_test = false;
  bool extended_all = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-o" && i + 1 < argc) {
      out_dir = argv[++i];
    } else if (arg == "--ruleset" && i + 1 < argc) {
      for (std::string& s : split(argv[++i])) selected.push_back(std::move(s));
    } else if (arg == "--self-test") {
      self_test = true;
    } else if (arg == "--extended-all") {
      extended_all = true;
    } else {
      std::cerr << "usage: simplifier_to_smt -o <dir> [--ruleset name,name,...] [--self-test] [--extended-all]\n";
      return 1;
    }
  }
  if (out_dir.empty()) {
    std::cerr << "error: -o <dir> is required\n";
    return 1;
  }

  std::vector<const ruleset*> to_run;
  if (self_test) {
    to_run.push_back(&selftest);
  } else if (selected.empty()) {
    for (const ruleset& i : rulesets) to_run.push_back(&i);
  } else {
    for (const std::string& s : selected) {
      const ruleset* found = nullptr;
      for (const ruleset& i : rulesets) {
        if (s == i.name) found = &i;
      }
      if (!found) {
        std::cerr << "error: unknown ruleset " << s << "\n";
        return 1;
      }
      to_run.push_back(found);
    }
  }

  std::error_code ec;
  std::filesystem::create_directories(out_dir, ec);
  if (ec) {
    std::cerr << "error: could not create " << out_dir << ": " << ec.message() << "\n";
    return 1;
  }
  std::ofstream manifest(out_dir + "/manifest.tsv");
  manifest << "id\truleset\tindex\treplacement\tkind\trule_text\n";
  for (const ruleset* i : to_run) {
    smt_rule_emitter emitter(out_dir, i->name, manifest, extended_all);
    i->emit(emitter);
  }
  manifest.close();
  if (!manifest) {
    std::cerr << "error: failed to write manifest\n";
    return 1;
  }
  return 0;
}
