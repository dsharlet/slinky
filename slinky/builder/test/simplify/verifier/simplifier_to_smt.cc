// This tool checks the rewrite rules in simplify_rules.h with SMT solvers. It writes
// one SMT-LIB2 query per replacement of each rule, and a query is `unsat` iff its
// replacement is correct.
//
// Integers are SMT Ints, and booleans are the Ints 0 and 1. Wildcards and constants are
// assumed to fit in int64, and so are the pattern's subexpressions, because the pattern
// is an expression that already exists. rewrite.h evaluates constant subexpressions of
// predicates and replacements (e.g. c0 + c1) in C++, and doesn't apply the rule if that
// overflows, so the query assumes that those fit in int64 too.
//
// Wildcards are finite. The only values in slinky that aren't are the literals
// positive_infinity(), negative_infinity() and indeterminate(), and simplify() (in
// simplify_exprs.cc) eliminates them before it applies the rules, so a wildcard never
// matches one. No rule mentions them either.
//
// Each operation in a rule that depends on a wildcard becomes a named term
// (define-fun nK () Int ...). An operation on constants only is written inline.
//
// A rewrite is correct if the pattern equals the replacement. Predicates on wildcards
// (e.g. is_non_negative(x)) can't be encoded exactly, so they are approximated from
// both sides (see pred_result).

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "slinky/base/util.h"
#include "slinky/builder/rewrite.h"
#include "slinky/builder/simplify_rules.h"
#include "slinky/runtime/print.h"

namespace slinky {

namespace {

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

std::string truthy(const std::string& a) { return str("distinct", a, "0"); }

std::string conj(const std::vector<std::string>& terms) {
  if (terms.empty()) return "true";
  if (terms.size() == 1) return terms[0];
  std::string result = "(and";
  for (const std::string& i : terms)
    result += " " + i;
  return result + ")";
}

// Sound approximations of a predicate P: P => may, and must => P.
struct pred_result {
  std::string may, must;
};

// An SMT term for a DSL node, which is an Int expression. If constant is true, the node
// doesn't depend on any wildcard (e.g. c0 + 1).
struct term {
  std::string s;
  bool constant = false;
};

// Dependencies of the terms translated since the last call to take().
struct side_info {
  // Conditions that must hold for the rule to fire, e.g. that constant folding doesn't overflow.
  std::vector<std::string> conds;
  // Declarations and assertions the terms depend on.
  std::vector<std::string> hyps;
  // Definitions of the nodes in the terms.
  std::vector<std::string> defs;
  // Lemma instances (see rewrite::hint_cong).
  std::vector<std::string> lemmas;
};

// Translates DSL nodes to SMT terms, mirroring the substitute overloads in
// rule_tester.h (for patterns) and rewrite.h (for replacements and predicates).
class translator {
public:
  // The names of the wildcards and constants the rule uses, indexed by N, or empty if it doesn't use them. These are
  // the names rules print, so a counterexample uses the same names as the rule text.
  std::array<std::string, rewrite::symbol_count> vars;
  std::array<std::string, rewrite::constant_count> consts;
  // The rule being translated, for error messages.
  std::string rule;
  // True while translating the pattern, whose non-constant subterms are assumed to be in range.
  bool in_pattern = false;
  std::vector<std::string> assumes;

  // Returns the side information collected since the last call, and starts collecting anew.
  side_info take() { return std::exchange(side, {}); }

  // Fail instead of skipping the rule, so that every rule is verified.
  [[noreturn]] void unsupported(const std::string& what) const {
    std::cerr << "error: " << rule << ": can't translate " << what << "\n";
    std::exit(1);
  }

private:
  int wildcard_uses = 0;
  int staircase_bounds = 0;
  int node_count = 0;
  // Structurally identical nodes share a name. Solvers do much better when lemmas are
  // stated over the same nodes as the rest of the query.
  std::map<std::string, term> nodes;
  side_info side;

  static term constant_term(std::string v) { return {std::move(v), true}; }

  // Names a node.
  term def(const std::string& value) {
    auto [i, inserted] = nodes.try_emplace(value);
    if (inserted) {
      const std::string n = "n" + std::to_string(node_count++);
      side.defs.push_back("(define-fun " + n + " () Int " + value + ")\n");
      i->second = {n, false};
    }
    return i->second;
  }

  // The term for an operation with the given value. Operations on constants are written inline, and the others are
  // named.
  term op_term(std::string value, bool constant) { return constant ? constant_term(std::move(value)) : def(value); }

  term note(int wildcards_before, term s) {
    if (in_pattern && wildcard_uses != wildcards_before) {
      // This term contains a wildcard, so it is part of the pattern, which is assumed to be in range.
      assert(!s.constant);
      assumes.push_back(str("in64", s.s));
    }
    return s;
  }

  // Binary arithmetic and logic, for pattern_binary, staircase, and_then and or_else.
  term binop(expr_node_type t, const term& a, const term& b) {
    return op_term(binop_value(t, a.s, b.s), a.constant && b.constant);
  }

  // The Int expression for a binary operation.
  static std::string binop_value(expr_node_type t, const std::string& a, const std::string& b) {
    switch (t) {
    case expr_node_type::add: return str("+", a, b);
    case expr_node_type::sub: return str("-", a, b);
    case expr_node_type::mul: return str("*", a, b);
    case expr_node_type::div: return str("ediv", a, b);
    case expr_node_type::mod: return str("emod", a, b);
    case expr_node_type::min: return str("ite", str("<=", a, b), a, b);
    case expr_node_type::max: return str("ite", str(">=", a, b), a, b);
    case expr_node_type::less: return str("b2i", str("<", a, b));
    case expr_node_type::less_equal: return str("b2i", str("<=", a, b));
    case expr_node_type::equal: return str("b2i", str("=", a, b));
    case expr_node_type::not_equal: return str("b2i", str("distinct", a, b));
    case expr_node_type::logical_and: return str("b2i", str("and", truthy(a), truthy(b)));
    case expr_node_type::logical_or: return str("b2i", str("or", truthy(a), truthy(b)));
    default: assert(false); return "0";
    }
  }

public:
  template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
  term to(T v) {
    return constant_term(lit(static_cast<index_t>(v)));
  }

  template <int N>
  term to(const rewrite::pattern_wildcard<N>&) {
    vars[N] = rewrite::pattern_wildcard<N>::name;
    ++wildcard_uses;
    return {vars[N], false};
  }

  template <int N>
  term to(const rewrite::pattern_constant<N>&) {
    consts[N] = rewrite::pattern_constant<N>::name;
    return constant_term(consts[N]);
  }

  // The default is one of the values of the free variable, so it needs no special case.
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
      // Like rewrite.h, the rule doesn't fire if this overflows. Folded mod never overflows: INT64_MIN % -1 is 0.
      if constexpr (t == expr_node_type::add || t == expr_node_type::sub || t == expr_node_type::mul ||
                    t == expr_node_type::div) {
        side.conds.push_back(str("in64", r.s));
      }
    }
    return note(w, r);
  }

  template <typename T, typename A>
  term to(const rewrite::pattern_unary<T, A>& p) {
    static_assert(T::static_type == expr_node_type::logical_not);
    const int w = wildcard_uses;
    term a = to(p.a);
    return note(w, op_term(str("b2i", str("=", a.s, "0")), a.constant));
  }

  template <typename C, typename T, typename F>
  term to(const rewrite::pattern_select<C, T, F>& p) {
    const int w = wildcard_uses;
    term c = to(p.c);
    term t = to(p.t);
    term f = to(p.f);
    return note(w, op_term(str("ite", truthy(c.s), t.s, f.s), c.constant && t.constant && f.constant));
  }

  term call_term(intrinsic fn, const std::vector<term>& args, int w) {
    switch (fn) {
    case intrinsic::abs: {
      const std::string& a = args[0].s;
      return note(w, op_term(str("ite", str("<", a, "0"), str("-", a), a), args[0].constant));
    }
    case intrinsic::and_then: return note(w, binop(expr_node_type::logical_and, args[0], args[1]));
    case intrinsic::or_else: return note(w, binop(expr_node_type::logical_or, args[0], args[1]));
    default: break;
    }
    std::stringstream ss;
    ss << "intrinsic " << fn;
    unsupported(ss.str());
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
    return op_term(str("b2i", truthy(a.s)), a.constant);
  }

  // staircase_sum_bounds (in arithmetic.cc) computes the exact range of
  // g(k) = ((k + a1)/b1)*c1 + ((k + a2)/b2)*c2, or reports that it is unbounded, in
  // which case the rule doesn't fire. Rather than encode that search, we declare a
  // fresh constant m, and assume that it bounds g whenever the C++ would return a
  // bound.
  //
  // The C++ only returns a bound when `pre` holds: |b1|, |b2| <= 1024 (which limits the
  // period search), and either both terms are constant or g is periodic. So `pre` is
  // also a condition for the rule to fire. The C++ can also give up because of int64
  // overflow, but that only removes cases in which the rule fires, so we can ignore it.
  // Whenever the C++ does return a bound, the bound is correct for unbounded integers.
  //
  // We only need the bound at the pattern's own wildcard x, so we assume it at k = x
  // rather than for all k. That is a weaker assumption, so it is still sound, and it
  // keeps the queries quantifier-free.
  template <typename A1, typename B1, typename C1, typename A2, typename B2, typename C2>
  term to(const rewrite::replacement_staircase_sum_bound<A1, B1, C1, A2, B2, C2>& r) {
    // These are all constants.
    std::string a1 = to(r.a1).s;
    std::string b1 = to(r.b1).s;
    std::string c1 = to(r.c1).s;
    std::string a2 = to(r.a2).s;
    std::string b2 = to(r.b2).s;
    std::string c2 = to(r.c2).s;
    term x;
    int wildcards = 0;
    for (const std::string& v : vars) {
      if (v.empty()) continue;
      x = {v, false};
      ++wildcards;
    }
    // We need exactly one wildcard: with more we don't know which one is the staircase's x.
    if (wildcards != 1) unsupported("a staircase sum bound in a rule with " + std::to_string(wildcards) + " wildcards");
    const std::string m = "m" + std::to_string(staircase_bounds++);
    auto term = [](const std::string& k, const std::string& a, const std::string& b, const std::string& c) {
      return str("*", str("ediv", str("+", k, a), b), c);
    };
    auto abs_le = [](const std::string& a, index_t limit) {
      return str("<=", str("ite", str("<", a, "0"), str("-", a), a), lit(limit));
    };
    // This must match the conditions under which staircase_sum_bounds in arithmetic.cc returns a bound.
    std::vector<std::string> pre;
    for (const std::string& b : {b1, b2})
      pre.push_back(abs_le(b, 1024));
    const std::string both_zero = str("and", str("=", b1, "0"), str("=", b2, "0"));
    const std::string periodic = conj(
        {str("=", str("*", c1, b2), str("-", str("*", c2, b1))), str("distinct", b1, "0"), str("distinct", b2, "0")});
    pre.push_back(str("or", both_zero, periodic));
    const std::string pre_all = conj(pre);
    side.conds.push_back(pre_all);
    const std::string g = str("+", term(x.s, a1, b1, c1), term(x.s, a2, b2, c2));
    const bool is_min = r.bound_sign < 0;
    side.hyps.push_back("(declare-const " + m + " Int)");
    side.hyps.push_back(str("assert", str("=>", pre_all, str(is_min ? ">=" : "<=", g, m))));
    return constant_term(m);
  }

  // Adds an instance of a lemma (see the hints in rewrite.h).
  void add_lemma(const char* name, std::initializer_list<term> args) {
    std::string call = std::string("(") + name;
    for (const term& a : args)
      call += " " + a.s;
    side.lemmas.push_back(call + ")");
  }

  template <typename U, typename V, typename D, typename K>
  void hint(const rewrite::hint_cong<U, V, D, K>& h) {
    add_lemma("cong", {to(h.u), to(h.v), to(h.d), to(h.k)});
  }

  template <typename A, typename D>
  void hint(const rewrite::hint_euclid<A, D>& h) {
    add_lemma("euclid", {to(h.a), to(h.d)});
  }

  template <typename S, typename T, typename D>
  void hint(const rewrite::hint_negdiv<S, T, D>& h) {
    add_lemma("negdiv", {to(h.s), to(h.t), to(h.d)});
  }

  // Adds the lemmas for a case's hints. Any conditions that come with their terms are dropped, so they can't become
  // premises.
  template <typename... Hints>
  void hints(const std::tuple<const Hints&...>& hints) {
    const std::vector<std::string> conds = side.conds;
    std::apply([&](const auto&... h) { (hint(h), ...); }, hints);
    side.conds = conds;
  }

  // Predicates. Wildcard predicates have no exact encoding, so we use may/must approximations.
  template <typename T, typename Fn>
  pred_result pred(const rewrite::replacement_predicate<T, Fn>& p) {
    const std::string a = to(p.a).s;
    const char* name = p.name;
    std::string may;
    if (!std::strcmp(name, "is_zero")) {
      may = str("=", a, "0");
    } else if (!std::strcmp(name, "is_non_negative")) {
      may = str(">=", a, "0");
    } else if (!std::strcmp(name, "is_non_positive")) {
      may = str("<=", a, "0");
    } else if (!std::strcmp(name, "is_boolean")) {
      may = str("or", str("=", a, "0"), str("=", a, "1"));
    } else if (!std::strcmp(name, "is_constant")) {
      // Any value may be a constant.
      may = "true";
    } else if (!std::strcmp(name, "no_overflow")) {
      // Translating the argument already added the condition that it doesn't overflow.
      may = "true";
    } else {
      unsupported(std::string("predicate ") + name);
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

  // Any other predicate must be a constant expression.
  template <typename P>
  pred_result pred(const P& p) {
    static_assert(is_folded<P>, "predicate must be a constant expression");
    std::string t = truthy(to(p).s);
    return {t, t};
  }
};

std::string sanitize(std::string s) {
  for (char& c : s) {
    if (c == '\t' || c == '\n' || c == '\r') c = ' ';
  }
  return s;
}

struct repl_data {
  term r;
  pred_result pr;
  // The side information of both the predicate and the replacement.
  side_info side;
  std::string text;
};

// Everything translated from a rule.
struct translated_rule {
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

// Each of these is true for all integers. See the hints in rewrite.h, and emit_lemma_proofs, which checks them.
const char* const lemma_prelude =
    "(define-fun cong ((u Int) (v Int) (d Int) (k Int)) Bool\n"
    "  (=> (and (distinct d 0) (= u (+ v (* d k))))\n"
    "      (and (= (emod u d) (emod v d)) (= (ediv u d) (+ (ediv v d) k)))))\n"
    "(define-fun euclid ((a Int) (d Int)) Bool (=> (distinct d 0) (= a (+ (* d (ediv a d)) (emod a d)))))\n"
    "(define-fun negdiv ((s Int) (t Int) (d Int)) Bool\n"
    "  (=> (and (> d 0) (= s (- (- d 1) t))) (= (ediv s d) (- (ediv t d)))))\n";

// Writes DIR/ID.smt2, exiting on failure.
void write_file(const std::string& dir, const std::string& id, const std::string& contents) {
  std::ofstream f(dir + "/" + id + ".smt2");
  f << contents;
  if (!f) {
    std::cerr << "failed to write " << id << ".smt2\n";
    std::exit(1);
  }
}

void write_row(std::ostream& manifest, const std::string& id, const std::string& ruleset, const std::string& text) {
  manifest << id << '\t' << ruleset << '\t' << text << '\n';
}

class smt_rule_emitter {
  std::string dir_;
  std::string ruleset_;
  std::ofstream& manifest_;
  int index_ = 0;

  template <typename P>
  void add_repls(translator&, std::vector<repl_data>&, const P&) {}
  template <typename P, typename R>
  void add_repls(translator& tr, std::vector<repl_data>& repls, const P& p, const R& r) {
    add_repl(tr, repls, p, rewrite::when(1, r));
  }
  template <typename P, typename Pr, typename R, typename... Hints, typename... Rest>
  void add_repls(translator& tr, std::vector<repl_data>& repls, const P& p,
      const rewrite::replacement_when<Pr, R, Hints...>& w, const Rest&... rest) {
    add_repl(tr, repls, p, w);
    add_repls(tr, repls, p, rest...);
  }

  template <typename P, typename Pr, typename R, typename... Hints>
  void add_repl(
      translator& tr, std::vector<repl_data>& repls, const P& p, const rewrite::replacement_when<Pr, R, Hints...>& w) {
    repl_data d;
    d.pr = tr.pred(w.pr);
    if constexpr (sizeof...(Hints) > 0) tr.hints(w.hints);
    d.r = tr.to(w.r);
    d.side = tr.take();
    std::stringstream ss;
    ss << p << " -> " << w.r << " if " << w.pr;
    d.text = sanitize(ss.str());
    repls.push_back(std::move(d));
  }

  static std::string rule_id(const std::string& ruleset, int index) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "_%03d", index);
    return ruleset + buf;
  }

  template <typename P, typename... Cases>
  translated_rule translate(const std::string& rule, const P& p, const Cases&... cases) {
    translated_rule result;
    translator& tr = result.tr;
    tr.rule = rule;
    tr.in_pattern = true;
    result.pattern = tr.to(p);
    tr.in_pattern = false;
    // A pattern has no conditions, hypotheses or lemmas (its range assumptions are in tr.assumes), only definitions.
    result.pattern_defs = tr.take().defs;
    add_repls(tr, result.repls, p, cases...);
    return result;
  }

public:
  smt_rule_emitter(std::string dir, std::string ruleset, std::ofstream& manifest)
      : dir_(std::move(dir)), ruleset_(std::move(ruleset)), manifest_(manifest) {}

  template <typename P, typename... Cases>
  bool operator()(const P& p, const Cases&... cases) {
    const std::string rule = rule_id(ruleset_, index_++);

    std::stringstream ss;
    ss << p;
    translated_rule rule_data = translate(rule + " (" + sanitize(ss.str()) + ")", p, cases...);

    emit_rule(rule, rule_data);
    return false;
  }

private:
  // Everything here is independent of the rule's types, so keep it out of the templated operator().
  SLINKY_NO_INLINE void emit_rule(const std::string& rule, const translated_rule& rule_data) {
    const translator& tr = rule_data.tr;
    const std::vector<repl_data>& repls = rule_data.repls;

    std::string decls;
    auto declare = [&](const std::string& v) {
      if (!v.empty()) decls += "(declare-const " + v + " Int)\n(assert (in64 " + v + "))\n";
    };
    for (const std::string& v : tr.vars)
      declare(v);
    for (const std::string& v : tr.consts)
      declare(v);
    for (const std::string& i : rule_data.pattern_defs)
      decls += i;
    for (const std::string& i : tr.assumes)
      decls += "(assert " + i + ")\n";

    std::vector<std::string> fell_through;  // not (replacement j fired), for j < i.
    // A query's fall-through conditions refer to the terms of earlier replacements, so
    // it includes the side information of every replacement up to its own.
    std::string hyps;
    bool has_lemmas = false;
    for (std::size_t i = 0; i < repls.size(); ++i) {
      const repl_data& d = repls[i];
      const std::string id = rule + "_r" + std::to_string(i);
      for (const std::string& h : d.side.hyps)
        hyps += h + "\n";
      for (const std::string& def : d.side.defs)
        hyps += def;
      for (const std::string& l : d.side.lemmas)
        hyps += "(assert " + l + ")\n";
      has_lemmas = has_lemmas || !d.side.lemmas.empty();

      // Replacement i is used when no earlier one fired, its predicate is true, and its conditions hold.
      std::vector<std::string> all = fell_through;
      all.push_back(d.pr.may);
      all.insert(all.end(), d.side.conds.begin(), d.side.conds.end());

      std::string body = "; " + id + "\n; " + d.text + "\n";
      body += prelude;
      if (has_lemmas) body += lemma_prelude;
      body += decls + hyps;
      body += "(assert " + conj(all) + ")\n";
      // Correct if the replacement has the same value as the pattern.
      body += "(assert " + str("distinct", rule_data.pattern.s, d.r.s) + ")\n";
      body += "(check-sat)\n(get-model)\n";
      write_file(dir_, id, body);
      write_row(manifest_, id, ruleset_, d.text);

      // This replacement is used (and later ones are not) when its predicate holds and its conditions hold.
      std::vector<std::string> fire = {d.pr.must};
      fire.insert(fire.end(), d.side.conds.begin(), d.side.conds.end());
      fell_through.push_back(str("not", conj(fire)));
    }
  }
};

// Writes a query per lemma that is unsat iff the lemma holds for all integers.
void emit_lemma_proofs(const std::string& dir, std::ofstream& manifest) {
  using lemma = std::pair<const char*, const char*>;
  const lemma lemmas[] = {{"cong", "uvdk"}, {"euclid", "ad"}, {"negdiv", "std"}};
  for (const auto& [name, args] : lemmas) {
    const std::string id = std::string("lemma_") + name;
    std::string call = std::string("(") + name;
    std::string body = std::string("; lemma ") + name + "\n" + prelude + lemma_prelude;
    for (const char* a = args; *a; ++a) {
      const std::string v(1, *a);
      body += "(declare-const " + v + " Int)\n";
      call += " " + v;
    }
    call += ')';
    body += "(assert (not " + call + "))\n(check-sat)\n(get-model)\n";
    write_file(dir, id, body);
    write_row(manifest, id, "lemma", call);
  }
}

// clang-format off
// Deliberately wrong rules. Every query generated from these must be sat.
template <typename Fn>
bool apply_selftest_rules(Fn&& apply) {
  return
      // Wrong when c0 != 1.
      apply((x / c0) * c0, x) ||
      apply(min(x, y), x) ||
      apply(x + c0 < y, x < y + c0) ||
      // Both replacements are wrong (the second only when c0 < 0, i.e. when we fell through).
      apply(min(x, x + may_be<0>(c0)), when(c0 >= 1, x + c0), x - c0) ||
      // Wildcard predicates. Wrong when x is 1 and c0 != 1.
      apply(x * c0, when(is_boolean(x), x)) ||
      // Wrong when x is a nonzero constant.
      apply(x + y, when(is_constant(x), y)) ||
      // Wrong when x is 0 and y != z.
      apply(select(x, y, z), when(is_non_negative(x), y)) ||
      // Wrong when x < 0.
      apply(abs(x), x) ||
      // Wrong when x is truthy and y is 0.
      apply(and_then(x, y), boolean(x)) ||
      // Wrong when c0 % c1 != 0. The hint must not make it look correct.
      apply((x + c0) % c1, when(c1 != 0, x % c1, euclid(c0, c1))) ||
      false;
}
// clang-format on

struct ruleset {
  const char* name;
  bool (*emit)(smt_rule_emitter&);
};

#define RULESET(name) {#name, apply_##name##_rules<smt_rule_emitter&>}

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

void print_usage(std::ostream& os) {
  os << "usage: simplifier_to_smt -o <dir> [--ruleset name,name,...]\n"
     << "Rulesets are the apply_<name>_rules functions in simplify_rules.h:";
  for (const ruleset& i : rulesets)
    os << " " << i.name;
  os << "\n";
}

}  // namespace

}  // namespace slinky

int main(int argc, char** argv) {
  using namespace slinky;

  std::string out_dir;
  std::vector<std::string> selected;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-o" && i + 1 < argc) {
      out_dir = argv[++i];
    } else if (arg == "--ruleset" && i + 1 < argc) {
      for (std::string& s : split(argv[++i]))
        selected.push_back(std::move(s));
    } else if (arg == "-h" || arg == "--help") {
      print_usage(std::cout);
      return 0;
    } else {
      print_usage(std::cerr);
      return 1;
    }
  }
  if (out_dir.empty()) {
    std::cerr << "error: -o <dir> is required\n";
    return 1;
  }

  // The self-test rules are always emitted.
  std::vector<const ruleset*> to_run = {&selftest};
  if (selected.empty()) {
    for (const ruleset& i : rulesets)
      to_run.push_back(&i);
  } else {
    for (const std::string& s : selected) {
      const ruleset* found = nullptr;
      for (const ruleset& i : rulesets) {
        if (s == i.name) found = &i;
      }
      if (!found) {
        std::cerr << "error: unknown ruleset " << s << "\n";
        print_usage(std::cerr);
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
  // Mark partial runs to tell the verifier to ignore missing rules.
  if (!selected.empty()) {
    manifest << "# rulesets:";
    for (const std::string& s : selected)
      manifest << " " << s;
    manifest << "\n";
  }
  manifest << "id\truleset\trule_text\n";
  emit_lemma_proofs(out_dir, manifest);
  for (const ruleset* i : to_run) {
    smt_rule_emitter emitter(out_dir, i->name, manifest);
    i->emit(emitter);
  }

  manifest.close();
  if (!manifest) {
    std::cerr << "error: failed to write manifest\n";
    return 1;
  }
  return 0;
}
