#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "builder/cse.h"

#include "builder/node_mutator.h"
#include "builder/substitute.h"
#include "runtime/expr.h"
#include "runtime/print.h"

#if 0
// For debugging
#define cse_debug(...)                                                                                                 \
  do {                                                                                                                 \
    __VA_ARGS__;                                                                                                       \
  } while (0)
#else
#define cse_debug(...)                                                                                                 \
  do {                                                                                                                 \
  } while (0)
#endif

// This is based on the simplifier in Halide: https://github.com/halide/Halide/blob/main/src/CSE.cpp
namespace slinky {

namespace {

// Like node_less, but only compare the pointer/value vs the deep structure
struct node_less_shallow {
  bool operator()(const expr& a, const expr& b) const { return a.get() < b.get(); }
  bool operator()(const stmt& a, const stmt& b) const { return a.get() < b.get(); }
};

// A base class for algorithms that walk recursively over the IR
// without visiting the same node twice. This is for passes that are
// capable of interpreting the IR as a DAG instead of a tree.
//
// TODO: should this be moved into node_mutator.h?
class node_graph_visitor : public expr_visitor, public stmt_visitor {
protected:
  // By default these methods add the node to the visited set, and
  // return whether or not it was already there. If it wasn't there,
  // it delegates to the appropriate visit method. You can override
  // them if you like.
  virtual void include(const expr& e) {
    if (e.defined()) {
      auto r = visited.insert(static_cast<const void*>(e.get()));
      if (r.second) {
        // Was newly inserted
        e.accept(this);
      }
    }
  }

  virtual void include(const stmt& s) {
    if (s.defined()) {
      auto r = visited.insert(static_cast<const void*>(s.get()));
      if (r.second) {
        // Was newly inserted
        s.accept(this);
      }
    }
  }

  void include(const interval_expr& i) {
    include(i.min);
    include(i.max);
  }

  void include(const dim_expr& d) {
    include(d.bounds);
    include(d.bounds);
    include(d.stride);
    include(d.fold_factor);
  }

  // These methods should call 'include' on the children to only
  // visit them if they haven't been visited already.
  void visit(const variable* op) override {
    // nothing
  }
  void visit(const constant* op) override {
    // nothing
  }
  void visit(const let* op) override {
    for (const auto& l : op->lets) {
      include(l.first);
      include(l.second);
    }
    include(op->body);
  }
  void visit(const add* op) override {
    include(op->a);
    include(op->b);
  }
  void visit(const sub* op) override {
    include(op->a);
    include(op->b);
  }
  void visit(const mul* op) override {
    include(op->a);
    include(op->b);
  }
  void visit(const div* op) override {
    include(op->a);
    include(op->b);
  }
  void visit(const mod* op) override {
    include(op->a);
    include(op->b);
  }
  void visit(const class min* op) override {
    include(op->a);
    include(op->b);
  }
  void visit(const class max* op) override {
    include(op->a);
    include(op->b);
  }
  void visit(const equal* op) override {
    include(op->a);
    include(op->b);
  }
  void visit(const not_equal* op) override {
    include(op->a);
    include(op->b);
  }
  void visit(const less* op) override {
    include(op->a);
    include(op->b);
  }
  void visit(const less_equal* op) override {
    include(op->a);
    include(op->b);
  }
  void visit(const logical_and* op) override {
    include(op->a);
    include(op->b);
  }
  void visit(const logical_or* op) override {
    include(op->a);
    include(op->b);
  }
  void visit(const logical_not* op) override { include(op->a); }
  void visit(const class select* op) override {
    include(op->condition);
    include(op->true_value);
    include(op->false_value);
  }
  void visit(const call* op) override {
    for (const auto& arg : op->args) {
      include(arg);
    }
  }
  void visit(const let_stmt* op) override {
    for (const auto& l : op->lets) {
      include(l.first);
      include(l.second);
    }
    include(op->body);
  }
  void visit(const block* op) override {
    for (const auto& s : op->stmts) {
      include(s);
    }
  }
  void visit(const loop* op) override {
    include(op->sym);
    include(op->bounds);
    include(op->step);
    include(op->body);
  }
  void visit(const call_stmt* op) override {}
  void visit(const copy_stmt* op) override {}
  void visit(const allocate* op) override {
    include(op->sym);
    include(op->elem_size);
    for (const auto& d : op->dims) {
      include(d);
    }
    include(op->body);
  }
  void visit(const make_buffer* op) override {
    include(op->base);
    include(op->elem_size);
    for (const auto& d : op->dims) {
      include(d);
    }
    include(op->body);
  }
  void visit(const clone_buffer* op) override {
    include(op->sym);
    include(op->src);
    include(op->body);
  }
  void visit(const crop_buffer* op) override {
    include(op->sym);
    include(op->src);
    for (const auto& b : op->bounds) {
      include(b);
    }
    include(op->body);
  }
  void visit(const crop_dim* op) override {
    include(op->sym);
    include(op->src);
    include(op->bounds);
    include(op->body);
  }
  void visit(const slice_buffer* op) override {
    include(op->sym);
    include(op->src);
    for (const auto& a : op->at) {
      include(a);
    }
    include(op->body);
  }
  void visit(const slice_dim* op) override {
    include(op->sym);
    include(op->src);
    include(op->at);
    include(op->body);
  }
  void visit(const transpose* op) override {
    include(op->sym);
    include(op->src);
    for (const auto& d : op->dims) {
      include(d);
    }
    include(op->body);
  }
  void visit(const check* op) override { include(op->condition); }

private:
  // The nodes visited so far
  std::set<const void*> visited;
};

// A mutator that caches and reapplies previously-done mutations, so
// that it can handle graphs of IR that have not had CSE done to
// them.
//
// TODO: should this be moved into node_mutator.h?
class graph_node_mutator : public node_mutator {
protected:
  std::map<expr, expr, node_less_shallow> expr_replacements;
  std::map<stmt, stmt, node_less_shallow> stmt_replacements;

public:
  stmt mutate(const stmt& s) override {
    auto p = stmt_replacements.emplace(s, stmt());
    if (p.second) {
      // N.B: Inserting into a map (as the recursive mutate call
      // does), does not invalidate existing iterators.
      p.first->second = node_mutator::mutate(s);
    }
    return p.first->second;
  }

  expr mutate(const expr& e) override {
    auto p = expr_replacements.emplace(e, expr());
    if (p.second) {
      p.first->second = node_mutator::mutate(e);
    }
    return p.first->second;
  }
};

// ----------------------

// Some expressions are not worth lifting out into lets, even if they
// occur redundantly many times. They may also be illegal to lift out
// (e.g. calls with side-effects).
//
// This list should at least avoid lifting the same cases as that of the
// simplifier for lets, otherwise CSE and the simplifier will fight each
// other pointlessly.
bool should_extract(const expr& e, bool lift_all) {
  if (as_constant(e) || as_variable(e)) {
    return false;
  } else if (lift_all) {
    return true;
  } else if (const add* a = e.as<add>()) {
    return !(as_constant(a->a) || as_constant(a->b));
  } else if (const sub* a = e.as<sub>()) {
    return !(as_constant(a->a) || as_constant(a->b));
  } else if (const mul* a = e.as<mul>()) {
    return !(as_constant(a->a) || as_constant(a->b));
  } else if (const div* a = e.as<div>()) {
    return !(as_constant(a->a) || as_constant(a->b));
  } else if (const mod* a = e.as<mod>()) {
    return !(as_constant(a->a) || as_constant(a->b));
  } else {
    return true;
  }
}

// A global-value-numbering of expressions. Returns canonical form of
// the expr and writes out a global value numbering as a side-effect.
class global_value_numbering : public node_mutator {
public:
  struct entry {
    expr e;
    int use_count = 0;
    // All consumer Exprs for which this is the last child expr.
    std::map<expr, int, node_less> uses;
    entry(const expr& e) : e(e) {}
  };
  std::vector<std::unique_ptr<entry>> entries;

  std::map<expr, int, node_less_shallow> shallow_numbering, output_numbering;
  std::map<expr, int, node_less> leaves;

  int number = 0;

  stmt mutate(const stmt& s) override {
    std::cerr << "Can't call global_value_numbering on a stmt: " << s << "\n";
    std::abort();
    return stmt();
  }

  expr mutate(const expr& e) override {
    // Early out if we've already seen this exact expr.
    {
      auto iter = shallow_numbering.find(e);
      if (iter != shallow_numbering.end()) {
        number = iter->second;
        return entries[number]->e;
      }
    }

    // We haven't seen this exact expr before. Rebuild it using
    // things already in the numbering.
    number = -1;
    expr new_e = node_mutator::mutate(e);

    // 'number' is now set to the numbering for the last child of
    // this expr (or -1 if there are no children). Next we see if
    // that child has an identical parent to this one.

    auto& use_map = number == -1 ? leaves : entries[number]->uses;
    auto p = use_map.emplace(new_e, (int)entries.size());
    auto iter = p.first;
    bool novel = p.second;
    if (novel) {
      // This is a never-before-seen expr
      number = (int)entries.size();
      iter->second = number;
      entries.emplace_back(new entry(new_e));
    } else {
      // This child already has a syntactically-equal parent
      number = iter->second;
      new_e = entries[number]->e;
    }

    // Memorize this numbering for the old and new forms of this expr
    shallow_numbering[e] = number;
    output_numbering[new_e] = number;
    return new_e;
  }
};

// Fill in the use counts in a global value numbering.
class compute_use_counts : public node_graph_visitor {
  global_value_numbering& gvn;
  bool lift_all;

public:
  compute_use_counts(global_value_numbering& g, bool l) : gvn(g), lift_all(l) {}

  using node_graph_visitor::include;
  using node_graph_visitor::visit;

  void include(const expr& e) override {
    // If it's not the sort of thing we want to extract as a let,
    // just use the generic visitor to increment use counts for
    // the children.
    cse_debug(std::cerr << "Include: " << e << "; should extract: " << should_extract(e, lift_all) << "\n");
    if (!should_extract(e, lift_all)) {
      e.accept(this);
      return;
    }

    // Find this thing's number.
    auto iter = gvn.output_numbering.find(e);
    if (iter != gvn.output_numbering.end()) {
      gvn.entries[iter->second]->use_count++;
    } else {
      std::cerr << "expr not in shallow numbering: " << e << "\n";
      std::abort();
    }

    // Visit the children if we haven't been here before.
    node_graph_visitor::include(e);
  }
};

// Rebuild an expression using a map of replacements. Works on graphs without exploding.
class replacer : public graph_node_mutator {
public:
  replacer() = default;
  replacer(const std::map<expr, expr, node_less_shallow>& r) : graph_node_mutator() { expr_replacements = r; }

  void erase(const expr& e) { expr_replacements.erase(e); }
};

class let_remover : public graph_node_mutator {
  using graph_node_mutator::visit;

  symbol_map<expr> scope;

  void visit(const variable* op) override {
    std::optional<expr> e = scope.lookup(op->sym);
    if (e) {
      set_result(*e);
    } else {
      set_result(op);
    }
  }

  void visit(const let* op) override {
    // When we enter a let, we invalidate all cached mutations
    // with values that reference this var due to shadowing. When
    // we leave a let, we similarly invalidate any cached
    // mutations we learned on the inside that reference the var.

    // A blunt way to handle this is to temporarily invalidate
    // *all* mutations, so we never see the same expr node
    // on the inside and outside of a Let.
    decltype(expr_replacements) tmp;
    tmp.swap(expr_replacements);

    std::vector<scoped_value_in_symbol_map<expr>> s;
    for (const auto& l : op->lets) {
      expr e = mutate(l.second);
      s.push_back(set_value_in_scope(scope, l.first, e));
    }
    auto result = mutate(op->body);
    tmp.swap(expr_replacements);
    set_result(result);
  }
};

class cse_every_expr_in_stmt : public node_mutator {
  node_context& ctx;
  bool lift_all;

public:
  using node_mutator::mutate;

  expr mutate(const expr& e) override { return common_subexpression_elimination(e, ctx, lift_all); }

  cse_every_expr_in_stmt(node_context& c, bool l) : ctx(c), lift_all(l) {}
};

}  // namespace

expr common_subexpression_elimination(const expr& e_in, node_context& ctx, bool lift_all) {
  expr e = e_in;

  if (!e.defined() || as_constant(e) || as_variable(e)) {
    // My, that was easy.
    return e;
  }

  cse_debug(std::cerr << "\n\n\nInput to CSE " << e << "\n");

  e = let_remover().mutate(e);

  cse_debug(std::cerr << "After removing lets: " << e << "\n");

  global_value_numbering gvn;
  e = gvn.mutate(e);

  cse_debug(std::cerr << "After gvn.mutate: " << e << "\n");

  compute_use_counts count_uses(gvn, lift_all);
  count_uses.include(e);

  cse_debug(std::cerr << "Canonical form without lets " << e << "\n");

  // Figure out which ones we'll pull out as lets and variables.
  std::vector<std::pair<var, expr>> lets;
  std::map<expr, expr, node_less_shallow> replacements;
  for (size_t i = 0; i < gvn.entries.size(); i++) {
    const auto& e = gvn.entries[i];
    if (e->use_count > 1) {
      var sym = ctx.insert_unique("cse");
      lets.emplace_back(sym, e->e);
      // Point references to this expr to the variable instead.
      replacements[e->e] = sym;
    }
    cse_debug(std::cerr << "GVN " << i << ": " << e->e << ", count=" << e->use_count << "\n");
  }

  // Rebuild the expr to include references to the variables:
  replacer r(replacements);
  e = r.mutate(e);

  cse_debug(std::cerr << "With variables " << e << "\n");

  // Wrap the final expr in the lets.
  std::vector<std::pair<var, expr>> new_lets;
  new_lets.reserve(lets.size());
  for (size_t i = lets.size(); i > 0; i--) {
    expr value = lets[i - 1].second;
    // Drop this variable as an acceptable replacement for this expr.
    r.erase(value);
    // Use containing lets in the value.
    value = r.mutate(lets[i - 1].second);
    new_lets.emplace_back(lets[i - 1].first, value);
  }
  std::reverse(new_lets.begin(), new_lets.end());

  if (!new_lets.empty()) {
    e = let::make(new_lets, e);
  }

  cse_debug(std::cerr << "With lets: " << e << "\n");

  return e;
}

stmt common_subexpression_elimination(const stmt& s, node_context& ctx, bool lift_all) {
  return cse_every_expr_in_stmt(ctx, lift_all).mutate(s);
}

}  // namespace slinky
