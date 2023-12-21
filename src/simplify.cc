#include "substitute.h"

#include <cassert>

#include "node_mutator.h"

namespace slinky {

namespace {

expr x = variable::make(0);
expr y = variable::make(1);

class simplifier : public node_mutator {
public:
  simplifier() {}

  void visit(const class min* op) {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    auto ca = is_constant(a);
    auto cb = is_constant(b);
    if (ca && cb) {
      e = std::min(*ca, *cb);
      return;
    }
    if (a.same_as(op->a) && b.same_as(op->b)) {
      e = op;
    } else {
      e = min(a, b);
    }

    static std::vector<std::pair<expr, expr>> rules = {
      {min(x, x), x},
      {min(load_buffer_meta::make(x, buffer_meta::min, y), load_buffer_meta::make(x, buffer_meta::max, y)), load_buffer_meta::make(x, buffer_meta::min, y)},
    };
    for (const auto& i : rules) {
      std::map<symbol_id, expr> matches;
      if (match(i.first, e, matches)) {
        e = substitute(i.second, matches);
        return;
      }
    }
  }

  void visit(const class max* op) {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    auto ca = is_constant(a);
    auto cb = is_constant(b);
    if (ca && cb) {
      e = std::max(*ca, *cb);
      return;
    }
    if (a.same_as(op->a) && b.same_as(op->b)) {
      e = op;
    } else {
      e = max(a, b);
    }

    static std::vector<std::pair<expr, expr>> rules = {
      {max(x, x), x},
      {max(load_buffer_meta::make(x, buffer_meta::min, y), load_buffer_meta::make(x, buffer_meta::max, y)), load_buffer_meta::make(x, buffer_meta::max, y)},
    };
    for (const auto& i : rules) {
      std::map<symbol_id, expr> matches;
      if (match(i.first, e, matches)) {
        e = substitute(i.second, matches);
        return;
      }
    }
  }
  
  void visit(const add* op) {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    auto ca = is_constant(a);
    auto cb = is_constant(b);
    if (ca && cb) {
      e = *ca + *cb;
      return;
    }
    if (a.same_as(op->a) && b.same_as(op->b)) {
      e = op;
    } else {
      e = a + b;
    }

    static std::vector<std::pair<expr, expr>> rules = {
      {(load_buffer_meta::make(x, buffer_meta::max, y) - load_buffer_meta::make(x, buffer_meta::min, y)) + 1, load_buffer_meta::make(x, buffer_meta::extent, y)},
    };
    for (const auto& i : rules) {
      std::map<symbol_id, expr> matches;
      if (match(i.first, e, matches)) {
        e = substitute(i.second, matches);
        return;
      }
    }
  }

  void visit(const sub* op) {
    expr a = mutate(op->a);
    expr b = mutate(op->b);
    auto ca = is_constant(a);
    auto cb = is_constant(b);
    if (ca && cb) {
      e = *ca - *cb;
      return;
    }
    if (a.same_as(op->a) && b.same_as(op->b)) {
      e = op;
    } else {
      e = a - b;
    }

    static std::vector<std::pair<expr, expr>> rules = {
      {x - x, 0},
      {(load_buffer_meta::make(x, buffer_meta::min, y) + load_buffer_meta::make(x, buffer_meta::extent, y)) - 1, load_buffer_meta::make(x, buffer_meta::max, y)},
    };
    for (const auto& i : rules) {
      std::map<symbol_id, expr> matches;
      if (match(i.first, e, matches)) {
        e = substitute(i.second, matches);
        return;
      }
    }
  }
};

}  // namespace

expr simplify(const expr& e) { return simplifier().mutate(e); }
stmt simplify(const stmt& s) { return simplifier().mutate(s); }

}  // namespace slinky
