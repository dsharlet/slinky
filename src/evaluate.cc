#include "evaluate.h"

#include <cassert>
#include <iostream>

#include "print.h"

namespace slinky {
 
class evaluator : public node_visitor {
public:
  index_t result = 0;
  eval_context& context;

  evaluator(eval_context& context) : context(context) {}

  // Assume `e` is defined, evaluate it and return the result.
  index_t eval_expr(const expr& e) {
    e.accept(this);
    return result;
  }

  // If `e` is defined, evaluate it and return the result. Otherwise, return default `def`.
  index_t eval_expr(const expr& e, index_t def) {
    if (e.defined()) {
      e.accept(this);
      return result;
    } else {
      return def;
    }
  }

  void visit(const variable* v) override {
    auto value = context.lookup(v->name);
    assert(value);
    result = *value;
  }

  void visit(const constant* c) override {
    result = c->value;
  }

  void visit(const let* l) override {
    scoped_value<index_t> s(context, l->name, eval_expr(l->value));
    l->body.accept(this);
  }

  void visit(const add* x) override { result = eval_expr(x->a) + eval_expr(x->b); }
  void visit(const sub* x) override { result = eval_expr(x->a) - eval_expr(x->b); }
  void visit(const mul* x) override { result = eval_expr(x->a) * eval_expr(x->b); }
  void visit(const div* x) override { result = euclidean_div(eval_expr(x->a), eval_expr(x->b)); }
  void visit(const mod* x) override { result = euclidean_mod(eval_expr(x->a), eval_expr(x->b)); }
  void visit(const class min* x) override { result = std::min(eval_expr(x->a), eval_expr(x->b)); }
  void visit(const class max* x) override { result = std::max(eval_expr(x->a), eval_expr(x->b)); }
  void visit(const equal* x) override { result = eval_expr(x->a) == eval_expr(x->b); }
  void visit(const not_equal* x) override { result = eval_expr(x->a) != eval_expr(x->b); }
  void visit(const less* x) override { result = eval_expr(x->a) < eval_expr(x->b); }
  void visit(const less_equal* x) override { result = eval_expr(x->a) <= eval_expr(x->b); }
  void visit(const bitwise_and* x) override { result = eval_expr(x->a) & eval_expr(x->b); }
  void visit(const bitwise_or* x) override { result = eval_expr(x->a) | eval_expr(x->b); }
  void visit(const bitwise_xor* x) override { result = eval_expr(x->a) ^ eval_expr(x->b); }
  void visit(const logical_and* x) override { result = eval_expr(x->a) && eval_expr(x->b); }
  void visit(const logical_or* x) override { result = eval_expr(x->a) || eval_expr(x->b); }
};

index_t evaluate(expr e, eval_context& context) {
  evaluator eval(context);
  e.accept(&eval);
  return eval.result;
}

index_t evaluate(expr e) {
  eval_context ctx;
  return evaluate(e, ctx);
}

}  // namespace slinky