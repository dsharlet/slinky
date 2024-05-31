#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cassert>
#include <map>
#include <string>

#include "builder/cse.h"
#include "builder/node_mutator.h"
#include "builder/simplify.h"
#include "builder/substitute.h"
#include "runtime/expr.h"
#include "runtime/print.h"

namespace slinky {

#ifndef NDEBUG
const node_context* set_default_print_context(const node_context* ctx);
#endif

namespace {

node_context symbols;

var x(symbols, "x");
var y(symbols, "y");
var t0(symbols, "t0");
var t1(symbols, "t1");
var t2(symbols, "t2");
var t3(symbols, "t3");
var t4(symbols, "t4");
var t5(symbols, "t5");
var t6(symbols, "t6");
var t7(symbols, "t7");
var t8(symbols, "t8");
var t9(symbols, "t9");

MATCHER_P(matches, expected, "") { return match(arg, expected); }

// Normalize all names in an expr so that expr compares can be done
// without worrying about mere name differences. This only works
// for exprs that are structurally identical (but we those that aren't
// wouldn't match anyway).
class normalize_var_names : public node_mutator {
  node_context ctx;
  int counter = 0;
  std::map<var, var> new_names;

  using node_mutator::visit;

  void visit(const variable* v) override {
    auto iter = new_names.find(var(v->sym));
    if (iter == new_names.end()) {
      set_result(v);
    } else {
      set_result(variable::make(iter->second));
    }
  }

  void visit(const let* let) override {
    for (const auto& l : let->lets) {
      std::string new_name_str = "n" + std::to_string(counter++);
      assert(!ctx.lookup(new_name_str));
      var new_name = ctx.insert(new_name_str);
      new_names[l.first] = new_name;
    }
    std::vector<std::pair<var, expr>> new_lets;
    for (const auto& l : let->lets) {
      new_lets.emplace_back(new_names[l.first], mutate(l.second));
    }
    set_result(let::make(new_lets, mutate(let->body)));
  }

public:
  normalize_var_names(const node_context& c) : ctx(c) {}
};

expr ssa_block(node_context& ctx, std::vector<expr> exprs) {
  std::vector<std::pair<var, expr>> lets;
  for (size_t i = exprs.size() - 1; i > 0; i--) {
    var sym = ctx.insert("t" + std::to_string(i - 1));
    lets.emplace_back(sym, exprs[i - 1]);
  }
  return let::make(lets, exprs.back());
}

void test(node_context& ctx, const expr& in, const expr& correct) {
#ifndef NDEBUG
  auto* old = set_default_print_context(&ctx);
#endif
  expr result = common_subexpression_elimination(in, ctx);
  expr actual = normalize_var_names(ctx).mutate(result);
  expr expected = normalize_var_names(ctx).mutate(correct);
  ASSERT_THAT(actual, matches(expected)) << "Incorrect CSE:\n"
                                         << in << "\nbecame:\n"
                                         << actual << "\ninstead of:\n"
                                         << expected << "\n";
#ifndef NDEBUG
  set_default_print_context(old);
#endif
}

}  // namespace

TEST(cse, no_op) {
  node_context ctx = symbols;
  expr e = ssa_block(ctx, {abs(x), t0 * t0});
  // This is fine as-is.
  test(ctx, e, e);
}

TEST(cse, simple) {
  node_context ctx = symbols;
  expr e = ((x * x + x) * (x * x + x)) + x * x;
  e += e;
  expr correct = ssa_block(ctx, {x * x,            // x*x
                                    x + t0,        // x*x + x
                                    t1 * t1 + t0,  // (x*x + x)*(x*x + x) + x*x
                                    t2 + t2});
  // Test a simple case.
  test(ctx, e, correct);

  // Check for idempotence (also checks a case with lets)
  test(ctx, correct, correct);
}

TEST(cse, redundant_lets) {
  node_context ctx = symbols;
  expr e = ssa_block(ctx, {x * x, x * x, t0 / t1, t1 / t1, t2 % t3, (t4 + x * x) + x * x});
  expr correct = ssa_block(ctx, {x * x, t0 / t0, (t1 % t1 + t0) + t0});
  test(ctx, e, correct);
}

TEST(cse, nested_lets) {
  node_context ctx = symbols;
  expr e1 = ssa_block(ctx, {x * x,              // a = x*x
                               t0 + x,          // b = a + x
                               t1 * t1 * t0});  // c = b * b * a
  expr e2 = ssa_block(ctx, {x * x,              // a again
                               t0 - x,          // d = a - x
                               t1 * t1 * t0});  // e = d * d * a
  expr e = ssa_block(ctx, {e1 + x * x,          // f = c + a
                              e1 + e2,          // g = c + e
                              t0 + t0 * t1});   // h = f + f * g

  expr correct = ssa_block(ctx, {x * x,                               // t0 = a = x*x
                                    x + t0,                           // t1 = b = a + x     = t0 + x
                                    t1 * t1 * t0,                     // t2 = c = b * b * a = t1 * t1 * t0
                                    t2 + t0,                          // t3 = f = c + a     = t2 + t0
                                    t0 - x,                           // t4 = d = a - x     = t0 - x
                                    t3 + t3 * (t2 + t4 * t4 * t0)});  // h (with g substituted in)
  test(ctx, e, correct);
}

TEST(cse, scales_ok) {
  node_context ctx = symbols;
  expr e = x;
  // Test against pathological runtimes (don't bother checking correctness)
  for (int i = 0; i < 100; i++) {
    e = e * e + e + i;
    e = e * e - e * i;
  }
  expr result = common_subexpression_elimination(e, ctx);
}

TEST(cse, select) {
  node_context ctx = symbols;

  expr buf = ctx.insert_unique("buf");
  expr index = select(x * x + y * y > 0, x * x + y * y + 2, x * x + y * y + 10);
  expr at_args[] = {index};
  expr load = buffer_at(buf, at_args);

  expr e = select(x * y > 10, x * y + 2, x * y + 3 + load);
  expr cse_index = select(t1 > 0, t1 + 2, t1 + 10);
  expr cse_at_args[] = {cse_index};
  expr cse_load = buffer_at(buf, cse_at_args);
  expr correct = ssa_block(ctx, {x * y, x * x + y * y, select(t0 > 10, t0 + 2, t0 + 3 + cse_load)});
  test(ctx, e, correct);
}

}  // namespace slinky
