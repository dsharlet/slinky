#include "simplify.h"
#include "buffer.h"
#include "evaluate.h"
#include "expr.h"
#include "print.h"
#include "substitute.h"
#include "test.h"

#include <cassert>

using namespace slinky;

node_context symbols;

expr x = make_variable(symbols, "x");
expr y = make_variable(symbols, "y");
expr z = make_variable(symbols, "z");
expr w = make_variable(symbols, "w");

template <typename T>
void dump_symbol_map(std::ostream& s, const symbol_map<T>& m) {
  s << "{";
  for (symbol_id n = 0; n < m.size(); ++n) {
    const std::optional<T>& value = m[n];
    if (value) {
      s << "{" << symbols.name(n) << ", " << *value << "},";
    }
  }
  s << "}";
}

void test_simplify(const expr& test, const expr& expected) {
  expr result = simplify(test);
  if (!match(result, expected)) {
    std::cout << "simplify failed" << std::endl;
    std::cout << test << std::endl;
    std::cout << "got: " << std::endl;
    std::cout << result << std::endl;
    std::cout << "expected: " << std::endl;
    std::cout << expected << std::endl;
    ASSERT(false);
  }
}

void test_simplify(const stmt& test, const stmt& expected) {
  stmt result = simplify(test);
  if (!match(result, expected)) {
    std::cout << "simplify failed" << std::endl;
    std::cout << test << std::endl;
    std::cout << "got: " << std::endl;
    std::cout << result << std::endl;
    std::cout << "expected: " << std::endl;
    std::cout << expected << std::endl;
    ASSERT(false);
  }
}

TEST(simplify) {
  test_simplify(expr(1) + 2, 3);
  test_simplify(expr(1) - 2, -1);
  test_simplify(expr(1) < 2, 1);
  test_simplify(expr(1) > 2, 0);

  test_simplify(min(1, 2), 1);
  test_simplify(max(1, 2), 2);
  test_simplify(min(x, y), min(x, y));
  test_simplify(max(x, y), max(x, y));
  test_simplify(min(x, x), x);
  test_simplify(max(x, x), x);
  test_simplify(min(x / 2, y / 2), min(x, y) / 2);
  test_simplify(max(x / 2, y / 2), max(x, y) / 2);
  test_simplify(min(negative_infinity(), x), negative_infinity());
  test_simplify(max(negative_infinity(), x), x);
  test_simplify(min(positive_infinity(), x), x);
  test_simplify(max(positive_infinity(), x), positive_infinity());

  test_simplify(x + 0, x);
  test_simplify(x - 0, x);
  test_simplify(0 + x + 0, x);
  test_simplify(x - 0, x);
  test_simplify(1 * x * 1, x);
  test_simplify(x * 0, 0);
  test_simplify(0 * x, 0);
  test_simplify(x / 1, x);

  test_simplify(x / x, x != 0);
  test_simplify(0 / x, 0);

  test_simplify(((x + 1) - (y - 1)) + 1, x - y + 3);

  test_simplify(select(x, y, y), y);
  test_simplify(select(x == x, y, z), y);
  test_simplify(select(x != x, y, z), z);

  test_simplify(x && false, false);
  test_simplify(x || true, true);
  test_simplify(false && x, false);
  test_simplify(true || x, true);

  test_simplify(x < x + 1, true);
  test_simplify(x - 1 < x + 1, true);

  test_simplify(abs(abs(x)), abs(x));

  test_simplify(select(z == z, x, y), x);
  test_simplify(select(z != z, x, y), y);

  test_simplify(select(x, y + 1, y + 2), y + select(x, 1, 2));
  test_simplify(select(x, 1, 2) + 1, select(x, 2, 3));
}

TEST(simplify_let) {
  // lets that should be removed
  test_simplify(let::make(0, y, z), z);                      // Dead let
  test_simplify(let::make(0, y * 2, x), y * 2);              // Single use, substitute
  test_simplify(let::make(0, y, (x + 1) / x), (y + 1) / y);  // Trivial value, substitute
  test_simplify(let::make(0, 10, x / x), 1);                 // Trivial value, substitute

  // lets that should be kept
  test_simplify(
      let::make(0, y * 2, (x + 1) / x), let::make(0, y * 2, (x + 1) / x));  // Non-trivial, used more than once.
}

TEST(simplify_load_buffer_meta) {
  test_simplify(buffer_extent(x, y) >= 0, true);
  test_simplify(max(buffer_max(x, y) + 1, buffer_min(x, y) - 1), buffer_max(x, y) + 1);
}

TEST(simplify_if_then_else) {
  test_simplify(if_then_else::make(x == x, check::make(y), check::make(z)), check::make(y));
  test_simplify(if_then_else::make(x != x, check::make(y), check::make(z)), check::make(z));
  test_simplify(block::make(if_then_else::make(x, check::make(y)), if_then_else::make(x, check::make(z))),
      if_then_else::make(x, block::make(check::make(y), check::make(z))));
}

TEST(simplify_bounds) {
  test_simplify(
      loop::make(*as_variable(x), bounds(y - 2, z),
          if_then_else::make(y - 2 <= x, check::make(z))),
      loop::make(*as_variable(x), bounds(y + -2, z),
          check::make(z)))
          ;
}

TEST(bounds_of) {
  // Test bounds_of by testing expressions of two operands, and setting the
  // bounds of the two operands to all possible cases of overlap. This approach
  // to testing should be great at finding cases where bounds are incorrectly tight,
  // but this test doesn't cover regressions that relax the bounds produced.
  int scale = 3;
  expr exprs[] = {
      x + y,
      x - y,
      x * y,
      x / y,
      slinky::min(x, y),
      slinky::max(x, y),
      x < y,
      x <= y,
      x == y,
      x != y,
      x < y && x != y,
      x < y || x == y,
  };

  for (const expr& e : exprs) {
    for (int x_min_sign : {-2, -1, 0, 1, 2}) {
      for (int x_max_sign : {-2, -1, 0, 1, 2}) {
        if (x_max_sign < x_min_sign) continue;
        int x_min = x_min_sign * scale;
        int x_max = x_max_sign * scale;
        for (int y_min_sign : {-2, -1, 0, 1, 2}) {
          for (int y_max_sign : {-2, -1, 0, 1, 2}) {
            if (y_max_sign < y_min_sign) continue;
            int y_min = y_min_sign * scale;
            int y_max = y_max_sign * scale;

            symbol_map<interval_expr> bounds;
            bounds[*as_variable(x)] = slinky::bounds(x_min, x_max);
            bounds[*as_variable(y)] = slinky::bounds(y_min, y_max);

            interval_expr bounds_e = bounds_of(e, bounds);

            eval_context ctx;
            for (int y_val = y_min; y_val <= y_max; ++y_val) {
              for (int x_val = x_min; x_val <= x_max; ++x_val) {
                ctx[*as_variable(x)] = x_val;
                ctx[*as_variable(y)] = y_val;

                index_t result = evaluate(e, ctx);
                index_t min = evaluate(bounds_e.min);
                index_t max = evaluate(bounds_e.max);

                if (result < min || result > max) {
                  std::cerr << "bounds_of failure: " << e << " -> " << bounds_e << std::endl;
                  std::cerr << result << " not in [" << min << ", " << max << "]" << std::endl;
                  std::cerr << "ctx: ";
                  dump_symbol_map(std::cerr, ctx);
                  std::cerr << std::endl;
                  std::cerr << "bounds: ";
                  dump_symbol_map(std::cerr, bounds);
                  std::cerr << std::endl;
                  std::abort();
                }
              }
            }
          }
        }
      }
    }
  }
}

std::vector<expr> vars = {x, y, z};
std::vector<symbol_id> bufs = {symbols.insert("buf0"), symbols.insert("buf1")};

template <typename T>
T random_pick(const std::vector<T>& from) {
  return from[rand() % from.size()];
}

constexpr int max_rank = 2;

constexpr int max_abs_constant = 256;

index_t random_constant() { return (rand() & (2 * max_abs_constant - 1)) - max_abs_constant; }

buffer_meta random_buffer_meta() {
  switch (rand() % 3) {
  case 0: return buffer_meta::min;
  case 1: return buffer_meta::extent;
  case 2: return buffer_meta::max;
  default: return buffer_meta::base;
  }
}

expr make_random_expr(int depth);

expr make_random_condition(int depth) {
  expr a = make_random_expr(depth - 1);
  expr b = make_random_expr(depth - 1);
  switch (rand() % 8) {
  default: return a == b;
  case 1: return a < b;
  case 2: return a <= b;
  case 3: return a != b;
  case 4: return make_random_condition(depth - 1) && make_random_condition(depth - 1);
  case 5: return make_random_condition(depth - 1) || make_random_condition(depth - 1);
  }
}

expr make_random_expr(int depth) {
  if (depth <= 0) {
    switch (rand() % 4) {
    default: return random_pick(vars);
    case 1: return constant::make(random_constant());
    case 2: return load_buffer_meta::make(variable::make(random_pick(bufs)), random_buffer_meta(), rand() % max_rank);
    }
  } else {
    expr a = make_random_expr(depth - 1);
    expr b = make_random_expr(depth - 1);
    switch (rand() % 9) {
    default: return a + b;
    case 1: return a - b;
    case 2: return a * b;
    case 3: return a / b;
    case 4: return a % b;
    case 5: return min(a, b);
    case 6: return max(a, b);
    case 7: return select(make_random_condition(depth - 1), a, b);
    case 8: return random_constant();
    }
  }
}

TEST(simplify_fuzz) {
  const int seed = time(nullptr);
  srand(seed);
  constexpr int tests = 100000;
  constexpr int checks = 10;

  eval_context ctx;

  std::vector<raw_buffer_ptr> buffers;
  for (int i = 0; i < static_cast<int>(bufs.size()); ++i) {
    buffers.emplace_back(raw_buffer::make(max_rank, 4));
  }
  for (int i = 0; i < static_cast<int>(bufs.size()); ++i) {
    ctx[bufs[i]] = reinterpret_cast<index_t>(buffers[i].get());
  }

  symbol_map<interval_expr> var_bounds;
  for (const expr& v : vars) {
    var_bounds[*as_variable(v)] = {-max_abs_constant, max_abs_constant};
  }

  for (int i = 0; i < tests; ++i) {
    expr test = make_random_expr(3);
    expr simplified = simplify(test);

    // Also test bounds_of.
    interval_expr bounds = bounds_of(test, var_bounds);
    bounds.min = simplify(bounds.min);
    bounds.max = simplify(bounds.max);

    for (int j = 0; j < checks; ++j) {
      for (const expr& v : vars) {
        ctx[*as_variable(v)] = random_constant();
      }
      for (auto& b : buffers) {
        for (int d = 0; d < max_rank; ++d) {
          // TODO: Add one to extent because the simplifier assumes buffer_max >= buffer_min. This is not
          // correct in the case of empty buffers. But do we need to handle empty buffers...?
          index_t min = random_constant();
          index_t max = std::max(min + 1, random_constant());
          b->dim(d).set_bounds(min, max);
        }
      }
      index_t a = evaluate(test, ctx);
      index_t b = evaluate(simplified, ctx);
      if (a != b) {
        std::cerr << "simplify failure (seed = " << seed << "): " << std::endl;
        print(std::cerr, test, &symbols);
        std::cerr << std::endl;
        print(std::cerr, simplified, &symbols);
        std::cerr << std::endl;
        dump_symbol_map(std::cerr, ctx);
        ASSERT_EQ(a, b);
      } else {
        index_t min = !is_infinity(bounds.min) ? evaluate(bounds.min, ctx) : std::numeric_limits<index_t>::min();
        index_t max = !is_infinity(bounds.max) ? evaluate(bounds.max, ctx) : std::numeric_limits<index_t>::max();
        if (a < min) {
          std::cerr << "bounds_of lower bound failure (seed = " << seed << "): " << std::endl;
          print(std::cerr, test, &symbols);
          std::cerr << std::endl;
          print(std::cerr, bounds.min, &symbols);
          std::cerr << std::endl;
          dump_symbol_map(std::cerr, ctx);
          std::cerr << std::endl;
          ASSERT_LE(min, a);
        }
        if (a > max) {
          std::cerr << "bounds_of upper bound failure (seed = " << seed << "): " << std::endl;
          print(std::cerr, test, &symbols);
          std::cerr << std::endl;
          print(std::cerr, bounds.max, &symbols);
          std::cerr << std::endl;
          dump_symbol_map(std::cerr, ctx);
          std::cerr << std::endl;
          ASSERT_LE(a, max);
        }
      }
    }
  }
}
