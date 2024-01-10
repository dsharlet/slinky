#include "simplify.h"
#include "buffer.h"
#include "evaluate.h"
#include "expr.h"
#include "print.h"
#include "substitute.h"
#include "test.h"

#include <cassert>

namespace slinky {

// Hackily get at this function in evaluate.cc that we don't want to put in the public API.
void dump_context_for_expr(
    std::ostream&, const symbol_map<index_t>&, const expr& = expr(), const node_context* symbols = nullptr);

}  // namespace slinky

using namespace slinky;

namespace {

node_context symbols;

var x(symbols, "x");
var y(symbols, "y");
var z(symbols, "z");
var w(symbols, "w");

}  // namespace

template <typename T>
void dump_symbol_map(std::ostream& s, const symbol_map<T>& m) {
  for (symbol_id n = 0; n < m.size(); ++n) {
    const std::optional<T>& value = m[n];
    if (value) {
      s << "  " << symbols.name(n) << " = " << *value << std::endl;
    }
  }
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
  test_simplify(min(min(x, 7), min(y, 7)), min(min(x, y), 7));
  test_simplify(min(min(x, 7), min(7, y)), min(min(x, y), 7));
  test_simplify(min(min(7, x), min(y, 7)), min(min(x, y), 7));
  test_simplify(min(min(7, x), min(7, y)), min(min(x, y), 7));

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

TEST(simplify_buffer_intrinsics) {
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
  test_simplify(loop::make(x.sym(), loop_mode::serial, bounds(y - 2, z), 2, if_then_else::make(y - 2 <= x, check::make(z))),
      loop::make(x.sym(), loop_mode::serial, bounds(y + -2, z), 2, check::make(z)));
  test_simplify(loop::make(x.sym(), loop_mode::serial, min_extent(x, z), z, check::make(y)), check::make(y));

  test_simplify(
      allocate::make(x.sym(), memory_type::heap, 1, {{bounds(2, 3), 4, 5}}, check::make(buffer_min(x, 0) == 2)),
      stmt());
  test_simplify(allocate::make(x.sym(), memory_type::heap, 1, {{bounds(2, 3), 4, 5}},
                    crop_dim::make(x.sym(), 0, bounds(1, 4), check::make(buffer_min(x, 0) == 2))),
      stmt());
  test_simplify(allocate::make(x.sym(), memory_type::heap, 1, {{bounds(y, z), 4, 5}},
                    crop_dim::make(x.sym(), 0, bounds(y - 1, z + 1), check::make(buffer_min(x, 0) == 2))),
      allocate::make(x.sym(), memory_type::heap, 1, {{bounds(y, z), 4, 5}}, check::make(y == 2)));
  test_simplify(allocate::make(x.sym(), memory_type::heap, 1, {{bounds(1, 10), 20, 0}, {bounds(y, z), 4, 5}},
                    slice_dim::make(x.sym(), 0, 5, check::make(buffer_min(x, 0) == y))),
      stmt());
}

TEST(bounds_of) {
  // Test bounds_of by testing expressions of up to two operands, and setting the
  // bounds of the two operands to all possible cases of overlap. This approach
  // to testing should be great at finding cases where bounds are incorrectly tight,
  // but this test doesn't cover regressions that relax the bounds produced.
  int scale = 3;
  expr exprs[] = {
      x + y,
      x - y,
      x * y,
      x / y,
      x % y,
      slinky::min(x, y),
      slinky::max(x, y),
      x < y,
      x <= y,
      x == y,
      x != y,
      !(x < y),
      x < y && x != y,
      x < y || x == y,
      abs(x),
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
            bounds[x] = slinky::bounds(x_min, x_max);
            bounds[y] = slinky::bounds(y_min, y_max);

            interval_expr bounds_e = bounds_of(e, bounds);

            eval_context ctx;
            for (int y_val = y_min; y_val <= y_max; ++y_val) {
              for (int x_val = x_min; x_val <= x_max; ++x_val) {
                ctx[x] = x_val;
                ctx[y] = y_val;

                index_t result = evaluate(e, ctx);
                index_t min = evaluate(bounds_e.min);
                index_t max = evaluate(bounds_e.max);

                if (result < min || result > max) {
                  std::cerr << "bounds_of failure: " << e << " -> " << bounds_e << std::endl;
                  std::cerr << result << " not in [" << min << ", " << max << "]" << std::endl;
                  std::cerr << "ctx: " << std::endl;
                  dump_context_for_expr(std::cerr, ctx, e, &symbols);
                  std::cerr << std::endl;
                  std::cerr << "bounds: " << std::endl;
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

void test_where(const expr& test, symbol_id var, const interval_expr& expected) {
  interval_expr result = where_true(test, var);
  if (!match(result, expected)) {
    std::cout << "where_true failed " << std::endl;
    std::cout << test << std::endl;
    std::cout << "got: " << std::endl;
    std::cout << result << std::endl;
    std::cout << "expected: " << std::endl;
    std::cout << expected << std::endl;
    ASSERT(false);
  }
}

void test_where_true(const expr& test, symbol_id var, const interval_expr& expected) {
  test_where(test, var, expected);
}

TEST(where_true) {
  test_where_true(x < 5, 0, bounds(negative_infinity(), 4));
  test_where_true(x < buffer_min(y, 0), 0, bounds(negative_infinity(), buffer_min(y, 0) + -1));
  test_where_true(x / 2 < 7, 0, bounds(negative_infinity(), 13));
  test_where_true(min(x, 6) < 7, 0, bounds(negative_infinity(), positive_infinity()));
  test_where_true(-10 <= x && x < 5, 0, bounds(-10, 4));
}

std::vector<var> vars = {x, y, z};
var b0(symbols, "b0");
var b1(symbols, "b1");
std::vector<var> bufs = {b0, b1};

template <typename T>
T random_pick(const std::vector<T>& from) {
  return from[rand() % from.size()];
}

constexpr int max_rank = 2;

constexpr int max_abs_constant = 256;

index_t random_constant() { return (rand() & (2 * max_abs_constant - 1)) - max_abs_constant; }

expr random_buffer_intrinsic() {
  switch (rand() % 3) {
  case 0: return buffer_min(random_pick(bufs), rand() % max_rank);
  case 1: return buffer_extent(random_pick(bufs), rand() % max_rank);
  case 2: return buffer_max(random_pick(bufs), rand() % max_rank);
  default: return buffer_base(random_pick(bufs));
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
  case 6: return !make_random_condition(depth - 1);
  }
}

expr make_random_expr(int depth) {
  if (depth <= 0) {
    switch (rand() % 4) {
    default: return random_pick(vars);
    case 1: return constant::make(random_constant());
    case 2: return random_buffer_intrinsic();
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
  constexpr int tests = 10000;
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
  for (const var& v : vars) {
    var_bounds[v] = {-max_abs_constant, max_abs_constant};
  }

  for (int i = 0; i < tests; ++i) {
    expr test = make_random_expr(3);
    expr simplified = simplify(test);

    // Also test bounds_of.
    interval_expr bounds = bounds_of(test, var_bounds);

    for (int j = 0; j < checks; ++j) {
      for (const var& v : vars) {
        ctx[v] = random_constant();
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
      index_t eval_test = evaluate(test, ctx);
      index_t eval_simplified = evaluate(simplified, ctx);
      if (eval_test != eval_simplified) {
        std::cerr << "simplify failure (seed = " << seed << "): " << std::endl;
        print(std::cerr, test, &symbols);
        std::cerr << " -> " << eval_test << std::endl;
        print(std::cerr, simplified, &symbols);
        std::cerr << " -> " << eval_simplified << std::endl;
        dump_context_for_expr(std::cerr, ctx, test, &symbols);
        ASSERT_EQ(eval_test, eval_simplified);
      } else {
        index_t min = !is_infinity(bounds.min) ? evaluate(bounds.min, ctx) : std::numeric_limits<index_t>::min();
        index_t max = !is_infinity(bounds.max) ? evaluate(bounds.max, ctx) : std::numeric_limits<index_t>::max();
        if (eval_test < min) {
          std::cerr << "bounds_of lower bound failure (seed = " << seed << "): " << std::endl;
          print(std::cerr, test, &symbols);
          std::cerr << " -> " << eval_test << std::endl;
          print(std::cerr, bounds.min, &symbols);
          std::cerr << " -> " << min << std::endl;
          dump_context_for_expr(std::cerr, ctx, test, &symbols);
          std::cerr << std::endl;
          ASSERT_LE(min, eval_test);
        }
        if (eval_test > max) {
          std::cerr << "bounds_of upper bound failure (seed = " << seed << "): " << std::endl;
          print(std::cerr, test, &symbols);
          std::cerr << " -> " << eval_test << std::endl;
          print(std::cerr, bounds.max, &symbols);
          std::cerr << " -> " << max << std::endl;
          dump_context_for_expr(std::cerr, ctx, test, &symbols);
          std::cerr << std::endl;
          ASSERT_LE(eval_test, max);
        }
      }
    }
  }
}
