#include "simplify.h"
#include "expr.h"
#include "print.h"
#include "substitute.h"
#include "evaluate.h"
#include "buffer.h"
#include "test.h"

#include <cassert>

using namespace slinky;

node_context symbols;

expr x = make_variable(symbols, "x");
expr y = make_variable(symbols, "y");
expr z = make_variable(symbols, "z");

void test_simplify(const expr& test, const expr& expected) {
  expr result = simplify(test);
  if (!match(result, expected)) {
    std::cout << "simplify(" << test << ") -> " << result << " != " << expected << std::endl;
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

  test_simplify(x + 0, x);
  test_simplify(x - 0, x);
  test_simplify(0 + x + 0, x);
  test_simplify(x - 0, x);
  test_simplify(1 * x * 1, x);
  test_simplify(x * 0, 0);
  test_simplify(0 * x, 0);
  test_simplify(x / 1, x);

  test_simplify(x / x, x / x);  // Not simplified due to possible division by zero.
  test_simplify(0 / x, 0 / x);  // Not simplified due to possible division by zero.

  test_simplify(((x + 1) - (y - 1)) + 1, x - y + 3);
}

TEST(simplify_let) {
  // lets that should be removed
  test_simplify(let::make(0, y, z), z);          // Dead let
  test_simplify(let::make(0, y * 2, x), y * 2);  // Single use, substitute
  test_simplify(let::make(0, y, x / x), y / y);  // Trivial value, substitute
  test_simplify(let::make(0, 10, x / x), 1);     // Trivial value, substitute

  // lets that should be kept
  test_simplify(let::make(0, y * 2, x / x), let::make(0, y * 2, x / x));  // Non-trivial, used more than once.
}

std::vector<expr> vars = {x, y, z};
std::vector<symbol_id> bufs = {symbols.insert("buf0"), symbols.insert("buf1")};

template <typename T>
T random_pick(const std::vector<T>& from) {
  return from[rand() % from.size()];
}

constexpr int max_rank = 2;

buffer_meta random_buffer_meta() {
  switch (rand() % 3) {
  case 0: return buffer_meta::min;
  case 1: return buffer_meta::extent;
  case 2: return buffer_meta::max;
  default: return buffer_meta::base;
  }
}

expr make_random_expr(int depth) {
  if (depth <= 0) {
    switch (rand() % 4) {
    default: return random_pick(vars);
    case 1: return constant::make(rand());
    case 2: return load_buffer_meta::make(variable::make(random_pick(bufs)), random_buffer_meta(), rand() % max_rank);
    }
  } else {
    expr a = make_random_expr(depth - rand() % 3);
    expr b = make_random_expr(depth - rand() % 3);
    switch (rand() % 7) {
    default: return a + b;
    case 1: return a - b;
    case 2: return a * b;
    case 3: return a / b;
    case 4: return a % b;
    case 5: return min(a, b);
    case 6: return max(a, b);
    }
  }
}

std::ostream& operator<<(std::ostream& s, const eval_context& ctx) {
  s << "{";
  for (symbol_id n = 0; n < ctx.size(); ++n) {
    const std::optional<index_t>& value = ctx[n];
    if (value) { 
      s << "{" << symbols.name(n) << ", " << *value << "},";
    }
  }
  return s << "}";
}

TEST(simplify_fuzz) {
  const int seed = time(nullptr);
  srand(seed);
  constexpr int tests = 10000;
  constexpr int checks = 10;
  
  eval_context ctx;

  std::vector<buffer_base_ptr> buffers;
  for (int i = 0; i < bufs.size(); ++i) {
    buffers.emplace_back(buffer_base::make(max_rank, 4));
  }
  for (int i = 0; i < bufs.size(); ++i) {
    ctx[bufs[i]] = reinterpret_cast<index_t>(buffers[i].get());
  }

  for (int i = 0; i < tests; ++i) {
    expr test = make_random_expr(3);
    expr simplified = simplify(test);

    for (int j = 0; j < checks; ++j) {
      for (const expr& v : vars) {
        ctx[*as_variable(v)] = rand() - RAND_MAX / 2;
      }
      for (auto& b : buffers) {
        for (int d = 0; d < max_rank; ++d) {
          b->dim(d).set_min_extent(rand() - RAND_MAX / 2, rand());
        }
      }
      index_t a = evaluate(test, ctx);
      index_t b = evaluate(simplified, ctx);
      if (a != b) {
        std::cerr << "Failure (seed = " << seed << "): " << std::endl;
        print(std::cerr, test, &symbols);
        std::cerr << std::endl;
        print(std::cerr, simplified, &symbols);
        std::cerr << std::endl;
        std::cerr << ctx << std::endl;
        ASSERT_EQ(a, b);
      }
    }
  }
}