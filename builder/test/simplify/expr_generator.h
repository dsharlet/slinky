#ifndef SLINKY_BUILDER_TEST_EXPR_GENERATOR_H
#define SLINKY_BUILDER_TEST_EXPR_GENERATOR_H

#include <cassert>
#include <random>

#include "runtime/evaluate.h"
#include "runtime/expr.h"

namespace slinky {

template <typename Rng>
class expr_generator {
  Rng& rng_;

  int var_count_;

  // Generate normally distributed constants biased a bit towards positive numbers. We have more simplifications for
  // positive constants.
  std::normal_distribution<> constant_dist_{4.0, 5.0};
  std::uniform_int_distribution<> var_dist_;

  static constexpr int max_abs_constant = 100;

  symbol_map<interval_expr> var_bounds_;

  template <typename T>
  T random_pick(const std::vector<T>& from) {
    return from[rng_() % from.size()];
  }

public:
  expr_generator(Rng& rng, int var_count) : rng_(rng), var_count_(var_count), var_dist_(0, var_count - 1) {
    for (int i = 0; i < var_count_; ++i) {
      var_bounds_[var(i)] = {-max_abs_constant, max_abs_constant};
    }
  }

  const symbol_map<interval_expr>& var_bounds() { return var_bounds_; }

  void init_context(eval_context& ctx) {
    for (int i = 0; i < var_count_; ++i) {
      ctx[var(i)] = random_constant();
    }
  }

  index_t random_constant(int max = max_abs_constant) {
    return std::clamp<index_t>(std::round(constant_dist_(rng_)), -max_abs_constant, max_abs_constant);
  }

  expr random_variable() { return variable::make(var(var_dist_(rng_))); }

  expr random_condition(int depth) {
    auto a = [&]() { return random_expr(depth - 1); };
    auto b = [&]() { return random_expr(depth - 1); };
    auto ac = [&]() { return random_condition(depth - 1); };
    auto bc = [&]() { return random_condition(depth - 1); };
    switch (rng_() % 7) {
    case 0: return a() == b();
    case 1: return a() < b();
    case 2: return a() <= b();
    case 3: return a() != b();
    case 4: return rng_() % 8 != 0 ? ac() && bc() : and_then(ac(), bc());
    case 5: return rng_() % 8 != 0 ? ac() || bc() : or_else(ac(), bc());
    case 6: return !random_condition(depth - 1);
    default: SLINKY_UNREACHABLE;
    }
  }

  expr random_expr(int depth) {
    if (depth <= 0) {
      switch (rng_() % 4) {
      default: return random_variable();
      case 1: return constant::make(random_constant());
      }
    } else {
      auto a = [&]() { return random_expr(depth - 1); };
      auto b = [&]() { return random_expr(depth - 1); };
      switch (rng_() % 11) {
      case 0: return a() + b();
      case 1: return a() - b();
      case 2: return a() * b();
      case 3: return a() / b();
      case 4: return a() % b();
      case 5: return min(a(), b());
      case 6: return max(a(), b());
      case 7: return select(random_condition(depth - 1), a(), b());
      case 8: return random_constant();
      case 9: return random_variable();
      case 10: return random_condition(depth);
      default: SLINKY_UNREACHABLE;
      }
    }
  }
};

}  // namespace slinky

#endif  // SLINKY_BUILDER_TEST_EXPR_GENERATOR_H