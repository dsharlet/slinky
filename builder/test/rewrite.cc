#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <map>
#include <tuple>

#include "builder/rewrite.h"

namespace slinky {

// Hackily get at this function in print.cc that we don't want to put in the public API.
const node_context* set_default_print_context(const node_context* ctx);

namespace rewrite {

pattern_wildcard<0> x;
pattern_wildcard<1> y;
pattern_wildcard<2> z;
pattern_wildcard<3> w;

pattern_constant<0> c0;
pattern_constant<1> c1;
pattern_constant<2> c2;

node_context symbols;
var a(symbols, "a");
var b(symbols, "b");
var c(symbols, "c");
var d(symbols, "d");

// Make test failures easier to read.
auto _ = []() {
  set_default_print_context(&symbols);
  return 0;
}();

MATCHER_P(matches, x, "") { return match(arg, x); }

// It's a bit tricky to test this. `match_context` doesn't extend the lifetime of the things it matches, so we can't
// return it from a function that takes an expr we target in a match. To fix this, we make a map from the context, and
// to do that, we need to make this helper type.
class context_item {
  bool is_wildcard;
  int idx;
  expr value;

public:
  template <int N>
  context_item(pattern_wildcard<N> p, expr value) : is_wildcard(true), idx(N), value(value) {}
  template <int N>
  context_item(pattern_constant<N> p, expr value) : is_wildcard(false), idx(N), value(value) {}

  bool operator==(const context_item& r) const {
    return is_wildcard == r.is_wildcard && idx == r.idx && match(value, r.value);
  }

  std::string to_string() const {
    static const char names[] = "xyzwuv";
    std::stringstream result;
    if (is_wildcard) {
      result << names[idx];
    } else {
      result << "c" << idx;
    }
    result << " -> " << value;
    return result.str();
  }
};

std::ostream& operator<<(std::ostream& os, const context_item& i) { return os << i.to_string(); }

template <typename Pattern>
std::vector<context_item> test_match(const Pattern& pattern, expr target) {
  match_context ctx;
  memset(&ctx, 0, sizeof(ctx));
  if (!match(ctx, pattern, target)) {
    return {};
  }

  std::vector<context_item> result;
  if (ctx.matched(x).defined()) result.push_back({x, ctx.matched(x)});
  if (ctx.matched(y).defined()) result.push_back({y, ctx.matched(y)});
  if (ctx.matched(z).defined()) result.push_back({z, ctx.matched(z)});
  if (ctx.matched(w).defined()) result.push_back({w, ctx.matched(w)});

  result.push_back({c0, ctx.matched(c0)});
  result.push_back({c1, ctx.matched(c1)});
  result.push_back({c2, ctx.matched(c2)});

  return result;
}

// We can't detect when constants are matched or not. So, we can only check that the match result is a superset of what
// we think it should be.
auto matches(std::vector<context_item> values) { return testing::IsSupersetOf(values); }
auto does_not_match() { return testing::IsEmpty(); }

TEST(match, basic) {
  // Wildcards
  ASSERT_THAT(test_match(x, a), matches({{x, a}}));
  ASSERT_THAT(test_match(x, a), matches({{x, a}}));
  ASSERT_THAT(test_match(x, a + 2), matches({{x, a + 2}}));

  // add
  ASSERT_THAT(test_match(x + y, a + b), matches({{x, a}, {y, b}}));
  ASSERT_THAT(test_match(x + y, a + 3), matches({{x, a}, {y, 3}}));
  ASSERT_THAT(test_match(x + c0, a + 3), matches({{x, a}, {c0, 3}}));
  ASSERT_THAT(test_match(x + c0, a), does_not_match());

  // sub
  ASSERT_THAT(test_match(x - y, a - b), matches({{x, a}, {y, b}}));
  ASSERT_THAT(test_match(x - y, a + b), does_not_match());

  // mul
  ASSERT_THAT(test_match(x * y, a * b), matches({{x, a}, {y, b}}));
  ASSERT_THAT(test_match(x * y, a * 2), matches({{x, a}, {y, 2}}));
  ASSERT_THAT(test_match(x * y, a / b), does_not_match());

  // staircase
  ASSERT_THAT(test_match(staircase(x, c0, c1, c2), ((a + 3) / 5) * 7), matches({{x, a}, {c0, 3}, {c1, 5}, {c2, 7}}));
  ASSERT_THAT(test_match(staircase(x, c0, c1, c2), (a + 3) / 5), matches({{x, a}, {c0, 3}, {c1, 5}, {c2, 1}}));
  ASSERT_THAT(test_match(staircase(x, c0, c1, c2), (a + 3) * 7), matches({{x, a}, {c0, 3}, {c1, 1}, {c2, 7}}));
  ASSERT_THAT(test_match(staircase(x, c0, c1, c2), (a / 5) * 7), matches({{x, a}, {c0, 0}, {c1, 5}, {c2, 7}}));
  ASSERT_THAT(test_match(staircase(x, c0, c1, c2), a * 7), matches({{x, a}, {c0, 0}, {c1, 1}, {c2, 7}}));
  ASSERT_THAT(test_match(staircase(x, c0, c1, c2), a / 5), matches({{x, a}, {c0, 0}, {c1, 5}, {c2, 1}}));
  ASSERT_THAT(test_match(staircase(x, c0, c1, c2), a + 3), matches({{x, a}, {c0, 3}, {c1, 1}, {c2, 1}}));
  ASSERT_THAT(test_match(staircase(x, c0, c1, c2), a), matches({{x, a}, {c0, 0}, {c1, 1}, {c2, 1}}));
}

TEST(match, optional) {
  ASSERT_THAT(test_match(may_be<0>(x) + c1, a + 3), matches({{x, a}, {c1, 3}}));
  ASSERT_THAT(test_match(may_be<0>(x) + c1, 3), matches({{x, 0}, {c1, 3}}));
  ASSERT_THAT(test_match(may_be<0>(x) + c1, a), does_not_match());
  ASSERT_THAT(test_match(min(x, y + may_be<0>(c0)), min(a, b + 2)), matches({{x, a}, {y, b}, {c0, 2}}));
  ASSERT_THAT(test_match(min(x, y + may_be<0>(c0)) < max(z, y + may_be<0>(c1)), min(a, b + 2) < max(c, b + 5)),
      matches({{x, a}, {y, b}, {c0, 2}, {z, c}, {c1, 5}}));
}

}  // namespace rewrite
}  // namespace slinky
