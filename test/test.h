#ifndef SLINKY_TEST_TEST_H
#define SLINKY_TEST_TEST_H

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>

namespace slinky {

// Base class of a test callback.
class test {
public:
  test(const std::string& name, std::function<void()> fn);
};

void add_test(const std::string& name, std::function<void()> fn);

// A stream class that builds a message, the destructor throws an
// assert_failure exception if the check fails.
class assert_stream {
  std::stringstream msg_;
  bool fail_;

public:
  assert_stream(bool condition, const std::string& check) : fail_(!condition) { msg_ << check; }
  ~assert_stream() noexcept(false) {
    if (fail_) { throw std::runtime_error(msg_.str()); }
  }

  template <class T>
  assert_stream& operator<<(const T& x) {
    if (fail_) { msg_ << x; }
    return *this;
  }
};

// Make a new test object. The body of the test should follow this
// macro, e.g. TEST(equality) { ASSERT(1 == 1); }
#define TEST(name)                                                                                                     \
  void test_##name##_body();                                                                                           \
  static ::slinky::test test_##name##_obj(#name, test_##name##_body);                                                  \
  void test_##name##_body()

#define ASSERT(condition) assert_stream(condition, #condition)

#define ASSERT_EQ(a, b) ASSERT(a == b) << "\n" << #a << "=" << a << "\n" << #b << "=" << b << " "

#define ASSERT_LT(a, b) ASSERT(a < b) << "\n" << #a << "=" << a << "\n" << #b << "=" << b << " "
#define ASSERT_GT(a, b) ASSERT(a > b) << "\n" << #a << "=" << a << "\n" << #b << "=" << b << " "
#define ASSERT_LE(a, b) ASSERT(a <= b) << "\n" << #a << "=" << a << "\n" << #b << "=" << b << " "
#define ASSERT_GE(a, b) ASSERT(a >= b) << "\n" << #a << "=" << a << "\n" << #b << "=" << b << " "

// This type generates compiler errors if it is copied.
struct move_only {
  move_only() = default;
  move_only(move_only&&) = default;
  move_only& operator=(move_only&&) = default;
  move_only(const move_only&) = delete;
  move_only& operator=(const move_only&) = delete;
};

}  // namespace slinky

#endif  // SLINKY_TEST_TEST_H
