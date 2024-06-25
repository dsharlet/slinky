#include <gtest/gtest.h>

#include <cassert>
#include <numeric>
#include <list>

#include "base/thread_pool.h"

namespace slinky {

int sum_arithmetic_sequence(int n) { return n * (n - 1) / 2; }

TEST(parallel_for, sum) {
  thread_pool_impl t;
  for (int n = 0; n < 100; ++n) {
    std::atomic<int> count = 0;
    std::atomic<int> sum = 0;
    t.parallel_for(n, [&](int i) {
      count++;
      sum += i;
    });
    ASSERT_EQ(count, n);
    ASSERT_EQ(sum, sum_arithmetic_sequence(n));
  }
}

TEST(parallel_for, sum_nested) {
  thread_pool_impl t;
  std::atomic<int> count = 0;
  std::atomic<int> sum = 0;
  t.parallel_for(10, [&](int i) {
    t.parallel_for(8, [&](int j) {
      count++;
      sum += j;
    });
  });
  ASSERT_EQ(count, 10 * 8);
  ASSERT_EQ(sum, 10 * sum_arithmetic_sequence(8));
}

TEST(atomic_call, sum) {
  thread_pool_impl t;
  int sum = 0;
  t.parallel_for(1000, [&](int i) { t.atomic_call([&]() { sum += i; }); });
  ASSERT_EQ(sum, sum_arithmetic_sequence(1000));
}

TEST(wait_for, barriers) {
  thread_pool_impl t;
  bool barrier0 = false;
  bool barrier1 = false;
  bool barrier2 = false;

  std::thread th([&]() {
    t.atomic_call([&]() { barrier0 = true; });
    t.wait_for([&]() -> bool { return barrier1; });
    t.atomic_call([&]() { barrier2 = true; });
  });
  t.wait_for([&]() -> bool { return barrier0; });
  t.atomic_call([&]() { barrier1 = true; });
  t.wait_for([&]() -> bool { return barrier2; });

  th.join();
}

TEST(parallel_for, sum_vector) {
  thread_pool_impl t;
  std::vector<int> values;
  for (int n = 0; n < 100; ++n) {
    std::atomic<int> count = 0;
    std::atomic<int> sum = 0;
    t.parallel_for(values.begin(), values.end(), [&](int i) {
      count++;
      sum += i;
    });
    ASSERT_EQ(count, n);
    ASSERT_EQ(sum, sum_arithmetic_sequence(n));
    values.push_back(n);
  }
}

TEST(parallel_for, sum_list) {
  thread_pool_impl t;
  std::list<int> values;
  for (int n = 0; n < 100; ++n) {
    std::atomic<int> count = 0;
    std::atomic<int> sum = 0;
    t.parallel_for(values.begin(), values.end(), [&](int i) {
      count++;
      sum += i;
    });
    ASSERT_EQ(count, n);
    ASSERT_EQ(sum, sum_arithmetic_sequence(n));
    values.push_back(n);
  }
}

}  // namespace slinky
