#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <iostream>

#include "base/thread_pool_impl.h"

namespace slinky {

int sum_arithmetic_sequence(int n) { return n * (n - 1) / 2; }

template <int K>
bool test_task_impl_done(bool ordered, int n) {
  std::vector<int> ran(n);

  thread_pool_impl::task_impl<K> p(ordered, n, [&](int i) { ran[i]++; });
  p.work();
  return std::all_of(ran.begin(), ran.end(), [](int i) { return i == 1; });
}

TEST(task_impl, done) {
  for (int n = 0; n < 100; ++n) {
    for (bool ordered : {false, true}) {
      ASSERT_TRUE(test_task_impl_done<1>(ordered, n));
      ASSERT_TRUE(test_task_impl_done<2>(ordered, n));
      ASSERT_TRUE(test_task_impl_done<4>(ordered, n));
      ASSERT_TRUE(test_task_impl_done<16>(ordered, n));
    }
  }
}

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

}  // namespace slinky
