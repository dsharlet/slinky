#ifndef SLINKY_BASE_ALLOCATOR_H
#define SLINKY_BASE_ALLOCATOR_H

#include <cstddef>
#include <memory>
#include <type_traits>

namespace slinky {

// This is an STL allocator that doesn't default construct, enabling an STL container to manage uninitialized memory.
// https://howardhinnant.github.io/allocator_boilerplate.html
template <class T>
class uninitialized_allocator {
public:
  using value_type = T;

  uninitialized_allocator() noexcept {}
  template <class U>
  uninitialized_allocator(uninitialized_allocator<U> const&) noexcept {}

  value_type* allocate(std::size_t n) { return static_cast<value_type*>(::operator new(n * sizeof(value_type))); }

  void deallocate(value_type* p, std::size_t) noexcept { ::operator delete(p); }

  template <class U, class... Args>
  void construct(U* p, Args&&... args) {
    if (sizeof...(args) > 0) {
      ::new (p) U(std::forward<Args>(args)...);
    }
  }
};

template <class T, class U>
bool operator==(uninitialized_allocator<T> const&, uninitialized_allocator<U> const&) noexcept {
  return true;
}

template <class T, class U>
bool operator!=(uninitialized_allocator<T> const& x, uninitialized_allocator<U> const& y) noexcept {
  return !(x == y);
}

}  // namespace slinky

#endif  // SLINKY_BASE_ARITHMETIC_H
