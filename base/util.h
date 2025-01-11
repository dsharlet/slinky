#ifndef SLINKY_BASE_UTIL_H
#define SLINKY_BASE_UTIL_H

#include <iostream>

namespace slinky {

// Some functions are templates that are usually unique specializations, which are beneficial to inline. The compiler
// will inline functions it knows are used only once, but it can't know this unless the functions have internal linkage.
#define SLINKY_UNIQUE static inline

#define SLINKY_ALLOCA(T, N) reinterpret_cast<T*>(__builtin_alloca((N) * sizeof(T)))
#define SLINKY_ALWAYS_INLINE __attribute__((always_inline))
#define SLINKY_NO_INLINE __attribute__((noinline))

#if !defined(__has_attribute)
#define SLINKY_HAS_ATTRIBUTE(x) 0
#else
#define SLINKY_HAS_ATTRIBUTE(x) __has_attribute(x)
#endif

#if SLINKY_HAS_ATTRIBUTE(trivial_abi)
#define SLINKY_TRIVIAL_ABI __attribute__((trivial_abi))
#else
#define SLINKY_TRIVIAL_ABI
#endif

#if SLINKY_HAS_ATTRIBUTE(pure)
#define SLINKY_PURE __attribute__((pure))
#else
#define SLINKY_PURE
#endif

#ifdef NDEBUG
// alloca() will cause stack-smashing code to be inserted;
// while laudable, we use alloca() in time-critical code
// and don't want it inserted there.
#define SLINKY_NO_STACK_PROTECTOR __attribute__((no_stack_protector))
#else
#define SLINKY_NO_STACK_PROTECTOR /* nothing */
#endif

class unreachable {
public:
  unreachable() = default;
  [[noreturn]] ~unreachable() {
#ifndef NDEBUG
    std::abort();
#else
    // https://en.cppreference.com/w/cpp/utility/unreachable
#if defined(_MSC_VER) && !defined(__clang__)
    __assume(false);
#else
    __builtin_unreachable();
#endif
#endif
  }

  template <typename T>
  unreachable& operator<<(const T& x) {
#ifndef NDEBUG
    std::cerr << x;
#endif
    return *this;
  }
};

#define SLINKY_UNREACHABLE unreachable() << "unreachable executed at " << __FILE__ << ", " << __LINE__ << ": "

}  // namespace slinky

#endif  // SLINKY_BASE_UTIL_H
