#ifndef SLINKY_BASE_UTIL_H
#define SLINKY_BASE_UTIL_H

namespace slinky {

#define SLINKY_ALLOCA(T, N) reinterpret_cast<T*>(alloca((N) * sizeof(T)))
#define SLINKY_ALWAYS_INLINE __attribute__((always_inline))
#define SLINKY_NO_INLINE __attribute__((noinline))

#ifdef NDEBUG
// alloca() will cause stack-smashing code to be inserted;
// while laudable, we use alloca() in time-critical code
// and don't want it inserted there.
#define SLINKY_NO_STACK_PROTECTOR __attribute__((no_stack_protector))
#else
#define SLINKY_NO_STACK_PROTECTOR /* nothing */
#endif

}  // namespace slinky

#endif  // SLINKY_BASE_UTIL_H
