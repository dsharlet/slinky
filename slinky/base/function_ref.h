#ifndef SLINKY_BASE_FUNCTION_REF_H
#define SLINKY_BASE_FUNCTION_REF_H

#include <cstddef>

namespace slinky {

template <typename Ret, typename... Args>
class function_ref;

// An implementation(-ish) of std::function_ref from C++26
template <typename Ret, typename... Args>
class function_ref<Ret(Args...)> {
  // Wrap the function object in something we can definitely call.
  template <typename F>
  static Ret get_impl(const F* fn, Args... args) {
    return (*fn)(args...);
  }

  typedef Ret (*impl_fn)(const void*, Args...);
  impl_fn impl_;
  const void* obj_;

public:
  function_ref() : impl_(nullptr), obj_(nullptr) {}
  function_ref(std::nullptr_t) : function_ref() {}
  template <typename F>
  function_ref(const F& f) : impl_(reinterpret_cast<impl_fn>(get_impl<F>)), obj_(&f) {}

  operator bool() const { return impl_ != nullptr; }

  Ret operator()(Args... args) const { return impl_(obj_, args...); }
};

}  // namespace slinky

#endif