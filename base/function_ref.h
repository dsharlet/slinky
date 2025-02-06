#ifndef SLINKY_BASE_FUNCTION_REF_H
#define SLINKY_BASE_FUNCTION_REF_H

namespace slinky {

template <typename Ret, typename... Args>
class function_ref;

// An implementation(-ish) of std::function_ref from C++26
template <typename Ret, typename... Args>
class function_ref<Ret(Args...)> {
  // Wrap the function object in something we can definitely call.
  template <typename F>
  static Ret get_impl(F* fn, Args... args) {
    return (*fn)(args...);
  }

  typedef Ret (*impl_fn)(const void*, Args...);
  impl_fn impl;
  const void* obj_;

public:
  function_ref() : impl(nullptr), obj_(nullptr) {}
  template <typename F>
  function_ref(const F& f) : impl(reinterpret_cast<impl_fn>(get_impl<F>)), obj_(&f) {}

  Ret operator()(Args... args) const { return impl(obj_, args...); }
};

}  // namespace slinky

#endif