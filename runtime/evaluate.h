#ifndef SLINKY_RUNTIME_EVALUATE_H
#define SLINKY_RUNTIME_EVALUATE_H

#include "runtime/expr.h"
#include "runtime/stmt.h"

namespace slinky {

class thread_pool;

class eval_context {
  // TODO: This should be uninitialized memory, not just for performance, but so we can detect uninitialized memory
  // usage when evaluating.
  std::vector<index_t> values_;

public:
  void reserve(std::size_t size) {
    if (size > values_.size()) {
      values_.resize(std::max(values_.size() * 2, size));
    }
  }

  index_t& operator[](var id) {
    reserve(id.id + 1);
    return values_[id.id];
  }
  index_t operator[](var id) const { return values_[id.id]; }

  index_t set(var id, index_t value) {
    index_t& value_ref = values_[id.id];
    index_t old_value = value_ref;
    value_ref = value;
    return old_value;
  }

  index_t lookup(var id) const {
    assert(id.id < values_.size());
    return values_[id.id];
  }
  const raw_buffer* lookup_buffer(var id) const { return reinterpret_cast<const raw_buffer*>(lookup(id)); }
  template <typename T>
  const buffer<T>* lookup_buffer(var id) const {
    const raw_buffer* buf = lookup_buffer(id);
    return buf ? &buf->cast<T>() : nullptr;
  }

  std::size_t size() const { return values_.size(); }

  // These two functions implement allocation. `allocate` is called before
  // running the body, and should assign `base` of the buffer to the address
  // of the min in each dimension. `free` is called after running the body,
  // passing the result of `allocate` in addition to the buffer.
  // If these functions are not defined, the default handler will call
  // `raw_buffer::allocate` and `::free`.
  std::function<void*(var, raw_buffer*)> allocate = [](var, raw_buffer* buf) { return buf->allocate(); };
  std::function<void(var, raw_buffer*, void*)> free = [](var, raw_buffer*, void* allocation) { ::free(allocation); };

  // Functions called when there is a failure in the pipeline.
  // If these functions are not defined, the default handler will write a
  // message to cerr and abort.
  std::function<void(const expr&)> check_failed;
  std::function<void(const call_stmt*)> call_failed;

  // A pointer to a thread pool, required for parallel
  slinky::thread_pool* thread_pool = nullptr;

  // Functions implementing buffer data movement:
  // - `copy` should copy from `src` to `dst`, filling `dst` with `padding` when out of bounds of `src`.
  // - `pad` should fill the area out of bounds of `src_dims` with `padding` in `dst`.
  std::function<void(const raw_buffer& src, const raw_buffer& dst, const void* padding)> copy =
      static_cast<void (*)(const raw_buffer&, const raw_buffer&, const void*)>(slinky::copy);
  std::function<void(const dim* in_bounds, const raw_buffer& dst, const void* padding)> pad =
      static_cast<void (*)(const dim*, const raw_buffer&, const void*)>(slinky::pad);

  // Functions called every time a stmt begins or ends evaluation.
  std::function<index_t(const char*)> trace_begin;
  std::function<void(index_t)> trace_end;
};

index_t evaluate(const expr& e, eval_context& context);
index_t evaluate(const stmt& s, eval_context& context);
index_t evaluate(const expr& e);
index_t evaluate(const stmt& s);

}  // namespace slinky

#endif  // SLINKY_RUNTIME_EVALUATE_H
