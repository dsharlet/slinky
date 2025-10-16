#ifndef SLINKY_RUNTIME_EVALUATE_H
#define SLINKY_RUNTIME_EVALUATE_H

#include "slinky/base/allocator.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

namespace slinky {

class thread_pool;

struct eval_config {
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

  // Functions implementing the `trace_begin` and `trace_end` intrinsics.
  std::function<index_t(const char*)> trace_begin;
  std::function<void(index_t)> trace_end;

  // Alignment to use for `raw_buffer::allocate` calls.
  std::size_t base_alignment = sizeof(std::max_align_t);

  // Alignment to use for `raw_buffer::init_strides` calls.
  std::size_t stride_alignment = 1;

  // Allocations with storage `memory_type::automatic` not bigger than this size (bytes) will be placed on the stack.
  std::size_t auto_stack_threshold = 4 * 1024;
};

class eval_context {
  // Leave uninitialized to avoid overhead and to detect uninitialized memory access via msan.
  std::vector<index_t, uninitialized_allocator<index_t>> values_;

public:
  eval_context();

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

  const eval_config* config;
};

index_t evaluate(const expr& e, eval_context& context);
index_t evaluate(const stmt& s, eval_context& context);
index_t evaluate(const expr& e);
index_t evaluate(const stmt& s);

}  // namespace slinky

#endif  // SLINKY_RUNTIME_EVALUATE_H
