#ifndef SLINKY_RUNTIME_EVALUATE_H
#define SLINKY_RUNTIME_EVALUATE_H

#include <optional>

#include "slinky/base/allocator.h"
#include "slinky/base/util.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/memory_pool.h"
#include "slinky/runtime/stmt.h"

namespace slinky {

class thread_pool;
class eval_context;

// The default implementations of `eval_config::allocate` and `eval_config::free`.
void* default_allocate(eval_context& ctx, var sym, raw_buffer* buf);
void default_free(eval_context& ctx, var sym, raw_buffer* buf, void* allocation);

struct eval_config {
  // These two functions implement allocation. `allocate` is called before
  // running the body, and should assign `base` of the buffer to the address
  // of the min in each dimension. `free` is called after running the body,
  // passing the result of `allocate` in addition to the buffer.
  // By default, they take heap blocks from the context's `pool`, aligned to `base_alignment`.
  std::function<void*(eval_context&, var, raw_buffer*)> allocate = default_allocate;
  std::function<void(eval_context&, var, raw_buffer*, void*)> free = default_free;

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

  // Alignment of the base pointer of allocations.
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

  SLINKY_INLINE void reserve(std::size_t size) {
    if (SLINKY_UNLIKELY(size > values_.size())) {
      values_.resize(std::max(values_.size() * 2, size));
    }
  }

  index_t& operator[](var id) {
    reserve(id.id + 1);
    return values_[id.id];
  }
  index_t operator[](var id) const { return values_[id.id]; }

  // This is always inlined to avoid msan false positives if the value hasn't been set already yet.
  SLINKY_INLINE index_t set(var id, index_t value) {
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

  // Heap blocks freed by this context are kept here for reuse instead of being returned to the system. Each context
  // has its own pool (copying a context gives the copy an empty pool), so blocks stay on the thread that freed them.
  // `pipeline::evaluate` trims the root context's pool at the end of each evaluation; the per-worker context copies
  // made for parallel loops trim theirs when they are destroyed at the end of the loop.
  memory_pool pool;
};

index_t evaluate(const expr& e, eval_context& context);
index_t evaluate(const stmt& s, eval_context& context);
index_t evaluate(const expr& e);
index_t evaluate(const stmt& s);

}  // namespace slinky

#endif  // SLINKY_RUNTIME_EVALUATE_H
