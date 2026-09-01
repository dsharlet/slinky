#ifndef SLINKY_RUNTIME_EVALUATE_H
#define SLINKY_RUNTIME_EVALUATE_H

#include <cstdlib>
#include <optional>

#include "slinky/base/allocator.h"
#include "slinky/base/util.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/memory_pool.h"
#include "slinky/runtime/stmt.h"

namespace slinky {

class thread_pool;

struct eval_config {
  // These two functions implement allocation of buffer memory. `allocate` returns a pointer to at least `size` bytes
  // with `alignof(std::max_align_t)` alignment; `free` releases a pointer returned by `allocate`. The caller aligns
  // buffer bases to `base_alignment` within the block, padding requests accordingly. When `use_memory_pool` is set,
  // freed blocks are retained in the context's `pool` for reuse: `allocate` is only called when no retained block
  // fits, and `free` when a block is released from the pool.
  std::function<void*(std::size_t)> allocate = [](std::size_t size) { return std::malloc(size); };
  std::function<void(void*)> free = [](void* allocation) { std::free(allocation); };

  // Whether to retain freed blocks in the context's `pool` for reuse. If false, every allocation calls `allocate`
  // and every free calls `free`.
  bool use_memory_pool = true;

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

  // Alignment of the base pointer of allocations. Blocks from `allocate` are assumed to have
  // `alignof(std::max_align_t)` alignment; stronger alignments are reached by padding the requested size and aligning
  // the base within the block.
  std::size_t base_alignment = alignof(std::max_align_t);

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
  ~eval_context() { trim_pool(); }

  // Copies carry the values and config, but not the pool: each copy starts with an empty pool, so the per-worker
  // context copies made for parallel loops never share retained blocks between threads.
  eval_context(const eval_context& other) : values_(other.values_), config(other.config) {}
  eval_context& operator=(const eval_context& other) {
    if (this == &other) return *this;
    trim_pool();
    values_ = other.values_;
    config = other.config;
    return *this;
  }

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

  // Heap blocks freed by this context are kept here for reuse instead of being returned to the system (see
  // `eval_config::use_memory_pool`). Each context has its own pool, so blocks stay on the thread that freed
  // them. The pool is trimmed when the context is destroyed (the per-worker context copies made for parallel loops
  // die at the end of their loop); callers that keep a context alive across evaluations can call `trim_pool` to
  // release retained blocks earlier.
  memory_pool pool;

  // Releases the blocks retained in `pool` through `config->free`.
  void trim_pool() { pool.trim(config->free); }
};

index_t evaluate(const expr& e, eval_context& context);
index_t evaluate(const stmt& s, eval_context& context);
index_t evaluate(const expr& e);
index_t evaluate(const stmt& s);

}  // namespace slinky

#endif  // SLINKY_RUNTIME_EVALUATE_H
