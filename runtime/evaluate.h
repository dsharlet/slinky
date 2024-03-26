#ifndef SLINKY_RUNTIME_EVALUATE_H
#define SLINKY_RUNTIME_EVALUATE_H

#include "runtime/expr.h"
#include "runtime/stmt.h"

namespace slinky {

// TODO: Probably shouldn't inherit here.
class eval_context : public symbol_map<index_t> {
public:
  // These two functions implement allocation. `allocate` is called before
  // running the body, and should assign `base` of the buffer to the address
  // of the min in each dimension. `free` is called after running the body,
  // passing the result of `allocate` in addition to the buffer.
  // If these functions are not defined, the default handler will call
  // `raw_buffer::allocate` and `::free`.
  std::function<void*(symbol_id, raw_buffer*)> allocate;
  std::function<void(symbol_id, raw_buffer*, void*)> free;

  // Functions called when there is a failure in the pipeline.
  // If these functions are not defined, the default handler will write a
  // message to cerr and abort.
  std::function<void(const expr&)> check_failed;
  std::function<void(const call_stmt*)> call_failed;

  // Functions implementing parallelism:
  // - `enqueue_many` should enqueue the task N times for asynchronous execution, where N is the maximum number of
  // instances that could be expected to run simultaneously.
  // - `enqueue` should enqueue a task N times.
  // - `wait_for` should wait until the given condition becomes true, executing tasks previously enqueued until it does.
  // These functions must be implemented if the statement being evaluated includes asynchronous nodes (parallel loops).
  using task = std::function<void()>;
  std::function<void(const task&)> enqueue_many;
  std::function<void(int, const task&)> enqueue;
  std::function<void(std::function<bool()>)> wait_for;
  std::function<void(task)> atomic_call;

  // Functions implementing buffer data movement:
  // - `copy` should copy from `src` to `dst`, filling `dst` with `padding` when out of bounds of `src`.
  // - `pad` should fill the area out of bounds of `src_dims` with `padding` in `dst`.
  std::function<void(const raw_buffer& src, const raw_buffer& dst, const void* padding)> copy = slinky::copy;
  std::function<void(const dim* in_bounds, const raw_buffer& dst, const void* padding)> pad = slinky::pad;

  const raw_buffer* lookup_buffer(symbol_id id) const { return reinterpret_cast<const raw_buffer*>(*lookup(id)); }
};

index_t evaluate(const expr& e, eval_context& context);
index_t evaluate(const stmt& s, eval_context& context);
index_t evaluate(const expr& e);
index_t evaluate(const stmt& s);

// Attempt to evaluate `e` as a constant. Returns `std::nullopt` if the expression is not constant.
std::optional<index_t> evaluate_constant(const expr& e);

// Returns true if `fn` can be evaluated.
bool can_evaluate(intrinsic fn);

}  // namespace slinky

#endif  // SLINKY_RUNTIME_EVALUATE_H
