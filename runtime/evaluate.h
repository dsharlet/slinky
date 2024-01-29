#ifndef SLINKY_RUNTIME_EVALUATE_H
#define SLINKY_RUNTIME_EVALUATE_H

#include "runtime/expr.h"

namespace slinky {

class eval_context {
public:
  // These two functions implement allocation. `allocate` is called before
  // running the body, and `free` is called after.
  // If these functions are not defined, the default handler will call
  // raw_buffer::allocate and raw_buffer::free.
  std::function<void(symbol_id, raw_buffer*)> allocate;
  std::function<void(symbol_id, raw_buffer*)> free;

  // Functions called when there is a failure in the pipeline.
  // If these functions are not defined, the default handler will write a
  // message to cerr and abort.
  std::function<void(const expr&)> check_failed;
  std::function<void(const call_stmt*)> call_failed;

  // Functions implementing parallelism:
  // - `enqueue_many` should enqueue the task N times for asynchronous execution, where N is the maximum number of
  // instances that could be expected to run simultaneously.
  // - `enqueue_one` should enqueue a single task for asynchronous execution.
  // - `wait_for` should wait until the given condition becomes true, executing tasks previously enqueued until it does.
  // These functions must be implemented if the statement being evaluated includes asynchronous nodes (parallel loops).
  using task = std::function<void()>;
  std::function<void(const task&)> enqueue_many;
  std::function<void(task)> enqueue_one;
  std::function<void(std::function<bool()>)> wait_for;

  const raw_buffer* lookup_buffer(symbol_id id) const {
    return reinterpret_cast<const raw_buffer*>(*symbols_.lookup(id));
  }

  symbol_map<index_t>& symbols() { return symbols_; }
  const symbol_map<index_t>& symbols() const { return symbols_; }

private:
  symbol_map<index_t> symbols_;
};

index_t evaluate(const expr& e, eval_context& context);
index_t evaluate(const stmt& s, eval_context& context);
index_t evaluate(const expr& e);
index_t evaluate(const stmt& s);

// Returns true if `fn` can be evaluated.
bool can_evaluate(intrinsic fn);

}  // namespace slinky

#endif  // SLINKY_RUNTIME_EVALUATE_H
