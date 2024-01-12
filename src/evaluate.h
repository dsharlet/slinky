#ifndef SLINKY_EVALUATE_H
#define SLINKY_EVALUATE_H

#include "expr.h"
#include "symbol_map.h"

namespace slinky {

// TODO: Probably shouldn't inherit here.
class eval_context : public symbol_map<index_t> {
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

  using task = std::function<void()>;
  // Function called to execute a task on as many threads as are available.
  std::function<void(const task&)> enqueue_many;
  // Function called to execute a single task in parallel.
  std::function<void(task)> enqueue_one;
  // Function called to indicate that this thread should wait until the given condition is true.
  std::function<void(std::function<bool()>)> wait_for;

  const raw_buffer* lookup_buffer(symbol_id id) const { return reinterpret_cast<const raw_buffer*>(*lookup(id)); }
};

index_t evaluate(const expr& e, eval_context& context);
index_t evaluate(const stmt& s, eval_context& context);
index_t evaluate(const expr& e);
index_t evaluate(const stmt& s);

// Returns true if `fn` can be evaluated.
bool can_evaluate(intrinsic fn);

}  // namespace slinky

#endif  // SLINKY_EVALUATE_H
