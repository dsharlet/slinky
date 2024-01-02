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
  std::function<void(const call_func*)> call_failed;
};

index_t evaluate(const expr& e, eval_context& context);
index_t evaluate(const stmt& s, eval_context& context);
index_t evaluate(const expr& e);
index_t evaluate(const stmt& s);

// Checks that the stmt or expr are valid:
// - No undefined variables are referenced.
// - No illegal undefined nodes.
// - Arithmetic is not performed on buffer pointers (data pointers are OK).
// - Arithmetic values are not treated as buffer pointers.
bool is_valid(const expr& e, const std::vector<symbol_id>& inputs, const node_context* symbols = nullptr);
bool is_valid(const stmt& s, const std::vector<symbol_id>& inputs, const node_context* symbols = nullptr);

// Returns true if `fn` can be evaluated.
bool can_evaluate(intrinsic fn);

}  // namespace slinky

#endif  // SLINKY_EVALUATE_H
