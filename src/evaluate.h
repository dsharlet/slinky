#ifndef SLINKY_EVALUATE_H
#define SLINKY_EVALUATE_H

#include "expr.h"
#include "symbol_map.h"

namespace slinky {

// TODO: Probably shouldn't inherit here.
class eval_context : public symbol_map<index_t> {
public:
  // These two functions implement allocate nodes.
  std::function<void(buffer_base*)> allocate;
  std::function<void(buffer_base*)> free;

  // If a check fails, this function is called.
  std::function<void(const expr&)> check_failed;
  
  // If a call to a func fails, this function is called.
  std::function<void(const call_func*)> call_failed;
};

index_t evaluate(const expr& e, eval_context& context);
index_t evaluate(const stmt& s, eval_context& context);
index_t evaluate(const expr& e);
index_t evaluate(const stmt& s);

}  // namespace slinky

#endif  // SLINKY_EVALUATE_H
