#ifndef SLINKY_EVALUATE_H
#define SLINKY_EVALUATE_H

#include "expr.h"
#include "symbol_map.h"

namespace slinky {

using eval_context = symbol_map<index_t>;

index_t evaluate(const expr& e, eval_context& context);
index_t evaluate(const stmt& s, eval_context& context);
index_t evaluate(const expr& e);
index_t evaluate(const stmt& s);

}  // namespace slinky

#endif  // SLINKY_EVALUATE_H
