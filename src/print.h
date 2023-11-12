#ifndef LOCALITY_PRINT_H
#define LOCALITY_PRINT_H

#include "expr.h"

namespace slinky {

void print(std::ostream& os, const expr& e, const node_context* ctx = nullptr);

std::ostream& operator<<(std::ostream& os, const expr& e);

}  // namespace slinky

#endif