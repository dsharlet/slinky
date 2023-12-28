#ifndef SLINKY_PRINT_H
#define SLINKY_PRINT_H

#include "expr.h"

namespace slinky {

void print(std::ostream& os, const expr& e, const node_context* ctx = nullptr);
void print(std::ostream& os, const stmt& s, const node_context* ctx = nullptr);

std::ostream& operator<<(std::ostream& os, const expr& e);
std::ostream& operator<<(std::ostream& os, const stmt& s);

std::ostream& operator<<(std::ostream& os, const interval& i);

}  // namespace slinky

#endif  // SLINKY_PRINT_H
