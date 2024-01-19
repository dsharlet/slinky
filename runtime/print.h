#ifndef SLINKY_RUNTIME_PRINT_H
#define SLINKY_RUNTIME_PRINT_H

#include "runtime/expr.h"

#include <tuple>

namespace slinky {

void print(std::ostream& os, const expr& e, const node_context* ctx = nullptr);
void print(std::ostream& os, const stmt& s, const node_context* ctx = nullptr);

std::ostream& operator<<(std::ostream& os, const expr& e);
std::ostream& operator<<(std::ostream& os, const stmt& s);

// Enables std::cout << std::tie(expr, ctx) << ...
std::ostream& operator<<(std::ostream& os, const std::tuple<const expr&, const node_context&>& e);
std::ostream& operator<<(std::ostream& os, const std::tuple<const stmt&, const node_context&>& s);

std::ostream& operator<<(std::ostream& os, const interval_expr& i);
std::ostream& operator<<(std::ostream& os, intrinsic fn);

}  // namespace slinky

#endif  // SLINKY_RUNTIME_PRINT_H
