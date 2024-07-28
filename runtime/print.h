#ifndef SLINKY_RUNTIME_PRINT_H
#define SLINKY_RUNTIME_PRINT_H

#include "runtime/expr.h"
#include "runtime/stmt.h"

#include <tuple>

namespace slinky {

void print(std::ostream& os, const expr& e, const node_context* ctx = nullptr);
void print(std::ostream& os, const stmt& s, const node_context* ctx = nullptr);

std::ostream& operator<<(std::ostream& os, const expr& e);
std::ostream& operator<<(std::ostream& os, const stmt& s);

// Enables std::cout << std::tie(expr, ctx) << ...
std::ostream& operator<<(std::ostream& os, const std::tuple<const expr&, const node_context&>& e);
std::ostream& operator<<(std::ostream& os, const std::tuple<const stmt&, const node_context&>& s);

std::ostream& operator<<(std::ostream& os, var sym);
std::ostream& operator<<(std::ostream& os, const interval_expr& i);
std::ostream& operator<<(std::ostream& os, const box_expr& i);
std::ostream& operator<<(std::ostream& os, intrinsic fn);
std::ostream& operator<<(std::ostream& os, memory_type type);
std::ostream& operator<<(std::ostream& os, modulus_remainder& type);

std::ostream& operator<<(std::ostream& os, const raw_buffer& buf);
std::ostream& operator<<(std::ostream& os, const dim& d);

// It's not legal to overload std::to_string(), or anything else in std;
// intended usage here is to do `using std::to_string;` followed by naked
// to_string() calls.
std::string to_string(var sym);
std::string to_string(intrinsic fn);
std::string to_string(memory_type type);

}  // namespace slinky

#endif  // SLINKY_RUNTIME_PRINT_H
