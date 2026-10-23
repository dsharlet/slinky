#include <atomic>
#include <iostream>
#include <sstream>

#include "slinky/base/arithmetic.h"
#include "slinky/base/chrome_trace.h"
#include "slinky/base/thread_pool_impl.h"
#include "slinky/builder/simplify.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/print.h"
#include "slinky/runtime/visualize.h"

int main() {
  using namespace slinky;

  if (euclidean_div(-7, 2) != -4) return 1;

  {
    std::ostringstream trace_output;
    chrome_trace trace(trace_output);
    scoped_trace scope(&trace, "smoke");
  }

  thread_pool_impl pool(2);
  std::atomic<int> sum = 0;
  pool.parallel_for(100, [&](std::size_t i) { sum += static_cast<int>(i); });
  if (sum != 4950) return 2;

  node_context symbols;
  var x(symbols, "x");
  expr simplified = simplify((x + 0) * 1);
  std::ostringstream printed;
  printed << std::make_tuple(std::cref(simplified), std::cref(symbols));
  if (printed.str() != "x") return 3;

  // Taking the address forces a reference to slinky::visualize (and so to the
  // runtime it builds on), even when the optimizer would drop an unused call.
  void (*volatile visualize_fn)(
      std::ostream&, const pipeline&, pipeline::buffers, pipeline::buffers, const node_context*) = &visualize;
  return visualize_fn == nullptr ? 4 : 0;
}
