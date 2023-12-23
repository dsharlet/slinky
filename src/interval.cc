#include "interval.h"

#include <limits>

namespace slinky {

interval interval::union_identity(std::numeric_limits<index_t>::max(), std::numeric_limits<index_t>::min());
interval interval::intersection_identity(std::numeric_limits<index_t>::min(), std::numeric_limits<index_t>::max());

box operator|(box a, const box& b) {
  assert(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] |= b[i];
  }
  return a;
}

box operator&(box a, const box& b) {
  assert(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] &= b[i];
  }
  return a;
}

}  // namespace slinky