#include "builder/rewrite.h"

namespace slinky {
namespace rewrite {

namespace {

pattern_expr global_positive_infinity{slinky::positive_infinity()};
pattern_expr global_negative_infinity{slinky::negative_infinity()};
pattern_expr global_indeterminate{slinky::indeterminate()};

}  // namespace

const pattern_expr& positive_infinity() { return global_positive_infinity; }
const pattern_expr& negative_infinity() { return global_negative_infinity; }
const pattern_expr& indeterminate() { return global_indeterminate; }

}  // namespace rewrite
}  // namespace slinky