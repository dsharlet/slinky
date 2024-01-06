#include "optimizations.h"

#include <cassert>
#include <iostream>

#include "evaluate.h"
#include "node_mutator.h"
#include "print.h"
#include "simplify.h"
#include "substitute.h"

namespace slinky {

namespace {

class copy_implementer : public node_mutator {
  node_context& ctx;

public:
  copy_implementer(node_context& ctx) : ctx(ctx) {}

  void visit(const copy_stmt* c) override { set_result(c); }
};  // namespace

}  // namespace

stmt implement_copies(const stmt& s, node_context& ctx) { return copy_implementer(ctx).mutate(s); }

}  // namespace slinky
