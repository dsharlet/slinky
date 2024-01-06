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

namespace {

template <typename Fn>
void for_each_stmt_forward(const stmt& s, const Fn& fn) {
  if (const block* b = s.as<block>()) {
    for_each_stmt_forward(b->a, fn);
    for_each_stmt_forward(b->b, fn);
  } else {
    fn(s);
  }
}

template <typename Fn>
void for_each_stmt_backward(const stmt& s, const Fn& fn) {
  if (const block* b = s.as<block>()) {
    for_each_stmt_backward(b->b, fn);
    for_each_stmt_backward(b->a, fn);
  } else {
    fn(s);
  }
}

stmt clone_with_new_body(const let_stmt* op, stmt new_body) {
  return let_stmt::make(op->sym, op->value, std::move(new_body));
}
stmt clone_with_new_body(const allocate* op, stmt new_body) {
  return allocate::make(op->storage, op->sym, op->elem_size, op->dims, std::move(new_body));
}
stmt clone_with_new_body(const make_buffer* op, stmt new_body) {
  return make_buffer::make(op->sym, op->base, op->elem_size, op->dims, std::move(new_body));
}
stmt clone_with_new_body(const crop_buffer* op, stmt new_body) {
  return crop_buffer::make(op->sym, op->bounds, std::move(new_body));
}
stmt clone_with_new_body(const crop_dim* op, stmt new_body) {
  return crop_dim::make(op->sym, op->dim, op->bounds, std::move(new_body));
}
stmt clone_with_new_body(const slice_buffer* op, stmt new_body) {
  return slice_buffer::make(op->sym, op->at, std::move(new_body));
}
stmt clone_with_new_body(const slice_dim* op, stmt new_body) {
  return slice_dim::make(op->sym, op->dim, op->at, std::move(new_body));
}
stmt clone_with_new_body(const truncate_rank* op, stmt new_body) {
  return truncate_rank::make(op->sym, op->rank, std::move(new_body));
}

class scope_reducer : public node_mutator {
  std::tuple<stmt, stmt, stmt> split_body(const stmt& body, std::span<const symbol_id> vars) {
    stmt before;
    stmt new_body_after;
    bool depended_on = false;
    // First, split the body into the before, and the new body + after.
    for_each_stmt_forward(body, [&](const stmt& s) {
      if (depended_on || depends_on(s, vars)) {
        new_body_after = block::make({new_body_after, s});
        depended_on = true;
      } else {
        before = block::make({before, s});
      }
    });

    // Now, split the new body + after into the new body and the after.
    stmt new_body;
    stmt after;
    depended_on = false;
    for_each_stmt_backward(new_body_after, [&](const stmt& s) {
      if (!depended_on && !depends_on(s, vars)) {
        after = block::make({s, after});
      } else {
        new_body = block::make({s, new_body});
        depended_on = true;
      }
    });

    return {before, new_body, after};
  }
  std::tuple<stmt, stmt, stmt> split_body(const stmt& body, symbol_id var) {
    symbol_id vars[] = {var};
    return split_body(body, vars);
  }

public:
  template <typename T>
  void visit_stmt(const T* op) {
    stmt body = mutate(op->body);

    stmt before, new_body, after;
    std::tie(before, new_body, after) = split_body(body, op->sym);

    if (body.same_as(op->body) && !before.defined() && !after.defined()) {
      set_result(op);
    } else if (new_body.defined()) {
      stmt result = clone_with_new_body(op, std::move(new_body));
      set_result(block::make({before, result, after}));
    } else {
      // The op was dead...?
      set_result(block::make({before, after}));
    }
  }

  void visit(const let_stmt* op) override { visit_stmt(op); }
  void visit(const allocate* op) override { visit_stmt(op); }
  void visit(const make_buffer* op) override { visit_stmt(op); }
  void visit(const crop_buffer* op) override { visit_stmt(op); }
  void visit(const crop_dim* op) override { visit_stmt(op); }
  void visit(const slice_buffer* op) override { visit_stmt(op); }
  void visit(const slice_dim* op) override { visit_stmt(op); }
  void visit(const truncate_rank* op) override { visit_stmt(op); }
};

}  // namespace

stmt reduce_scopes(const stmt& s) { return scope_reducer().mutate(s); }

}  // namespace slinky
