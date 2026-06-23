#include <cassert>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "slinky/base/ref_count.h"
#include "slinky/base/util.h"
#include "slinky/base/util.h"
#include "slinky/builder/node_mutator.h"
#include "slinky/builder/optimizations.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

namespace slinky {

namespace {

class arena : public ref_counted<arena> {
  size_t next_;

  arena() : next_(sizeof(arena)) {}

public:
  static constexpr std::size_t block_size = 4096;

  // A chunk is aligned to `block_size`, so it can align anything it holds up to this much.
  static constexpr std::size_t max_alignment = block_size;

  static slinky::ref_count<arena> make() {
    return slinky::ref_count<arena>(new (allocate_bytes(block_size, block_size)) arena());
  }

  // Because chunks are aligned to `block_size`, the chunk an object lives in is the beginning of the block containing
  // it. This lets nodes find the chunk that owns them without storing a pointer to it.
  static arena* from(const void* p) {
    return reinterpret_cast<arena*>(reinterpret_cast<std::uintptr_t>(p) & ~(block_size - 1));
  }

  // Returns memory for a node, padding to align it, or null if there is not enough space in this chunk.
  void* allocate(std::size_t size, std::size_t alignment) {
    assert(alignment <= max_alignment);
    std::size_t offset = (next_ + alignment - 1) & ~(alignment - 1);
    if (offset + size > block_size) {
      // Out of space
      return nullptr;
    }
    next_ = offset + size;
    return reinterpret_cast<char*>(this) + offset;
  }

  static void destroy(arena* a) {
    a->~arena();
    deallocate_bytes(a, block_size);
  }
};

constexpr std::size_t array_alignment = alignof(void*);

// A node of type `T` that lives in an `arena`.
template <typename T>
class arena_node : public T {
public:
  arena_node(const T& src) : T(src) {
    static_assert(alignof(arena_node) <= arena::max_alignment, "node is overaligned for a chunk");
    static_assert(sizeof(arena_node) % array_alignment == 0, "node arrays would be misaligned");
    arena::from(this)->add_ref();
  }

  // These nodes find their arena from their own address, so they can only be constructed in an arena.
  static void* operator new(std::size_t) = delete;
  static void* operator new(std::size_t, void* mem) { return mem; }

  void destroy() override {
    this->~arena_node();
    arena::from(this)->release();
  }
};

template <typename T>
std::size_t size_of(span<T> x) {
  static_assert(alignof(T) <= array_alignment, "array would not be aligned");
  return (x.size() * sizeof(T) + array_alignment - 1) & ~(array_alignment - 1);
}

// The size of a node in the arena. Nodes that own arrays need room for them too.
template <typename T>
std::size_t size_of(const T&) {
  return sizeof(arena_node<T>);
}
std::size_t size_of(const let& n) { return sizeof(arena_node<let>) + size_of(n.lets); }
std::size_t size_of(const call& n) { return sizeof(arena_node<call>) + size_of(n.args); }
std::size_t size_of(const let_stmt& n) { return sizeof(arena_node<let_stmt>) + size_of(n.lets); }
std::size_t size_of(const block& n) { return sizeof(arena_node<block>) + size_of(n.stmts); }
std::size_t size_of(const call_stmt& n) {
  return sizeof(arena_node<call_stmt>) + size_of(n.inputs) + size_of(n.outputs) + size_of(n.scalars);
}
std::size_t size_of(const copy_stmt& n) {
  return sizeof(arena_node<copy_stmt>) + size_of(n.src_x) + size_of(n.dst_x);
}
std::size_t size_of(const allocate& n) { return sizeof(arena_node<allocate>) + size_of(n.dims); }
std::size_t size_of(const make_buffer& n) { return sizeof(arena_node<make_buffer>) + size_of(n.dims); }
std::size_t size_of(const crop_buffer& n) { return sizeof(arena_node<crop_buffer>) + size_of(n.bounds); }
std::size_t size_of(const slice_buffer& n) { return sizeof(arena_node<slice_buffer>) + size_of(n.at); }
std::size_t size_of(const transpose& n) { return sizeof(arena_node<transpose>) + size_of(n.dims); }

// Copy `src` into `storage`, and advance `storage` past it.
template <typename T>
span<T> make_span(void*& storage, span<T> src) {
  T* result = static_cast<T*>(storage);
  for (std::size_t i = 0; i < src.size(); ++i) {
    new (result + i) T(src[i]);
  }
  storage = static_cast<char*>(storage) + size_of(src);
  return span<T>(result, src.size());
}

template <typename T>
const T* clone_into(void* mem, const T& n) {
  return new (mem) arena_node<T>(n);
}
const let* clone_into(void* mem, const let& n) {
  auto result = new (mem) arena_node<let>(n);
  void* arrays = result + 1;
  result->lets = make_span(arrays, n.lets);
  return result;
}
const call* clone_into(void* mem, const call& n) {
  auto result = new (mem) arena_node<call>(n);
  void* arrays = result + 1;
  result->args = make_span(arrays, n.args);
  return result;
}
const let_stmt* clone_into(void* mem, const let_stmt& n) {
  auto result = new (mem) arena_node<let_stmt>(n);
  void* arrays = result + 1;
  result->lets = make_span(arrays, n.lets);
  return result;
}
const block* clone_into(void* mem, const block& n) {
  auto result = new (mem) arena_node<block>(n);
  void* arrays = result + 1;
  result->stmts = make_span(arrays, n.stmts);
  return result;
}
const call_stmt* clone_into(void* mem, const call_stmt& n) {
  auto result = new (mem) arena_node<call_stmt>(n);
  void* arrays = result + 1;
  result->inputs = make_span(arrays, n.inputs);
  result->outputs = make_span(arrays, n.outputs);
  result->scalars = make_span(arrays, n.scalars);
  return result;
}
const copy_stmt* clone_into(void* mem, const copy_stmt& n) {
  auto result = new (mem) arena_node<copy_stmt>(n);
  void* arrays = result + 1;
  result->src_x = make_span(arrays, n.src_x);
  result->dst_x = make_span(arrays, n.dst_x);
  return result;
}
const allocate* clone_into(void* mem, const allocate& n) {
  auto result = new (mem) arena_node<allocate>(n);
  void* arrays = result + 1;
  result->dims = make_span(arrays, n.dims);
  return result;
}
const make_buffer* clone_into(void* mem, const make_buffer& n) {
  auto result = new (mem) arena_node<make_buffer>(n);
  void* arrays = result + 1;
  result->dims = make_span(arrays, n.dims);
  return result;
}
const crop_buffer* clone_into(void* mem, const crop_buffer& n) {
  auto result = new (mem) arena_node<crop_buffer>(n);
  void* arrays = result + 1;
  result->bounds = make_span(arrays, n.bounds);
  return result;
}
const slice_buffer* clone_into(void* mem, const slice_buffer& n) {
  auto result = new (mem) arena_node<slice_buffer>(n);
  void* arrays = result + 1;
  result->at = make_span(arrays, n.at);
  return result;
}
const transpose* clone_into(void* mem, const transpose& n) {
  auto result = new (mem) arena_node<transpose>(n);
  void* arrays = result + 1;
  result->dims = make_span(arrays, n.dims);
  return result;
}

class node_compactor : public node_mutator {
  // The current arena we are building.
  slinky::ref_count<arena> arena_;

  std::unordered_map<const base_expr_node*, expr> exprs_;
  std::unordered_map<const base_stmt_node*, stmt> stmts_;

  // Reserve space in the current arena, making a new arena if needed.
  void* reserve(std::size_t size, std::size_t alignment) {
    void* result = arena_ != nullptr ? arena_->allocate(size, alignment) : nullptr;
    if (!result) {
      slinky::ref_count<arena> chunk = arena::make();
      result = chunk->allocate(size, alignment);
      if (!result) return nullptr;
      arena_ = std::move(chunk);
    }
    return result;
  }

  expr take_result(const base_expr_node*) { return std::move(const_cast<expr&>(mutated_expr())); }
  stmt take_result(const base_stmt_node*) { return std::move(const_cast<stmt&>(mutated_stmt())); }

  template <typename T>
  void compact(const T* op) {
    // Reserve space in the arena for this node, and remember the arena this node belongs to.
    std::size_t size = size_of(*op);
    void* mem = reserve(size, alignof(arena_node<T>));
    slinky::ref_count<arena> chunk = arena_;

    node_mutator::visit(op);

    if (!mem) {
      // This node is too big for a chunk, just use the node we got.
      return;
    }

    // Make a clone of the node in the reserved arena allocation, and use that as the result.
    auto mutated = take_result(op);
    const T* cloneable = mutated.template as<T>();
    assert(cloneable);
    assert(size_of(*cloneable) == size);
    set_result(decltype(mutated)(clone_into(mem, *cloneable)));
  }

public:
  using node_mutator::mutate;
  expr mutate(const expr& e) override {
    if (!e.defined()) return e;
    // `variable` and `constant` are stored inline in the `expr` itself, not as nodes. There is nothing to move into
    // an arena, and no address to memoize them by -- `get()` is null for all of them, so they would all collide on
    // the same cache entry.
    const base_expr_node* key = e.get();
    if (!key) return e;
    auto i = exprs_.find(key);
    if (i != exprs_.end()) return i->second;

    expr result = node_mutator::mutate(e);
    exprs_[key] = result;
    return result;
  }

  stmt mutate(const stmt& s) override {
    if (!s.defined()) return s;
    auto i = stmts_.find(s.get());
    if (i != stmts_.end()) return i->second;

    stmt result = stmt_mutator::mutate(s);
    stmts_[s.get()] = result;
    return result;
  }

  void visit(const constant_buffer* op) override { compact(op); }
  void visit(const let* op) override { compact(op); }
  void visit(const add* op) override { compact(op); }
  void visit(const sub* op) override { compact(op); }
  void visit(const mul* op) override { compact(op); }
  void visit(const div* op) override { compact(op); }
  void visit(const mod* op) override { compact(op); }
  void visit(const class min* op) override { compact(op); }
  void visit(const class max* op) override { compact(op); }
  void visit(const equal* op) override { compact(op); }
  void visit(const not_equal* op) override { compact(op); }
  void visit(const less* op) override { compact(op); }
  void visit(const less_equal* op) override { compact(op); }
  void visit(const logical_and* op) override { compact(op); }
  void visit(const logical_or* op) override { compact(op); }
  void visit(const logical_not* op) override { compact(op); }
  void visit(const class select* op) override { compact(op); }
  void visit(const call* op) override { compact(op); }

  void visit(const let_stmt* op) override { compact(op); }
  void visit(const block* op) override { compact(op); }
  void visit(const loop* op) override { compact(op); }
  void visit(const call_stmt* op) override { compact(op); }
  void visit(const copy_stmt* op) override { compact(op); }
  void visit(const allocate* op) override { compact(op); }
  void visit(const make_buffer* op) override { compact(op); }
  void visit(const clone_buffer* op) override { compact(op); }
  void visit(const crop_buffer* op) override { compact(op); }
  void visit(const crop_dim* op) override { compact(op); }
  void visit(const slice_buffer* op) override { compact(op); }
  void visit(const slice_dim* op) override { compact(op); }
  void visit(const transpose* op) override { compact(op); }
  void visit(const async* op) override { compact(op); }
  void visit(const check* op) override { compact(op); }
};

}  // namespace

stmt compact_nodes(const stmt& s) {
  if (!s.defined()) return s;

  return node_compactor().mutate(s);
}

}  // namespace slinky
