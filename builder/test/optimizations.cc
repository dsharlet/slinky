#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cassert>

#include "builder/optimizations.h"
#include "builder/substitute.h"
#include "runtime/expr.h"
#include "runtime/print.h"

namespace slinky {

namespace {

node_context symbols;

var x(symbols, "x");
var y(symbols, "y");
var z(symbols, "z");
var w(symbols, "w");
var u(symbols, "u");
var v(symbols, "v");

MATCHER_P(matches, expected, "") { return match(arg, expected); }

class node_counter : public node_mutator {
public:
  std::set<const void*> visited;

  expr mutate(const expr& e) override {
    if (visited.count(e.get())) return e;
    visited.insert(e.get());
    return node_mutator::mutate(e);
  }
  stmt mutate(const stmt& s) override {
    if (visited.count(s.get())) return s;
    visited.insert(s.get());
    return node_mutator::mutate(s);
  }
  using node_mutator::mutate;
};

int count_unique_nodes(const stmt& s) {
  node_counter c;
  c.mutate(s);
  return c.visited.size();
}
int count_unique_nodes(const expr& s) {
  node_counter c;
  c.mutate(s);
  return c.visited.size();
}

MATCHER_P(unique_node_count_is, n, "") { return count_unique_nodes(arg) == n; }

}  // namespace

TEST(optimizations, fuse_siblings) {
  auto use_buffer = [](var x) { return call_stmt::make(nullptr, {}, {x}, {}); };

  ASSERT_THAT(fuse_siblings(block::make({
                  allocate::make(x, memory_type::heap, 1, {}, use_buffer(x)),
                  allocate::make(y, memory_type::heap, 1, {}, use_buffer(y)),
              })),
      matches(allocate::make(x, memory_type::heap, 1, {}, block::make({use_buffer(x), use_buffer(x)}))));

  ASSERT_THAT(fuse_siblings(block::make({
                  allocate::make(x, memory_type::heap, 1, {}, use_buffer(x)),
                  allocate::make(y, memory_type::heap, 2, {}, use_buffer(y)),
              })),
      matches(fuse_siblings(block::make({
          allocate::make(x, memory_type::heap, 1, {}, use_buffer(x)),
          allocate::make(y, memory_type::heap, 2, {}, use_buffer(y)),
      }))));

  ASSERT_THAT(fuse_siblings(block::make({
                  allocate::make(x, memory_type::heap, 1, {}, use_buffer(x)),
                  allocate::make(y, memory_type::stack, 1, {}, use_buffer(y)),
              })),
      matches(fuse_siblings(block::make({
          allocate::make(x, memory_type::heap, 1, {}, use_buffer(x)),
          allocate::make(y, memory_type::stack, 1, {}, use_buffer(y)),
      }))));

  ASSERT_THAT(fuse_siblings(block::make({
                  allocate::make(x, memory_type::heap, 1, {{}}, use_buffer(x)),
                  allocate::make(y, memory_type::heap, 1, {}, use_buffer(y)),
              })),
      matches(fuse_siblings(block::make({
          allocate::make(x, memory_type::heap, 1, {{}}, use_buffer(x)),
          allocate::make(y, memory_type::heap, 1, {}, use_buffer(y)),
      }))));

  ASSERT_THAT(fuse_siblings(block::make({
                  allocate::make(x, memory_type::heap, 1, {}, use_buffer(x)),
                  use_buffer(z),
                  allocate::make(y, memory_type::heap, 1, {}, use_buffer(y)),
              })),
      matches(block::make({
          allocate::make(x, memory_type::heap, 1, {}, use_buffer(x)),
          use_buffer(z),
          allocate::make(y, memory_type::heap, 1, {}, use_buffer(y)),
      })));
  ASSERT_THAT(fuse_siblings(block::make({
                  crop_dim::make(x, y, 0, {0, 10}, crop_dim::make(z, x, 1, {0, 10}, use_buffer(z))),
                  crop_dim::make(z, y, 0, {0, 10}, crop_dim::make(w, z, 1, {0, 10}, use_buffer(w))),
              })),
      matches(crop_dim::make(
          x, y, 0, {0, 10}, crop_dim::make(z, x, 1, {0, 10}, block::make({use_buffer(z), use_buffer(z)})))));
}

TEST(optimizations, remove_pure_dims) {
  {
    // Test support for crop_dim.
    stmt call = call_stmt::make(nullptr, {y}, {y}, {.min_rank = 0});
    stmt crop = crop_dim::make(y, z, 1, {0, 0}, call);

    stmt result = remove_pure_dims(crop);
    ASSERT_THAT(
        result, matches(crop_dim::make(y, z, 1, {0, 0}, slice_buffer::make(y, y, {expr{}, buffer_min(y, 1)}, call))));
  }

  {
    // Test support for crop_buffer.
    stmt call = call_stmt::make(nullptr, {y}, {y}, {.min_rank = 0});
    stmt crop = crop_buffer::make(y, z, {{}, {0, 0}}, call);

    stmt result = remove_pure_dims(crop);
    ASSERT_THAT(
        result, matches(crop_buffer::make(y, z, {{}, {0, 0}}, slice_buffer::make(y, y, {{}, buffer_min(y, 1)}, call))));
  }

  {
    // Test support for allocate.
    stmt call = call_stmt::make(nullptr, {y}, {y}, {.min_rank = 0});
    stmt alloc = allocate::make(y, memory_type::heap, 1, {{interval_expr{0, 5}}, {interval_expr{0, 0}}}, call);

    stmt result = remove_pure_dims(alloc);
    ASSERT_THAT(result, matches(allocate::make(y, memory_type::heap, 1, {{interval_expr{0, 5}}, {interval_expr{0, 0}}},
                            slice_buffer::make(y, y, {{}, buffer_min(y, 1)}, call))));
  }

  {
    // Test slicing inputs as well as outputs.
    stmt call = call_stmt::make(nullptr, {x}, {y}, {.min_rank = 0});
    stmt crop = crop_dim::make(y, z, 1, {0, 0}, call);

    stmt result = remove_pure_dims(crop);
    ASSERT_THAT(result, matches(crop_dim::make(y, z, 1, {0, 0},
                            slice_buffer::make(x, x, {expr{}, buffer_min(y, 1)},
                                slice_buffer::make(y, y, {expr{}, buffer_min(y, 1)}, call)))));
  }

  {
    // Test propagation of sliceable dims through clone_buffer.
    stmt call = call_stmt::make(nullptr, {y}, {y}, {.min_rank = 0});
    stmt alloc = allocate::make(
        z, memory_type::heap, 1, {{interval_expr{0, 5}}, {interval_expr{0, 0}}}, clone_buffer::make(y, z, call));

    stmt result = remove_pure_dims(alloc);
    ASSERT_THAT(result, matches(allocate::make(z, memory_type::heap, 1, {{interval_expr{0, 5}}, {interval_expr{0, 0}}},
                            clone_buffer::make(y, z, slice_buffer::make(y, y, {{}, buffer_min(y, 1)}, call)))));
  }

  {
    // Test propagation of sliceable dims through transpose.
    stmt call = call_stmt::make(nullptr, {y}, {y}, {.min_rank = 0});
    stmt alloc = allocate::make(
        z, memory_type::heap, 1, {{interval_expr{0, 5}}, {interval_expr{0, 0}}}, transpose::make(y, z, {1, 0}, call));

    stmt result = remove_pure_dims(alloc);
    ASSERT_THAT(result, matches(allocate::make(z, memory_type::heap, 1, {{interval_expr{0, 5}}, {interval_expr{0, 0}}},
                            transpose::make(y, z, {1, 0}, slice_buffer::make(y, y, {buffer_min(y, 0)}, call)))));
  }

  {
    // Test slicing more than one dimension.
    stmt call = call_stmt::make(nullptr, {y}, {y}, {.min_rank = 0});
    stmt crop1 = crop_dim::make(z, w, 1, {0, 0}, crop_dim::make(y, z, 2, {0, 0}, call));

    stmt result = remove_pure_dims(crop1);
    ASSERT_THAT(result, matches(crop_dim::make(z, w, 1, {0, 0},
                            crop_dim::make(y, z, 2, {0, 0},
                                slice_buffer::make(y, y, {expr{}, buffer_min(y, 1), buffer_min(y, 2)}, call)))));
  }

  {
    // Test slicing the correct dimension under shadowing.
    stmt call = call_stmt::make(nullptr, {y}, {y}, {.min_rank = 0});
    stmt crop1 = crop_dim::make(y, z, 1, {0, 0}, crop_dim::make(y, z, 2, {0, 0}, call));

    stmt result = remove_pure_dims(crop1);
    ASSERT_THAT(result,
        matches(crop_dim::make(y, z, 1, {0, 0},
            crop_dim::make(y, z, 2, {0, 0}, slice_buffer::make(y, y, {expr{}, expr{}, buffer_min(y, 2)}, call)))));
  }

  {
    // Test slicing two dimensions under shadowing and overlaping input/output.
    stmt call = call_stmt::make(nullptr, {x}, {x}, {.min_rank = 0});
    stmt crop1 = crop_dim::make(x, x, 1, {0, 0}, crop_dim::make(x, x, 2, {0, 0}, call));
    stmt result = remove_pure_dims(crop1);
    ASSERT_THAT(result, matches(crop_dim::make(x, x, 1, {0, 0},
                            crop_dim::make(x, x, 2, {0, 0},
                                slice_buffer::make(x, x, {expr{}, buffer_min(x, 1), buffer_min(x, 2)}, call)))));
  }
}

TEST(optimizations, optimize_symbols) {
  auto make_dummy_decl = [](var x, stmt body) { return allocate::make(x, memory_type::heap, 1, {}, body); };

  {
    // We don't know about x, we can't mutate it.
    node_context ctx = symbols;
    ASSERT_THAT(optimize_symbols(crop_dim::make(y, x, 0, {0, 0}, check::make(y)), ctx),
        matches(crop_dim::make(y, x, 0, {0, 0}, check::make(y))));
  }

  {
    // We know about x, we can mutate it.
    node_context ctx = symbols;
    ASSERT_THAT(optimize_symbols(make_dummy_decl(x, crop_dim::make(y, x, 0, {0, 0}, check::make(y))), ctx),
        matches(make_dummy_decl(x, crop_dim::make(x, x, 0, {0, 0}, check::make(x)))));
  }

  {
    node_context ctx = symbols;
    ASSERT_THAT(
        optimize_symbols(
            make_dummy_decl(x, crop_dim::make(y, x, 0, {0, 0}, crop_dim::make(z, y, 0, {0, 0}, check::make(z)))), ctx),
        matches(make_dummy_decl(x, crop_dim::make(x, x, 0, {0, 0}, crop_dim::make(x, x, 0, {0, 0}, check::make(x))))));
  }

  {
    node_context ctx = symbols;
    ASSERT_THAT(optimize_symbols(make_dummy_decl(y, crop_dim::make(x, y, 0, {0, 0}, check::make(y))), ctx),
        matches(make_dummy_decl(y, crop_dim::make(x, y, 0, {0, 0}, check::make(y)))));
  }
}

TEST(optimizations, deshadow_speed) {
  node_context ctx = symbols;
  stmt s = call_stmt::make(nullptr, {x}, {y}, {});
  for (int i = 0; i < 1000; ++i) {
    s = crop_dim::make(y, y, 0, {0, 0}, s);
  }
  stmt s2 = deshadow(s, {}, ctx);
}

TEST(optimizations, canonicalize_nodes) {
  ASSERT_THAT(canonicalize_nodes(x + x), unique_node_count_is(2));
  ASSERT_THAT(canonicalize_nodes(x + y), unique_node_count_is(3));
  ASSERT_THAT(canonicalize_nodes(
                  block::make({copy_stmt::make(x, {z, w}, y, {z, w}, {}), copy_stmt::make(x, {z, w}, y, {z, w}, {})})),
      unique_node_count_is(4));
  ASSERT_THAT(
      canonicalize_nodes(block::make({call_stmt::make(nullptr, {x}, {y}, {}), call_stmt::make(nullptr, {x}, {y}, {})})),
      unique_node_count_is(3));
}

TEST(optimizations, parallelize_tasks) {
  // Two independent consumers.
  ASSERT_THAT(parallelize_tasks(block::make({
                  call_stmt::make(nullptr, {x}, {y}, {}),
                  call_stmt::make(nullptr, {x}, {z}, {}),
              })),
      matches(async::make(var(), call_stmt::make(nullptr, {x}, {y}, {}), call_stmt::make(nullptr, {x}, {z}, {}))));

  // One producer, one consumer.
  ASSERT_THAT(parallelize_tasks(block::make({
                  call_stmt::make(nullptr, {x}, {y}, {}),
                  call_stmt::make(nullptr, {y}, {z}, {}),
              })),
      matches(block::make({
          call_stmt::make(nullptr, {x}, {y}, {}),
          call_stmt::make(nullptr, {y}, {z}, {}),
      })));

  // One producer, one consumer computed in-place.
  ASSERT_THAT(parallelize_tasks(block::make({
                  call_stmt::make(nullptr, {x}, {y}, {}),
                  call_stmt::make(nullptr, {y}, {y}, {}),
              })),
      matches(block::make({
          call_stmt::make(nullptr, {x}, {y}, {}),
          call_stmt::make(nullptr, {y}, {y}, {}),
      })));

  // One producer, two independent consumers.
  ASSERT_THAT(parallelize_tasks(block::make({
                  call_stmt::make(nullptr, {x}, {y}, {}),
                  call_stmt::make(nullptr, {y}, {z}, {}),
                  call_stmt::make(nullptr, {y}, {w}, {}),
              })),
      matches(block::make({
          call_stmt::make(nullptr, {x}, {y}, {}),
          async::make(var(), call_stmt::make(nullptr, {y}, {z}, {}), call_stmt::make(nullptr, {y}, {w}, {})),
      })));
  // One producer, three independent consumers.
  ASSERT_THAT(parallelize_tasks(block::make({
                  call_stmt::make(nullptr, {x}, {y}, {}),
                  call_stmt::make(nullptr, {y}, {z}, {}),
                  call_stmt::make(nullptr, {y}, {w}, {}),
                  call_stmt::make(nullptr, {y}, {u}, {}),
              })),
      matches(block::make({
          call_stmt::make(nullptr, {x}, {y}, {}),
          async::make(var(), call_stmt::make(nullptr, {y}, {z}, {}),
              async::make(var(), call_stmt::make(nullptr, {y}, {w}, {}), call_stmt::make(nullptr, {y}, {u}, {}))),
      })));

  // A check that both independent consumers depends on.
  ASSERT_THAT(parallelize_tasks(block::make({
                  check::make(buffer_min(x, 0) < buffer_max(x, 0)),
                  call_stmt::make(nullptr, {x}, {y}, {}),
                  call_stmt::make(nullptr, {x}, {z}, {}),
              })),
      matches(block::make({
          check::make(buffer_min(x, 0) < buffer_max(x, 0)),
          async::make(var(), call_stmt::make(nullptr, {x}, {y}, {}), call_stmt::make(nullptr, {x}, {z}, {})),
      })));

  // A producer-consumer, with an unrelated stage.
  ASSERT_THAT(parallelize_tasks(block::make({
                  call_stmt::make(nullptr, {x}, {y}, {}),
                  call_stmt::make(nullptr, {y}, {z}, {}),
                  call_stmt::make(nullptr, {x}, {w}, {}),
              })),
      matches(block::make({
          async::make(var(),
              block::make({
                  call_stmt::make(nullptr, {x}, {y}, {}),
                  call_stmt::make(nullptr, {y}, {z}, {}),
              }),
              call_stmt::make(nullptr, {x}, {w}, {})),
      })));

  // Two unrelated producer-consumers.
  ASSERT_THAT(parallelize_tasks(block::make({
                  call_stmt::make(nullptr, {x}, {y}, {}),
                  call_stmt::make(nullptr, {y}, {z}, {}),
                  call_stmt::make(nullptr, {x}, {w}, {}),
                  call_stmt::make(nullptr, {w}, {u}, {}),
              })),
      matches(async::make(var(),
          block::make({
              call_stmt::make(nullptr, {x}, {y}, {}),
              call_stmt::make(nullptr, {y}, {z}, {}),
          }),
          block::make({
              call_stmt::make(nullptr, {x}, {w}, {}),
              call_stmt::make(nullptr, {w}, {u}, {}),
          }))));

  // Two unrelated producer-consumers, where both consumers are consumed by the same stage.
  ASSERT_THAT(parallelize_tasks(block::make({
                  call_stmt::make(nullptr, {x}, {y}, {}),
                  call_stmt::make(nullptr, {y}, {z}, {}),
                  call_stmt::make(nullptr, {x}, {w}, {}),
                  call_stmt::make(nullptr, {w}, {u}, {}),
                  call_stmt::make(nullptr, {z, u}, {v}, {}),
              })),
      matches(block::make({
          async::make(var(),
              block::make({
                  call_stmt::make(nullptr, {x}, {y}, {}),
                  call_stmt::make(nullptr, {y}, {z}, {}),
              }),
              block::make({
                  call_stmt::make(nullptr, {x}, {w}, {}),
                  call_stmt::make(nullptr, {w}, {u}, {}),
              })),
          call_stmt::make(nullptr, {z, u}, {v}, {}),
      })));

  // One producer, two independent consumers, one of which is computed in-place.
  ASSERT_THAT(parallelize_tasks(block::make({
                  call_stmt::make(nullptr, {x}, {y}, {}),
                  call_stmt::make(nullptr, {y}, {z}, {}),
                  call_stmt::make(nullptr, {y}, {y}, {}),
              })),
      matches(block::make({
          call_stmt::make(nullptr, {x}, {y}, {}),
          call_stmt::make(nullptr, {y}, {z}, {}),
          call_stmt::make(nullptr, {y}, {y}, {}),
      })));
}

}  // namespace slinky
