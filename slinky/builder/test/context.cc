#include "slinky/builder/test/context.h"

#include <fstream>
#include <sstream>

#include "slinky/base/chrome_trace.h"
#include "slinky/base/thread_pool_impl.h"
#include "slinky/runtime/buffer.h"

namespace slinky {

void setup_tracing(eval_config& cfg, const std::string& filename) {
  struct tracer {
    std::string trace_file;
    // Store the trace in a stringstream and write it at the end, to avoid overhead influencing the trace.
    std::stringstream buffer;
    chrome_trace trace;

    tracer(const std::string& filename) : trace_file(filename), trace(buffer) {}
    ~tracer() {
      std::ofstream file(trace_file);
      file << buffer.str();
    }
  };

  auto t = std::make_shared<tracer>(filename);

  cfg.trace_begin = [t](const char* op) -> index_t {
    t->trace.begin(op);
    // chrome_trace expects trace_begin and trace_end to pass the string, while slinky's API expects to pass a token to
    // trace_end returned by trace_begin. Because `index_t` must be able to hold a pointer, we'll just use the token to
    // store the pointer.
    return reinterpret_cast<index_t>(op);
  };
  cfg.trace_end = [t](index_t token) { t->trace.end(reinterpret_cast<const char*>(token)); };
}

test_context::test_context() {
  static thread_pool_impl threads;

  config.allocate = [this](var, raw_buffer* b) {
    void* allocation = b->allocate();
    heap.track_allocate(b->size_bytes());
    return allocation;
  };
  config.free = [this](var, raw_buffer* b, void* allocation) {
    ::free(allocation);
    heap.track_free(b->size_bytes());
  };

  copy = [this](const raw_buffer& src, const raw_buffer& dst, const raw_buffer& pad) {
    ++copy_calls;
    copy_elements += dst.elem_count();
    slinky::copy(src, dst, pad);
  };

  config.thread_pool = &threads;

  // Many tests count the number of allocations that go on the heap, don't let memory_type::automatic corrupt those
  // counts.
  config.auto_stack_threshold = 0;

  eval_context::config = &config;
}

}  // namespace slinky
