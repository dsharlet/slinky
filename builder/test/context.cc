#include "builder/test/context.h"

#include <fstream>
#include <sstream>

#include "base/chrome_trace.h"
#include "base/thread_pool.h"

namespace slinky {

void setup_tracing(eval_context& ctx, const std::string& filename) {
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

  ctx.trace_begin = [t](const char* op) -> index_t {
    t->trace.begin(op);
    // chrome_trace expects trace_begin and trace_end to pass the string, while slinky's API expects to pass a token to
    // trace_end returned by trace_begin. Because `index_t` must be able to hold a pointer, we'll just use the token to
    // store the pointer.
    return reinterpret_cast<index_t>(op);
  };
  ctx.trace_end = [t](index_t token) { t->trace.end(reinterpret_cast<const char*>(token)); };
}

test_context::test_context() {
  static thread_pool threads;

  allocate = [this](var, raw_buffer* b) {
    void* allocation = b->allocate();
    heap.track_allocate(b->size_bytes());
    return allocation;
  };
  free = [this](var, raw_buffer* b, void* allocation) {
    ::free(allocation);
    heap.track_free(b->size_bytes());
  };

  enqueue_many = [&](thread_pool::task t) { threads.enqueue(threads.thread_count(), std::move(t)); };
  enqueue = [&](int n, thread_pool::task t) { threads.enqueue(n, std::move(t)); };
  wait_for = [&](const std::function<bool()>& condition) { return threads.wait_for(condition); };
  atomic_call = [&](const thread_pool::task& t) { return threads.atomic_call(t); };
}

}  // namespace slinky
