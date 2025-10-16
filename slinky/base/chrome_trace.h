#ifndef SLINKY_BASE_CHROME_TRACE_H
#define SLINKY_BASE_CHROME_TRACE_H

#include <chrono>
#include <ctime>
#include <iostream>
#include <mutex>

namespace slinky {

// A minimal wrapper for generating chrome trace files:
// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
class chrome_trace {
  std::ostream& os_;
  std::mutex mtx_;
  using timestamp = std::chrono::high_resolution_clock::time_point;
  timestamp t0_;
  std::clock_t cpu_t0_;

  void write_event(const char* name, const char* cat, char type);

public:
  chrome_trace(std::ostream& os);
  ~chrome_trace();

  void begin(const char* name);
  void end(const char* name);

  // Return the global instance of tracing, or nullptr if none. Trace files will be written to the path in the
  // `SLINKY_TRACE` environment variable.
  static chrome_trace* global();
};

// Call `chrome_trace::begin` upon construction, and `chrome_trace::end` upon destruction.
class scoped_trace {
  chrome_trace* trace;
  const char* name;

public:
  scoped_trace(chrome_trace* trace, const char* name) : trace(trace), name(name) {
    if (trace) {
      trace->begin(name);
    }
  }
  scoped_trace(const char* name) : scoped_trace(chrome_trace::global(), name) {}

  ~scoped_trace() {
    if (trace) {
      trace->end(name);
    }
  }
};

}  // namespace slinky

#endif  // SLINKY_BASE_CHROME_TRACE_H
