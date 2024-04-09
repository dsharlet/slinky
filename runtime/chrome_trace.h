#ifndef SLINKY_RUNTIME_CHROME_TRACE_H
#define SLINKY_RUNTIME_CHROME_TRACE_H

#include <iostream>
#include <chrono>
#include <mutex>
#include <string>

namespace slinky {
 
// A minimal wrapper for generating chrome trace files:
// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
class chrome_trace {
  std::ostream& os_;
  bool event_written_;
  // Note that using a mutex to avoid races in the output possibly distorts the tracing output.
  // Not easy to avoid this.
  std::mutex mtx_;
  using timestamp = std::chrono::high_resolution_clock::time_point;
  timestamp t0_;

  void write_event(const char* name, const char* cat, char type);

public:
  chrome_trace(std::ostream& os);
  ~chrome_trace();

  void begin(const char* name);
  void end(const char* name);
};

}  // namespace slinky

#endif  // SLINKY_RUNTIME_CHROME_TRACE_H
