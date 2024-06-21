#include "base/chrome_trace.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <thread>

namespace slinky {

namespace {

std::atomic<int> next_id = 0;

}  // namespace

chrome_trace::chrome_trace(std::ostream& os) : os_(os), id_(next_id++) {
  os_ << "[{\"name\":\"chrome_trace\",\"cat\":\"slinky\",\"ph\":\"B\",\"pid\":0,\"tid\":0,\"ts\":0}";
  os_ << ",\n{\"name\":\"chrome_trace\",\"cat\":\"slinky\",\"ph\":\"E\",\"pid\":0,\"tid\":0,\"ts\":0}";
  t0_ = std::chrono::high_resolution_clock::now();
}
chrome_trace::~chrome_trace() {
  // Flush any unwritten buffers.
  for (const auto& i : buffers_) {
    os_ << i.second;
  }
  os_ << "]\n";
}

void chrome_trace::write_event(const char* name, const char* cat, char type) {
  auto t = std::chrono::high_resolution_clock::now();
  auto ts = std::chrono::duration_cast<std::chrono::microseconds>(t - t0_).count();

  // The only way to convert a thread ID to a string is via operator<<, which is slow, so we do it as a thread_local
  // initializer.
  thread_local std::string pid_tid_str = []() {
    std::stringstream ss;
    ss << "\",\"pid\":0,\"tid\":\"" << std::this_thread::get_id() << "\"";
    return ss.str();
  }();

  // To avoid overhead when multiple threads are writing traces, we try to keep a pointer to the buffer for this trace
  // object and thread cached locally.
  thread_local int current_buffer_owner = -1;
  thread_local std::string* buffer = nullptr;

  if (!buffer || current_buffer_owner != id_) {
    // We should only need to pay the cost of this lookup if multiple different chrome_trace objects are writing traces
    // on the same thread at the same time.
    std::unique_lock l(mtx_);
    buffer = &buffers_[std::this_thread::get_id()];
    current_buffer_owner = id_;
  }

  // It would be an error to put a comma here as the first item in the output, but we put a dummy {} object at the
  // beginning of the array.
  *buffer += ",\n{\"name\":\"";
  *buffer += name;
  *buffer += "\",\"cat\":\"";
  *buffer += cat;
  *buffer += "\",\"ph\":\"";
  *buffer += type;
  *buffer += pid_tid_str;
  *buffer += ",\"ts\":";
  *buffer += std::to_string(ts);
  *buffer += '}';

  if (buffer->size() > 4096 * 16) {
    // Flush our buffer.
    std::unique_lock l(mtx_);
    os_.write(buffer->data(), buffer->size());
    buffer->clear();
  }
}

void chrome_trace::begin(const char* name) { write_event(name, "slinky", 'B'); }
void chrome_trace::end(const char* name) { write_event(name, "slinky", 'E'); }

chrome_trace* chrome_trace::global() {
  static std::unique_ptr<std::ofstream> file;
  static std::unique_ptr<chrome_trace> trace;
  if (!trace) {
    const char* path = getenv("SLINKY_TRACE");
    if (path) {
      std::cout << "Tracing to: " << path << std::endl;
      file = std::make_unique<std::ofstream>(path);
      trace = std::make_unique<chrome_trace>(*file);
    }
  }
  return trace.get();
}

}  // namespace slinky
