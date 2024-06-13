#include "base/chrome_trace.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <thread>

namespace slinky {

chrome_trace::chrome_trace(std::ostream& os) : os_(os) {
  os_ << "[{}";
  t0_ = std::chrono::high_resolution_clock::now();
}
chrome_trace::~chrome_trace() { os_ << "]\n"; }

void chrome_trace::write_event(const char* name, const char* cat, char type) {
  auto t = std::chrono::high_resolution_clock::now();
  auto ts = std::chrono::duration_cast<std::chrono::microseconds>(t - t0_).count();

  // The only way to convert a thread ID to a string is via operator<<, which is slow, so we do it as a thread_local
  // initializer.
  thread_local std::string tid_str = []() {
    auto tid = std::this_thread::get_id();
    std::stringstream ss;
    ss << tid;
    return ss.str();
  }();

  // Assemble a buffer of what we want to write.
  thread_local std::string buffer;
  buffer.clear();
  // It would be an error to put a comma here as the first item in the output, but we put a dummy {} object at the
  // beginning of the array.
  buffer += ",\n{\"name\":\"";
  buffer += name;
  buffer += "\",\"cat\":\"";
  buffer += cat;
  buffer += "\",\"ph\":\"";
  buffer += type;
  buffer += "\",\"pid\":0,\"tid\":";
  buffer += tid_str;
  buffer += ",\"ts\":";
  buffer += std::to_string(ts);
  buffer += '}';

  // Write our buffer.
  std::unique_lock l(mtx_);
  os_.write(buffer.data(), buffer.size());
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
