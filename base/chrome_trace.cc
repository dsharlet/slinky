#include "base/chrome_trace.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <thread>

namespace slinky {

chrome_trace::chrome_trace(std::ostream& os) : os_(os), event_written_(false) {
  os_ << "[";
  t0_ = std::chrono::high_resolution_clock::now();
}
chrome_trace::~chrome_trace() { os_ << "]\n"; }

void chrome_trace::write_event(const char* name, const char* cat, char type) {
  auto t = std::chrono::high_resolution_clock::now();
  auto ts = std::chrono::duration_cast<std::chrono::microseconds>(t - t0_).count();

  // The only way to convert a thread ID to a string is via operator<<, which is slow, so do it once as a thread local.
  thread_local std::string tid_str = []() {
    auto tid = std::this_thread::get_id();
    std::stringstream ss;
    ss << tid;
    return ss.str();
  }();

  // The idea here is to assemble a buffer of what we want to write, then lock the mutex, and write our buffer.
  thread_local std::string buffer;
  buffer.clear();
  buffer += "{\"name\":\"";
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

  std::unique_lock l(mtx_);
  if (event_written_) {
    os_ << ",\n";
  }
  event_written_ = true;
  os_ << buffer;
}

void chrome_trace::begin(const char* name) { write_event(name, "stmt", 'B'); }
void chrome_trace::end(const char* name) { write_event(name, "stmt", 'E'); }

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
