#include "base/chrome_trace.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <thread>

namespace slinky {

chrome_trace::chrome_trace(std::ostream& os) : os_(os), event_written_(false) {
  os_ << "[";
  t0_ = std::chrono::high_resolution_clock::now();
}
chrome_trace::~chrome_trace() { os_ << "]\n"; }

void chrome_trace::write_event(const char* name, const char* cat, char type) {
  auto tid = std::this_thread::get_id();
  auto t = std::chrono::high_resolution_clock::now();
  auto ts = std::chrono::duration_cast<std::chrono::microseconds>(t - t0_).count();
  std::unique_lock l(mtx_);
  if (event_written_) {
    os_ << ",\n";
  }
  event_written_ = true;
  os_ << "{\"name\":\"" << name << "\",\"cat\":\"" << cat << "\",\"ph\":\"" << type << "\",\"pid\":0,\"tid\":" << tid
      << ",\"ts\":" << ts << "}";
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
