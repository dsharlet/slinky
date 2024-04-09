#include <chrono>
#include <cstdint>
#include <thread>

#include "runtime/chrome_trace.h"

namespace slinky {

chrome_trace::chrome_trace(std::ostream& os) : os_(os), event_written_(false) {
  os_ << "[";
  t0_ = std::chrono::high_resolution_clock::now();
}
chrome_trace::~chrome_trace() { os_ << "]\n"; }

void chrome_trace::write_event(const char* name, const char* cat, char type) {
  auto tid = std::this_thread::get_id();
  auto t = std::chrono::high_resolution_clock::now();
  auto ts = std::chrono::duration_cast<std::chrono::nanoseconds>(t - t0_).count() / 1000.0;
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

}  // namespace slinky
