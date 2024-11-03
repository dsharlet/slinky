#include "base/chrome_trace.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <thread>

namespace slinky {

namespace {

// Unfortunately, std::clock returns the CPU time for the whole process, not the current thread.
std::clock_t clock_per_thread_us() {
  #ifdef _MSC_VER
  // CLOCK_THREAD_CPUTIME_ID not defined under MSVC
  return 0;
  #else
  timespec t;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return t.tv_sec * 1000000 + t.tv_nsec / 1000;
  #endif
}

}  // namespace

chrome_trace::chrome_trace(std::ostream& os) : os_(os) {
  os_ << "[{\"name\":\"chrome_trace\",\"cat\":\"slinky\",\"ph\":\"B\",\"pid\":0,\"tid\":0,\"ts\":0}";
  os_ << ",\n{\"name\":\"chrome_trace\",\"cat\":\"slinky\",\"ph\":\"E\",\"pid\":0,\"tid\":0,\"ts\":0}";
  t0_ = std::chrono::high_resolution_clock::now();
  cpu_t0_ = clock_per_thread_us();
}
chrome_trace::~chrome_trace() {
  os_ << "]\n";
}

void chrome_trace::write_event(const char* name, const char* cat, char type) {
  auto t = std::chrono::high_resolution_clock::now();
  std::clock_t cpu_t = clock_per_thread_us();
  auto ts = std::chrono::duration_cast<std::chrono::microseconds>(t - t0_).count();
  std::clock_t cpu_ts = cpu_t - cpu_t0_;

  // The only way to convert a thread ID to a string is via operator<<, which is slow, so we do it as a thread_local
  // initializer.
  static std::atomic<int> next_thread_id = 0;
  thread_local std::string pid_tid_str = []() {
    std::stringstream ss;
    ss << "\",\"pid\":0,\"tid\":" << next_thread_id++;
    return ss.str();
  }();

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
  buffer += pid_tid_str;
  buffer += ",\"ts\":";
  buffer += std::to_string(ts);
  buffer += ",\"tts\":";
  buffer += std::to_string(cpu_ts);
  buffer += '}';

  // Flush our buffer.
  std::unique_lock l(mtx_);
  os_.write(buffer.data(), buffer.size());
}

void chrome_trace::begin(const char* name) { write_event(name, "slinky", 'B'); }
void chrome_trace::end(const char* name) { write_event(name, "slinky", 'E'); }

chrome_trace* chrome_trace::global() {
  static const char* path = getenv("SLINKY_TRACE");
  if (!path) return nullptr;

  static auto file = std::make_unique<std::ofstream>(path);
  static auto trace = std::make_unique<chrome_trace>(*file);
  return trace.get();
}

}  // namespace slinky
