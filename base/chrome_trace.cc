#include "base/chrome_trace.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>

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

const char* message_format =
    "%s{\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"%c\",\"pid\":0,\"tid\":%d,\"ts\":%ld,\"tts\":%ld}";

}  // namespace

chrome_trace::chrome_trace(std::ostream& os) : os_(os) {
  char buffer[1024];
  int size = snprintf(buffer, sizeof(buffer), message_format, "[", "chrome_trace", "slinky", 'B', 0, 0, 0);
  os_.write(buffer, size);
  size = snprintf(buffer, sizeof(buffer), message_format, ",\n", "chrome_trace", "slinky", 'E', 0, 0, 0);
  os_.write(buffer, size);
  t0_ = std::chrono::high_resolution_clock::now();
  cpu_t0_ = clock_per_thread_us();
}
chrome_trace::~chrome_trace() { os_ << "]\n"; }

void chrome_trace::write_event(const char* name, const char* cat, char type) {
  auto t = std::chrono::high_resolution_clock::now();
  std::clock_t cpu_t = clock_per_thread_us();
  auto ts = std::chrono::duration_cast<std::chrono::microseconds>(t - t0_).count();
  std::clock_t cpu_ts = cpu_t - cpu_t0_;

  // std::thread ids are super long, use our own integer id instead.
  static std::atomic<int> next_thread_id = 0;
  thread_local int tid = next_thread_id++;

  char buffer[1024];
  // It would be an error to put a comma here as the first item in the output, but we put a dummy {} object at the
  // beginning of the array.
  int size = snprintf(buffer, sizeof(buffer), message_format, ",\n", name, cat, type, tid, ts, cpu_ts);

  // Flush our buffer.
  std::unique_lock l(mtx_);
  os_.write(buffer, size);
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
