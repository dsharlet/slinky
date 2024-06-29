#include "base/chrome_trace.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>

namespace slinky {

namespace {

std::atomic<int> next_id = 0;

// Unfortunately, std::clock returns the CPU time for the whole process, not the current thread.
std::clock_t clock_per_thread_us() {
  timespec t;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return t.tv_sec * 1000000 + t.tv_nsec / 1000;
}

}  // namespace

namespace proto {

enum class wire_type {
  varint = 0,
  i64 = 1,
  len = 2,
  i32 = 5,
};

enum class TrackDescriptor {
  /*optional uint64*/ uuid = 1,
  /*optional uint64*/ parent_uuid = 5,
  // oneof {
  /*optional string*/ name = 2,
  /*optional string*/ static_name = 10,
  // }
  /*optional ProcessDescriptor*/ process = 3,
  /*optional ThreadDescriptor*/ thread = 4,
};

enum class ThreadDescriptor {
  /*optional int32*/ pid = 1,
  /*optional int32*/ tid = 2,
  /*optional string*/ thread_name = 5,
};

enum class EventType {
  TYPE_UNSPECIFIED = 0,
  TYPE_SLICE_BEGIN = 1,
  TYPE_SLICE_END = 2,
  TYPE_INSTANT = 3,
  TYPE_COUNTER = 4,
};

enum class TrackEvent {
  // Optional name of the event for its display in trace viewer. May be left
  // unspecified for events with typed arguments.
  //
  // Note that metrics should not rely on event names, as they are prone to
  // changing. Instead, they should use typed arguments to identify the events
  // they are interested in.
  // oneof {
  /*uint64*/ name_iid = 10,
  /*string*/ name = 23,
  //}

  /*optional EventType*/ type = 9,

  /*optional uint64*/ track_uuid = 11,

  // Deprecated. Use the |timestamp| and |timestamp_clock_id| fields in
  // TracePacket instead.
  // oneof timestamp {

  /*repeated uint64*/ extra_counter_track_uuids = 31,
  /*repeated int64*/ extra_counter_values = 12,

  // Deprecated. Use |extra_counter_values| and |extra_counter_track_uuids| to
  // encode thread time instead.
  //
  // CPU time for the current thread (e.g., CLOCK_THREAD_CPUTIME_ID) in
  // microseconds.
  // oneof thread_time {
};

enum class TracePacket {
  /*optional uint64*/ timestamp = 8,

  // Specifies the ID of the clock used for the TracePacket |timestamp|. Can be
  // one of the built-in types from ClockSnapshot::BuiltinClocks, or a
  // producer-defined clock id.
  // If unspecified and if no default per-sequence value has been provided via
  // TracePacketDefaults, it defaults to BuiltinClocks::BOOTTIME.
  /*optional uint32*/ timestamp_clock_id = 58,
  /*TrackEvent*/ track_event = 11,
  /*TrackDescriptor*/ track_descriptor = 60,

  /*optional TracePacketDefaults*/ trace_packet_defaults = 59,

  /*uint32*/ trusted_packet_sequence_id = 10,
};

// varint is 7 bits at a time, with the MSB indicating if there is another 7 bits remaining.
void write_varint(buffer& buf, uint64_t value) {
  constexpr uint8_t continuation = 0x80;
  std::size_t begin = buf.size();
  while (value > 0x7f) {
    buf.push_back(static_cast<uint8_t>(value | continuation));
    value >>= 7;
  }
  buf.push_back(static_cast<uint8_t>(value));
  std::reverse(buf.begin() + begin, buf.end());
}

// sint uses "zigzag" encoding: positive x -> 2*x, negative x -> -2*x - 1
//void write_varint(buffer& buf, int64_t value) {
  //write_varint(buf, static_cast<uint64_t>(value < 0 ? -2 * value - 1 : 2 * value));
//}

void write_tag(buffer& buf, uint64_t field_number, wire_type type) {
  write_varint(buf, (field_number << 3) | static_cast<uint64_t>(type));
}

void write_len_tag(buffer& buf, uint64_t field_number, uint64_t len) {
  write_varint(buf, len);
  write_tag(buf, field_number, wire_type::len);
}

void write(buffer& buf, uint64_t field_number, uint64_t value) {
  write_varint(buf, value);
  write_tag(buf, field_number, wire_type::varint);
}

//void write(buffer& buf, uint64_t field_number, int64_t value) {
//  write_tag(buf, field_number, wire_type::varint);
//  write_varint(buf, value);
//}

void write(buffer& buf, uint64_t field_number, const char* str) {
  std::size_t len = strlen(str);
  std::size_t begin = buf.size();
  buf.insert(buf.end(), str, str + len);
  std::reverse(buf.begin() + begin, buf.end());
  write_len_tag(buf, field_number, len);
}

}  // namespace proto

chrome_trace::chrome_trace(std::ostream& os) : os_(os), id_(next_id++) {
  proto::buffer buf;
  proto::write(buf, static_cast<uint64_t>(proto::TrackDescriptor::uuid), id_);
  proto::write_len_tag(buf, static_cast<uint64_t>(proto::TracePacket::track_descriptor), buf.size());
  proto::write_len_tag(buf, 0, buf.size());
  std::reverse(buf.begin(), buf.end());
  os.write(buf.data(), buf.size());

  t0_ = std::chrono::high_resolution_clock::now();
  cpu_t0_ = clock_per_thread_us();
}
chrome_trace::~chrome_trace() {
  // Flush any unwritten buffers.
  for (auto& i : buffers_) {
    std::reverse(i.second.begin(), i.second.end());
    os_.write(i.second.data(), i.second.size());
  }
}

namespace {

void write_event(proto::buffer& buf, int id, int tid, const char* name, proto::EventType type, uint64_t ts, uint64_t cpu_ts) {
}

}  // namespace

int chrome_trace::get_thread_id() {
  static std::atomic<int> next_thread_id = 0;
  thread_local int tid = next_thread_id++;
  return tid;
}

proto::buffer& chrome_trace::get_buffer() {
  // To avoid overhead when multiple threads are writing traces, we try to keep a pointer to the buffer for this trace
  // object and thread cached locally.
  thread_local int current_buffer_owner = -1;
  thread_local proto::buffer* buf = nullptr;

  bool new_thread = false;

  if (!buf || current_buffer_owner != id_) {
    // We should only need to pay the cost of this lookup if multiple different chrome_trace objects are writing traces
    // on the same thread at the same time.
    std::unique_lock l(mtx_);
    new_thread = buffers_.find(std::this_thread::get_id()) == buffers_.end();
    buf = &buffers_[std::this_thread::get_id()];
    current_buffer_owner = id_;
  }
  if (new_thread) {
    // Write the thread descriptor once.
    proto::buffer buf;
    proto::write(buf, static_cast<uint64_t>(proto::TrackDescriptor::uuid), get_thread_id());
    proto::write(buf, static_cast<uint64_t>(proto::TrackDescriptor::parent_uuid), id_);
    proto::write_len_tag(buf, static_cast<uint64_t>(proto::TracePacket::track_descriptor), buf.size());
    proto::write_len_tag(buf, 0, buf.size());
    std::reverse(buf.begin(), buf.end());
    os_.write(buf.data(), buf.size());
  }
  return *buf;
}

void chrome_trace::begin(const char* name) {
  auto t = std::chrono::high_resolution_clock::now();
  std::clock_t cpu_t = clock_per_thread_us();
  auto ts = std::chrono::duration_cast<std::chrono::microseconds>(t - t0_).count();
  std::clock_t cpu_ts = cpu_t - cpu_t0_;
  auto& buf = get_buffer();
  std::size_t begin = buf.size();
  proto::write(
      buf, static_cast<uint64_t>(proto::TrackEvent::type), static_cast<uint64_t>(proto::EventType::TYPE_SLICE_BEGIN));
  proto::write(buf, static_cast<uint64_t>(proto::TrackEvent::track_uuid), get_thread_id());
  proto::write(buf, static_cast<uint64_t>(proto::TrackEvent::name), name);
  proto::write_len_tag(buf, static_cast<uint64_t>(proto::TracePacket::track_event), buf.size() - begin);
  proto::write(buf, static_cast<uint64_t>(proto::TracePacket::timestamp), ts);
  proto::write(buf, static_cast<uint64_t>(proto::TracePacket::trusted_packet_sequence_id), 1);
  proto::write_len_tag(buf, 0, buf.size() - begin);
  if (true || buf.size() > 4096 * 16) {
    // Flush our buffer.
    std::reverse(buf.begin(), buf.end());
    std::unique_lock l(mtx_);
    os_.write(buf.data(), buf.size());
    buf.clear();
  }
}
void chrome_trace::end() {
  auto t = std::chrono::high_resolution_clock::now();
  std::clock_t cpu_t = clock_per_thread_us();
  auto ts = std::chrono::duration_cast<std::chrono::microseconds>(t - t0_).count();
  std::clock_t cpu_ts = cpu_t - cpu_t0_;
  auto& buf = get_buffer();
  std::size_t begin = buf.size();
  proto::write(
      buf, static_cast<uint64_t>(proto::TrackEvent::type), static_cast<uint64_t>(proto::EventType::TYPE_SLICE_END));
  proto::write(buf, static_cast<uint64_t>(proto::TrackEvent::track_uuid), get_thread_id());
  proto::write_len_tag(buf, static_cast<uint64_t>(proto::TracePacket::track_event), buf.size() - begin);
  proto::write(buf, static_cast<uint64_t>(proto::TracePacket::timestamp), ts);
  proto::write(buf, static_cast<uint64_t>(proto::TracePacket::trusted_packet_sequence_id), 1);
  proto::write_len_tag(buf, 0, buf.size() - begin);
  if (true || buf.size() > 4096 * 16) {
    // Flush our buffer.
    std::reverse(buf.begin(), buf.end());
    std::unique_lock l(mtx_);
    os_.write(buf.data(), buf.size());
    buf.clear();
  }
}

chrome_trace* chrome_trace::global() {
  static const char* path = getenv("SLINKY_TRACE");
  if (!path) return nullptr;

  static auto file = std::make_unique<std::ofstream>(path, std::ios::binary);
  static auto trace = std::make_unique<chrome_trace>(*file);
  return trace.get();
}

}  // namespace slinky
