#ifndef SLINKY_BASE_THREAD_SYNCHRONIZATION_H
#define SLINKY_BASE_THREAD_SYNCHRONIZATION_H

#include <condition_variable>
#include <mutex>

namespace slinky {

using mutex = std::mutex;
using condition_variable = std::condition_variable;
using unique_lock = std::unique_lock<std::mutex>;

}  // namespace slinky

#endif  // SLINKY_BASE_THREAD_SYNCHRONIZATION_H
