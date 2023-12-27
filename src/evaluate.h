#ifndef SLINKY_EVALUATE_H
#define SLINKY_EVALUATE_H

#include "expr.h"
#include "symbol_map.h"

namespace slinky {

// TODO: Probably shouldn't inherit here.
class eval_context : public symbol_map<index_t> {
public:
  struct memory_info {
    std::size_t live_count = 0;
    std::size_t live_size = 0;
    std::size_t total_count = 0;
    std::size_t total_size = 0;
    std::size_t peak_count = 0;
    std::size_t peak_size = 0;

    class alloc_tracker {
      memory_info* info;
      std::size_t size;

      void free() {
        if (info) {
          info->live_count -= 1;
          info->live_size -= size;
          info = nullptr;
        }
      }

    public:
      alloc_tracker() : info(nullptr) {}
      alloc_tracker(memory_info* info, std::size_t size) : info(info), size(size) { 
        info->live_count += 1;
        info->live_size += size;
        info->total_count += 1;
        info->total_size += size;
        info->peak_count = std::max(info->peak_count, info->live_count);
        info->peak_size = std::max(info->peak_size, info->live_size);
      }

      alloc_tracker(const alloc_tracker&) = delete;
      alloc_tracker& operator=(const alloc_tracker& m) = delete;
      alloc_tracker(alloc_tracker&& m) : info(m.info), size(m.size) { m.info = nullptr; }
      alloc_tracker& operator=(alloc_tracker&& m) {
        free();
        info = m.info; 
        size = m.size;
        m.info = nullptr;
        return *this;
      }

      ~alloc_tracker() { free(); }
    };

    alloc_tracker track_allocate(std::size_t size) { 
      return alloc_tracker(this, size);
    }
  };

  memory_info heap;
  memory_info stack;
};

index_t evaluate(const expr& e, eval_context& context);
index_t evaluate(const stmt& s, eval_context& context);
index_t evaluate(const expr& e);
index_t evaluate(const stmt& s);

}  // namespace slinky

#endif  // SLINKY_EVALUATE_H
