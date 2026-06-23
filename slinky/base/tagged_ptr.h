#ifndef SLINKY_BASE_TAGGED_PTR_H
#define SLINKY_BASE_TAGGED_PTR_H

#include <cassert>
#include <cstdint>

#include "slinky/base/util.h"

namespace slinky {

// A `tagged_ptr` packs a small tag and either a pointer to `T` or an inline
// payload into a single pointer-sized word. The low `TagBits` bits hold the
// tag; the remaining high bits hold either an aligned `T*` (when the tag is 0)
// or an arbitrary inline payload.
//
// This relies on pointers to `T` being aligned to at least `2^TagBits` bytes,
// so the low bits of a real pointer are always zero. A word of all zeros is a
// null pointer with tag 0.
template <typename T, int TagBits = 2>
class SLINKY_TRIVIAL_ABI tagged_ptr {
  std::uintptr_t bits_;

public:
  static constexpr int tag_bits = TagBits;
  static constexpr std::uintptr_t tag_mask = (std::uintptr_t(1) << TagBits) - 1;

  SLINKY_INLINE tagged_ptr() : bits_(0) {}
  SLINKY_INLINE tagged_ptr(const T* p) : bits_(reinterpret_cast<std::uintptr_t>(p)) {
    assert((bits_ & tag_mask) == 0);
  }

  SLINKY_INLINE static tagged_ptr from_payload(unsigned tag, std::uintptr_t payload) {
    assert(tag != 0 && (tag & ~tag_mask) == 0);
    tagged_ptr result;
    result.bits_ = (payload << TagBits) | tag;
    return result;
  }

  SLINKY_INLINE unsigned tag() const { return static_cast<unsigned>(bits_ & tag_mask); }
  SLINKY_INLINE std::uintptr_t bits() const { return bits_; }

  SLINKY_INLINE const T* pointer() const {
    assert(tag() == 0);
    return reinterpret_cast<const T*>(bits_); 
  }

  SLINKY_INLINE std::uintptr_t upayload() const {
    assert(tag() != 0);
    return bits_ >> TagBits;
  }
  SLINKY_INLINE std::intptr_t spayload() const {
    assert(tag() != 0);
    return static_cast<std::intptr_t>(bits_) >> TagBits;
  }

  SLINKY_INLINE bool operator==(tagged_ptr other) const { return bits_ == other.bits_; }
  SLINKY_INLINE bool operator!=(tagged_ptr other) const { return bits_ != other.bits_; }
  SLINKY_INLINE explicit operator bool() const { return bits_ != 0; }
};

}  // namespace slinky

#endif  // SLINKY_BASE_TAGGED_PTR_H
