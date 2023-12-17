#ifndef LOCALITY_INTERVAL_H
#define LOCALITY_INTERVAL_H

#include "expr.h"

namespace slinky {

struct interval {
  expr min, max;

  interval() {}
  explicit interval(const expr& point) : min(point), max(point) {}
  interval(expr min, expr max) : min(std::move(min)), max(std::move(max)) {}

  expr extent() const {
    return max - min + 1;
  }
  void set_extent(expr extent) {
    max = min + extent - 1;
  }

  interval& operator*=(expr scale) {
    min *= scale;
    max *= scale;
    return *this;
  }

  interval& operator/=(expr scale) {
    min /= scale;
    max /= scale;
    return *this;
  }

  interval& operator+=(expr offset) {
    min += offset;
    max += offset;
    return *this;
  }

  interval& operator-=(expr offset) {
    min -= offset;
    max -= offset;
    return *this;
  }

  interval operator*(expr scale) const {
    interval result(*this);
    result *= scale;
    return result;
  }

  interval operator/(expr scale) const {
    interval result(*this);
    result /= scale;
    return result;
  }

  interval operator+(expr offset) const {
    interval result(*this);
    result += offset;
    return result;
  }

  interval operator-(expr offset) const {
    interval result(*this);
    result -= offset;
    return result;
  }
};

}  // namespace slinky

#endif  // LOCALITY_INTERVAL_H