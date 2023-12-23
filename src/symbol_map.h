#ifndef SLINKY_SYMBOL_MAP_H
#define SLINKY_SYMBOL_MAP_H

#include <optional>

#include "expr.h"

namespace slinky {

template <typename T>
class symbol_map {
  std::vector<std::optional<T>> values;

  void grow(std::size_t size) {
    if (size >= values.size()) { values.resize(std::max(values.size() * 2, size + 1)); }
  }

public:
  symbol_map() {}

  std::optional<T> lookup(symbol_id name) const {
    if (name < values.size()) { return values[name]; }
    return std::nullopt;
  }

  std::optional<T> operator[](symbol_id name) const { return lookup(name); }
  std::optional<T>& operator[](symbol_id name) {
    grow(name);
    return values[name];
  }

  bool contains(symbol_id name) const {
    if (name >= values.size()) { return false; }
    return !!values[name];
  }

  // Returns the previous state of `name`.
  std::optional<T> set(symbol_id name, std::optional<T> value) {
    if (value) {
      grow(name);
      std::swap(values[name], value);
      return value;
    } else {
      if (name < values.size()) {
        std::swap(values[name], value);
        return value;
      } else {
        return std::nullopt;
      }
    }
  }
};

// Set a value in an eval_context upon construction, and restore the old value upon destruction.
template <typename T>
class scoped_value {
  symbol_map<T>* context;
  symbol_id name;
  std::optional<T> old_value;

public:
  scoped_value(symbol_map<T>& context, symbol_id name, T value) : context(&context), name(name) {
    old_value = context.set(name, std::move(value));
  }

  scoped_value(scoped_value&& other) : context(other.context), name(other.name), old_value(std::move(other.old_value)) {
    // Don't let other.~scoped_value() unset this value.
    other.context = nullptr;
  }
  scoped_value(const scoped_value&) = delete;
  scoped_value& operator=(const scoped_value&) = delete;
  scoped_value& operator=(scoped_value&& other) {
    context = other.context;
    name = other.name;
    old_value = std::move(other.old_value);
    // Don't let other.~scoped_value() unset this value.
    other.context = nullptr;
  }

  ~scoped_value() {
    if (context) { context->set(name, std::move(old_value)); }
  }
};

}  // namespace slinky

#endif  // SLINKY_SYMBOL_MAP_H
