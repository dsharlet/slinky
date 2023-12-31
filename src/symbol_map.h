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
  std::optional<T> lookup(const var& v) const { return lookup(v.name()); }

  const T& lookup(symbol_id name, const T& def) const {
    if (name < values.size()) { return values[name]; }
    return def;
  }
  const T& lookup(const var& v, const T& def) const { return lookup(v.name(), def); }

  std::optional<T> operator[](symbol_id name) const {return lookup(name); } 
  std::optional<T> operator[](const var& v) const { return lookup(v.name()); }
  std::optional<T>& operator[](symbol_id name) {
    grow(name);
    return values[name];
  }
  std::optional<T>& operator[](const var& v) {
    return operator[](v.name());
  }

  bool contains(symbol_id name) const {
    if (name >= values.size()) { return false; }
    return !!values[name];
  }
  bool contains(const var& v) const { return contains(v.name()); }

  std::size_t size() const { return values.size(); }
  auto begin() { return values.begin(); }
  auto end() { return values.end(); }
  auto begin() const { return values.begin(); }
  auto end() const { return values.end(); }
};

// Set a value in an eval_context upon construction, and restore the old value upon destruction.
template <typename T>
class scoped_value_in_symbol_map {
  symbol_map<T>* context;
  symbol_id name;
  std::optional<T> old_value;

public:
  scoped_value_in_symbol_map(symbol_map<T>& context, symbol_id name, T value) : context(&context), name(name) {
    std::optional<T>& ctx_value = context[name];
    old_value = std::move(ctx_value);
    ctx_value = std::move(value);
  }
  scoped_value_in_symbol_map(symbol_map<T>& context, symbol_id name, std::optional<T> value)
      : context(&context), name(name) {
    std::optional<T>& ctx_value = context[name];
    old_value = std::move(ctx_value);
    ctx_value = std::move(value);
  }

  scoped_value_in_symbol_map(scoped_value_in_symbol_map&& other)
      : context(other.context), name(other.name), old_value(std::move(other.old_value)) {
    // Don't let other.~scoped_value() unset this value.
    other.context = nullptr;
  }
  scoped_value_in_symbol_map(const scoped_value_in_symbol_map&) = delete;
  scoped_value_in_symbol_map& operator=(const scoped_value_in_symbol_map&) = delete;
  scoped_value_in_symbol_map& operator=(scoped_value_in_symbol_map&& other) {
    context = other.context;
    name = other.name;
    old_value = std::move(other.old_value);
    // Don't let other.~scoped_value_in_symbol_map() unset this value.
    other.context = nullptr;
  }

  ~scoped_value_in_symbol_map() {
    if (context) { (*context)[name] = std::move(old_value); }
  }
};

template <typename T>
scoped_value_in_symbol_map<T> set_value_in_scope(symbol_map<T>& context, symbol_id name, T value) {
  return scoped_value_in_symbol_map<T>(context, name, value);
}
template <typename T>
scoped_value_in_symbol_map<T> set_value_in_scope(symbol_map<T>& context, symbol_id name, std::optional<T> value) {
  return scoped_value_in_symbol_map<T>(context, name, value);
}

}  // namespace slinky

#endif  // SLINKY_SYMBOL_MAP_H
