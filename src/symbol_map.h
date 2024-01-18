#ifndef SLINKY_SYMBOL_MAP_H
#define SLINKY_SYMBOL_MAP_H

#include <optional>

#include "src/expr.h"

namespace slinky {

template <typename T>
class symbol_map {
  std::vector<std::optional<T>> values;

  void grow(std::size_t size) {
    if (size >= values.size()) { values.resize(std::max(values.size() * 2, size + 1)); }
  }

public:
  symbol_map() {}
  symbol_map(std::initializer_list<std::pair<symbol_id, T>> init) {
    for (const std::pair<symbol_id, T>& i : init) {
      operator[](i.first) = i.second;
    }
  }

  std::optional<T> lookup(symbol_id sym) const {
    if (sym < values.size()) { return values[sym]; }
    return std::nullopt;
  }
  std::optional<T> lookup(const var& v) const { return lookup(v.sym()); }

  const T& lookup(symbol_id sym, const T& def) const {
    if (sym < values.size() && values[sym]) { return *values[sym]; }
    return def;
  }
  const T& lookup(const var& v, const T& def) const { return lookup(v.sym(), def); }

  std::optional<T> operator[](symbol_id sym) const {return lookup(sym); } 
  std::optional<T> operator[](const var& v) const { return lookup(v.sym()); }
  std::optional<T>& operator[](symbol_id sym) {
    grow(sym);
    return values[sym];
  }
  std::optional<T>& operator[](const var& v) {
    return operator[](v.sym());
  }

  bool contains(symbol_id sym) const {
    if (sym >= values.size()) { return false; }
    return !!values[sym];
  }
  bool contains(const var& v) const { return contains(v.sym()); }

  std::size_t size() const { return values.size(); }
  auto begin() { return values.begin(); }
  auto end() { return values.end(); }
  auto begin() const { return values.begin(); }
  auto end() const { return values.end(); }
  void clear() { values.clear(); }
};

// Set a value in an eval_context upon construction, and restore the old value upon destruction.
template <typename T>
class scoped_value_in_symbol_map {
  symbol_map<T>* context_;
  symbol_id sym_;
  std::optional<T> old_value_;

public:
  scoped_value_in_symbol_map(symbol_map<T>& context, symbol_id sym, T value) : context_(&context), sym_(sym) {
    std::optional<T>& ctx_value = context[sym];
    old_value_ = std::move(ctx_value);
    ctx_value = std::move(value);
  }
  scoped_value_in_symbol_map(symbol_map<T>& context, symbol_id sym, std::optional<T> value)
      : context_(&context), sym_(sym) {
    std::optional<T>& ctx_value = context[sym];
    old_value_ = std::move(ctx_value);
    ctx_value = std::move(value);
  }

  scoped_value_in_symbol_map(scoped_value_in_symbol_map&& other)
      : context_(other.context), sym_(other.sym), old_value_(std::move(other.old_value_)) {
    // Don't let other.~scoped_value() unset this value.
    other.context = nullptr;
  }
  scoped_value_in_symbol_map(const scoped_value_in_symbol_map&) = delete;
  scoped_value_in_symbol_map& operator=(const scoped_value_in_symbol_map&) = delete;
  scoped_value_in_symbol_map& operator=(scoped_value_in_symbol_map&& other) {
    context_ = other.context;
    sym_ = other.sym;
    old_value_ = std::move(other.old_value_);
    // Don't let other.~scoped_value_in_symbol_map() unset this value.
    other.context = nullptr;
  }

  const std::optional<T>& old_value() const { return old_value_; }

  ~scoped_value_in_symbol_map() {
    if (context_) {
      (*context_)[sym_] = std::move(old_value_);
    }
  }
};

template <typename T>
scoped_value_in_symbol_map<T> set_value_in_scope(symbol_map<T>& context, symbol_id sym, T value) {
  return scoped_value_in_symbol_map<T>(context, sym, value);
}
template <typename T>
scoped_value_in_symbol_map<T> set_value_in_scope(symbol_map<T>& context, symbol_id sym, std::optional<T> value) {
  return scoped_value_in_symbol_map<T>(context, sym, value);
}

}  // namespace slinky

#endif  // SLINKY_SYMBOL_MAP_H
