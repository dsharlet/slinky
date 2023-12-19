#ifndef LOCALITY_EVALUATE_H
#define LOCALITY_EVALUATE_H

#include <optional>

#include "expr.h"

namespace slinky {

template <typename T>
class symbol_map {
  std::vector<std::optional<T>> values;

public:
  symbol_map() {}

  std::optional<T> lookup(symbol_id name) const {
    if (name < values.size()) {
      return values[name];
    }
    return std::nullopt;
  }

  // Returns the previous state of `name`.
  std::optional<T> set(symbol_id name, std::optional<T> value) {
    if (value) {
      if (name >= values.size()) {
        values.resize(std::max(values.size() * 2, name + 1));
      }
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
  symbol_map<T>& context;
  symbol_id name;
  std::optional<T> old_value;

public:
  scoped_value(symbol_map<T>& context, symbol_id name, T value) : context(context), name(name) {
    old_value = context.set(name, std::move(value));
  }

  scoped_value(scoped_value&&) = default;
  scoped_value(const scoped_value&) = delete;
  scoped_value& operator=(const scoped_value&) = delete;

  ~scoped_value() {
    context.set(name, std::move(old_value));
  }
};

using eval_context = symbol_map<index_t>;

index_t evaluate(const expr& e, eval_context& context);
index_t evaluate(const stmt& s, eval_context& context);
index_t evaluate(const expr& e);
index_t evaluate(const stmt& s);

}  // namespace slinky

#endif