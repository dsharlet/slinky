#include "expr.h"

namespace slinky {

// Get the name of a symbol_id.
std::string node_context::name(symbol_id i) const { 
  if (i < id_to_name.size()) {
    return id_to_name[i];
  } else {
    return "<" + std::to_string(i) + ">";
  }
}

// Get or insert a new symbol_id for a name.
symbol_id node_context::insert(const std::string& name) {
  auto i = name_to_id.find(name);
  if (i == name_to_id.end()) {
    symbol_id id = id_to_name.size();
    id_to_name.push_back(name);
    name_to_id[name] = id;
    return id;
  }
  return i->second;
}
symbol_id node_context::lookup(const std::string& name) const {
  auto i = name_to_id.find(name);
  if (i != name_to_id.end()) {
    return i->second;
  } else {
    return -1;
  }
}

template <typename T>
const T* make_bin_op(expr a, expr b) {
  T* n = new T();
  n->a = std::move(a);
  n->b = std::move(b);
  return n;
}

expr let::make(symbol_id name, expr value, expr body) {
  let* n = new let();
  n->name = name;
  n->value = std::move(value);
  n->body = std::move(body);
  return n;
}

expr variable::make(symbol_id name) {
  variable* n = new variable();
  n->name = name;
  return n;
}

const constant* make_constant(index_t value) {
  constant* n = new constant();
  n->value = value;
  return n;
}

expr constant::make(index_t value) {
  return make_constant(value);
}

expr::expr(index_t value) : expr(make_constant(value)) {}

expr constant::make(const void* value) {
  return make(reinterpret_cast<index_t>(value));
}

expr add::make(expr a, expr b) { return make_bin_op<add>(std::move(a), std::move(b)); }
expr sub::make(expr a, expr b) { return make_bin_op<sub>(std::move(a), std::move(b)); }
expr mul::make(expr a, expr b) { return make_bin_op<mul>(std::move(a), std::move(b)); }
expr div::make(expr a, expr b) { return make_bin_op<div>(std::move(a), std::move(b)); }
expr mod::make(expr a, expr b) { return make_bin_op<mod>(std::move(a), std::move(b)); }
expr min::make(expr a, expr b) { return make_bin_op<min>(std::move(a), std::move(b)); }
expr max::make(expr a, expr b) { return make_bin_op<max>(std::move(a), std::move(b)); }
expr equal::make(expr a, expr b) { return make_bin_op<equal>(std::move(a), std::move(b)); }
expr not_equal::make(expr a, expr b) { return make_bin_op<not_equal>(std::move(a), std::move(b)); }
expr less::make(expr a, expr b) { return make_bin_op<less>(std::move(a), std::move(b)); }
expr less_equal::make(expr a, expr b) { return make_bin_op<less_equal>(std::move(a), std::move(b)); }
expr bitwise_and::make(expr a, expr b) { return make_bin_op<bitwise_and>(std::move(a), std::move(b)); }
expr bitwise_or::make(expr a, expr b) { return make_bin_op<bitwise_or>(std::move(a), std::move(b)); }
expr bitwise_xor::make(expr a, expr b) { return make_bin_op<bitwise_xor>(std::move(a), std::move(b)); }
expr logical_and::make(expr a, expr b) { return make_bin_op<logical_and>(std::move(a), std::move(b)); }
expr logical_or::make(expr a, expr b) { return make_bin_op<logical_or>(std::move(a), std::move(b)); }

expr make_variable(node_context& ctx, const std::string& name) {
  return variable::make(ctx.insert(name));
}

expr operator+(expr a, expr b) { return add::make(std::move(a), std::move(b)); }
expr operator-(expr a, expr b) { return sub::make(std::move(a), std::move(b)); }
expr operator*(expr a, expr b) { return mul::make(std::move(a), std::move(b)); }
expr operator/(expr a, expr b) { return div::make(std::move(a), std::move(b)); }
expr operator%(expr a, expr b) { return mod::make(std::move(a), std::move(b)); }
expr min(expr a, expr b) { return min::make(std::move(a), std::move(b)); }
expr max(expr a, expr b) { return max::make(std::move(a), std::move(b)); }
expr operator==(expr a, expr b) { return equal::make(std::move(a), std::move(b)); }
expr operator!=(expr a, expr b) { return not_equal::make(std::move(a), std::move(b)); }
expr operator<(expr a, expr b) { return less::make(std::move(a), std::move(b)); }
expr operator<=(expr a, expr b) { return less_equal::make(std::move(a), std::move(b)); }
expr operator>(expr a, expr b) { return less::make(std::move(b), std::move(a)); }
expr operator>=(expr a, expr b) { return less_equal::make(std::move(b), std::move(a)); }
expr operator&(expr a, expr b) { return bitwise_and::make(std::move(a), std::move(b)); }
expr operator|(expr a, expr b) { return bitwise_or::make(std::move(a), std::move(b)); }
expr operator^(expr a, expr b) { return bitwise_xor::make(std::move(a), std::move(b)); }
expr operator&&(expr a, expr b) { return logical_and::make(std::move(a), std::move(b)); }
expr operator||(expr a, expr b) { return logical_or::make(std::move(a), std::move(b)); }

}  // namespace slinky
