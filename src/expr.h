#ifndef SLINKY_EXPR_H
#define SLINKY_EXPR_H

#include "buffer.h"

#include <cstdlib>
#include <initializer_list>
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <span>
#include <functional>

namespace slinky {

using symbol_id = std::size_t;

// We don't want to be doing string lookups in the inner loops. A node_context
// uniquely maps strings to symbol_id.
class node_context {
  std::map<std::string, symbol_id> name_to_id;
  std::vector<std::string> id_to_name;

public:
  // Get the name of a symbol_id.
  std::string name(symbol_id i) const;

  // Get or insert a new symbol_id for a name.
  symbol_id insert(const std::string& name);
  symbol_id lookup(const std::string& name) const;
};

enum class node_type {
  variable,
  constant,
  let,
  add,
  sub,
  mul,
  div,
  mod,
  min,
  max,
  equal,
  not_equal,
  less,
  less_equal,
  bitwise_and,
  bitwise_or,
  bitwise_xor,
  logical_and,
  logical_or,

  call,
  let_stmt,
  block,
  loop,
  if_then_else,
  allocate,
  check,
};

enum class memory_type {
  stack,
  heap,
};

class node_visitor;

class base_node : public std::enable_shared_from_this<base_node> {
public:
  base_node(node_type type) : type(type) {}
  virtual ~base_node() {}

  virtual void accept(node_visitor* v) const = 0;

  node_type type;

  template <typename T>
  const T* as() const {
    if (type == T::static_type) {
      return reinterpret_cast<const T*>(this);
    } else {
      return nullptr;
    }
  }
};

class base_expr_node : public base_node {
public:
  base_expr_node(node_type type) : base_node(type) {}
};

class base_stmt_node : public base_node {
public:
  base_stmt_node(node_type type) : base_node(type) {}
};

template <typename T>
class expr_node : public base_expr_node {
public:
  expr_node() : base_expr_node(T::static_type) {}
};

template <typename T>
class stmt_node : public base_stmt_node {
public:
  stmt_node() : base_stmt_node(T::static_type) {}
};

class expr;

expr operator+(expr a, expr b);
expr operator-(expr a, expr b);
expr operator*(expr a, expr b);
expr operator/(expr a, expr b);
expr operator%(expr a, expr b);
expr operator&(expr a, expr b);
expr operator|(expr a, expr b);
expr operator^(expr a, expr b);

class expr {
public:
  std::shared_ptr<const base_expr_node> e;

  expr() = default;
  expr(const expr&) = default;
  expr(expr&&) = default;
  expr(index_t x);
  expr(int x) : expr(static_cast<index_t>(x)) {}
  expr(const base_expr_node* e) : e(e) {}

  expr& operator=(const expr&) = default;
  expr& operator=(expr&&) = default;

  void accept(node_visitor* v) const { e->accept(v); }

  bool defined() const { return e != nullptr; }

  template <typename T>
  const T* as() const {
    if (e->type == T::static_type) {
      return reinterpret_cast<const T*>(e.get());
    } else {
      return nullptr;
    }
  }

  expr& operator+=(const expr& r) { *this = *this + r; return *this; }
  expr& operator-=(const expr& r) { *this = *this - r; return *this; }
  expr& operator*=(const expr& r) { *this = *this * r; return *this; }
  expr& operator/=(const expr& r) { *this = *this / r; return *this; }
  expr& operator%=(const expr& r) { *this = *this % r; return *this; }
  expr& operator&=(const expr& r) { *this = *this & r; return *this; }
  expr& operator^=(const expr& r) { *this = *this ^ r; return *this; }
  expr& operator|=(const expr& r) { *this = *this | r; return *this; }
};

class stmt {
public:
  std::shared_ptr<const base_stmt_node> s;

  stmt() = default;
  stmt(const stmt&) = default;
  stmt(stmt&&) = default;
  stmt(const base_stmt_node* s) : s(s) {}
  stmt(std::initializer_list<stmt> stmts);

  stmt& operator=(const stmt&) = default;
  stmt& operator=(stmt&&) = default;

  void accept(node_visitor* v) const { s->accept(v); }

  bool defined() const { return s != nullptr; }

  template <typename T>
  const T* as() const {
    if (s->type == T::static_type) {
      return reinterpret_cast<const T*>(s.get());
    }
    else {
      return nullptr;
    }
  }
};

class let : public expr_node<let> {
public:
  symbol_id name;
  expr value;
  expr body;

  void accept(node_visitor* v) const;

  static expr make(symbol_id name, expr value, expr body);

  static constexpr node_type static_type = node_type::let;
};

class variable : public expr_node<variable> {
public:
  symbol_id name;

  void accept(node_visitor* v) const;

  static expr make(symbol_id name);

  static constexpr node_type static_type = node_type::variable;
};

class constant : public expr_node<constant> {
public:
  index_t value;

  void accept(node_visitor* v) const;

  static expr make(index_t value);
  static expr make(const void* value);

  static constexpr node_type static_type = node_type::variable;
};

#define DECLARE_BINARY_OP(op) \
class op : public expr_node<op> { \
public: \
  expr a, b; \
  void accept(node_visitor* v) const; \
  static expr make(expr a, expr b); \
  static constexpr node_type static_type = node_type::op; \
};

DECLARE_BINARY_OP(add)
DECLARE_BINARY_OP(sub)
DECLARE_BINARY_OP(mul)
DECLARE_BINARY_OP(div)
DECLARE_BINARY_OP(mod)
DECLARE_BINARY_OP(min)
DECLARE_BINARY_OP(max)
DECLARE_BINARY_OP(equal)
DECLARE_BINARY_OP(not_equal)
DECLARE_BINARY_OP(less)
DECLARE_BINARY_OP(less_equal)
DECLARE_BINARY_OP(bitwise_and)
DECLARE_BINARY_OP(bitwise_or)
DECLARE_BINARY_OP(bitwise_xor)
DECLARE_BINARY_OP(logical_and)
DECLARE_BINARY_OP(logical_or)

#undef DECLARE_BINARY_OP

class call : public stmt_node<call> {
public:
  typedef index_t(*callable_t)(std::span<const index_t>, std::span<buffer_base*>);
  using callable = std::function<index_t(std::span<const index_t>, std::span<buffer_base*>)>;

  callable fn;
  std::vector<expr> scalar_args;
  std::vector<expr> buffer_args;

  void accept(node_visitor* v) const;

  static stmt make(callable fn, std::vector<expr> scalar_args, std::vector<expr> buffer_args);

  static constexpr node_type static_type = node_type::call;
};

class let_stmt : public stmt_node<let_stmt> {
public:
  symbol_id name;
  expr value;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(symbol_id name, expr value, stmt body);

  static constexpr node_type static_type = node_type::let_stmt;
};

class block : public stmt_node<block> {
public:
  stmt a, b;

  void accept(node_visitor* v) const;

  static stmt make(stmt a, stmt b);

  static constexpr node_type static_type = node_type::block;
};

class loop : public stmt_node<loop> {
public:
  symbol_id name;
  expr n;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(symbol_id name, expr n, stmt body);

  static constexpr node_type static_type = node_type::loop;
};

class if_then_else : public stmt_node<if_then_else> {
public:
  expr condition;
  stmt true_body;
  stmt false_body;

  void accept(node_visitor* v) const;

  static stmt make(expr condition, stmt true_body, stmt false_body = stmt());

  static constexpr node_type static_type = node_type::if_then_else;
};

class allocate : public stmt_node<allocate> {
public:
  memory_type type;
  symbol_id name;
  expr size;
  stmt body;

  void accept(node_visitor* v) const;

  static stmt make(memory_type type, symbol_id name, expr size, stmt body);

  static constexpr node_type static_type = node_type::allocate;
};

class check : public stmt_node<check> {
public:
  expr condition;

  void accept(node_visitor* v) const;

  static stmt make(expr condition);

  static constexpr node_type static_type = node_type::check;
};

class node_visitor {
public:
  virtual ~node_visitor() {}

  virtual void visit(const variable*) = 0;
  virtual void visit(const constant*) = 0;
  virtual void visit(const let*) = 0;
  virtual void visit(const add*) = 0;
  virtual void visit(const sub*) = 0;
  virtual void visit(const mul*) = 0;
  virtual void visit(const div*) = 0;
  virtual void visit(const mod*) = 0;
  virtual void visit(const min*) = 0;
  virtual void visit(const max*) = 0;
  virtual void visit(const equal*) = 0;
  virtual void visit(const not_equal*) = 0;
  virtual void visit(const less*) = 0;
  virtual void visit(const less_equal*) = 0;
  virtual void visit(const bitwise_and*) = 0;
  virtual void visit(const bitwise_or*) = 0;
  virtual void visit(const bitwise_xor*) = 0;
  virtual void visit(const logical_and*) = 0;
  virtual void visit(const logical_or*) = 0;

  virtual void visit(const let_stmt*) = 0;
  virtual void visit(const block*) = 0;
  virtual void visit(const loop*) = 0;
  virtual void visit(const if_then_else*) = 0;
  virtual void visit(const call*) = 0;
  virtual void visit(const allocate*) = 0;
  virtual void visit(const check*) = 0;
};

inline void variable::accept(node_visitor* v) const { v->visit(this); }
inline void constant::accept(node_visitor* v) const { v->visit(this); }
inline void let::accept(node_visitor* v) const { v->visit(this); }
inline void add::accept(node_visitor* v) const { v->visit(this); }
inline void sub::accept(node_visitor* v) const { v->visit(this); }
inline void mul::accept(node_visitor* v) const { v->visit(this); }
inline void div::accept(node_visitor* v) const { v->visit(this); }
inline void mod::accept(node_visitor* v) const { v->visit(this); }
inline void min::accept(node_visitor* v) const { v->visit(this); }
inline void max::accept(node_visitor* v) const { v->visit(this); }
inline void equal::accept(node_visitor* v) const { v->visit(this); }
inline void not_equal::accept(node_visitor* v) const { v->visit(this); }
inline void less::accept(node_visitor* v) const { v->visit(this); }
inline void less_equal::accept(node_visitor* v) const { v->visit(this); }
inline void bitwise_and::accept(node_visitor* v) const { v->visit(this); }
inline void bitwise_or::accept(node_visitor* v) const { v->visit(this); }
inline void bitwise_xor::accept(node_visitor* v) const { v->visit(this); }
inline void logical_and::accept(node_visitor* v) const { v->visit(this); }
inline void logical_or::accept(node_visitor* v) const { v->visit(this); }

inline void let_stmt::accept(node_visitor* v) const { v->visit(this); }
inline void block::accept(node_visitor* v) const { v->visit(this); }
inline void loop::accept(node_visitor* v) const { v->visit(this); }
inline void if_then_else::accept(node_visitor* v) const { v->visit(this); }
inline void call::accept(node_visitor* v) const { v->visit(this); }
inline void allocate::accept(node_visitor* v) const { v->visit(this); }
inline void check::accept(node_visitor* v) const { v->visit(this); }

expr make_variable(node_context& ctx, const std::string& name);

expr operator==(expr a, expr b);
expr operator!=(expr a, expr b);
expr operator<(expr a, expr b);
expr operator<=(expr a, expr b);
expr operator>(expr a, expr b);
expr operator>=(expr a, expr b);
expr operator&&(expr a, expr b);
expr operator||(expr a, expr b);
expr min(expr a, expr b);
expr max(expr a, expr b);

}  // namespace slinky

#endif  // SLINKY_EXPR_H
