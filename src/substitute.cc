#include "substitute.h"

#include <cassert>
#include <iostream>

#include "node_mutator.h"

namespace slinky {

class substitutor : public node_mutator {
public:
  const std::map<symbol_id, expr>& replacements;

  substitutor(const std::map<symbol_id, expr>& replacements) : replacements(replacements) {}

  void visit(const variable* v) override {
    auto i = replacements.find(v->name);
    if (i != replacements.end()) {
      e = i->second;
    } else {
      e = v;
    }
  }
};

expr substitute(const expr& e, const std::map<symbol_id, expr>& replacements) {
  return substitutor(replacements).mutate(e);
}

stmt substitute(const stmt& s, const std::map<symbol_id, expr>& replacements) {
  return substitutor(replacements).mutate(s);
}

}  // namespace slinky
