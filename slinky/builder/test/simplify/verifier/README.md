# Simplifier rule verifier

This directory proves the rewrite rules in [`simplify_rules.h`](../../../simplify_rules.h) correct with SMT solvers.
It complements the fuzz tests in [`rule_tester.h`](../rule_tester.h). A fuzzer tries many random inputs, so it can miss
the rare one that breaks a rule. An SMT solver reasons about all inputs at once, so an `unsat` result is a proof that
no input breaks the rule, under the assumptions described in [The SMT encoding](#the-smt-encoding).

## Architecture

There are two stages:

1. **`simplifier_to_smt`** (C++) translates the rule DSL to SMT-LIB2. It writes one query per (rule, replacement) to an
   output directory, plus a `manifest.tsv` that lists each query's id, ruleset and rule text. Each query is `unsat` iff
   the replacement is correct (i.e. there is no _counterexample_ for the rule). A rule that uses something the encoding
   can't model (e.g. most intrinsic calls) is an error, so every rule is verified.
2. **`verify_simplifier_rules.py`** runs one solver (z3 or cvc5) on every query in the manifest and checks the
   results against [`known_gaps.toml`](known_gaps.toml), which lists the rules that the verifier can't prove: those
   z3 refutes (in `[counterexample]`, with a counterexample), and those it can't decide (in `[inconclusive]`).

The run fails if:

- a rule has an unexpected counterexample;
- a known counterexample is now proven, or no longer matches any rule;
- with z3, a rule is inconclusive and isn't listed in `[inconclusive]`, or is listed but is now decided or no longer
  matches any rule;
- a self-test rule isn't refuted (see [Self-test](#self-test));
- a solver reports an error, including a counterexample whose model fails validation.

Timeouts and `unknown` results are reported as inconclusive. A rule that z3 can't decide isn't verified, so z3's must
match `[inconclusive]`. z3 runs with a resource limit (its `rlimit`, which counts its steps) and answers `unknown` when
a query exceeds it, so which rules are inconclusive depends on z3's version, not on the machine. For z3, the wall-clock
timeout is only a backstop. Other solvers' inconclusive results don't fail the run.

z3 is the reference solver: CI ([`verify_rules.yml`](../../../../../.github/workflows/verify_rules.yml)) fails if the z3
job does, and the known-gaps file records z3's results. cvc5 runs as an advisory job that fails only when it
disagrees with z3, which doesn't fail the workflow. Both solvers validate their counterexamples, but neither certifies a
proof, so a disagreement is worth investigating: it means one of them has a bug.

## Running it

Build the emitter and write the queries:

```sh
cmake --build build --target simplifier_to_smt
build/slinky/builder/test/simplify/verifier/simplifier_to_smt -o /tmp/rules
```

Pass `--ruleset min,max` to write only some rulesets. The ruleset names are the `apply_<name>_rules` functions in
`simplify_rules.h`; `simplifier_to_smt --help` lists them.

Check them with a solver on your `PATH` (z3 is the default, and by far the fastest):

```sh
slinky/builder/test/simplify/verifier/verify_simplifier_rules.py /tmp/rules
```

Useful options:

- `--solver z3|cvc5` picks the solver.
- `--timeout SECONDS` is the per-query timeout (default: 300 for z3, where it's a backstop for the rlimit, and 60 for
  cvc5).
- `--jobs N` sets the number of queries to run in parallel (default: CPU count).
- `--known-gaps PATH` uses a different known-gaps file (default: `known_gaps.toml`).
- `--update-known-gaps` rewrites the known-gaps file to match the latest results, keeping the notes of existing
  entries. It only works with z3, and is incompatible with `simplifier_to_smt`'s `--ruleset` option.

To look at a single rule, find its id in the output or in `manifest.tsv` and run a solver on its query directly, e.g.
`z3 /tmp/rules/min_012_r0.smt2`. Each query starts with a comment giving the rule, and ends with `(get-model)`, so a
`sat` result prints a counterexample in SMT-LIB format. The script normalizes counterexamples to one line, e.g.
`x=+inf, y=3, c0=2`.

## The SMT encoding

A rule is `apply(pattern, when(p0, r0), when(p1, r1), ..., rn)`: the pattern is rewritten to the first replacement `ri`
whose predicate `pi` holds, or to the last replacement `rn` (which is optional) if none of them do. The emitter writes a
query for each replacement that asserts the conditions under which the simplifier would use it, and that its result
differs from the pattern's. If the solver finds that unsatisfiable, the replacement is correct.

### Values

Integers are SMT `Int`s, with booleans as 0/1. The rules use *extended* semantics, in which a wildcard (`x`, `y`, ...)
ranges over the integers plus `+inf` and `-inf`. A value is an element of the SMT datatype

```
(declare-datatype X ((fin (val Int)) (pinf) (ninf) (ind)))
```

where `(fin v)` is the finite value `v`, `pinf` is `+inf`, `ninf` is `-inf` and `ind` is indeterminate. A wildcard
`x` is a constant of type `X`, which is never indeterminate, because `simplify()` returns early on indeterminate
operands, and which is in the `int64` range when it is finite:

```
(declare-const x X)
(assert (not ((_ is ind) x)))
(assert (=> ((_ is fin) x) (in64 (val x))))
```

Constants (`c0`, `c1`, ...) are always finite, so they are plain `Int`s, written `(fin c0)` where they meet an `X`.

### Operations

The prelude of each query defines a helper `xop` for each operation, which follow `simplify()` in
[`simplify_exprs.cc`](../../../simplify_exprs.cc):

- indeterminate propagates through everything (a poison value);
- `inf + -inf`, `inf * 0`, `inf / inf`, `inf / 0` and `mod` with an infinity are indeterminate;
- `finite / inf` is 0;
- `min`, `max` and comparisons use the total order `-inf < Z < +inf`;
- infinities are truthy.

Division and modulo are Euclidean, and `x / 0 == x % 0 == 0`, as in slinky. These are the prelude's `ediv` and `emod`.

Each non-trivial node of a rule becomes a named term, `(define-fun nK () X (xadd x (fin c0)))`. Structurally identical
nodes share a name, which helps the solvers, especially with lemmas (see below). When all of a node's operands are known
to be finite (e.g. arithmetic on constants), the emitter skips the helpers and writes plain integer arithmetic. Such a
term is an `Int` expression, and is only wrapped in `fin` where it meets a term that isn't known to be finite.

Optional subpatterns such as `may_be<0>(c0)` need no special handling: when the pattern matches without them, they
take their default value, and the solver already considers that value for the free constant.

### Overflow

The rules are applied to `int64` expressions, but SMT Ints are unbounded. The encoding bridges the two like this:

- Finite wildcards and constants are assumed to be in the `int64` range (`in64`).
- Every finite subterm of the pattern that contains a wildcard is assumed to be in range too: the pattern is an
  expression that already exists, and the rules don't need to be correct for one that overflows.
- Subterms that `rewrite.h` constant folds (those evaluated on `index_t`, e.g. `c0 + c1` in a replacement or a
  predicate) get the same overflow checks that `rewrite.h` does. A rule doesn't fire if folding overflows, so "nothing
  overflows" is a premise of the query.

### Predicates and fall-through

The query for replacement `i` assumes that:

1. its predicate `pi` holds;
2. its folded subterms don't overflow;
3. every earlier replacement `j < i` didn't fire.

Predicates on wildcards (`is_non_negative(x)`, `is_constant(x)`, ...) are decided by the simplifier from what it knows
about an expression, so they have no exact SMT encoding. Instead, the emitter computes two approximations of each
predicate `P`: *may* (`P => may`) and *must* (`must => P`). For example, `is_non_negative(x)` *may* hold when `x` is
`+inf` or a finite value `>= 0`, and its *must* is `false`. When the argument is a constant expression, such as
`is_non_negative(c0 - c1)`, the predicate is exact and *may* and *must* are the same. Condition 1 uses *may*, and
condition 3 uses *not must*. Both overapproximate the cases in which replacement `i` is used, so an `unsat` result
covers every case the simplifier can actually reach. `!` swaps the two approximations, and `&&`/`||` combine them
pointwise.

### The correctness condition

The final assertion is that the replacement is wrong:

```
(assert (not (or ((_ is ind) pattern) (= pattern replacement))))
```

That is, a rewrite is correct if the pattern is indeterminate (any result is allowed), or the replacement is equal to
it.

### Staircase bounds

Some rules use `staircase_sum_min`/`staircase_sum_max`, the bounds that `staircase_sum_bounds` in
[`arithmetic.cc`](../../../../base/arithmetic.cc) computes for `g(k) = ((k + a1)/b1)*c1 + ((k + a2)/b2)*c2`. Rather
than encoding that search, the emitter introduces a fresh constant `m`, and assumes that it bounds `g` when the C++
returns a bound. `staircase_sum_bounds` takes `int`, so it sees the arguments truncated to 32 bits, and it returns a
bound when, for those, `|b1|, |b2| <= 1024`, and either `b1 == b2 == 0` (both terms are constant) or `b1, b2 != 0` and
`c1*b2 == -c2*b1` (`g` is periodic). Otherwise, `rewrite.h` uses `INT64_MIN` or `INT64_MAX` as the bound, and the rule
can still fire. The encoding assumes that the `int` arithmetic in `staircase_sum_bounds` doesn't overflow. The bound
is only instantiated at the rule's own wildcard `k = x`, which is a weaker hypothesis than a bound for all `k`, but
keeps the queries quantifier-free.

### Lemmas

Nonlinear integer arithmetic (division and modulo by symbolic constants) is hard for SMT solvers. A rule can help the
solver by passing hints to `when` after the replacement, as in `when(c0 > 0, r, euclid(c1, c0))` (see
[`rewrite.h`](../../../rewrite.h)). Hints are ignored when rules are applied. Each hint applies a lemma that holds for
all integers:

- `cong(u, v, d, k)`: if `d != 0` and `u == v + d*k`, then `u % d == v % d` and `u / d == v / d + k`;
- `euclid(a, d)`: if `d != 0`, then `a == d*(a/d) + a%d`;
- `negdiv(s, t, d)`: if `d > 0` and `s == d - 1 - t`, then `s / d == -(t / d)`.

The emitter asserts each hint, instantiated with the rule's terms, as a hypothesis of the query (for finite values
only). This is sound because the lemmas are proven separately: every run also writes the queries `lemma_cong`,
`lemma_euclid` and `lemma_negdiv`, which are `unsat` iff each lemma holds.

### Self-test

A bug in the encoding could add contradictory assumptions, which would make every query `unsat`, so every rule would
look proven. To guard against that, the emitter always writes queries for a few deliberately wrong rules. They exercise
each part of the encoding: constants, fall-through between replacements, wildcard predicates and infinities. All of them
must be `sat`, and the verifier fails the run if any of them isn't.
