#!/usr/bin/env python3
"""
Check the SMT queries written by simplifier_to_smt with one solver.

Each query is unsat iff its replacement is correct. The rules that the verifier can't prove are listed in the
known-gaps file: those z3 refutes (with a counterexample), and those it can't decide within its rlimit. z3 is the
reference solver: the file records its results, only it may update the file, and a z3 run fails if its results don't
match the file. Another solver checks that it agrees with z3, so a run with it fails if it finds a counterexample z3
didn't, or proves a rule z3 refuted; its inconclusive results don't matter.

Usage:
  simplifier_to_smt -o DIR
  verify_simplifier_rules.py [--solver z3|cvc5] [--jobs N] DIR

The manifest also has queries from deliberately wrong rules (ruleset selftest), which must all be sat. They guard
against a vacuous encoding, in which contradictory assumptions would make every real rule look proven. A self-test
query that isn't sat fails the run; the known gaps don't apply to them.

Exit code is 0 on success and 1 on failure (see --known-gaps).
"""

import argparse
import asyncio
import collections
import dataclasses
import enum
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tomllib

# Each solver is a command line, in which {f} is the path of the query. z3 answers unknown when a query exceeds its
# rlimit, which counts its steps, so which rules are inconclusive doesn't depend on the machine.
PRESETS = {
    "z3": ["z3", "-smt2", "model_validate=true", "rlimit=250000000", "{f}"],
    "cvc5": ["cvc5", "--lang", "smt2", "--produce-models", "--check-models", "{f}"],
}
# Default seconds per query. For z3, this is only a backstop, so even a slow machine reaches the rlimit first. (z3
# doesn't check the rlimit everywhere, so a few queries do run until the timeout.)
TIMEOUTS = {"z3": 300, "cvc5": 60}


class Status(enum.StrEnum):
    PROVEN = "proven"
    COUNTEREXAMPLE = "counterexample"
    EXPECTED = "expected"  # A known counterexample, or one for a self-test rule. Only used for display.
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    ERROR = "error"


SELFTEST = "selftest"  # The ruleset of the deliberately wrong rules, which must all be sat.
REFERENCE = "z3"  # The solver whose results the known-gaps file records.
UNDECIDED = (Status.UNKNOWN, Status.TIMEOUT)


class Reason(enum.StrEnum):
    UNEXPECTED = "unexpected counterexample"
    OUT_OF_DATE = "known counterexample out of date"
    INCONCLUSIVE = "unexpected inconclusive result"
    INCONCLUSIVE_OUT_OF_DATE = "known inconclusive result out of date"
    NOT_REFUTED = "self-test rule not refuted"
    ERROR = "error"


# These are fixed by --update-known-gaps instead of failing the run.
FIXABLE = (Reason.UNEXPECTED, Reason.OUT_OF_DATE, Reason.INCONCLUSIVE, Reason.INCONCLUSIVE_OUT_OF_DATE)


@dataclasses.dataclass(frozen=True)
class Query:
    id: str
    ruleset: str
    path: pathlib.Path
    rule_text: str


@dataclasses.dataclass
class Result:
    query: Query
    status: Status
    model: str = None
    detail: str = ""


def parse_output(text):
    """Returns (verdict, rest) where verdict is sat/unsat/unknown or None."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if s:
            if s in ("sat", "unsat", "unknown"):
                return s, "\n".join(lines[i + 1 :]).strip()
            break
    return None, text


def reported_error(text, etext):
    """True if a solver reported an error, e.g. z3 with model_validate=true when the model is bogus."""
    if "validat" in etext.lower():
        return True
    return any(line.lstrip().startswith("(error") for line in (text + "\n" + etext).splitlines())


def load_queries(directory):
    """Returns (queries, partial), where partial is true if simplifier_to_smt only wrote some rulesets (--ruleset)."""
    with (directory / "manifest.tsv").open(encoding="utf-8") as f:
        line = f.readline()
        # A partial run starts the manifest with a comment listing its rulesets.
        partial = line.startswith("#")
        if partial:
            line = f.readline()
        header = line.rstrip("\n").split("\t")
        rows = [
            dict(zip(header, line.rstrip("\n").split("\t", len(header) - 1), strict=True)) for line in f if line.strip()
        ]
    queries = [Query(r["id"], r["ruleset"], directory / (r["id"] + ".smt2"), r["rule_text"]) for r in rows]
    return queries, partial


def load_gaps(path):
    """Returns (counterexample, inconclusive), which map the rule texts in each table of the known-gaps file to their
    notes."""
    with path.open("rb") as f:
        gaps = tomllib.load(f)
    return gaps.get("counterexample", {}), gaps.get("inconclusive", {})


# The value of a wildcard or constant in a model. Negative integers are printed as (- 3). The solvers differ in
# whitespace, so this must not depend on it.
MODEL_VALUE = re.compile(r"\(define-fun ([a-z]|c\d+) \(\) Int\s+(\(- \d+\)|\d+)\)")


def format_model(model):
    """Renders a solver's model on one line, e.g. 'x=-1, y=3, c0=2'."""
    values = {name: int(re.sub(r"[()\s]", "", text)) for name, text in MODEL_VALUE.findall(model)}
    return ", ".join(f"{name}={values[name]}" for name in sorted(values, key=lambda n: (n[0] == "c", n)))


GAPS_HEADER = """\
# The rules that the verifier can't prove, as printed by simplifier_to_smt, each with a note. z3 refutes the rules in
# [counterexample] (the note is a counterexample), and can't decide the ones in [inconclusive] within its rlimit.
# Maintain with verify_simplifier_rules.py --update-known-gaps, which keeps the notes of existing entries.
"""


def updated(table, old, new, note):
    """Returns old (rule text to note) with only the rule texts in new (rule text to result). Existing entries keep
    their notes, so rerunning doesn't churn the file, and new ones are appended with note(result)."""
    added = [t for t in new if t not in old]
    removed = [t for t in old if t not in new]
    print(f"\n[{table}]: {len(added)} added, {len(removed)} removed")
    for t in added:
        print("  added:   " + t)
    for t in removed:
        print("  removed: " + t)
    return {t: old[t] for t in old if t in new} | {t: note(new[t]) for t in added}


def write_gaps(path, counterexample, inconclusive):
    # A JSON string literal is also a TOML basic string, so json.dumps quotes the keys and values.
    lines = [GAPS_HEADER]
    for table, entries in (("counterexample", counterexample), ("inconclusive", inconclusive)):
        lines.append(f"[{table}]")
        lines += [f"{json.dumps(text)} = {json.dumps(note)}" for text, note in entries.items()]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


async def run_solver(q, solver, timeout):  # noqa: ASYNC109 (a timeout is a verdict, not an exception)
    """Runs the solver on one query."""
    argv = [a.replace("{f}", str(q.path)) for a in PRESETS[solver]]
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except OSError as e:
        return Result(q, Status.ERROR, detail=f"failed to start: {e}")
    try:
        async with asyncio.timeout(timeout):
            out, err = await proc.communicate()
    except TimeoutError:
        return Result(q, Status.TIMEOUT)
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
    text = out.decode("utf-8", "replace")
    etext = err.decode("utf-8", "replace")
    verdict, rest = parse_output(text)
    if verdict is None:
        snippet = (text.strip() or etext.strip())[:500]
        return Result(q, Status.ERROR, detail=f"exit {proc.returncode}, unparseable output: {snippet}")
    if verdict == "unsat":
        return Result(q, Status.PROVEN)
    if verdict == "unknown":
        return Result(q, Status.UNKNOWN)
    if proc.returncode != 0 or reported_error(text, etext):
        snippet = (etext.strip() or text.strip())[:500]
        return Result(q, Status.ERROR, detail=f"sat rejected (exit {proc.returncode}): {snippet}")
    return Result(q, Status.COUNTEREXAMPLE, model=rest)


async def run_all(queries, solver, args):
    limit = asyncio.Semaphore(max(1, args.jobs))
    total = len(queries)
    done = 0

    async def run_one(q):
        nonlocal done
        async with limit:
            res = await run_solver(q, solver, args.timeout)
        done += 1
        if done % 50 == 0 or done == total:
            print(f"[{done}/{total}] queries done", file=sys.stderr)
        return res

    return await asyncio.gather(*map(run_one, queries))


def failure_reason(r, known, inconclusive):
    """Why a result fails the run, or None if it is fine. known and inconclusive are the rule texts of the known
    counterexamples and of the known inconclusive results; inconclusive is None if the solver isn't the reference."""
    text = r.query.rule_text
    if r.query.ruleset == SELFTEST:
        return None if r.status == Status.COUNTEREXAMPLE else Reason.NOT_REFUTED
    if r.status == Status.ERROR:
        return Reason.ERROR
    if r.status == Status.COUNTEREXAMPLE and text not in known:
        return Reason.UNEXPECTED
    if r.status == Status.PROVEN and text in known:
        return Reason.OUT_OF_DATE
    if inconclusive is not None and (r.status in UNDECIDED) != (text in inconclusive):
        return Reason.INCONCLUSIVE if r.status in UNDECIDED else Reason.INCONCLUSIVE_OUT_OF_DATE
    return None


def failure_entry(r):
    """The lines describing a failing result: its id and rule, then the evidence."""
    head = f"{r.query.id}: {r.query.rule_text}"
    if r.status == Status.COUNTEREXAMPLE:
        # The raw model on one line if it isn't in the form format_model expects.
        return f"{head}\n{format_model(r.model) or ' '.join(r.model.split())}"
    if r.status == Status.ERROR:
        return f"{head}\n{r.detail}"
    return f"{head}\n{r.status}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", type=pathlib.Path, help="directory of queries written by simplifier_to_smt")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="queries in flight (default: CPU count)")
    ap.add_argument(
        "--timeout", type=float, help="seconds per query (default: 300 for z3, a backstop for its rlimit; 60 for cvc5)"
    )
    ap.add_argument("--solver", choices=list(PRESETS), default="z3", help="solver to run (default z3)")
    ap.add_argument(
        "--known-gaps",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("known_gaps.toml"),
        metavar="PATH",
        help="file of the rules the verifier can't prove (default: known_gaps.toml next to this script)",
    )
    ap.add_argument(
        "--update-known-gaps",
        action="store_true",
        help="rewrite the --known-gaps file to match the results: add new counterexamples (with their model) and "
        "inconclusive rules, and remove the rules that no longer belong or no longer exist. "
        "Needs a full manifest, i.e. one written without --ruleset (z3 only)",
    )
    args = ap.parse_args()
    if not (args.directory / "manifest.tsv").exists():
        ap.error(f"no manifest.tsv in {args.directory}")

    solver = args.solver
    if args.timeout is None:
        args.timeout = TIMEOUTS[solver]
    if not shutil.which(PRESETS[solver][0]):
        ap.error(f"solver {solver} not found: {PRESETS[solver][0]} is not on PATH")
    print(f"solver {solver}: {' '.join(PRESETS[solver])}", file=sys.stderr)

    queries, partial = load_queries(args.directory)
    if args.update_known_gaps and partial:
        ap.error("the manifest is from a partial run (--ruleset); can't update the known gaps from it")
    if args.update_known_gaps and solver != REFERENCE:
        ap.error(f"the known gaps are {REFERENCE}'s results; can't update them with {solver}")
    known, inconclusive = load_gaps(args.known_gaps)
    known_texts = set(known)
    inconclusive_texts = set(inconclusive) if solver == REFERENCE else None

    try:
        results = asyncio.run(run_all(queries, solver, args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 1

    reasons = collections.defaultdict(list)
    undecided = []  # Inconclusive results that don't fail the run.
    for r in results:
        reason = failure_reason(r, known_texts, inconclusive_texts)
        if reason:
            reasons[reason].append(r.query.rule_text if reason == Reason.OUT_OF_DATE else failure_entry(r))
        elif r.status in UNDECIDED:
            undecided.append(r.query.id)
    if not partial:
        rule_texts = {q.rule_text for q in queries}
        for reason, texts in ((Reason.OUT_OF_DATE, known_texts), (Reason.INCONCLUSIVE_OUT_OF_DATE, inconclusive_texts)):
            unmatched = [t for t in texts or () if t not in rule_texts]
            if unmatched:
                reasons[reason] += unmatched
    if args.update_known_gaps:
        reasons = {k: v for k, v in reasons.items() if k not in FIXABLE}

    def label(r):
        expected = r.query.ruleset == SELFTEST or r.query.rule_text in known_texts
        return Status.EXPECTED if r.status == Status.COUNTEREXAMPLE and expected else r.status

    counts = collections.Counter(map(label, results))
    print(f"{len(results)} queries: " + ", ".join(f"{counts[s]} {s}" for s in Status if counts[s]))
    if undecided:
        print("inconclusive (not failures): " + ", ".join(undecided))
    for reason, entries in reasons.items():
        print(f"FAIL: {reason} ({len(entries)})")
        for entry in entries:
            print("  " + entry.replace("\n", "\n    "))
    ok = not reasons

    if args.update_known_gaps and ok:
        rules = [r for r in results if r.query.ruleset != SELFTEST]
        sat = {r.query.rule_text: r for r in rules if r.status == Status.COUNTEREXAMPLE}
        unsure = {r.query.rule_text: r for r in rules if r.status in UNDECIDED}
        write_gaps(
            args.known_gaps,
            updated("counterexample", known, sat, lambda r: format_model(r.model)),
            updated("inconclusive", inconclusive, unsure, lambda r: str(r.status)),
        )

    print("\nOK" if ok else "\nFAILED")
    if not ok and solver != REFERENCE:
        print(f"{solver} disagrees with {REFERENCE}: either solver may be wrong, so check the queries above with both.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
