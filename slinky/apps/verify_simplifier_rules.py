#!/usr/bin/env python3
"""Check the SMT queries written by simplifier_to_smt with a portfolio of solvers.

Each query is unsat iff the corresponding rewrite rule is correct. Every query
is handed to all configured solvers concurrently; the first definitive answer
(sat/unsat) wins and the remaining solver processes are killed.

Usage:
  simplifier_to_smt -o DIR
  verify_simplifier_rules.py [--solver z3] [--solver cvc5] [--jobs N] DIR

Exit code is 0 on success and 1 on failure (see --known-failures and
--expect-all-sat).
"""

import argparse
import collections
import concurrent.futures
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

PRESETS = {
    "z3": ("z3", "z3 -smt2 model_validate=true {f}"),
    "cvc5": ("cvc5", "cvc5 --lang smt2 --produce-models {f}"),
    "yices": ("yices-smt2", "yices-smt2 --mcsat {f}"),
}

PROVEN = "proven"
COUNTEREXAMPLE = "counterexample"
EXPECTED_FAILURE = "expected_failure"
UNKNOWN = "unknown"
TIMEOUT = "timeout"
ERROR = "error"
SKIPPED = "skipped"
DISAGREEMENT = "disagreement"
STATUSES = [PROVEN, COUNTEREXAMPLE, EXPECTED_FAILURE, UNKNOWN, TIMEOUT, ERROR, SKIPPED, DISAGREEMENT]

_live = set()
_live_lock = threading.Lock()
# Set on Ctrl-C. Workers check it before and after launching a solver, so nothing
# started in a race with kill_all() survives.
_stop = threading.Event()


def _kill(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def kill_all():
    with _live_lock:
        procs = list(_live)
    for p in procs:
        _kill(p)


class Solver:
    def __init__(self, name, template):
        self.name = name
        self.argv_template = shlex.split(template)

    def argv(self, path):
        return [a.replace("{f}", path) for a in self.argv_template]


def parse_solver(spec):
    if spec in PRESETS:
        return Solver(spec, PRESETS[spec][1])
    if "{f}" not in spec:
        raise ValueError("unknown solver preset %r (known: %s) and not a template containing {f}"
                         % (spec, ", ".join(PRESETS)))
    return Solver(os.path.basename(shlex.split(spec)[0]), spec)


def parse_output(text):
    """Returns (verdict, rest) where verdict is sat/unsat/unknown or None."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if s in ("sat", "unsat", "unknown"):
            return s, "\n".join(lines[i + 1:]).strip()
        return None, text
    return None, text


class Query:
    def __init__(self, qid, ruleset, kind, path, rule_text=""):
        self.id = qid
        self.ruleset = ruleset
        self.kind = kind
        self.path = path
        self.rule_text = rule_text


def load_queries(directory):
    queries = []
    manifest = os.path.join(directory, "manifest.tsv")
    seen_files = set()
    if os.path.exists(manifest):
        with open(manifest, encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                cols = line.split("\t", len(header) - 1)
                cols += [""] * (len(header) - len(cols))
                row = dict(zip(header, cols))
                kind = row.get("kind", "query")
                path = None
                if kind != "skipped":
                    # A missing file is reported as an error by run_query.
                    path = os.path.join(directory, row["id"] + ".smt2")
                    seen_files.add(row["id"] + ".smt2")
                queries.append(Query(row["id"], row.get("ruleset", ""), kind, path,
                                     row.get("rule_text", "")))
    for name in sorted(os.listdir(directory)):
        if name.endswith(".smt2") and name not in seen_files:
            stem = name[:-len(".smt2")]
            m = re.match(r"^(.*?)_\d+_r\d+", stem)
            queries.append(Query(stem, m.group(1) if m else "", "query", os.path.join(directory, name)))
    return queries


def read_rule_text_from_file(path):
    """Fallback: the leading comment block of the query file."""
    lines = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith(";"):
                    lines.append(line[1:].strip())
                else:
                    break
    except OSError:
        pass
    return "\n".join(lines)


def run_query(q, solvers, timeout, cross_check):
    """Runs the portfolio on one query. Returns a result dict."""
    res = {"id": q.id, "ruleset": q.ruleset, "kind": q.kind, "path": q.path,
           "rule_text": q.rule_text, "status": ERROR, "solver": None, "model": None,
           "answers": {}, "detail": "", "seconds": 0.0}
    if _stop.is_set():
        res["detail"] = "interrupted"
        return res
    if not os.path.exists(q.path):
        res["detail"] = "query file missing: " + q.path
        return res

    start = time.monotonic()
    running = {}  # solver name -> (proc, outfile)
    answers = {}  # solver name -> (verdict, rest)
    errors = {}
    try:
        for s in solvers:
            if _stop.is_set():
                break
            out = tempfile.TemporaryFile()
            err = tempfile.TemporaryFile()
            try:
                p = subprocess.Popen(s.argv(q.path), stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                     start_new_session=True)
            except OSError as e:
                errors[s.name] = "failed to start: %s" % e
                out.close()
                err.close()
                continue
            with _live_lock:
                _live.add(p)
            running[s.name] = (p, out, err)
            if _stop.is_set():  # kill_all() may have run before p was registered
                break

        timed_out = False
        while running:
            finished = []
            for name, (p, out, err) in running.items():
                if p.poll() is not None:
                    finished.append(name)
            for name in finished:
                p, out, err = running.pop(name)
                out.seek(0)
                text = out.read().decode("utf-8", "replace")
                err.seek(0)
                etext = err.read().decode("utf-8", "replace")
                out.close()
                err.close()
                with _live_lock:
                    _live.discard(p)
                verdict, rest = parse_output(text)
                if verdict is None:
                    errors[name] = "exit %s, unparseable output: %s" % (
                        p.returncode, (text.strip() or etext.strip())[:500])
                elif verdict == "sat" and (re.search(r"\(error|validat", text + etext, re.I)
                                           or p.returncode != 0):
                    # E.g. z3 with model_validate=true when the model is bogus.
                    errors[name] = "sat rejected (exit %s): %s" % (
                        p.returncode, (etext.strip() or text.strip())[:500])
                else:
                    answers[name] = (verdict, rest)
            # Only unsat may end the query early: a lone sat must be confirmed.
            if not cross_check and any(v == "unsat" for v, _ in answers.values()):
                break
            if not running or _stop.is_set():
                break
            if time.monotonic() - start > timeout:
                timed_out = True
                break
            time.sleep(0.01)
    finally:
        for name, (p, out, err) in list(running.items()):
            _kill(p)
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            out.close()
            err.close()
            with _live_lock:
                _live.discard(p)
        running.clear()
    res["seconds"] = round(time.monotonic() - start, 3)
    res["answers"] = {n: v for n, (v, _) in answers.items()}
    if errors:
        res["detail"] = "; ".join("%s: %s" % kv for kv in sorted(errors.items()))

    definitive = {n: v for n, (v, _) in answers.items() if v in ("sat", "unsat")}
    if len(set(definitive.values())) > 1:
        res["status"] = DISAGREEMENT
        res["detail"] = "solvers disagree: " + ", ".join("%s=%s" % kv for kv in sorted(definitive.items()))
        res["model"] = "\n\n".join("--- %s (%s) ---\n%s" % (n, v, r) for n, (v, r) in sorted(answers.items())
                                   if v == "sat")
    elif definitive:
        winner = sorted(definitive)[0]
        res["solver"] = winner
        if definitive[winner] == "unsat":
            res["status"] = PROVEN
        else:
            res["status"] = COUNTEREXAMPLE
            res["model"] = answers[winner][1]
            res["confirmed_by"] = sorted(definitive)
            res["unconfirmed_by"] = sorted(s.name for s in solvers if s.name not in definitive)
    elif timed_out:
        res["status"] = TIMEOUT
    elif answers:
        res["status"] = UNKNOWN
        res["solver"] = sorted(answers)[0]
    else:
        res["status"] = ERROR
    return res


def load_known_failures(path):
    known = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            qid, _, reason = line.partition("#")
            known[qid.strip()] = reason.strip()
    return known


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", help="directory of queries written by simplifier_to_smt")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="queries in flight (default: CPU count)")
    ap.add_argument("--timeout", type=float, default=60, help="seconds per query (default 60)")
    ap.add_argument("--solver", action="append", default=[],
                    help="preset (%s) or command template containing {f}; repeatable. "
                         "Default: every preset found on PATH" % ", ".join(PRESETS))
    ap.add_argument("--json", metavar="PATH", help="write a machine-readable report")
    ap.add_argument("--known-failures", metavar="PATH", help="file of query ids expected to be sat")
    ap.add_argument("--expect-all-sat", action="store_true",
                    help="succeed only if every query is sat (emitter self-test)")
    ap.add_argument("--filter", metavar="REGEX", help="only run queries whose id matches REGEX (re.search)")
    ap.add_argument("--copy-failures", metavar="DIR", help="copy the .smt2 files of failing queries to DIR")
    ap.add_argument("--cross-check", action="store_true",
                    help="wait for every solver instead of stopping at the first answer, "
                         "so that all disagreements are detected (a sat is always awaited for confirmation; "
                         "only unsat ends a query early otherwise)")
    args = ap.parse_args()

    try:
        if args.solver:
            solvers = [parse_solver(s) for s in args.solver]
        else:
            solvers = [parse_solver(n) for n, (exe, _) in PRESETS.items() if shutil.which(exe)]
    except ValueError as e:
        ap.error(str(e))
    if not solvers:
        ap.error("no solvers found on PATH; install z3 or cvc5, or pass --solver")
    for s in solvers:
        try:
            vp = subprocess.Popen([s.argv_template[0], "--version"], stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True,
                                  start_new_session=True)
            try:
                vout, verr = vp.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                _kill(vp)
                vp.communicate()
                raise
            except KeyboardInterrupt:
                _kill(vp)
                print("interrupted", file=sys.stderr)
                return 1
            ver = ((vout or verr).strip().splitlines() or ["?"])[0]
        except (OSError, subprocess.TimeoutExpired) as e:
            ver = "version unavailable: %s" % e
        print("solver %s: %s [%s]" % (s.name, ver, " ".join(s.argv_template)), file=sys.stderr)

    if not os.path.isdir(args.directory):
        ap.error("not a directory: " + args.directory)
    queries = load_queries(args.directory)
    if args.filter:
        rx = re.compile(args.filter)
        queries = [q for q in queries if rx.search(q.id)]
    known = load_known_failures(args.known_failures) if args.known_failures else {}

    results = []
    todo = []
    for q in queries:
        if q.kind == "skipped":
            results.append({"id": q.id, "ruleset": q.ruleset, "kind": q.kind, "path": None,
                            "rule_text": q.rule_text, "status": SKIPPED, "solver": None, "model": None,
                            "answers": {}, "detail": "", "seconds": 0.0})
        else:
            if not q.rule_text:
                q.rule_text = read_rule_text_from_file(q.path)
            todo.append(q)

    done = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            try:
                futs = [ex.submit(run_query, q, solvers, args.timeout, args.cross_check) for q in todo]
                for fut in concurrent.futures.as_completed(futs):
                    r = fut.result()
                    results.append(r)
                    done += 1
                    if done % 50 == 0 or done == len(todo):
                        print("[%d/%d] queries done" % (done, len(todo)), file=sys.stderr)
            except KeyboardInterrupt:
                # Leaving the with block joins the workers, so stop them first: drop the queued
                # queries and kill the running solvers (which are in their own sessions and never
                # saw the SIGINT).
                _stop.set()
                ex.shutdown(wait=False, cancel_futures=True)
                kill_all()
                raise
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 1
    finally:
        kill_all()

    order = {q.id: i for i, q in enumerate(queries)}
    results.sort(key=lambda r: order.get(r["id"], 0))

    # Classify against the allowlist.
    failures = []  # (id, why)
    stale = []
    for r in results:
        r["failing"] = False
        if r["status"] == COUNTEREXAMPLE and r["id"] in known:
            r["status"] = EXPECTED_FAILURE
        if r["status"] == COUNTEREXAMPLE:
            r["failing"] = not args.expect_all_sat
        elif r["status"] == PROVEN and r["id"] in known:
            stale.append(r["id"])
            r["failing"] = True
        elif r["status"] in (DISAGREEMENT, ERROR):
            r["failing"] = True
        if args.expect_all_sat and r["status"] not in (COUNTEREXAMPLE, EXPECTED_FAILURE, SKIPPED):
            r["failing"] = True
    ran = [r for r in results if r["status"] != SKIPPED]
    known_missing = sorted(set(known) - {r["id"] for r in results}) if not args.filter else []

    # Summary table.
    rulesets = collections.OrderedDict()
    for r in results:
        rulesets.setdefault(r["ruleset"] or "(none)", collections.Counter())[r["status"]] += 1
    cols = [s for s in STATUSES if any(c[s] for c in rulesets.values())]
    width = max([len("ruleset")] + [len(n) for n in rulesets] + [len("TOTAL")])
    print()
    print("%-*s  %s  %6s" % (width, "ruleset", "  ".join("%*s" % (max(len(c), 5), c) for c in cols), "total"))
    total = collections.Counter()
    for name, c in rulesets.items():
        total.update(c)
        print("%-*s  %s  %6d" % (width, name, "  ".join("%*d" % (max(len(s), 5), c[s]) for s in cols),
                                 sum(c.values())))
    print("%-*s  %s  %6d" % (width, "TOTAL", "  ".join("%*d" % (max(len(s), 5), total[s]) for s in cols),
                             sum(total.values())))

    # Details.
    ces = [r for r in results if r["status"] in (COUNTEREXAMPLE, EXPECTED_FAILURE)]
    if ces:
        print("\n=== COUNTEREXAMPLES (%d) ===" % len(ces))
        for r in ces:
            tag = " [expected failure: %s]" % known[r["id"]] if r["status"] == EXPECTED_FAILURE else ""
            conf = " [sat from: %s; no sat from: %s]" % (
                ", ".join(r["confirmed_by"]), ", ".join(r["unconfirmed_by"]) or "-")
            print("\n--- %s (solver: %s)%s%s" % (r["id"], r["solver"], tag, conf))
            if r["rule_text"]:
                print(r["rule_text"])
            if r["model"]:
                print("model:")
                print(r["model"])
    dis = [r for r in results if r["status"] == DISAGREEMENT]
    if dis:
        print("\n" + "!" * 70)
        print("!!! SOLVER DISAGREEMENTS (%d): a solver or the encoding is WRONG !!!" % len(dis))
        print("!" * 70)
        for r in dis:
            print("\n--- %s: %s" % (r["id"], r["detail"]))
            if r["rule_text"]:
                print(r["rule_text"])
            if r["model"]:
                print(r["model"])
    errs = [r for r in results if r["status"] == ERROR]
    if errs:
        print("\n=== ERRORS (%d) ===" % len(errs))
        for r in errs:
            print("%s: %s" % (r["id"], r["detail"] or "no solver produced an answer"))
    for qid in stale:
        print("\nstale allowlist entry, remove it: %s is now proven" % qid)
    for qid in known_missing:
        print("warning: known-failures entry %s does not match any query" % qid)
    inconclusive = [r for r in results if r["status"] in (UNKNOWN, TIMEOUT)]
    if inconclusive:
        print("\n=== INCONCLUSIVE (%d; not failures) ===" % len(inconclusive))
        for r in inconclusive:
            print("%s: %s" % (r["id"], r["status"]))

    failing = [r for r in results if r["failing"]]
    ok = not failing
    if args.expect_all_sat and not ran:
        ok = False
        print("\nFAIL: --expect-all-sat but no queries were run")
    if args.expect_all_sat:
        bad = [r for r in ran if r["status"] not in (COUNTEREXAMPLE, EXPECTED_FAILURE)]
        if bad:
            print("\nFAIL: --expect-all-sat: %d queries were not sat: %s" % (
                len(bad), ", ".join(r["id"] for r in bad[:20])))
    else:
        unexpected = [r for r in failing if r["status"] == COUNTEREXAMPLE]
        if unexpected:
            print("\nFAIL: %d unexpected counterexample(s): %s" % (
                len(unexpected), ", ".join(r["id"] for r in unexpected)))
    if dis or errs or stale:
        print("\nFAIL: %d disagreement(s), %d error(s), %d stale allowlist entr%s" % (
            len(dis), len(errs), len(stale), "y" if len(stale) == 1 else "ies"))

    if args.copy_failures and failing:
        os.makedirs(args.copy_failures, exist_ok=True)
        for r in failing:
            if os.path.exists(r["path"]):
                shutil.copy(r["path"], args.copy_failures)

    if args.json:
        report = {
            "solvers": [s.name for s in solvers],
            "timeout": args.timeout,
            "ok": ok,
            "summary": {k: dict(v) for k, v in rulesets.items()},
            "totals": dict(total),
            "results": results,
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print("\n%s" % ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
