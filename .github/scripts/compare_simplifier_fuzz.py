#!/usr/bin/env python3
"""Compares simplifier fuzz test throughput between two builds.

Runs the `simplify.fuzz` test of two `slinky_builder_simplify_simplify_test` binaries in
alternating order, counts the iterations each one completes in its fixed time
budget, and writes a Markdown report comparing the two.

Each round runs both binaries back to back with the same --gtest_random_seed,
so they see the same sequence of expressions, and alternates which one goes
first to cancel out ordering effects. The headline number is the median over
rounds of the per-round ratio, which is robust to slow drift in machine speed
(the two runs in a round are adjacent in time) and to outlier rounds.
"""

import argparse
import json
import re
import statistics
import subprocess
import sys

ITERATIONS_RE = re.compile(r"fuzz_test: executed (\d+) iterations")


def run_fuzz(binary, seed, cpu):
    cmd = [binary, "--gtest_filter=simplify.fuzz", f"--gtest_random_seed={seed}"]
    if cpu is not None:
        cmd = ["taskset", "--cpu-list", str(cpu)] + cmd
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        sys.exit(f"{binary} failed with exit code {result.returncode} (seed={seed})")
    counts = ITERATIONS_RE.findall(result.stdout)
    if len(counts) != 1:
        sys.stderr.write(result.stdout)
        sys.exit(f"expected one iteration count from {binary}, found {len(counts)}")
    return int(counts[0])


def quartiles(values):
    q1, q2, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q1, q2, q3


def percent(ratio):
    return f"{(ratio - 1) * 100:+.1f}%"


def report(args, base, pr):
    ratios = [p / b for b, p in zip(base, pr)]
    median_ratio = statistics.median(ratios)
    r_q1, _, r_q3 = quartiles(ratios)
    b_q1, b_median, b_q3 = quartiles(base)
    p_q1, p_median, p_q3 = quartiles(pr)

    # More iterations means a faster simplifier.
    if median_ratio <= 1 - args.threshold / 100:
        verdict = f":warning: **Possible regression**: the PR simplifies {percent(median_ratio)} expressions per second."
    elif median_ratio >= 1 + args.threshold / 100:
        verdict = f":rocket: **Possible improvement**: the PR simplifies {percent(median_ratio)} expressions per second."
    else:
        verdict = f":white_check_mark: **No significant change** ({percent(median_ratio)}, threshold ±{args.threshold:g}%)."

    lines = [
        "## Simplifier fuzz throughput",
        "",
        verdict,
        "",
        "| | Base | PR | Change |",
        "|:--|--:|--:|--:|",
        f"| Median iterations | {b_median:,.0f} | {p_median:,.0f} | {percent(p_median / b_median)} |",
        f"| Interquartile range | {b_q1:,.0f} – {b_q3:,.0f} | {p_q1:,.0f} – {p_q3:,.0f} | |",
        f"| **Median per-round ratio** | | | **{percent(median_ratio)}** |",
        f"| Ratio interquartile range | | | {percent(r_q1)} – {percent(r_q3)} |",
        "",
        f"Iterations of `simplify.fuzz` completed in its time budget, over {len(base)} alternating rounds "
        f"(after {args.warmup} warmup round{'s' if args.warmup != 1 else ''}). "
        "This is a coarse, noisy check; treat changes within a few percent as noise.",
        "",
        "<details><summary>Raw data</summary>",
        "",
        "| Round | Seed | First | Base | PR | Ratio |",
        "|--:|--:|:--|--:|--:|--:|",
    ]
    for i, (b, p) in enumerate(zip(base, pr)):
        round_index = i + args.warmup
        first = "base" if round_index % 2 == 0 else "PR"
        lines.append(f"| {i + 1} | {round_index + 1} | {first} | {b:,} | {p:,} | {percent(p / b)} |")
    lines += ["", f"Base: `{args.base_label}`, PR: `{args.pr_label}`", "", "</details>", ""]
    return "\n".join(lines), median_ratio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="simplify test binary built from the base revision")
    parser.add_argument("--pr", required=True, help="simplify test binary built from the PR revision")
    parser.add_argument("--base-label", default="base", help="description of the base revision for the report")
    parser.add_argument("--pr-label", default="PR", help="description of the PR revision for the report")
    parser.add_argument("--rounds", type=int, default=21, help="number of measured rounds")
    parser.add_argument("--warmup", type=int, default=1, help="number of unmeasured rounds to run first")
    parser.add_argument("--threshold", type=float, default=5, help="percent change to flag")
    parser.add_argument("--cpu", type=int, help="pin each run to this CPU with taskset")
    parser.add_argument("--markdown", required=True, help="path to write the Markdown report")
    parser.add_argument("--json", help="path to write the raw results")
    args = parser.parse_args()

    if args.rounds < 2:
        parser.error("--rounds must be at least 2")

    base, pr = [], []
    for i in range(args.warmup + args.rounds):
        # gtest treats a seed of 0 as "use the time", so start at 1.
        seed = i + 1
        if i % 2 == 0:
            b = run_fuzz(args.base, seed, args.cpu)
            p = run_fuzz(args.pr, seed, args.cpu)
        else:
            p = run_fuzz(args.pr, seed, args.cpu)
            b = run_fuzz(args.base, seed, args.cpu)
        tag = "warmup" if i < args.warmup else f"round {i - args.warmup + 1}"
        print(f"{tag}: seed={seed} base={b} pr={p} ratio={p / b:.3f}", flush=True)
        if i >= args.warmup:
            base.append(b)
            pr.append(p)

    markdown, median_ratio = report(args, base, pr)
    with open(args.markdown, "w") as f:
        f.write(markdown)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"base": base, "pr": pr, "median_ratio": median_ratio}, f, indent=2)
    print(markdown)


if __name__ == "__main__":
    main()
