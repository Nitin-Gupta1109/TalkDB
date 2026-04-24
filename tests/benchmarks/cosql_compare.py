"""
Compare multiple CoSQL result JSONs side-by-side.

Reads the per-turn rows, computes aggregate metrics for each, and prints:
  1. A headline comparison table (EX / SAFE / IM / DSAFE / Refusal rate)
  2. A failure-mode breakdown (where did each run still lose turns?)
  3. A per-DB breakdown (which schemas did each improvement actually help?)

Usage:
    python -m tests.benchmarks.cosql_compare \
        tests/benchmarks/cosql/results/cosql_dev_baseline.json \
        tests/benchmarks/cosql/results/cosql_dev_linker.json \
        tests/benchmarks/cosql/results/cosql_dev_rewriter.json \
        tests/benchmarks/cosql/results/cosql_dev_approve.json \
        tests/benchmarks/cosql/results/cosql_dev_all.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)
    return {"path": path, "rows": data["rows"], "summary": data["summary"]}


def per_db_ex(rows: list[dict]) -> dict[str, tuple[int, int]]:
    """Return {db_id: (ex_correct, total_turns)}."""
    by_db: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_db[r["db_id"]].append(r)
    return {db: (sum(r.get("ex", 0) for r in rs), len(rs)) for db, rs in by_db.items()}


def failure_reasons(rows: list[dict]) -> Counter:
    """Bucket non-EX, non-refused turns by a coarse failure class.

    Wrong shape       gen SQL runs but rows differ (actual hallucination/join/agg error)
    Refused confidence system refused (principled)
    Refused infra     harness/engine error (not credited as safe)
    Correct           EX=1
    """
    counts: Counter = Counter()
    for r in rows:
        if r.get("ex") == 1:
            counts["correct"] += 1
        elif r.get("refused") == 1:
            reason = r.get("refused_reason") or "unknown"
            counts[f"refused_{reason}"] += 1
        else:
            counts["wrong_result"] += 1
    return counts


def print_headline(runs: list[dict]) -> None:
    label_w = max(len(r["path"].stem) for r in runs) + 2
    print("\n" + "=" * 88)
    print("HEADLINE METRICS")
    print("=" * 88)
    header = f"{'run':<{label_w}s} {'turns':>6s} {'EX':>8s} {'SAFE':>8s} {'REF':>8s} {'IM':>8s} {'DSAFE':>8s}"
    print(header)
    print("-" * len(header))
    baseline = runs[0]["summary"]
    for run in runs:
        s = run["summary"]
        ex_delta = f"{(s['ex'] - baseline['ex']) * 100:+.1f}" if run is not runs[0] else "    "
        safe_delta = f"{(s['safe'] - baseline['safe']) * 100:+.1f}" if run is not runs[0] else "    "
        im_delta = f"{(s['im'] - baseline['im']) * 100:+.1f}" if run is not runs[0] else "    "
        print(
            f"{run['path'].stem:<{label_w}s} "
            f"{s['turns']:>6d} "
            f"{s['ex']*100:>6.1f}% "
            f"{s['safe']*100:>6.1f}% "
            f"{s['refusal_rate']*100:>6.1f}% "
            f"{s['im']*100:>6.1f}% "
            f"{s['dsafe']*100:>6.1f}%"
        )
        if run is not runs[0]:
            print(
                f"{'  ↳ vs baseline':<{label_w}s} "
                f"{'':<6s} "
                f"  {ex_delta}%   {safe_delta}%          {im_delta}%"
            )


def print_failure_breakdown(runs: list[dict]) -> None:
    print("\n" + "=" * 88)
    print("FAILURE-MODE BREAKDOWN (turns-level)")
    print("=" * 88)
    all_classes = set()
    per_run = []
    for run in runs:
        cls = failure_reasons(run["rows"])
        per_run.append(cls)
        all_classes.update(cls)
    order = ["correct", "wrong_result", "refused_confidence", "refused_no_sql", "refused_infra_error", "refused_unknown"]
    ordered = [c for c in order if c in all_classes] + sorted(all_classes - set(order))
    label_w = max(len(r["path"].stem) for r in runs) + 2
    header = f"{'run':<{label_w}s}" + "".join(f"{c:>22s}" for c in ordered)
    print(header)
    print("-" * len(header))
    for run, cls in zip(runs, per_run, strict=True):
        turns = sum(cls.values()) or 1
        line = f"{run['path'].stem:<{label_w}s}"
        for c in ordered:
            count = cls.get(c, 0)
            line += f"{count:>4d} ({count*100/turns:>4.1f}%)    "
        print(line)


def print_per_db(runs: list[dict]) -> None:
    print("\n" + "=" * 88)
    print("PER-DB EX — where did each improvement actually move the needle?")
    print("=" * 88)
    baseline_per_db = per_db_ex(runs[0]["rows"])
    per_run_per_db = [per_db_ex(run["rows"]) for run in runs]
    label_w = max(len(r["path"].stem) for r in runs) + 2
    all_dbs = sorted(baseline_per_db.keys())

    header = f"{'db_id':<32s} {'turns':>6s}" + "".join(f"{r['path'].stem[-12:]:>14s}" for r in runs)
    print(header)
    print("-" * len(header))
    for db in all_dbs:
        _, total = baseline_per_db[db]
        line = f"{db:<32s} {total:>6d}"
        for per_db in per_run_per_db:
            correct, _ = per_db.get(db, (0, 0))
            pct = correct * 100 / total if total else 0
            line += f"{correct:>3d}/{total:<3d}={pct:>4.1f}% "
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="Path to one or more cosql_*.json files. First is the baseline.")
    args = ap.parse_args()

    runs = [load(Path(p)) for p in args.results]
    print_headline(runs)
    print_failure_breakdown(runs)
    print_per_db(runs)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
