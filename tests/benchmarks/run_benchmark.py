"""
Benchmark runner for the ecommerce suite (easy + hard tiers).

Grading:
- Execution accuracy (EX): gold SQL result ⊆ generated SQL result, per row, same row count.
  Gen rows may include EXTRA columns the question didn't require — this is fine,
  since many valid answers include context columns (name alongside id, etc.).
  Still strict about row count and value correctness.
- Lenient (keyword): generated SQL contains expected structural tokens.

Usage:
    python -m tests.benchmarks.run_benchmark
    python -m tests.benchmarks.run_benchmark --save-json results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from collections import Counter, defaultdict
from typing import Any

from sqlalchemy import create_engine, text

from talkdb.config.settings import get_settings
from talkdb.core.engine import Engine
from tests.benchmarks.ecommerce_suite import ALL_CASES, Case, ConvCase
from tests.benchmarks.hard_suite import ALL_HARD


def run_gold(conn_str: str, sql: str) -> list[tuple]:
    """Execute gold SQL and return normalized rows (tuples of values, sorted)."""
    eng = create_engine(conn_str)
    with eng.connect() as c:
        result = c.execute(text(sql))
        rows = [tuple(row) for row in result]
    eng.dispose()
    return _normalize(rows)


def _norm_value(v: Any) -> Any:
    """Canonicalize a cell for comparison (round floats, coerce bool to int)."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return round(v, 4)
    return v


def _normalize(rows: list[tuple]) -> list[tuple]:
    """Sort rows and normalize floats (used for the strict path and for gold SQL normalization)."""
    return sorted([tuple(_norm_value(v) for v in row) for row in rows])


def results_match(gold_rows: list[tuple], gen_rows: list[dict]) -> bool:
    """
    Containment match: for each gold row's multiset of values, there must be a unique
    generated row whose multiset of values is a superset. In other words, the gen row
    contains everything the gold row did, and may include extra context columns.

    Why: a correct SQL that returns (id, name, revenue) answers the same question as
    one that returns (id, revenue) PLUS useful extra context. Penalizing the superset
    answer is a grader artifact, not a correctness signal.

    Still strict about:
    - Row count (must match).
    - Values (must match within float tolerance).
    - Multiplicities (duplicates counted).
    """
    if len(gold_rows) != len(gen_rows):
        return False

    gen_multisets = [Counter(_norm_value(v) for v in r.values()) for r in gen_rows]
    used = [False] * len(gen_multisets)

    for gold_row in gold_rows:
        gold_counter = Counter(_norm_value(v) for v in gold_row)
        matched = False
        for i, gm in enumerate(gen_multisets):
            if used[i]:
                continue
            if all(gm.get(v, 0) >= count for v, count in gold_counter.items()):
                used[i] = True
                matched = True
                break
        if not matched:
            return False
    return True


def keywords_match(sql: str, keywords: list[str]) -> bool:
    low = sql.lower()
    return all(k.lower() in low for k in keywords)


def extract_dual_path_signal(warnings: list[str] | None) -> str | None:
    """Parse the dual-path warning (added by the engine) into a short label, or None if absent."""
    for w in warnings or []:
        wl = w.lower()
        if not wl.startswith("dual_path"):
            continue
        if "full agreement" in wl or "(path a and path b produced matching" in wl:
            return "full"
        if "partial agreement" in wl:
            return "partial"
        if "disagreement" in wl:
            return "disagreement"
        if "path b refused" in wl:
            return "path_b_refused"
        if "path b execution failed" in wl:
            return "path_b_failed"
        if "unsafe" in wl:
            return "path_b_unsafe"
    return None


async def grade_case(engine: Engine, case: Case, conn_str: str) -> dict:
    gold_rows = run_gold(conn_str, case.gold_sql)
    t0 = time.perf_counter()
    result = await engine.ask(case.question)
    elapsed = time.perf_counter() - t0

    gen_rows = result.results
    execution_match = bool(result.sql) and results_match(gold_rows, gen_rows)
    lenient_match = bool(result.sql) and keywords_match(result.sql, case.keywords)

    return {
        "id": case.id,
        "category": case.category,
        "question": case.question,
        "gen_sql": result.sql,
        "refused": not result.sql,
        "confidence": result.confidence,
        "execution_match": execution_match,
        "lenient_match": lenient_match,
        "dual_path_signal": extract_dual_path_signal(result.warnings),
        "gold_row_count": len(gold_rows),
        "gen_row_count": len(gen_rows),
        "elapsed_s": round(elapsed, 2),
        "explanation": result.explanation,
    }


async def grade_conv(engine: Engine, case: ConvCase, conn_str: str) -> dict:
    """Grade a multi-turn case by running all turns in a shared session; evaluate the final turn."""
    gold_rows = run_gold(conn_str, case.gold_sql)
    session_id = f"bench-{uuid.uuid4().hex[:8]}"
    t0 = time.perf_counter()
    result = None
    for q in case.turns:
        result = await engine.ask(q, session_id=session_id)
    elapsed = time.perf_counter() - t0

    assert result is not None
    gen_rows = result.results
    execution_match = bool(result.sql) and results_match(gold_rows, gen_rows)
    lenient_match = bool(result.sql) and keywords_match(result.sql, case.keywords)

    return {
        "id": case.id,
        "category": case.category,
        "question": " | ".join(case.turns),
        "gen_sql": result.sql,
        "refused": not result.sql,
        "confidence": result.confidence,
        "execution_match": execution_match,
        "lenient_match": lenient_match,
        "dual_path_signal": extract_dual_path_signal(result.warnings),
        "gold_row_count": len(gold_rows),
        "gen_row_count": len(gen_rows),
        "elapsed_s": round(elapsed, 2),
        "explanation": result.explanation,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-json", default=None, help="Write per-case results to this path.")
    args = parser.parse_args()

    settings = get_settings()
    engine = Engine(settings)
    conn_str = settings.default_db

    all_cases = [*ALL_CASES, *ALL_HARD]
    print(f"Running {len(all_cases)} cases against {conn_str}")
    print(f"dual_path_enabled={settings.dual_path_enabled}\n")
    results: list[dict] = []
    t_start = time.perf_counter()
    for case in all_cases:
        try:
            if isinstance(case, ConvCase):
                r = await grade_conv(engine, case, conn_str)
            else:
                r = await grade_case(engine, case, conn_str)
        except Exception as e:  # noqa: BLE001
            r = {
                "id": case.id,
                "category": case.category,
                "question": case.question if isinstance(case, Case) else " | ".join(case.turns),
                "gen_sql": "",
                "refused": True,
                "execution_match": False,
                "lenient_match": False,
                "error": str(e),
                "elapsed_s": 0.0,
            }
        results.append(r)
        marker = "OK " if r["execution_match"] else ("~  " if r["lenient_match"] else "FAIL")
        print(f"  [{marker}] {r['id']:<6} {r['category']:<20} ({r['elapsed_s']}s) conf={r.get('confidence','-')}")

    total_elapsed = time.perf_counter() - t_start

    # --- Report ---
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    print("\n" + "=" * 60)
    print(f"{'Category':<22} {'EX':<10} {'Lenient':<10} {'Refused':<10}")
    print("-" * 60)
    total_ex = total_le = total_ref = 0
    categories = [
        "simple_aggregation", "filter_lookup", "single_join", "multi_join", "conversational",
        "hard_aggregation", "hard_temporal", "hard_ranking", "hard_subquery", "hard_conversational",
    ]
    for cat in categories:
        cat_results = by_cat.get(cat, [])
        n = len(cat_results)
        if n == 0:
            continue
        ex = sum(1 for r in cat_results if r["execution_match"])
        le = sum(1 for r in cat_results if r["lenient_match"])
        ref = sum(1 for r in cat_results if r["refused"])
        print(f"{cat:<22} {ex}/{n} ({ex*100//n}%)  {le}/{n} ({le*100//n}%)  {ref}/{n}")
        total_ex += ex
        total_le += le
        total_ref += ref

    total = len(results)
    print("-" * 60)
    print(f"{'TOTAL':<22} {total_ex}/{total} ({total_ex*100//total}%)  {total_le}/{total} ({total_le*100//total}%)  {total_ref}/{total}")
    print(f"Elapsed: {total_elapsed:.1f}s  |  EX = execution accuracy  |  Lenient = SQL keyword match")

    # Dual-path effectiveness: precision/recall on 'is this answer wrong?' classifier.
    signals = [r.get("dual_path_signal") for r in results]
    if any(s is not None for s in signals):
        tp = tn = fp = fn = unreliable = 0
        for r in results:
            sig = r.get("dual_path_signal")
            if sig is None:
                continue
            correct = r["execution_match"]
            non_agreement = sig in ("partial", "disagreement")
            if sig in ("path_b_refused", "path_b_failed", "path_b_unsafe"):
                unreliable += 1
                continue
            if non_agreement and not correct:
                tp += 1
            elif non_agreement and correct:
                fp += 1
            elif sig == "full" and correct:
                tn += 1
            elif sig == "full" and not correct:
                fn += 1
        print(
            f"\nDual-path flag (as error detector): "
            f"TP={tp}  FP={fp}  TN={tn}  FN={fn}  unreliable={unreliable}"
        )
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        print(f"  precision={precision:.2f}  recall={recall:.2f}")

    # Show failures for inspection.
    failures = [r for r in results if not r["execution_match"]]
    if failures:
        print("\nFailures (execution mismatch):")
        for r in failures:
            print(f"  [{r['id']}] {r['question']}")
            print(f"    gen: {r.get('gen_sql', '')[:200]}")
            if r.get("refused"):
                print(f"    refused: {r.get('explanation', '')}")
            else:
                print(f"    gold_rows={r['gold_row_count']} gen_rows={r['gen_row_count']}")

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump({"results": results, "elapsed_s": total_elapsed}, f, indent=2, default=str)
        print(f"\nWrote per-case results to {args.save_json}")


if __name__ == "__main__":
    asyncio.run(main())
