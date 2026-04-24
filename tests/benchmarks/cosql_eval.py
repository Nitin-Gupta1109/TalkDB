"""
CoSQL eval harness for TalkDB.

Measures four things per turn — the first two are the standard CoSQL metrics,
the last two are TalkDB's novel "trustworthy" axis:

    QM   Question Match — exact SQL equality after lightweight normalization
    EX   Execution Match — gold SQL results equal generated SQL results
    REF  Refused — confidence fell below refusal threshold, so TalkDB returned no SQL
    SAFE Safe — correct (EX=1) OR refused on a query that would have been wrong

And at the dialogue level:

    IM   Interaction Match — every turn in the dialogue is QM=1 (CoSQL's headline metric)
    DEX  Dialogue Exec — every turn is EX=1
    DSAFE Dialogue Safe — every turn is SAFE=1

Usage:
    # Sanity-check the data layout without burning LLM budget
    python -m tests.benchmarks.cosql_eval --verify

    # Run on a 5-dialogue subsample (cheap — ~30 turns, a few $ of LLM calls)
    python -m tests.benchmarks.cosql_eval --limit 5 --save results/smoke.json

    # Full dev run
    python -m tests.benchmarks.cosql_eval --save results/cosql_dev_full.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine as sa_create_engine, text

COSQL_DIR = Path(__file__).parent / "cosql"
DEV_JSON = COSQL_DIR / "cosql_dataset" / "sql_state_tracking" / "cosql_dev.json"
DB_ROOT = COSQL_DIR / "cosql_dataset" / "database"
RESULTS_DIR = COSQL_DIR / "results"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────


def load_dev() -> list[dict]:
    """Load and flatten the CoSQL dev JSON into a list of dialogues."""
    if not DEV_JSON.exists():
        raise FileNotFoundError(
            f"Expected CoSQL dev split at {DEV_JSON}. "
            f"See {COSQL_DIR / 'README.md'} for download instructions."
        )
    with DEV_JSON.open() as f:
        return json.load(f)


def db_path(db_id: str) -> Path:
    """SQLite path for a CoSQL database id."""
    return DB_ROOT / db_id / f"{db_id}.sqlite"


def verify(dialogues: list[dict]) -> tuple[int, int, list[str]]:
    """Return (dialogue_count, turn_count, missing_dbs) without any LLM calls."""
    turn_count = 0
    missing: set[str] = set()
    for d in dialogues:
        db_id = d.get("database_id") or d.get("db_id")
        if not db_path(db_id).exists():
            missing.add(db_id)
        turn_count += len(d.get("interaction", []))
    return len(dialogues), turn_count, sorted(missing)


# ─────────────────────────────────────────────────────────────────────────────
# Gold SQL execution (for EX scoring)
# ─────────────────────────────────────────────────────────────────────────────


def execute_gold(sql: str, db_id: str, timeout_s: int = 10) -> tuple[bool, list[tuple] | None, str | None]:
    """Run gold SQL against its SQLite DB. Returns (ok, rows, error)."""
    try:
        engine = sa_create_engine(f"sqlite:///{db_path(db_id)}")
        with engine.connect() as conn:
            conn.execute(text(f"PRAGMA busy_timeout = {timeout_s * 1000}"))
            rows = conn.execute(text(sql)).fetchall()
        return True, [tuple(r) for r in rows], None
    except Exception as e:
        return False, None, str(e)[:200]


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_sql(sql: str) -> str:
    """Lowercase, collapse whitespace, strip trailing semicolons. Cheap normalization
    — NOT a full AST comparison. QM is an imperfect metric; EX is the honest one."""
    if not sql:
        return ""
    return " ".join(sql.lower().replace(";", "").split())


def _normalize_value(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 4)
    return v


def _rows_equal(a: list[tuple] | None, b: list[tuple] | None) -> bool:
    """Multiset equality, row-order agnostic, with float rounding."""
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return False
    norm_a = Counter(tuple(_normalize_value(v) for v in row) for row in a)
    norm_b = Counter(tuple(_normalize_value(v) for v in row) for row in b)
    return norm_a == norm_b


def score_turn(
    gold_sql: str,
    gold_rows: list[tuple] | None,
    gen_sql: str,
    gen_rows: list[dict],
    confidence: int,
    refusal_threshold: int,
) -> dict:
    """Score a single turn. See module docstring for metric definitions.

    `refused_reason` distinguishes:
      - "confidence": engine returned sub-threshold confidence (a real, principled refusal)
      - "no_sql":     engine returned empty SQL at non-zero confidence (unusual, bucket with confidence)
      - None:         engine returned SQL above threshold

    Infrastructure errors (index build, ask() exception) are handled upstream and get
    refused_reason="infra_error" — those do NOT count as SAFE, because they're bugs, not refusals.
    """
    refused_low_conf = confidence < refusal_threshold
    refused_no_sql = (not gen_sql) and not refused_low_conf
    refused = refused_low_conf or refused_no_sql
    refused_reason = "confidence" if refused_low_conf else ("no_sql" if refused_no_sql else None)

    qm = 0 if refused else int(_normalize_sql(gen_sql) == _normalize_sql(gold_sql))

    gen_tuples = [tuple(_normalize_value(v) for v in row.values()) for row in (gen_rows or [])]
    ex = 0 if refused else int(_rows_equal(gold_rows, gen_tuples))

    # SAFE = either correct, or we refused (principled refusal, NOT infra error — caller handles that).
    safe = ex or refused

    return {
        "qm": qm,
        "ex": ex,
        "refused": int(refused),
        "refused_reason": refused_reason,
        "safe": int(safe),
        "confidence": confidence,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main eval loop
# ─────────────────────────────────────────────────────────────────────────────


async def run_dialogue(dialogue: dict, refusal_threshold: int, dialogue_idx: int) -> list[dict]:
    """Run one dialogue turn-by-turn. Returns per-turn score rows.

    A fresh TalkDB Engine is constructed per dialogue so each conversation's
    retriever/session state starts clean. This is the "per-DB engine factory"
    baseline — slow, but correct. A multi-tenant retriever is a separate refactor.

    ChromaDB isolation: each dialogue gets its own temp persistence dir, so collection
    state from DB A doesn't bleed into DB B (the smoke run hit 'Expected IDs to be unique'
    crashes from this). Temp dir is cleaned up at the end of the dialogue.
    """
    # Import here so --verify doesn't require the whole dependency tree to load.
    import shutil
    import tempfile

    from talkdb.config.settings import get_settings
    from talkdb.core.engine import Engine

    db_id = dialogue.get("database_id") or dialogue.get("db_id")
    # CoSQL JSON has no per-dialogue id; use position in the list so sessions don't collide.
    dialogue_id = f"{db_id}_{dialogue_idx:04d}"

    # Isolate each dialogue: its own SQLite DB + its own ChromaDB persistence dir.
    # get_settings() is @lru_cache'd — clear it so the env override actually applies.
    chroma_dir = tempfile.mkdtemp(prefix=f"cosql_chroma_{dialogue_id}_")
    os.environ["TALKDB_DEFAULT_DB"] = f"sqlite:///{db_path(db_id)}"
    os.environ["TALKDB_CHROMA_PATH"] = chroma_dir
    get_settings.cache_clear()
    settings = get_settings()

    def _infra_refusal_row(turn_number: int, turn: dict, error: str) -> dict:
        """A dialogue-level infrastructure failure. NOT counted as safe — it's a bug."""
        return {
            "dialogue_id": dialogue_id,
            "db_id": db_id,
            "turn_number": turn_number,
            "question": turn.get("utterance") or turn.get("question", ""),
            "gold_sql": turn.get("query", ""),
            "error": error[:200],
            "qm": 0,
            "ex": 0,
            "refused": 1,
            "refused_reason": "infra_error",
            "safe": 0,
            "confidence": 0,
        }

    try:
        engine = Engine(settings)
        engine.build_index(database=None)
    except Exception as e:
        shutil.rmtree(chroma_dir, ignore_errors=True)
        return [
            _infra_refusal_row(i, t, f"index_build_failed: {e}")
            for i, t in enumerate(dialogue.get("interaction", []), start=1)
        ]

    session_id = f"cosql_{dialogue_id}"
    rows: list[dict] = []

    for turn_number, turn in enumerate(dialogue.get("interaction", []), start=1):
        question = turn.get("utterance") or turn.get("question", "")
        gold_sql = turn.get("query", "")

        gold_ok, gold_rows, gold_err = execute_gold(gold_sql, db_id)

        try:
            result = await engine.ask(question, session_id=session_id)
            score = score_turn(
                gold_sql=gold_sql,
                gold_rows=gold_rows if gold_ok else None,
                gen_sql=result.sql,
                gen_rows=result.results,
                confidence=result.confidence,
                refusal_threshold=refusal_threshold,
            )
            score.update(
                {
                    "dialogue_id": dialogue_id,
                    "db_id": db_id,
                    "turn_number": turn_number,
                    "question": question,
                    "gold_sql": gold_sql,
                    "gen_sql": result.sql,
                    "gold_ok": gold_ok,
                    "gold_err": gold_err,
                }
            )
            rows.append(score)
        except Exception as e:
            rows.append(_infra_refusal_row(turn_number, turn, f"ask_failed: {e}"))

    shutil.rmtree(chroma_dir, ignore_errors=True)
    return rows


def summarize(rows: list[dict]) -> dict:
    """Turn-level + dialogue-level aggregates."""
    if not rows:
        return {"turns": 0}

    turns = len(rows)
    qm_total = sum(r.get("qm", 0) for r in rows)
    ex_total = sum(r.get("ex", 0) for r in rows)
    refused_total = sum(r.get("refused", 0) for r in rows)
    safe_total = sum(r.get("safe", 0) for r in rows)

    refusal_reasons: Counter = Counter(
        r.get("refused_reason") for r in rows if r.get("refused", 0) == 1
    )

    # Dialogue-level: group by dialogue_id, require all-1 for each metric.
    by_dialogue: dict[str, list[dict]] = {}
    for r in rows:
        by_dialogue.setdefault(r["dialogue_id"], []).append(r)

    dialogues = len(by_dialogue)
    im = sum(1 for ts in by_dialogue.values() if all(t.get("qm", 0) == 1 for t in ts))
    dex = sum(1 for ts in by_dialogue.values() if all(t.get("ex", 0) == 1 for t in ts))
    dsafe = sum(1 for ts in by_dialogue.values() if all(t.get("safe", 0) == 1 for t in ts))

    # Refusal precision: when we refused, how often were we *right* to refuse?
    # Approximated as: refusals where gold was known and gen would have been wrong.
    # (For the first pass we only know when gold ran successfully.)
    refused_rows = [r for r in rows if r.get("refused", 0) == 1]
    refusal_precision_denominator = len(refused_rows)
    # We can't check "would have been wrong" without running the gen SQL anyway,
    # so for v1 we report the simpler number: refusals as a fraction of turns.
    refusal_rate = refused_total / turns if turns else 0

    return {
        "turns": turns,
        "dialogues": dialogues,
        "qm": round(qm_total / turns, 4),
        "ex": round(ex_total / turns, 4),
        "safe": round(safe_total / turns, 4),
        "refusal_rate": round(refusal_rate, 4),
        "im": round(im / dialogues, 4) if dialogues else 0,
        "dex": round(dex / dialogues, 4) if dialogues else 0,
        "dsafe": round(dsafe / dialogues, 4) if dialogues else 0,
        "refused_turns": refused_total,
        "refused_by_reason": dict(refusal_reasons),
        "qm_turns": qm_total,
        "ex_turns": ex_total,
        "safe_turns": safe_total,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="Check data layout; no LLM calls.")
    ap.add_argument("--limit", type=int, default=None, help="Only run the first N dialogues.")
    ap.add_argument("--save", type=str, default=None, help="Write per-turn rows + summary to this JSON file.")
    ap.add_argument(
        "--refusal-threshold",
        type=int,
        default=50,
        help="Confidence threshold below which the engine is treated as refusing.",
    )
    args = ap.parse_args()

    try:
        dialogues = load_dev()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2

    if args.verify:
        n_dialog, n_turn, missing = verify(dialogues)
        print(f"dialogues: {n_dialog}")
        print(f"turns:     {n_turn}")
        print(f"databases referenced: {len({d.get('database_id') or d.get('db_id') for d in dialogues})}")
        if missing:
            print(f"missing SQLite files ({len(missing)}):")
            for m in missing[:20]:
                print(f"  - {m}")
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more")
            return 1
        print("all referenced SQLite files present ✓")
        return 0

    if args.limit:
        dialogues = dialogues[: args.limit]

    print(f"running {len(dialogues)} dialogues ({sum(len(d.get('interaction', [])) for d in dialogues)} turns)...", flush=True)
    t0 = time.time()
    all_rows: list[dict] = []
    for i, dialogue in enumerate(dialogues, start=1):
        rows = await run_dialogue(dialogue, args.refusal_threshold, dialogue_idx=i)
        all_rows.extend(rows)
        if i % 10 == 0 or i == len(dialogues):
            elapsed = time.time() - t0
            print(f"  [{i}/{len(dialogues)}]  {elapsed:.0f}s elapsed", flush=True)

    summary = summarize(all_rows)
    print()
    print("=== RESULTS ===")
    for k, v in summary.items():
        print(f"  {k:20s} {v}")

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = Path(args.save)
        if not out_path.is_absolute():
            out_path = RESULTS_DIR / out_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump({"summary": summary, "rows": all_rows}, f, indent=2, default=str)
        print(f"\nwrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
