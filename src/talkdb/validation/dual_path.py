"""
Dual-path verification.

Given a primary SQL (Path A), generate a second SQL via a structurally different
prompt strategy (Path B — decompose-then-compose), execute both, compare the results.

Agreement → confidence boost. Divergence → flag. Catches semantic errors that syntactic
validation can't (e.g. a wrong join condition that still passes schema checks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DualPathResult:
    """Outcome of comparing the primary SQL's result to an independently-generated secondary SQL's result."""

    agreement_level: str  # "full" | "partial" | "disagreement" | "path_b_failed"
    path_b_sql: str
    path_b_rows: list[dict[str, Any]] = field(default_factory=list)
    path_b_columns: list[str] = field(default_factory=list)
    confidence_adjustment: float = 0.0  # In [-1.0, +1.0], fed into confidence scorer
    agreement_score: float = 0.0        # In [0.0, 1.0], used by the confidence signal
    divergence_note: str | None = None


def compare_results(
    path_a_columns: list[str],
    path_a_rows: list[dict[str, Any]],
    path_b_columns: list[str],
    path_b_rows: list[dict[str, Any]],
    *,
    float_tolerance: float = 1e-4,
) -> DualPathResult:
    """
    Compare two result sets. Both paths are expected to already be validated and executed.
    """
    # Shape check: do columns match (as sets, ignoring order and case)?
    a_cols = {c.lower() for c in path_a_columns}
    b_cols = {c.lower() for c in path_b_columns}

    if a_cols != b_cols:
        # Different columns — often a sign the model misunderstood the question differently.
        overlap = a_cols & b_cols
        if overlap:
            note = (
                f"Path A columns {sorted(a_cols)} vs Path B {sorted(b_cols)} "
                f"(overlap: {sorted(overlap)})"
            )
        else:
            note = f"Path A columns {sorted(a_cols)} vs Path B {sorted(b_cols)} — no overlap"
        return DualPathResult(
            agreement_level="disagreement",
            path_b_sql="",
            path_b_rows=path_b_rows,
            path_b_columns=path_b_columns,
            confidence_adjustment=-0.3,
            agreement_score=0.0,
            divergence_note=note,
        )

    if len(path_a_rows) != len(path_b_rows):
        return DualPathResult(
            agreement_level="disagreement",
            path_b_sql="",
            path_b_rows=path_b_rows,
            path_b_columns=path_b_columns,
            confidence_adjustment=-0.3,
            agreement_score=0.0,
            divergence_note=f"Path A returned {len(path_a_rows)} rows, Path B returned {len(path_b_rows)}",
        )

    # Values match? Normalize each row to a tuple keyed by sorted column name so order doesn't matter.
    a_sorted = _normalize_rows(path_a_rows, path_a_columns)
    b_sorted = _normalize_rows(path_b_rows, path_b_columns)

    mismatches = 0
    for ar, br in zip(a_sorted, b_sorted, strict=False):
        if not _rows_equivalent(ar, br, float_tolerance):
            mismatches += 1

    if mismatches == 0:
        return DualPathResult(
            agreement_level="full",
            path_b_sql="",
            path_b_rows=path_b_rows,
            path_b_columns=path_b_columns,
            confidence_adjustment=+0.5,
            agreement_score=1.0,
        )

    # Partial: same shape but some values differ.
    score = 1.0 - (mismatches / max(len(a_sorted), 1))
    return DualPathResult(
        agreement_level="partial",
        path_b_sql="",
        path_b_rows=path_b_rows,
        path_b_columns=path_b_columns,
        confidence_adjustment=-0.1,
        agreement_score=score,
        divergence_note=f"{mismatches} of {len(a_sorted)} rows differ in value",
    )


def _normalize_rows(rows: list[dict[str, Any]], columns: list[str]) -> list[tuple]:
    """Return rows as tuples keyed by sorted column names, so column order doesn't affect equality."""
    if not columns:
        return []
    sorted_cols = sorted(c.lower() for c in columns)
    # Build a lookup from lower-case column -> original column name (first match wins).
    lower_to_orig: dict[str, str] = {}
    for c in columns:
        lower_to_orig.setdefault(c.lower(), c)
    tuples = [tuple(row.get(lower_to_orig[c]) for c in sorted_cols) for row in rows]
    return sorted(tuples, key=lambda t: tuple(_sort_key(v) for v in t))


def _sort_key(v: Any) -> tuple:
    """Stable sort key across mixed types so we can order rows without TypeErrors."""
    if v is None:
        return (0, "")
    if isinstance(v, bool):
        return (1, int(v))
    if isinstance(v, int | float):
        return (2, float(v))
    return (3, str(v))


def _rows_equivalent(a: tuple, b: tuple, tol: float) -> bool:
    if len(a) != len(b):
        return False
    for av, bv in zip(a, b, strict=False):
        if isinstance(av, float) or isinstance(bv, float):
            try:
                if abs(float(av) - float(bv)) > tol:
                    return False
            except (TypeError, ValueError):
                if av != bv:
                    return False
        elif av != bv:
            return False
    return True
