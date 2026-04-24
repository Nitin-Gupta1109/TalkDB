"""
Insight analyzer. Pure pandas/numpy computation — NO LLM calls.

Produces a structured `AnalysisResult` from a query's result rows. The narrator
(separate module) turns this into prose; the charter turns it into a picture.

By keeping the math out of the LLM, we guarantee the numbers in insights are correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from talkdb.core.intent import Intent, IntentType


@dataclass
class SingleValueInsight:
    label: str
    value: float | int | str


@dataclass
class TimeSeriesInsight:
    date_column: str
    value_column: str
    points: int
    first_value: float
    last_value: float
    total_change: float
    total_change_pct: float | None  # None if first value is 0
    trend: str  # "up" | "down" | "flat"
    min_value: float
    max_value: float
    anomalies: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CategoricalInsight:
    label_column: str
    value_column: str
    groups: int
    total: float
    top_label: str | None
    top_value: float | None
    top_share_pct: float | None
    top3_share_pct: float | None
    mean: float
    median: float
    std: float


@dataclass
class AnalysisResult:
    row_count: int
    columns: list[str]
    column_types: dict[str, str]
    intent_type: str
    key_findings: list[str]
    single_value: SingleValueInsight | None = None
    time_series: TimeSeriesInsight | None = None
    categorical: CategoricalInsight | None = None


class InsightAnalyzer:
    """Turn rows into structured, numeric findings. Never calls an LLM."""

    def analyze(
        self,
        rows: list[dict[str, Any]],
        columns: list[str],
        intent: Intent,
    ) -> AnalysisResult:
        df = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)
        df = _coerce_datetime_strings(df)

        column_types = {c: _dtype_label(df[c]) for c in df.columns}
        key_findings: list[str] = []

        result = AnalysisResult(
            row_count=len(df),
            columns=list(df.columns),
            column_types=column_types,
            intent_type=intent.type.value,
            key_findings=key_findings,
        )

        if df.empty:
            key_findings.append("Query returned no rows.")
            return result

        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        text_cols = [
            c for c in df.columns
            if c not in numeric_cols and c not in date_cols
        ]

        # Single-value result.
        if len(df) == 1 and len(numeric_cols) == 1 and len(df.columns) <= 2:
            col = numeric_cols[0]
            value = df[col].iloc[0]
            if isinstance(value, float | int | np.floating | np.integer):
                value = float(value)
            label = _label_for_single_value(df, col, text_cols)
            result.single_value = SingleValueInsight(label=label, value=value)
            key_findings.append(f"{label} is {_fmt_number(value)}.")
            return result

        # Time series: a date column + at least one numeric column, sorted by date.
        if date_cols and numeric_cols:
            date_col = date_cols[0]
            value_col = numeric_cols[0]
            ts_df = df.sort_values(date_col).reset_index(drop=True)
            values = ts_df[value_col].astype(float)
            if len(values) >= 3:
                result.time_series = _compute_time_series(ts_df, date_col, value_col, values)
                key_findings.extend(_time_series_findings(result.time_series))
                return result

        # Categorical: one text column + one numeric column (ranking / distribution).
        if text_cols and numeric_cols and len(df) >= 2:
            label_col = text_cols[0]
            value_col = numeric_cols[0]
            result.categorical = _compute_categorical(df, label_col, value_col, intent)
            key_findings.extend(_categorical_findings(result.categorical, intent))
            return result

        key_findings.append(
            f"Returned {len(df)} row(s) with columns {', '.join(df.columns)}."
        )
        return result


def _coerce_datetime_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    If a column is all strings but most values parse as ISO dates, convert it.
    Runs only on object-dtype columns to avoid touching existing numeric/datetime columns.
    """
    for col in df.columns:
        s = df[col]
        if (
            pd.api.types.is_numeric_dtype(s)
            or pd.api.types.is_datetime64_any_dtype(s)
            or pd.api.types.is_bool_dtype(s)
        ):
            continue
        sample = s.dropna().astype(str).head(20)
        if sample.empty:
            continue
        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= 0.8:  # 80%+ parse as dates
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _dtype_label(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    return "text"


def _label_for_single_value(df: pd.DataFrame, numeric_col: str, text_cols: list[str]) -> str:
    """Prefer the accompanying text column ('tier'/'month') as the label; fall back to the numeric column name."""
    if text_cols:
        return f"{text_cols[0]}={df[text_cols[0]].iloc[0]}" if len(df) == 1 else text_cols[0]
    return numeric_col


def _compute_time_series(
    df: pd.DataFrame, date_col: str, value_col: str, values: pd.Series
) -> TimeSeriesInsight:
    first = float(values.iloc[0])
    last = float(values.iloc[-1])
    change = last - first
    change_pct = (change / first * 100) if first != 0 else None
    if abs(change_pct or 0) < 2:
        trend = "flat"
    elif change > 0:
        trend = "up"
    else:
        trend = "down"

    # Anomaly detection: MAD-based robust z-score. MAD doesn't get inflated by the outlier
    # the way a rolling window does, so spikes are actually flagged.
    anomalies: list[dict[str, Any]] = []
    if len(values) >= 4:
        median = float(values.median())
        deviations = (values - median).abs()
        mad = float(deviations.median())
        # Fall back to stdev when MAD is 0 (happens when >=half the values are identical).
        scale = mad if mad > 0 else float(values.std(ddof=0)) or 1.0
        # For normal data, MAD ≈ 0.6745 × stdev, so threshold 3.5 ~ 2.3 stdev equivalent.
        threshold = 3.5 if mad > 0 else 2.0
        robust_z = (values - median) / scale
        for i, z in enumerate(robust_z):
            if pd.notna(z) and abs(z) >= threshold:
                anomalies.append(
                    {
                        "index": int(i),
                        "date": str(df[date_col].iloc[i]),
                        "value": float(values.iloc[i]),
                        "z_score": round(float(z), 2),
                    }
                )

    return TimeSeriesInsight(
        date_column=date_col,
        value_column=value_col,
        points=len(values),
        first_value=first,
        last_value=last,
        total_change=change,
        total_change_pct=round(change_pct, 2) if change_pct is not None else None,
        trend=trend,
        min_value=float(values.min()),
        max_value=float(values.max()),
        anomalies=anomalies,
    )


def _compute_categorical(
    df: pd.DataFrame, label_col: str, value_col: str, intent: Intent
) -> CategoricalInsight:
    values = df[value_col].astype(float)
    total = float(values.sum())

    if intent.type == IntentType.RANKING:
        ordered = df.sort_values(value_col, ascending=False).reset_index(drop=True)
    else:
        ordered = df.copy().reset_index(drop=True)

    top_label = str(ordered[label_col].iloc[0]) if not ordered.empty else None
    top_value = float(ordered[value_col].iloc[0]) if not ordered.empty else None
    top_share = (top_value / total * 100) if top_value is not None and total != 0 else None

    top3 = ordered.head(3)[value_col].astype(float).sum()
    top3_share = (top3 / total * 100) if total != 0 else None

    return CategoricalInsight(
        label_column=label_col,
        value_column=value_col,
        groups=len(df),
        total=total,
        top_label=top_label,
        top_value=top_value,
        top_share_pct=round(top_share, 1) if top_share is not None else None,
        top3_share_pct=round(top3_share, 1) if top3_share is not None else None,
        mean=float(values.mean()),
        median=float(values.median()),
        std=float(values.std(ddof=0)),
    )


def _time_series_findings(ts: TimeSeriesInsight) -> list[str]:
    out: list[str] = []
    trend_word = {"up": "up", "down": "down", "flat": "flat"}[ts.trend]
    if ts.total_change_pct is not None:
        out.append(
            f"{ts.value_column} trended {trend_word} {ts.total_change_pct:+.1f}% "
            f"over {ts.points} periods (from {_fmt_number(ts.first_value)} to {_fmt_number(ts.last_value)})."
        )
    else:
        out.append(
            f"{ts.value_column} went from {_fmt_number(ts.first_value)} to {_fmt_number(ts.last_value)}."
        )
    if ts.anomalies:
        names = ", ".join(str(a["date"])[:10] for a in ts.anomalies)
        out.append(f"{len(ts.anomalies)} anomaly point(s) detected: {names}.")
    return out


def _categorical_findings(cat: CategoricalInsight, intent: Intent) -> list[str]:
    out: list[str] = []
    if cat.top_label is not None and cat.top_share_pct is not None:
        out.append(
            f"Top {cat.label_column}: {cat.top_label} at {_fmt_number(cat.top_value)} "
            f"({cat.top_share_pct:.1f}% of total)."
        )
    if cat.top3_share_pct is not None and cat.groups >= 4:
        out.append(f"Top 3 account for {cat.top3_share_pct:.1f}% of total {cat.value_column}.")
    if cat.std > 0 and cat.mean > 0 and cat.std / cat.mean > 0.75:
        out.append(
            f"High dispersion across groups (std/mean = {cat.std / cat.mean:.2f})."
        )
    return out


def _fmt_number(v: float | int | None) -> str:
    if v is None:
        return "n/a"
    v = float(v)
    if abs(v) >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"{v / 1_000:.2f}k"
    if v == int(v):
        return f"{int(v)}"
    return f"{v:.2f}"
