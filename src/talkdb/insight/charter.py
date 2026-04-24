"""
Auto-generate charts from query results.

Decision tree driven by (intent, data shape). Returns `None` when a chart would
not add value (single-value result, text-only result, too few rows).

matplotlib + seaborn for styling. PNG output base64-encoded. Lightweight,
no JS runtime, portable across MCP clients.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any

import matplotlib

# Non-interactive backend — required for server-side rendering without a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from talkdb.core.intent import Intent, IntentType  # noqa: E402
from talkdb.insight.analyzer import AnalysisResult  # noqa: E402

sns.set_theme(style="whitegrid", palette="deep")
_MAX_ROWS_FOR_BAR = 20  # Bar charts become unreadable beyond ~20 bars.


@dataclass
class ChartSpec:
    chart_type: str  # "line" | "horizontal_bar" | "vertical_bar" | "metric_card" | "histogram"
    image_base64: str
    title: str


class InsightCharter:
    def generate(
        self,
        rows: list[dict[str, Any]],
        columns: list[str],
        intent: Intent,
        analysis: AnalysisResult,
    ) -> tuple[ChartSpec | None, str]:
        """Returns (chart, reason). reason is non-empty iff chart is None."""
        if not rows:
            return None, "empty result — no chart"

        df = pd.DataFrame(rows, columns=columns)

        if analysis.single_value is not None:
            return self._metric_card(analysis.single_value.label, analysis.single_value.value), ""

        date_cols = [c for c, t in analysis.column_types.items() if t == "datetime"]
        numeric_cols = [c for c, t in analysis.column_types.items() if t in ("int", "float")]
        text_cols = [
            c for c in df.columns if c not in numeric_cols and c not in date_cols
        ]

        if not numeric_cols:
            return None, "no numeric column — table view is more useful"
        if len(df) < 3 and analysis.single_value is None:
            return None, "too few rows for a chart (<3)"

        # Time-series
        if date_cols and numeric_cols and analysis.time_series is not None:
            return self._line_chart(df, date_cols[0], numeric_cols[0], analysis), ""

        # Ranking / distribution
        if text_cols and numeric_cols:
            if len(df) > _MAX_ROWS_FOR_BAR:
                # Trim to the top N for readability.
                df = df.sort_values(numeric_cols[0], ascending=False).head(_MAX_ROWS_FOR_BAR)
            if intent.type == IntentType.RANKING or len(df) > 8:
                return self._horizontal_bar(df, text_cols[0], numeric_cols[0]), ""
            return self._vertical_bar(df, text_cols[0], numeric_cols[0]), ""

        # Pure numeric distribution (e.g., list of amounts)
        if numeric_cols and len(df) >= 10 and not text_cols:
            return self._histogram(df, numeric_cols[0]), ""

        return None, "no suitable chart for this result shape"

    def _line_chart(
        self, df: pd.DataFrame, date_col: str, value_col: str, analysis: AnalysisResult
    ) -> ChartSpec:
        ordered = df.sort_values(date_col)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(ordered[date_col], ordered[value_col], marker="o", linewidth=2)

        if analysis.time_series and analysis.time_series.anomalies:
            anom_indices = [a["index"] for a in analysis.time_series.anomalies]
            anom_x = [ordered[date_col].iloc[i] for i in anom_indices if 0 <= i < len(ordered)]
            anom_y = [ordered[value_col].iloc[i] for i in anom_indices if 0 <= i < len(ordered)]
            ax.scatter(anom_x, anom_y, color="crimson", zorder=5, s=80, label="anomaly")
            ax.legend(loc="best", frameon=True)

        title = f"{value_col} over time"
        ax.set_title(title)
        ax.set_xlabel(date_col)
        ax.set_ylabel(value_col)
        fig.autofmt_xdate()
        fig.tight_layout()
        return ChartSpec("line", _encode(fig), title)

    def _horizontal_bar(self, df: pd.DataFrame, label_col: str, value_col: str) -> ChartSpec:
        ordered = df.sort_values(value_col, ascending=True)
        fig, ax = plt.subplots(figsize=(9, max(3.5, 0.35 * len(ordered) + 1.5)))
        ax.barh(ordered[label_col].astype(str), ordered[value_col])
        title = f"{value_col} by {label_col}"
        ax.set_title(title)
        ax.set_xlabel(value_col)
        fig.tight_layout()
        return ChartSpec("horizontal_bar", _encode(fig), title)

    def _vertical_bar(self, df: pd.DataFrame, label_col: str, value_col: str) -> ChartSpec:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(df[label_col].astype(str), df[value_col])
        title = f"{value_col} by {label_col}"
        ax.set_title(title)
        ax.set_ylabel(value_col)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        return ChartSpec("vertical_bar", _encode(fig), title)

    def _histogram(self, df: pd.DataFrame, value_col: str) -> ChartSpec:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.hist(df[value_col], bins=min(20, max(5, len(df) // 5)))
        title = f"Distribution of {value_col}"
        ax.set_title(title)
        ax.set_xlabel(value_col)
        ax.set_ylabel("frequency")
        fig.tight_layout()
        return ChartSpec("histogram", _encode(fig), title)

    def _metric_card(self, label: str, value: float | int | str) -> ChartSpec:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.text(0.5, 0.70, _format_metric(value), ha="center", va="center", fontsize=42, fontweight="bold")
        ax.text(0.5, 0.25, label, ha="center", va="center", fontsize=14, color="#666")
        fig.tight_layout()
        return ChartSpec("metric_card", _encode(fig), label)


def _encode(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _format_metric(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    v = float(value)
    if abs(v) >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"{v / 1_000:.2f}k"
    if v == int(v):
        return f"{int(v)}"
    return f"{v:.2f}"
