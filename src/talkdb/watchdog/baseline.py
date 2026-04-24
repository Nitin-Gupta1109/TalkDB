"""Rolling baseline computation for watch alert conditions."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta

from talkdb.watchdog.storage import WatchdogStorage


@dataclass
class BaselineResult:
    baseline: float | None
    sample_size: int
    kind: str
    stdev: float | None = None


class BaselineComputer:
    """Computes rolling baselines from the watch's historical values."""

    def __init__(self, storage: WatchdogStorage):
        self.storage = storage

    def compute(self, watch_name: str, baseline_type: str, *, now: datetime | None = None) -> BaselineResult:
        now = now or datetime.utcnow()
        history = self.storage.history(watch_name, limit=500)
        if not history:
            return BaselineResult(baseline=None, sample_size=0, kind=baseline_type)

        if baseline_type in ("7_day_avg", "30_day_avg"):
            days = 7 if baseline_type == "7_day_avg" else 30
            cutoff = now - timedelta(days=days)
            values = [v for (t, v) in history if t >= cutoff]
            if not values:
                return BaselineResult(baseline=None, sample_size=0, kind=baseline_type)
            stdev = statistics.stdev(values) if len(values) > 1 else None
            return BaselineResult(
                baseline=statistics.mean(values),
                sample_size=len(values),
                kind=baseline_type,
                stdev=stdev,
            )

        if baseline_type == "previous_period":
            # Simplest interpretation: the previous recorded value.
            values = [v for (_, v) in history]
            if len(values) < 2:
                return BaselineResult(baseline=None, sample_size=0, kind=baseline_type)
            return BaselineResult(baseline=values[1], sample_size=1, kind=baseline_type)

        if baseline_type == "same_day_last_week":
            target = now - timedelta(days=7)
            closest = min(history, key=lambda th: abs((th[0] - target).total_seconds()))
            if abs((closest[0] - target).total_seconds()) > 2 * 86400:
                return BaselineResult(baseline=None, sample_size=0, kind=baseline_type)
            return BaselineResult(baseline=closest[1], sample_size=1, kind=baseline_type)

        # Fallback: mean of all history.
        values = [v for (_, v) in history]
        stdev = statistics.stdev(values) if len(values) > 1 else None
        return BaselineResult(
            baseline=statistics.mean(values),
            sample_size=len(values),
            kind=baseline_type,
            stdev=stdev,
        )
