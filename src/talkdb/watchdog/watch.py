"""Watch + AlertCondition data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

ConditionKind = Literal["threshold", "change_percent", "anomaly", "pulse"]
BaselineType = Literal["7_day_avg", "30_day_avg", "previous_period", "same_day_last_week"]


@dataclass
class AlertCondition:
    """
    When to fire an alert.

    Exactly one of these condition modes is active per watch:
      - threshold: value crosses an absolute bound.
      - change_percent: value deviates from a baseline by X%.
      - anomaly: value is N stdev from rolling mean.
      - pulse: always fire on each run (reporting, not alerting).
    """

    kind: ConditionKind = "pulse"

    # threshold mode
    threshold_value: float | None = None
    threshold_direction: Literal["above", "below"] | None = None

    # change_percent mode — negative percentage means drop.
    change_percent: float | None = None
    baseline_type: BaselineType | None = None

    # anomaly mode
    anomaly_std_devs: float = 2.0

    # Human-readable summary for display / alert message body.
    description: str | None = None


@dataclass
class Watch:
    name: str
    question: str
    sql: str                          # Pre-validated SQL, frozen at creation time.
    database: str | None
    schedule: str                     # Raw schedule string (e.g. "every 1 hour", "daily at 9am").
    alert_condition: AlertCondition
    webhook_url: str | None = None
    delivery_channels: list[str] = field(default_factory=lambda: ["stdout"])  # "stdout" | "webhook" | "slack"
    slack_webhook_url: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    last_run: datetime | None = None
    last_value: float | None = None
    last_status: str | None = None     # "ok" | "alert" | "error"
    last_message: str | None = None


@dataclass
class WatchRun:
    """Outcome of a single scheduled execution of a watch."""

    watch_name: str
    ran_at: datetime
    value: float | None
    baseline: float | None
    baseline_type: str | None
    triggered: bool
    status: str                        # "ok" | "alert" | "error"
    message: str
    sql_executed: str
    error: str | None = None
