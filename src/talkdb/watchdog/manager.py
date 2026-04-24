"""
WatchdogManager — high-level orchestration.

- Creates watches: parse NL alert condition, pre-validate SQL via the engine, persist.
- Runs watches: execute SQL, record history, compute baseline, evaluate condition, alert.
- Registers watches with the scheduler on start.

Separate from Engine (uses engine for SQL execution but owns its own storage / alerter / scheduler).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

import litellm

from talkdb.config.settings import Settings
from talkdb.watchdog.alerter import Alert, Alerter, build_message
from talkdb.watchdog.baseline import BaselineComputer
from talkdb.watchdog.scheduler import WatchdogScheduler
from talkdb.watchdog.storage import WatchdogStorage
from talkdb.watchdog.watch import AlertCondition, Watch, WatchRun

if TYPE_CHECKING:
    from talkdb.core.engine import Engine

_log = logging.getLogger("talkdb.watchdog")

_CONDITION_SYSTEM_PROMPT = """You convert natural-language alert conditions into a JSON struct.

Return STRICT JSON with exactly these keys (null where not applicable):
{
  "kind": "threshold" | "change_percent" | "anomaly" | "pulse",
  "threshold_value": number | null,
  "threshold_direction": "above" | "below" | null,
  "change_percent": number | null,      // negative = drop, positive = rise
  "baseline_type": "7_day_avg" | "30_day_avg" | "previous_period" | "same_day_last_week" | null,
  "anomaly_std_devs": number | null,
  "description": string                 // short English summary of the condition
}

Examples:
"drops more than 20% below 7-day average" -> {"kind":"change_percent","change_percent":-20,"baseline_type":"7_day_avg", ...}
"above 1000" -> {"kind":"threshold","threshold_value":1000,"threshold_direction":"above", ...}
"more than 2 stdev from mean" -> {"kind":"anomaly","anomaly_std_devs":2.0, ...}
""  or no condition given -> {"kind":"pulse", ...}

Return ONLY the JSON object. No preamble. No markdown fences.
"""


class WatchdogManager:
    def __init__(self, engine: "Engine", settings: Settings):
        self.engine = engine
        self.settings = settings
        self.storage = WatchdogStorage(settings.watchdog_db)
        self.baseline_computer = BaselineComputer(self.storage)
        self.alerter = Alerter()
        self.scheduler = WatchdogScheduler(runner=self._run_by_name)

    # ----- Lifecycle -----

    def start(self, load_existing: bool = True) -> None:
        self.scheduler.start()
        if load_existing:
            for watch in self.storage.all(active_only=True):
                self.scheduler.add_watch(watch)

    def shutdown(self) -> None:
        self.scheduler.shutdown()

    # ----- CRUD -----

    async def add_watch(
        self,
        *,
        name: str,
        question: str,
        schedule: str = "every 1 hour",
        alert_condition: str = "",
        database: str | None = None,
        delivery_channels: list[str] | None = None,
        webhook_url: str | None = None,
        slack_webhook_url: str | None = None,
        run_now: bool = True,
    ) -> Watch:
        # 1. Pre-validate SQL by asking the engine once. This is the only LLM-driven SQL generation
        #    for this watch; scheduled runs will just re-execute the saved SQL.
        result = await self.engine.ask(question, database=database)
        if not result.sql:
            raise ValueError(f"Could not generate SQL for watch '{name}': {result.explanation or 'refused'}")

        # 2. Parse the natural-language alert condition.
        condition = await self._parse_condition(alert_condition) if alert_condition.strip() else AlertCondition(kind="pulse")

        channels = delivery_channels or (["webhook"] if webhook_url else (["slack"] if slack_webhook_url else ["stdout"]))

        watch = Watch(
            name=name,
            question=question,
            sql=result.sql,
            database=database,
            schedule=schedule,
            alert_condition=condition,
            delivery_channels=channels,
            webhook_url=webhook_url,
            slack_webhook_url=slack_webhook_url,
        )
        self.storage.upsert(watch)
        self.scheduler.add_watch(watch)
        if run_now:
            await self.run_watch(name)
        return watch

    def remove_watch(self, name: str) -> bool:
        self.scheduler.remove_watch(name)
        return self.storage.delete(name)

    def list_watches(self) -> list[Watch]:
        return self.storage.all()

    def get_watch(self, name: str) -> Watch | None:
        return self.storage.get(name)

    # ----- Execution -----

    async def run_watch(self, name: str) -> WatchRun:
        watch = self.storage.get(name)
        if watch is None:
            raise ValueError(f"No watch named '{name}'")

        ran_at = datetime.utcnow()
        try:
            connector = self.engine.connector_for(watch.database)
            columns, rows = connector.execute(
                watch.sql, read_only=True, timeout_seconds=self.settings.query_timeout
            )
        except Exception as e:  # noqa: BLE001
            msg = f"execution error: {e}"
            self.storage.mark_run(name, ran_at=ran_at, value=None, status="error", message=msg)
            return WatchRun(
                watch_name=name, ran_at=ran_at, value=None, baseline=None,
                baseline_type=None, triggered=False, status="error", message=msg,
                sql_executed=watch.sql, error=str(e),
            )

        value = _extract_primary_value(columns, rows)
        if value is None:
            msg = "could not extract a numeric value from the result (expected single numeric)"
            self.storage.mark_run(name, ran_at=ran_at, value=None, status="error", message=msg)
            return WatchRun(
                watch_name=name, ran_at=ran_at, value=None, baseline=None,
                baseline_type=None, triggered=False, status="error", message=msg,
                sql_executed=watch.sql,
            )

        # Compute baseline BEFORE recording this run, so the new value doesn't pollute the baseline.
        baseline_result = None
        if watch.alert_condition.baseline_type:
            baseline_result = self.baseline_computer.compute(
                watch.name, watch.alert_condition.baseline_type, now=ran_at
            )

        self.storage.record_history(name, value, timestamp=ran_at)

        triggered, message = _evaluate_condition(
            value=value,
            condition=watch.alert_condition,
            baseline=baseline_result.baseline if baseline_result else None,
            stdev=baseline_result.stdev if baseline_result else None,
        )

        if triggered:
            alert = build_message(
                watch,
                value=value,
                baseline=(baseline_result.baseline if baseline_result else None),
                kind=(baseline_result.kind if baseline_result else None),
            )
            alert.message = f"{message} {alert.message}".strip()
            await self.alerter.send(watch, alert)

        status = "alert" if triggered else "ok"
        self.storage.mark_run(name, ran_at=ran_at, value=value, status=status, message=message)

        return WatchRun(
            watch_name=name,
            ran_at=ran_at,
            value=value,
            baseline=(baseline_result.baseline if baseline_result else None),
            baseline_type=(baseline_result.kind if baseline_result else None),
            triggered=triggered,
            status=status,
            message=message,
            sql_executed=watch.sql,
        )

    async def _run_by_name(self, name: str) -> None:
        try:
            await self.run_watch(name)
        except Exception:  # noqa: BLE001
            _log.exception("scheduled run of watch %s failed", name)

    # ----- NL condition parsing -----

    async def _parse_condition(self, text: str) -> AlertCondition:
        try:
            response = await litellm.acompletion(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": _CONDITION_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0,
                max_tokens=200,
            )
            content = (response.choices[0].message.content or "").strip()
            # Strip markdown fences if any.
            if content.startswith("```"):
                content = content.strip("`")
                if content.lower().startswith("json"):
                    content = content[4:].strip()
            data = json.loads(content)
            return AlertCondition(**{k: v for k, v in data.items() if v is not None})
        except Exception as e:  # noqa: BLE001
            _log.warning("alert condition parse failed (%s). Falling back to pulse.", e)
            return AlertCondition(kind="pulse", description=text)


def _extract_primary_value(columns: list[str], rows: list[dict[str, Any]]) -> float | None:
    """The first numeric cell of the first row. If no numeric column, return None."""
    if not rows or not columns:
        return None
    row = rows[0]
    for c in columns:
        v = row.get(c)
        if isinstance(v, bool):
            continue
        if isinstance(v, int | float):
            return float(v)
    return None


def _evaluate_condition(
    value: float,
    condition: AlertCondition,
    baseline: float | None,
    stdev: float | None,
) -> tuple[bool, str]:
    if condition.kind == "pulse":
        return True, f"Current value: {value:.4g}."

    if condition.kind == "threshold":
        if condition.threshold_value is None:
            return False, "threshold condition missing threshold_value"
        if condition.threshold_direction == "above":
            trig = value > condition.threshold_value
            return trig, f"{value:.4g} {'>' if trig else '<='} {condition.threshold_value:.4g} (threshold above)"
        trig = value < condition.threshold_value
        return trig, f"{value:.4g} {'<' if trig else '>='} {condition.threshold_value:.4g} (threshold below)"

    if condition.kind == "change_percent":
        if baseline is None or baseline == 0 or condition.change_percent is None:
            return False, "no baseline yet (need prior history)"
        deviation_pct = (value - baseline) / baseline * 100
        if condition.change_percent < 0:
            # Alert when the drop is AT LEAST as steep as configured.
            trig = deviation_pct <= condition.change_percent
        else:
            trig = deviation_pct >= condition.change_percent
        return trig, f"deviation {deviation_pct:+.1f}% vs {condition.baseline_type}"

    if condition.kind == "anomaly":
        if baseline is None or stdev is None or stdev == 0:
            return False, "insufficient history for anomaly detection"
        z = (value - baseline) / stdev
        trig = abs(z) >= condition.anomaly_std_devs
        return trig, f"z-score {z:+.2f} vs {condition.anomaly_std_devs} stdev threshold"

    return False, f"unknown condition kind: {condition.kind}"
