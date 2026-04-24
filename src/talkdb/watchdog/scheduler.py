"""
APScheduler integration. Parses human-readable schedule strings, registers each
active watch with AsyncIOScheduler, and runs them in the same asyncio event loop
as the MCP server.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Awaitable, Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from talkdb.watchdog.watch import Watch

if TYPE_CHECKING:
    from apscheduler.triggers.base import BaseTrigger

_log = logging.getLogger("talkdb.watchdog")

_INTERVAL_RE = re.compile(r"^\s*every\s+(\d+)\s*(second|minute|hour|day)s?\s*$", re.I)
_DAILY_AT_RE = re.compile(r"^\s*daily\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*$", re.I)


class ScheduleParseError(ValueError):
    pass


def parse_schedule(schedule: str) -> "BaseTrigger":
    """
    Accepts three shapes:
      "every N (second|minute|hour|day)"  -> IntervalTrigger
      "daily at HH[:MM] [am|pm]"          -> CronTrigger
      "<cron expression with 5 fields>"   -> CronTrigger
    """
    text = schedule.strip()

    m = _INTERVAL_RE.match(text)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        return IntervalTrigger(**{f"{unit}s": n})

    m = _DAILY_AT_RE.match(text)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        meridiem = (m.group(3) or "").lower()
        if meridiem == "pm" and hour < 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
        return CronTrigger(hour=hour, minute=minute)

    parts = text.split()
    if len(parts) == 5:
        # Standard cron expression.
        return CronTrigger.from_crontab(text)

    raise ScheduleParseError(
        f"Cannot parse schedule {text!r}. Use 'every N minute/hour/day', "
        f"'daily at HH[:MM] [am|pm]', or a 5-field cron expression."
    )


class WatchdogScheduler:
    """
    Owns an AsyncIOScheduler. The runner callback is injected by WatchdogManager
    so the scheduler module stays free of engine/storage dependencies.
    """

    def __init__(self, runner: Callable[[str], Awaitable[None]]):
        self._runner = runner
        self._scheduler: AsyncIOScheduler | None = None

    def start(self) -> None:
        if self._scheduler is None:
            self._scheduler = AsyncIOScheduler()
        if not self._scheduler.running:
            self._scheduler.start()

    def shutdown(self) -> None:
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    def add_watch(self, watch: Watch) -> None:
        if self._scheduler is None:
            self.start()
        assert self._scheduler is not None
        try:
            trigger = parse_schedule(watch.schedule)
        except ScheduleParseError as e:
            _log.warning("skipping watch %s: %s", watch.name, e)
            return

        self._scheduler.add_job(
            self._runner,
            trigger,
            args=[watch.name],
            id=f"watch:{watch.name}",
            replace_existing=True,
            misfire_grace_time=60,
            coalesce=True,
        )

    def remove_watch(self, name: str) -> None:
        if self._scheduler is None:
            return
        job_id = f"watch:{name}"
        if self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)

    def jobs(self) -> list[dict]:
        if self._scheduler is None:
            return []
        return [
            {
                "id": j.id,
                "next_run_time": j.next_run_time.isoformat() if j.next_run_time else None,
            }
            for j in self._scheduler.get_jobs()
        ]

    @property
    def running(self) -> bool:
        return bool(self._scheduler and self._scheduler.running)
