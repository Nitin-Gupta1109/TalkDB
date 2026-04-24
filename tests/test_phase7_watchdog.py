"""Phase 7 unit tests: schedule parser, baseline, storage, condition evaluator."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from talkdb.watchdog.baseline import BaselineComputer
from talkdb.watchdog.manager import _evaluate_condition, _extract_primary_value
from talkdb.watchdog.scheduler import ScheduleParseError, parse_schedule
from talkdb.watchdog.storage import WatchdogStorage
from talkdb.watchdog.watch import AlertCondition, Watch


class TestScheduleParser:
    def test_interval_hours(self):
        assert isinstance(parse_schedule("every 2 hours"), IntervalTrigger)

    def test_interval_minute_no_plural(self):
        assert isinstance(parse_schedule("every 15 minute"), IntervalTrigger)

    def test_daily_at_24h(self):
        t = parse_schedule("daily at 18:30")
        assert isinstance(t, CronTrigger)

    def test_daily_at_12h_pm(self):
        t = parse_schedule("daily at 9pm")
        assert isinstance(t, CronTrigger)

    def test_cron_expression(self):
        t = parse_schedule("0 9 * * 1")
        assert isinstance(t, CronTrigger)

    def test_invalid(self):
        with pytest.raises(ScheduleParseError):
            parse_schedule("occasionally")


class TestStorage:
    def _fresh(self, tmp_path: Path) -> WatchdogStorage:
        return WatchdogStorage(path=str(tmp_path / "w.sqlite"))

    def _sample_watch(self, name: str = "w1") -> Watch:
        return Watch(
            name=name,
            question="How many orders today?",
            sql="SELECT COUNT(*) FROM orders",
            database=None,
            schedule="every 1 hour",
            alert_condition=AlertCondition(kind="pulse"),
        )

    def test_upsert_and_get(self, tmp_path: Path):
        s = self._fresh(tmp_path)
        s.upsert(self._sample_watch("w1"))
        w = s.get("w1")
        assert w is not None and w.name == "w1"

    def test_history_and_delete(self, tmp_path: Path):
        s = self._fresh(tmp_path)
        s.upsert(self._sample_watch("w1"))
        s.record_history("w1", 100.0, timestamp=datetime(2026, 4, 1))
        s.record_history("w1", 110.0, timestamp=datetime(2026, 4, 2))
        assert len(s.history("w1")) == 2
        assert s.delete("w1")
        assert s.get("w1") is None

    def test_list_active_only(self, tmp_path: Path):
        s = self._fresh(tmp_path)
        w1 = self._sample_watch("w1")
        w2 = self._sample_watch("w2")
        w2.is_active = False
        s.upsert(w1)
        s.upsert(w2)
        assert len(s.all(active_only=True)) == 1
        assert len(s.all()) == 2


class TestBaseline:
    def test_7_day_avg(self, tmp_path: Path):
        s = WatchdogStorage(path=str(tmp_path / "w.sqlite"))
        w = Watch(name="rev", question="", sql="", database=None, schedule="every 1 hour", alert_condition=AlertCondition())
        s.upsert(w)
        now = datetime(2026, 4, 10)
        for i, v in enumerate([100.0, 110.0, 95.0, 105.0, 102.0, 108.0, 100.0, 90.0]):
            s.record_history("rev", v, timestamp=now - timedelta(days=i))
        bc = BaselineComputer(s)
        r = bc.compute("rev", "7_day_avg", now=now)
        # 7-day avg: values within last 7 days (not counting day 8: 90.0) = first 7 values ~ 102.85
        assert r.baseline is not None
        assert 100 < r.baseline < 115

    def test_empty_history(self, tmp_path: Path):
        s = WatchdogStorage(path=str(tmp_path / "w.sqlite"))
        w = Watch(name="rev", question="", sql="", database=None, schedule="every 1 hour", alert_condition=AlertCondition())
        s.upsert(w)
        bc = BaselineComputer(s)
        r = bc.compute("rev", "7_day_avg")
        assert r.baseline is None


class TestExtractValue:
    def test_first_numeric(self):
        v = _extract_primary_value(["n"], [{"n": 42}])
        assert v == 42.0

    def test_skips_non_numeric(self):
        v = _extract_primary_value(["label", "n"], [{"label": "x", "n": 100}])
        assert v == 100.0

    def test_no_rows(self):
        assert _extract_primary_value(["n"], []) is None

    def test_skips_bool(self):
        v = _extract_primary_value(["is_ok", "n"], [{"is_ok": True, "n": 7}])
        assert v == 7.0


class TestConditions:
    def test_pulse_always_fires(self):
        trig, _ = _evaluate_condition(42.0, AlertCondition(kind="pulse"), baseline=None, stdev=None)
        assert trig

    def test_threshold_above(self):
        c = AlertCondition(kind="threshold", threshold_value=100, threshold_direction="above")
        trig, _ = _evaluate_condition(150.0, c, baseline=None, stdev=None)
        assert trig
        trig2, _ = _evaluate_condition(50.0, c, baseline=None, stdev=None)
        assert not trig2

    def test_change_percent_drop(self):
        c = AlertCondition(kind="change_percent", change_percent=-20, baseline_type="7_day_avg")
        trig, _ = _evaluate_condition(value=80.0, condition=c, baseline=100.0, stdev=None)
        # 80 is -20% of 100 → equals threshold, should trigger.
        assert trig

    def test_change_percent_no_baseline(self):
        c = AlertCondition(kind="change_percent", change_percent=-20, baseline_type="7_day_avg")
        trig, msg = _evaluate_condition(value=80.0, condition=c, baseline=None, stdev=None)
        assert not trig
        assert "no baseline" in msg.lower()

    def test_anomaly(self):
        c = AlertCondition(kind="anomaly", anomaly_std_devs=2.0)
        trig, _ = _evaluate_condition(value=200.0, condition=c, baseline=100.0, stdev=10.0)
        assert trig  # z = 10
        trig2, _ = _evaluate_condition(value=105.0, condition=c, baseline=100.0, stdev=10.0)
        assert not trig2  # z = 0.5
