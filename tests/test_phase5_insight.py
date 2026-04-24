"""Phase 5 unit tests: analyzer (pandas math, no LLM) and charter (decision tree)."""

from __future__ import annotations

import base64

import pytest

from talkdb.core.intent import Intent, IntentType
from talkdb.insight.analyzer import InsightAnalyzer
from talkdb.insight.charter import InsightCharter


def _intent(t: IntentType, single: bool = False) -> Intent:
    return Intent(type=t, is_single_value=single, matched_keywords=[])


class TestAnalyzer:
    def test_empty_result(self):
        a = InsightAnalyzer()
        r = a.analyze([], ["n"], _intent(IntentType.AGGREGATION, True))
        assert r.row_count == 0
        assert any("no rows" in f.lower() for f in r.key_findings)

    def test_single_value(self):
        a = InsightAnalyzer()
        r = a.analyze([{"n": 200}], ["n"], _intent(IntentType.AGGREGATION, True))
        assert r.single_value is not None
        assert r.single_value.value == 200
        assert "200" in r.key_findings[0]

    def test_time_series_trend_up(self):
        a = InsightAnalyzer()
        rows = [
            {"month": "2026-01-01", "rev": 100.0},
            {"month": "2026-02-01", "rev": 110.0},
            {"month": "2026-03-01", "rev": 121.0},
            {"month": "2026-04-01", "rev": 133.0},
        ]
        r = a.analyze(rows, ["month", "rev"], _intent(IntentType.DISTRIBUTION))
        assert r.time_series is not None
        assert r.time_series.trend == "up"
        assert r.time_series.total_change_pct is not None
        assert r.time_series.total_change_pct > 30

    def test_categorical_ranking(self):
        a = InsightAnalyzer()
        rows = [
            {"tier": "gold", "rev": 500.0},
            {"tier": "silver", "rev": 300.0},
            {"tier": "bronze", "rev": 200.0},
        ]
        r = a.analyze(rows, ["tier", "rev"], _intent(IntentType.RANKING))
        assert r.categorical is not None
        assert r.categorical.top_label == "gold"
        assert r.categorical.top_share_pct == 50.0  # 500 / 1000
        assert r.categorical.groups == 3

    def test_anomaly_detection(self):
        a = InsightAnalyzer()
        rows = [
            {"date": "2026-01-01", "v": 100.0},
            {"date": "2026-02-01", "v": 102.0},
            {"date": "2026-03-01", "v": 98.0},
            {"date": "2026-04-01", "v": 500.0},  # spike
            {"date": "2026-05-01", "v": 105.0},
        ]
        r = a.analyze(rows, ["date", "v"], _intent(IntentType.DISTRIBUTION))
        assert r.time_series is not None
        assert len(r.time_series.anomalies) >= 1
        # The spike at index 3 must be one of the anomalies.
        anomaly_indices = {a["index"] for a in r.time_series.anomalies}
        assert 3 in anomaly_indices


class TestCharter:
    def _png_decodes(self, b64: str) -> bool:
        try:
            data = base64.b64decode(b64)
            return data.startswith(b"\x89PNG")
        except Exception:
            return False

    def test_empty_skipped(self):
        a = InsightAnalyzer()
        c = InsightCharter()
        analysis = a.analyze([], ["n"], _intent(IntentType.AGGREGATION, True))
        spec, reason = c.generate([], ["n"], _intent(IntentType.AGGREGATION, True), analysis)
        assert spec is None
        assert reason

    def test_single_value_is_metric_card(self):
        a = InsightAnalyzer()
        c = InsightCharter()
        rows = [{"revenue": 734283.16}]
        analysis = a.analyze(rows, ["revenue"], _intent(IntentType.AGGREGATION, True))
        spec, reason = c.generate(rows, ["revenue"], _intent(IntentType.AGGREGATION, True), analysis)
        assert spec is not None and spec.chart_type == "metric_card"
        assert self._png_decodes(spec.image_base64)

    def test_time_series_is_line(self):
        a = InsightAnalyzer()
        c = InsightCharter()
        rows = [
            {"month": "2026-01-01", "rev": 100.0},
            {"month": "2026-02-01", "rev": 110.0},
            {"month": "2026-03-01", "rev": 121.0},
            {"month": "2026-04-01", "rev": 133.0},
        ]
        analysis = a.analyze(rows, ["month", "rev"], _intent(IntentType.DISTRIBUTION))
        spec, _ = c.generate(rows, ["month", "rev"], _intent(IntentType.DISTRIBUTION), analysis)
        assert spec is not None and spec.chart_type == "line"
        assert self._png_decodes(spec.image_base64)

    def test_ranking_is_horizontal_bar(self):
        a = InsightAnalyzer()
        c = InsightCharter()
        rows = [
            {"name": f"c{i}", "rev": 100 - i * 10}
            for i in range(5)
        ]
        analysis = a.analyze(rows, ["name", "rev"], _intent(IntentType.RANKING))
        spec, _ = c.generate(rows, ["name", "rev"], _intent(IntentType.RANKING), analysis)
        assert spec is not None and spec.chart_type == "horizontal_bar"

    def test_small_distribution_is_vertical_bar(self):
        a = InsightAnalyzer()
        c = InsightCharter()
        rows = [{"tier": "gold", "rev": 500}, {"tier": "silver", "rev": 300}, {"tier": "bronze", "rev": 200}]
        analysis = a.analyze(rows, ["tier", "rev"], _intent(IntentType.DISTRIBUTION))
        spec, _ = c.generate(rows, ["tier", "rev"], _intent(IntentType.DISTRIBUTION), analysis)
        assert spec is not None and spec.chart_type == "vertical_bar"

    def test_text_only_no_chart(self):
        a = InsightAnalyzer()
        c = InsightCharter()
        rows = [{"name": "Alice"}, {"name": "Bob"}, {"name": "Carol"}]
        analysis = a.analyze(rows, ["name"], _intent(IntentType.LOOKUP))
        spec, reason = c.generate(rows, ["name"], _intent(IntentType.LOOKUP), analysis)
        assert spec is None
        assert "numeric" in reason.lower()

    def test_too_few_rows_no_chart(self):
        a = InsightAnalyzer()
        c = InsightCharter()
        rows = [{"tier": "gold", "rev": 500}, {"tier": "silver", "rev": 300}]
        analysis = a.analyze(rows, ["tier", "rev"], _intent(IntentType.DISTRIBUTION))
        spec, reason = c.generate(rows, ["tier", "rev"], _intent(IntentType.DISTRIBUTION), analysis)
        assert spec is None
        assert "few rows" in reason.lower()
