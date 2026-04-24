"""Alert dispatch: stdout, webhook, Slack (which is just a webhook)."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass

import httpx

from talkdb.watchdog.watch import Watch

_log = logging.getLogger("talkdb.watchdog")


@dataclass
class Alert:
    watch_name: str
    value: float | None
    baseline: float | None
    baseline_kind: str | None
    deviation_pct: float | None
    message: str
    suggested_follow_up: str | None = None


class Alerter:
    async def send(self, watch: Watch, alert: Alert) -> list[str]:
        """Dispatch the alert to every configured channel. Returns list of channels delivered successfully."""
        delivered: list[str] = []
        for channel in watch.delivery_channels:
            channel = channel.strip().lower()
            try:
                if channel == "stdout":
                    self._to_stdout(watch, alert)
                    delivered.append("stdout")
                elif channel == "webhook" and watch.webhook_url:
                    await self._to_webhook(watch.webhook_url, self._payload(alert))
                    delivered.append("webhook")
                elif channel == "slack" and watch.slack_webhook_url:
                    await self._to_webhook(watch.slack_webhook_url, self._slack_payload(watch, alert))
                    delivered.append("slack")
            except Exception as e:  # noqa: BLE001 — delivery errors should not crash the scheduler
                _log.warning("alert delivery to %s failed: %s", channel, e)
        return delivered

    @staticmethod
    def _to_stdout(watch: Watch, alert: Alert) -> None:
        # Use stderr, not stdout — stdout is reserved for the MCP JSON-RPC protocol
        # when the server is running over stdio. Writing to stdout corrupts the stream.
        print(f"🔴 TalkDB watch alert — {watch.name}", file=sys.stderr)
        print(f"   {alert.message}", file=sys.stderr)
        if alert.suggested_follow_up:
            print(f"   Suggested follow-up: {alert.suggested_follow_up}", file=sys.stderr)

    @staticmethod
    async def _to_webhook(url: str, payload: dict) -> None:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()

    @staticmethod
    def _payload(alert: Alert) -> dict:
        return {
            "watch_name": alert.watch_name,
            "value": alert.value,
            "baseline": alert.baseline,
            "baseline_kind": alert.baseline_kind,
            "deviation_pct": alert.deviation_pct,
            "message": alert.message,
            "suggested_follow_up": alert.suggested_follow_up,
        }

    @staticmethod
    def _slack_payload(watch: Watch, alert: Alert) -> dict:
        """Standard Slack incoming-webhook payload."""
        lines = [f"*{watch.name}*", alert.message]
        if alert.suggested_follow_up:
            lines.append(f"Follow-up: _{alert.suggested_follow_up}_")
        return {"text": "\n".join(lines)}


def build_message(watch: Watch, value: float | None, baseline: float | None, kind: str | None) -> Alert:
    """Compose a human-readable alert message from the watch run outcome."""
    deviation_pct: float | None = None
    if baseline is not None and baseline != 0 and value is not None:
        deviation_pct = round((value - baseline) / baseline * 100, 1)

    value_str = _fmt(value)
    baseline_str = f"{_fmt(baseline)} ({kind})" if baseline is not None else "n/a"
    if deviation_pct is None:
        core = f"Current value: {value_str}."
    else:
        direction = "above" if deviation_pct > 0 else "below"
        core = f"Current value: {value_str} — {abs(deviation_pct)}% {direction} baseline {baseline_str}."

    follow_up: str | None = None
    if deviation_pct is not None and abs(deviation_pct) >= 10:
        follow_up = f"Why is '{watch.question}' {direction} baseline?"

    return Alert(
        watch_name=watch.name,
        value=value,
        baseline=baseline,
        baseline_kind=kind,
        deviation_pct=deviation_pct,
        message=core,
        suggested_follow_up=follow_up,
    )


def _fmt(v: float | None) -> str:
    if v is None:
        return "n/a"
    if abs(v) >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"{v / 1_000:.2f}k"
    if v == int(v):
        return str(int(v))
    return f"{v:.2f}"
