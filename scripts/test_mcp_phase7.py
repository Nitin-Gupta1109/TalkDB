"""
Phase 7 MCP verification: watch -> establish baseline -> force a drop -> run_watch -> alert fires.

Steps:
 1. Clear any prior watchdog state.
 2. Create a watch on 'completed orders today' with a 'drops more than 20% below 7-day avg' condition.
 3. Seed baseline history (7 prior days) directly in the watchdog SQLite — so we don't have to wait a week.
 4. Invoke run_watch.
 5. Assert alert fires (status == 'alert') with useful message + suggested follow-up.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

WATCHDOG_DB = Path("data/watchdog.sqlite")


def seed_baseline_history(watch_name: str, baseline_values: list[float]) -> None:
    """Directly inject 7 days of baseline into the watchdog SQLite."""
    from talkdb.watchdog.storage import WatchdogStorage

    storage = WatchdogStorage(path=str(WATCHDOG_DB))
    today = datetime.utcnow().replace(hour=12, minute=0, second=0, microsecond=0)
    for i, v in enumerate(baseline_values):
        storage.record_history(watch_name, v, timestamp=today - timedelta(days=i + 1))


async def main() -> None:
    if WATCHDOG_DB.exists():
        WATCHDOG_DB.unlink()

    params = StdioServerParameters(command="talkdb", args=["serve", "--transport", "stdio"])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        # 1. Create the watch. add_watch runs it once to establish a baseline point.
        print("=== Create watch ===")
        r = await session.call_tool(
            "watch",
            {
                "name": "refunded_count",
                "question": "How many refunded orders are there?",
                "schedule": "every 1 hour",
                "alert_condition": "drops more than 10% below 7-day average",
            },
        )
        created = json.loads(r.content[0].text)
        print(json.dumps(created, indent=2))
        assert "error" not in created, f"watch creation failed: {created}"

        # 2. Seed baseline history so the change_percent condition has data to compare against.
        print("\n=== Seed 7 days of baseline history (values around 100) ===")
        seed_baseline_history("refunded_count", [100.0, 102.0, 98.0, 101.0, 99.0, 103.0, 97.0])

        # 3. Manually run the watch (the actual value from the seeded DB is small — ~40,
        #    which is well below 100 baseline).
        print("\n=== Run watch — expect alert ===")
        r = await session.call_tool("run_watch", {"name": "refunded_count"})
        run = json.loads(r.content[0].text)
        print(json.dumps(run, indent=2))

        # 4. List watches to confirm last_status propagated.
        print("\n=== list_watches ===")
        r = await session.call_tool("list_watches", {})
        # MCP returns each list element as its own content block. Join them back into a list.
        watches = [json.loads(c.text) for c in r.content]
        for w in watches:
            print(f"  {w['name']}: last_status={w['last_status']} last_value={w['last_value']}")

        # 5. Assertions.
        assert run["value"] is not None, "watch did not return a numeric value"
        assert run["baseline"] is not None, "baseline was not computed"
        assert run["status"] == "alert", f"expected alert, got {run['status']}"
        assert "deviation" in run["message"].lower() or "%" in run["message"], run["message"]

        # 6. Cleanup.
        print("\n=== Remove watch ===")
        r = await session.call_tool("remove_watch", {"name": "refunded_count"})
        print(json.loads(r.content[0].text))

        print("\nAll checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
