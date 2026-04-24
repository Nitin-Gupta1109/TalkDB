"""
Phase 5 MCP verification: `analyze` returns SQL + rows + narrative + chart.

Saves the chart PNG to disk for visual inspection.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

OUT_DIR = Path("data/phase5_charts")


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    params = StdioServerParameters(command="talkdb", args=["serve", "--transport", "stdio"])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        cases = [
            ("q1_single_value", "What is our total revenue?"),
            ("q2_ranking", "Top 5 product categories by total completed revenue"),
            ("q3_distribution", "Total revenue by customer tier, for completed orders"),
            ("q4_time_series", "Revenue by month, for completed orders"),
        ]

        for case_id, question in cases:
            print(f"\n=== {case_id}: {question!r} ===")
            r = await session.call_tool("analyze", {"question": question})
            result = json.loads(r.content[0].text)

            print(f"  sql: {result['sql']}")
            print(f"  rows: {result['row_count']}  confidence: {result['confidence']}")
            print(f"  key_findings: {result.get('key_findings')}")
            print(f"  insight: {result.get('insight')}")
            if result.get("chart"):
                chart = result["chart"]
                print(f"  chart: {chart['type']} — '{chart['title']}'")
                png_path = OUT_DIR / f"{case_id}.png"
                png_path.write_bytes(base64.b64decode(chart["image_base64"]))
                print(f"    saved: {png_path}")
            else:
                print(f"  chart: skipped ({result.get('chart_skipped_reason')})")


if __name__ == "__main__":
    asyncio.run(main())
