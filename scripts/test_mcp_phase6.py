"""
Phase 6 MCP verification.

- Enable dual-path, ask the same 'revenue by category' question that misfired in Phase 5.
  The divergence (if any) should surface as a warning + lower confidence.
- Call correct_query with a known-good SQL, then re-ask — the retriever should surface
  the corrected pattern and the generated SQL should follow it.
"""

from __future__ import annotations

import asyncio
import json
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    env = {**os.environ, "TALKDB_DUAL_PATH_ENABLED": "true"}
    params = StdioServerParameters(
        command="talkdb", args=["serve", "--transport", "stdio"], env=env
    )

    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        question = "Top 5 product categories by total completed revenue"

        print("=== Round 1: ask with dual-path enabled ===")
        r = await session.call_tool("ask", {"question": question})
        before = json.loads(r.content[0].text)
        print(f"  sql: {before['sql']}")
        print(f"  rows: {before['row_count']}  confidence: {before['confidence']}")
        print(f"  warnings: {before.get('warnings')}")

        correct_sql = (
            "SELECT p.category, SUM(oi.quantity * oi.unit_price) AS revenue "
            "FROM order_items oi "
            "JOIN orders o ON oi.order_id = o.id "
            "JOIN products p ON oi.product_id = p.id "
            "WHERE o.status = 'completed' "
            "GROUP BY p.category ORDER BY revenue DESC LIMIT 5"
        )

        print("\n=== Submit correction via correct_query ===")
        r = await session.call_tool(
            "correct_query",
            {
                "original_question": question,
                "wrong_sql": before["sql"] or "SELECT 1",
                "correct_sql": correct_sql,
            },
        )
        feedback = json.loads(r.content[0].text)
        print(f"  {feedback}")

        print("\n=== Round 2: re-ask the same question ===")
        r = await session.call_tool("ask", {"question": question})
        after = json.loads(r.content[0].text)
        print(f"  sql: {after['sql']}")
        print(f"  rows: {after['row_count']}  confidence: {after['confidence']}")
        print(f"  warnings: {after.get('warnings')}")

        # Sanity checks
        assert before["sql"], "round 1 produced no SQL"
        assert after["sql"], "round 2 produced no SQL"
        # After the correction, the generated SQL should include 'order_items'
        # (the right bridge table), matching the pattern we saved.
        assert "order_items" in after["sql"].lower(), (
            f"round 2 SQL did not use order_items even after correction. SQL: {after['sql']}"
        )
        print("\nAll checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
