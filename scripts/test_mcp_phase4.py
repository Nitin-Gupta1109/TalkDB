"""
Phase 4 MCP verification: ask -> follow_up -> follow_up chain over stdio.

Reproduces the CLAUDE.md "done" checkpoint for Phase 4:
  "revenue by month" -> "just Q4" -> "break that down by region"
Each follow-up must produce SQL that builds on the prior context.
"""

from __future__ import annotations

import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    params = StdioServerParameters(command="talkdb", args=["serve", "--transport", "stdio"])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        # Turn 1: standalone question. Capture the session_id.
        print("Turn 1: 'Revenue by month'")
        r = await session.call_tool(
            "ask",
            {"question": "Revenue by month this year", "session_id": "phase4-demo"},
        )
        t1 = json.loads(r.content[0].text)
        print(f"  sql: {t1['sql']}")
        print(f"  rows: {t1['row_count']}")
        print(f"  explanation: {t1.get('explanation')}")
        print()

        # Turn 2: follow-up that materially narrows the prior query.
        print("Turn 2 (follow-up): 'just platinum and gold tier customers'")
        r = await session.call_tool(
            "follow_up",
            {"refinement": "just platinum and gold tier customers", "session_id": "phase4-demo"},
        )
        t2 = json.loads(r.content[0].text)
        print(f"  sql: {t2['sql']}")
        print(f"  rows: {t2['row_count']}")
        print(f"  explanation: {t2.get('explanation')}")
        print()

        # Turn 3: stack another refinement.
        print("Turn 3 (follow-up): 'break that down by product category'")
        r = await session.call_tool(
            "follow_up",
            {"refinement": "break that down by product category", "session_id": "phase4-demo"},
        )
        t3 = json.loads(r.content[0].text)
        print(f"  sql: {t3['sql']}")
        print(f"  rows: {t3['row_count']}")
        print(f"  explanation: {t3.get('explanation')}")
        print()

        # Inspect the stored session state.
        print("get_session state:")
        r = await session.call_tool("get_session", {"session_id": "phase4-demo"})
        state = json.loads(r.content[0].text)
        print(f"  turn_count: {state['turn_count']}")
        for t in state["turns"]:
            print(f"  turn {t['turn_number']}: {t['question']}")
            if t["question"] != t["rewritten_question"]:
                print(f"    -> rewritten: {t['rewritten_question']}")
            print(f"    sql: {t['sql']}")

        # Simple checks — each turn must have produced SQL, each later turn should differ structurally.
        assert t1["sql"], "turn 1 produced no SQL"
        assert t2["sql"], "turn 2 produced no SQL"
        assert t3["sql"], "turn 3 produced no SQL"
        assert t1["sql"] != t2["sql"], "turn 2 should differ from turn 1 (added tier filter)"
        assert t2["sql"] != t3["sql"], "turn 3 should differ from turn 2 (added category grouping)"
        assert "tier" in t2["sql"].lower(), "turn 2 should reference customer tier"
        assert "category" in t3["sql"].lower(), "turn 3 should include category grouping"
        assert "tier" in t3["sql"].lower(), "turn 3 should still carry the tier filter from turn 2"
        print("\nAll checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
