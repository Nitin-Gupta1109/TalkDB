"""Phase 2 MCP verification: does `ask` route through the retriever + semantic metrics over stdio?"""

from __future__ import annotations

import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

CASES = [
    (
        "What is our total revenue?",
        lambda sql: "status" in sql.lower() and "completed" in sql.lower(),
        "should apply the `status = 'completed'` filter from the revenue metric",
    ),
    (
        "What is the average order value?",
        lambda sql: "avg" in sql.lower() and "status" in sql.lower() and "completed" in sql.lower(),
        "should use AVG + completed filter from the average_order_value metric",
    ),
    (
        "How many active customers do we have?",
        lambda sql: "distinct" in sql.lower() and ("90" in sql or "days" in sql.lower()),
        "should use DISTINCT + 90-day window from the active_customers metric",
    ),
    (
        "Revenue by product category",
        lambda sql: "group by" in sql.lower() and "category" in sql.lower(),
        "should GROUP BY category (matches an example in the YAML)",
    ),
]


async def main() -> None:
    params = StdioServerParameters(command="talkdb", args=["serve", "--transport", "stdio"])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        passed = 0
        for question, check, reason in CASES:
            r = await session.call_tool("ask", {"question": question})
            result = json.loads(r.content[0].text)
            sql = result.get("sql", "")
            ok = bool(sql) and check(sql)
            marker = "PASS" if ok else "FAIL"
            print(f"[{marker}] {question}")
            print(f"    sql: {sql}")
            print(f"    confidence: {result['confidence']}  rows: {result['row_count']}")
            print(f"    expected: {reason}")
            if not ok and result.get("explanation"):
                print(f"    explanation: {result['explanation']}")
            print()
            passed += int(ok)

        print(f"{passed}/{len(CASES)} cases passed")


if __name__ == "__main__":
    asyncio.run(main())
