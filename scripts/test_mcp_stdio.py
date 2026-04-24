"""
Spawn `talkdb serve --transport stdio` as a subprocess and exercise each MCP tool
via the official MCP Python SDK client. Verifies the full protocol: initialize,
list_tools, call_tool.
"""

from __future__ import annotations

import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    params = StdioServerParameters(
        command="talkdb",
        args=["serve", "--transport", "stdio"],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init_result = await session.initialize()
            print(f"[init] server: {init_result.serverInfo.name} v{init_result.serverInfo.version}")

            tools = await session.list_tools()
            print(f"\n[tools] {len(tools.tools)} registered:")
            for t in tools.tools:
                print(f"  - {t.name}: {t.description.strip().splitlines()[0] if t.description else ''}")

            print("\n[call] list_databases()")
            r = await session.call_tool("list_databases", {})
            print(_text(r))

            print("\n[call] describe_database()")
            r = await session.call_tool("describe_database", {})
            info = json.loads(_text(r))
            print(f"  dialect: {info['dialect']}")
            print(f"  tables: {[t['name'] for t in info['tables']]}")

            print("\n[call] ask('How many customers are there?')")
            r = await session.call_tool("ask", {"question": "How many customers are there?"})
            result = json.loads(_text(r))
            print(f"  sql: {result['sql']!r}")
            print(f"  rows: {result['results']}")
            print(f"  confidence: {result['confidence']}")
            print(f"  explanation: {result.get('explanation')}")
            print(f"  warnings: {result.get('warnings')}")

            print("\n[call] validate_sql(good SQL)")
            r = await session.call_tool("validate_sql", {"sql": "SELECT COUNT(*) FROM customers"})
            v = json.loads(_text(r))
            print(f"  valid: {v['valid']}")
            print(f"  tables_referenced: {v['tables_referenced']}")
            print(f"  sample_rows: {v['sample_rows']}")

            print("\n[call] validate_sql(hallucinated column)")
            r = await session.call_tool(
                "validate_sql",
                {"sql": "SELECT first_name, last_name FROM customers"},
            )
            v = json.loads(_text(r))
            print(f"  valid: {v['valid']}")
            print(f"  issues: {v['issues']}")

            print("\n[call] validate_sql(unsafe: DROP)")
            r = await session.call_tool("validate_sql", {"sql": "DROP TABLE customers"})
            v = json.loads(_text(r))
            print(f"  valid: {v['valid']}")
            print(f"  issues: {v['issues']}")


def _text(call_result) -> str:
    """Extract text payload from an MCP tool call result."""
    parts: list[str] = []
    for content in call_result.content:
        if hasattr(content, "text"):
            parts.append(content.text)
    return "\n".join(parts)


if __name__ == "__main__":
    asyncio.run(main())
