"""
Schema linker: a pre-pass that asks the LLM which tables are needed for a question.

Reduces hallucination on schemas with many tables — if the retriever surfaces
50 columns from 10 tables and only 2 tables are actually relevant, the generator
gets confused. The linker narrows the playing field before retrieval results
reach the generator.

Cost: one extra LLM call per question (~$0.001 on gpt-4o-mini). Worth it when
schemas have > ~8 tables; for tiny schemas it's pure overhead.
"""

from __future__ import annotations

import json
import re

import litellm

from talkdb.schema.models import DatabaseSchema

LINKER_SYSTEM = """\
You are a schema linker for a text-to-SQL system. Given a user question and a
list of tables, identify the MINIMAL set of tables required to answer the question.

Rules:
- Return ONLY a JSON list of table names. No explanation.
- Include tables needed for joins, even if not directly queried.
- If unsure between two similar tables, include both — false positives cost less than false negatives.
- If the question is ambiguous, return the tables for the most literal reading.

Example:
  Question: "How many orders were placed last month?"
  Tables: customers, orders, order_items, products
  Answer: ["orders"]

  Question: "Top 5 customers by revenue"
  Tables: customers, orders, order_items, products
  Answer: ["customers", "orders"]
"""


class SchemaLinker:
    """Given a question + a schema, return the names of the tables that are relevant."""

    def __init__(self, model: str, max_tokens: int = 200):
        self._model = model
        self._max_tokens = max_tokens

    async def link(self, question: str, schema: DatabaseSchema) -> list[str]:
        """Return a list of table names. On any failure, returns [] (caller should
        interpret that as 'no filter' — fall back to full retrieval)."""
        if not schema.tables:
            return []

        schema_text = self._render_schema(schema)
        try:
            resp = await litellm.acompletion(
                model=self._model,
                messages=[
                    {"role": "system", "content": LINKER_SYSTEM},
                    {"role": "user", "content": f"Question: {question}\n\nTables:\n{schema_text}\n\nAnswer:"},
                ],
                temperature=0,
                max_tokens=self._max_tokens,
            )
            raw = resp.choices[0].message.content or ""
        except Exception:  # noqa: BLE001 — linker is best-effort
            return []

        tables = self._parse(raw)
        # Defensive: only return names that actually exist in the schema.
        valid = {t.name for t in schema.tables}
        return [t for t in tables if t in valid]

    def _render_schema(self, schema: DatabaseSchema) -> str:
        lines = []
        for t in schema.tables:
            cols = ", ".join(c.name for c in t.columns[:8])
            desc = f" — {t.description}" if t.description else ""
            lines.append(f"  {t.name}: columns=[{cols}]{desc}")
        return "\n".join(lines)

    def _parse(self, raw: str) -> list[str]:
        # Expect a bare JSON list, but tolerate code fences or stray text around it.
        m = re.search(r"\[[^\[\]]*\]", raw)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        return [str(x).strip() for x in data if isinstance(x, str)]
