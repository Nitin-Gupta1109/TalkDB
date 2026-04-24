"""Thin wrapper around sqlglot for dialect conversion."""

from __future__ import annotations

import sqlglot


def transpile(sql: str, *, from_dialect: str, to_dialect: str) -> str:
    """
    Convert SQL between dialects. Returns the original SQL untouched if either
    dialect is unknown to sqlglot, or if transpilation produces nothing.
    """
    if from_dialect == to_dialect:
        return sql
    try:
        out = sqlglot.transpile(sql, read=from_dialect, write=to_dialect)
    except (sqlglot.errors.ParseError, ValueError):
        # ValueError fires when sqlglot doesn't know the dialect name.
        return sql
    return out[0] if out else sql
