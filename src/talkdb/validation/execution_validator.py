"""
Execution validator: run the generated SQL with a safety LIMIT and timeout.
Catches runtime errors cheaply before the engine commits to a full execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlalchemy.exc import SQLAlchemyError
from sqlglot import expressions as exp

from talkdb.connectors.base import BaseConnector


@dataclass
class ExecutionResult:
    ok: bool
    columns: list[str]
    rows: list[dict[str, Any]]
    error: str | None = None
    sql_executed: str = ""


class ExecutionValidator:
    """
    Runs SQL inside a read-only transaction with LIMIT applied.
    The goal is to catch runtime errors (bad function names, type mismatches,
    timeouts) without fetching a large result set.
    """

    def __init__(self, connector: BaseConnector, sample_limit: int = 10, timeout_seconds: int = 10):
        self.connector = connector
        self.sample_limit = sample_limit
        self.timeout_seconds = timeout_seconds

    def validate(self, sql: str) -> ExecutionResult:
        wrapped = _apply_limit(sql, self.sample_limit, dialect=self.connector.dialect)
        try:
            columns, rows = self.connector.execute(
                wrapped,
                read_only=True,
                timeout_seconds=self.timeout_seconds,
            )
        except SQLAlchemyError as e:
            return ExecutionResult(ok=False, columns=[], rows=[], error=str(e.__cause__ or e), sql_executed=wrapped)
        except Exception as e:  # noqa: BLE001
            return ExecutionResult(ok=False, columns=[], rows=[], error=str(e), sql_executed=wrapped)
        return ExecutionResult(ok=True, columns=columns, rows=rows, sql_executed=wrapped)


def _apply_limit(sql: str, limit: int, dialect: str) -> str:
    """
    Apply (or tighten) a LIMIT clause. If the SQL already has a LIMIT smaller than
    `limit`, it is left alone. Falls back to naive suffix append on parse failure.
    """
    try:
        parsed = sqlglot.parse_one(sql, read=dialect)
    except sqlglot.errors.ParseError:
        return f"{sql.rstrip().rstrip(';')} LIMIT {limit}"

    if not isinstance(parsed, exp.Select):
        return sql

    existing = parsed.args.get("limit")
    if existing is not None:
        try:
            existing_val = int(existing.expression.this)  # type: ignore[union-attr]
            if existing_val <= limit:
                return parsed.sql(dialect=dialect)
        except (AttributeError, ValueError, TypeError):
            pass

    limited = parsed.limit(limit)
    return limited.sql(dialect=dialect)
