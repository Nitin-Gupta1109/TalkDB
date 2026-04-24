from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL, make_url


class BaseConnector(ABC):
    """Abstract SQL database connector. Subclasses set the dialect label."""

    dialect: str = "generic"

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(self.connection_string, future=True)
        return self._engine

    def execute(
        self,
        sql: str,
        *,
        read_only: bool = True,
        timeout_seconds: int | None = None,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """
        Execute a SQL statement. Returns (column_names, rows).

        read_only=True wraps the execution in a transaction that is always rolled back,
        so even if the statement somehow mutates state it is discarded.
        """
        with self.engine.connect() as conn:
            if timeout_seconds is not None:
                self._apply_timeout(conn, timeout_seconds)
            if read_only:
                trans = conn.begin()
                try:
                    result = conn.execute(text(sql))
                    columns = list(result.keys())
                    rows = [dict(row._mapping) for row in result]
                finally:
                    trans.rollback()
            else:
                result = conn.execute(text(sql))
                columns = list(result.keys())
                rows = [dict(row._mapping) for row in result]
                conn.commit()
        return columns, rows

    def _apply_timeout(self, conn, timeout_seconds: int) -> None:
        """Hook for dialect-specific statement timeouts. Default no-op."""
        return None

    @abstractmethod
    def quote_identifier(self, name: str) -> str:
        ...

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None


def get_connector(connection_string: str) -> BaseConnector:
    """Factory: pick the connector class based on the connection string's dialect."""
    from talkdb.connectors.postgres import PostgresConnector
    from talkdb.connectors.sqlite import SQLiteConnector

    url: URL = make_url(connection_string)
    backend = url.get_backend_name()
    if backend == "sqlite":
        return SQLiteConnector(connection_string)
    if backend in ("postgresql", "postgres"):
        return PostgresConnector(connection_string)
    raise ValueError(f"Unsupported database backend: {backend}")
