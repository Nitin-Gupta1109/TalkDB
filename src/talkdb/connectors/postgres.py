from sqlalchemy import text

from talkdb.connectors.base import BaseConnector


class PostgresConnector(BaseConnector):
    dialect = "postgresql"

    def _apply_timeout(self, conn, timeout_seconds: int) -> None:
        conn.execute(text(f"SET LOCAL statement_timeout = {timeout_seconds * 1000}"))

    def quote_identifier(self, name: str) -> str:
        escaped = name.replace('"', '""')
        return f'"{escaped}"'
