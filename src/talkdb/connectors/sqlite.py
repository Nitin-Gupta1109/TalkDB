from talkdb.connectors.base import BaseConnector


class SQLiteConnector(BaseConnector):
    dialect = "sqlite"

    def quote_identifier(self, name: str) -> str:
        escaped = name.replace('"', '""')
        return f'"{escaped}"'
