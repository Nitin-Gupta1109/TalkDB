from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from talkdb.connectors.base import BaseConnector
from talkdb.schema.models import ColumnInfo, DatabaseSchema, ForeignKeyInfo, TableInfo


class SchemaIntrospector:
    """Extract tables, columns, foreign keys, and sample values from a live database."""

    def __init__(self, connector: BaseConnector, sample_size: int = 5):
        self.connector = connector
        self.sample_size = sample_size

    def introspect(self, include_samples: bool = True) -> DatabaseSchema:
        inspector = inspect(self.connector.engine)

        tables: list[TableInfo] = []
        foreign_keys: list[ForeignKeyInfo] = []

        for table_name in inspector.get_table_names():
            pk_cols = set(inspector.get_pk_constraint(table_name).get("constrained_columns") or [])
            fk_data = inspector.get_foreign_keys(table_name)
            fk_by_col: dict[str, str] = {}
            for fk in fk_data:
                to_table = fk.get("referred_table")
                to_cols = fk.get("referred_columns") or []
                from_cols = fk.get("constrained_columns") or []
                if to_table and from_cols and to_cols:
                    foreign_keys.append(
                        ForeignKeyInfo(
                            from_table=table_name,
                            from_columns=list(from_cols),
                            to_table=to_table,
                            to_columns=list(to_cols),
                        )
                    )
                    for src, dst in zip(from_cols, to_cols, strict=False):
                        fk_by_col[src] = f"{to_table}.{dst}"

            columns: list[ColumnInfo] = []
            for col in inspector.get_columns(table_name):
                name = col["name"]
                samples = self._sample_values(table_name, name) if include_samples else []
                columns.append(
                    ColumnInfo(
                        name=name,
                        data_type=str(col.get("type", "")),
                        is_nullable=bool(col.get("nullable", True)),
                        is_primary_key=name in pk_cols,
                        is_foreign_key=name in fk_by_col,
                        foreign_key_references=fk_by_col.get(name),
                        description=col.get("comment"),
                        sample_values=samples,
                    )
                )

            tables.append(
                TableInfo(
                    name=table_name,
                    columns=columns,
                    row_count=self._approximate_row_count(table_name),
                )
            )

        return DatabaseSchema(
            tables=tables,
            foreign_keys=foreign_keys,
            dialect=self.connector.dialect,
        )

    def _sample_values(self, table: str, column: str) -> list[str]:
        quoted_table = self.connector.quote_identifier(table)
        quoted_column = self.connector.quote_identifier(column)
        sql = (
            f"SELECT DISTINCT {quoted_column} FROM {quoted_table} "
            f"WHERE {quoted_column} IS NOT NULL LIMIT {self.sample_size}"
        )
        try:
            _, rows = self.connector.execute(sql, read_only=True)
        except SQLAlchemyError:
            return []
        return [str(list(row.values())[0]) for row in rows]

    def _approximate_row_count(self, table: str) -> int | None:
        quoted = self.connector.quote_identifier(table)
        try:
            with self.connector.engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {quoted}"))
                return int(result.scalar() or 0)
        except SQLAlchemyError:
            return None
