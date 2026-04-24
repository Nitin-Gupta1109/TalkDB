"""
Schema validator: parse SQL, walk the AST, verify every referenced table and
column exists in the introspected schema. Suggests corrections for near-misses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches

import sqlglot
from sqlglot import expressions as exp

from talkdb.schema.models import DatabaseSchema


@dataclass
class SchemaIssue:
    kind: str  # "unknown_table" | "unknown_column" | "parse_error"
    identifier: str
    suggestion: str | None = None


@dataclass
class SchemaValidationResult:
    valid: bool
    issues: list[SchemaIssue] = field(default_factory=list)
    tables_referenced: list[str] = field(default_factory=list)
    columns_referenced: list[tuple[str | None, str]] = field(default_factory=list)

    def error_message(self) -> str:
        """Compose a human-readable message suitable for sending back to the LLM as retry feedback."""
        lines: list[str] = []
        for issue in self.issues:
            if issue.kind == "parse_error":
                lines.append(f"Parse error: {issue.identifier}")
            elif issue.kind == "unknown_table":
                suggestion = f" Did you mean '{issue.suggestion}'?" if issue.suggestion else ""
                lines.append(f"Unknown table: {issue.identifier}.{suggestion}")
            elif issue.kind == "unknown_column":
                suggestion = f" Did you mean '{issue.suggestion}'?" if issue.suggestion else ""
                lines.append(f"Unknown column: {issue.identifier}.{suggestion}")
        return "\n".join(lines)


class SchemaValidator:
    def __init__(self, schema: DatabaseSchema):
        self.schema = schema
        self._tables_by_name = {t.name.lower(): t for t in schema.tables}
        # Flat set of all column names (across tables) for fallback matching.
        self._all_column_names: set[str] = set()
        for t in schema.tables:
            for c in t.columns:
                self._all_column_names.add(c.name.lower())

    def validate(self, sql: str) -> SchemaValidationResult:
        try:
            parsed = sqlglot.parse(sql, read=self.schema.dialect)
        except sqlglot.errors.ParseError as e:
            return SchemaValidationResult(
                valid=False,
                issues=[SchemaIssue(kind="parse_error", identifier=str(e))],
            )

        tables_referenced: list[str] = []
        columns_referenced: list[tuple[str | None, str]] = []
        issues: list[SchemaIssue] = []
        alias_to_table: dict[str, str] = {}

        for statement in parsed:
            if statement is None:
                continue

            # Pass 1: collect every Table node (this also picks up CTE references,
            # which we filter below using `with_cte_names`).
            with_cte_names = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}

            # Collect SELECT list aliases so ORDER BY / GROUP BY / HAVING references to
            # `SELECT x AS foo ... ORDER BY foo` don't get flagged as unknown columns.
            select_aliases: set[str] = set()
            for alias_node in statement.find_all(exp.Alias):
                alias_name = alias_node.alias
                if alias_name:
                    select_aliases.add(alias_name.lower())

            for table_node in statement.find_all(exp.Table):
                name = table_node.name
                if not name:
                    continue
                alias = table_node.alias_or_name
                if name.lower() in with_cte_names:
                    # Treat CTE references as implicit — they're defined in the statement itself.
                    if alias:
                        alias_to_table[alias.lower()] = name.lower()
                    continue
                tables_referenced.append(name)
                if name.lower() not in self._tables_by_name:
                    suggestion = _closest_match(name, list(self._tables_by_name))
                    issues.append(SchemaIssue(kind="unknown_table", identifier=name, suggestion=suggestion))
                if alias:
                    alias_to_table[alias.lower()] = name.lower()

            # Pass 2: check column references. A Column node may be qualified (t.col) or bare (col).
            for col_node in statement.find_all(exp.Column):
                col_name = col_node.name
                if not col_name or col_name == "*":
                    continue
                table_ref = col_node.table or None

                # Resolve alias -> real table name.
                resolved_table: str | None = None
                if table_ref:
                    resolved_table = alias_to_table.get(table_ref.lower(), table_ref.lower())
                    if resolved_table in with_cte_names:
                        # Columns of CTEs aren't introspected from the physical schema.
                        continue

                columns_referenced.append((resolved_table, col_name))

                if resolved_table:
                    table_info = self._tables_by_name.get(resolved_table)
                    if table_info is None:
                        continue  # Already flagged as unknown_table.
                    if not any(c.name.lower() == col_name.lower() for c in table_info.columns):
                        suggestion = _closest_match(
                            col_name, [c.name for c in table_info.columns]
                        )
                        issues.append(
                            SchemaIssue(
                                kind="unknown_column",
                                identifier=f"{resolved_table}.{col_name}",
                                suggestion=f"{resolved_table}.{suggestion}" if suggestion else None,
                            )
                        )
                else:
                    # Bare column reference — accept if it exists anywhere in the schema,
                    # OR if it matches a SELECT-list alias in the same statement.
                    if (
                        col_name.lower() not in self._all_column_names
                        and col_name.lower() not in select_aliases
                    ):
                        suggestion = _closest_match(col_name, list(self._all_column_names))
                        issues.append(
                            SchemaIssue(
                                kind="unknown_column",
                                identifier=col_name,
                                suggestion=suggestion,
                            )
                        )

        return SchemaValidationResult(
            valid=not issues,
            issues=issues,
            tables_referenced=tables_referenced,
            columns_referenced=columns_referenced,
        )


def _closest_match(name: str, candidates: list[str], cutoff: float = 0.7) -> str | None:
    if not candidates:
        return None
    matches = get_close_matches(name.lower(), [c.lower() for c in candidates], n=1, cutoff=cutoff)
    if not matches:
        return None
    # Return the candidate with the matching lowercase form, preserving original case.
    for c in candidates:
        if c.lower() == matches[0]:
            return c
    return matches[0]
