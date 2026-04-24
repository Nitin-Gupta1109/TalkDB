from pydantic import BaseModel, Field


class ColumnInfo(BaseModel):
    name: str
    data_type: str
    is_nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_references: str | None = None
    description: str | None = None
    sample_values: list[str] = Field(default_factory=list)


class TableInfo(BaseModel):
    name: str
    schema_name: str | None = None
    columns: list[ColumnInfo] = Field(default_factory=list)
    row_count: int | None = None
    description: str | None = None

    def column(self, name: str) -> ColumnInfo | None:
        for col in self.columns:
            if col.name == name:
                return col
        return None


class ForeignKeyInfo(BaseModel):
    from_table: str
    from_columns: list[str]
    to_table: str
    to_columns: list[str]


class DatabaseSchema(BaseModel):
    tables: list[TableInfo] = Field(default_factory=list)
    foreign_keys: list[ForeignKeyInfo] = Field(default_factory=list)
    dialect: str

    def table(self, name: str) -> TableInfo | None:
        for t in self.tables:
            if t.name == name:
                return t
        return None

    def to_prompt_text(self) -> str:
        """Render the schema as compact text for inclusion in an LLM prompt."""
        lines: list[str] = []
        for t in self.tables:
            header = f"TABLE {t.name}"
            if t.description:
                header += f" -- {t.description}"
            lines.append(header)
            for col in t.columns:
                marks: list[str] = []
                if col.is_primary_key:
                    marks.append("PK")
                if col.is_foreign_key and col.foreign_key_references:
                    marks.append(f"FK->{col.foreign_key_references}")
                if not col.is_nullable:
                    marks.append("NOT NULL")
                mark_str = f" [{', '.join(marks)}]" if marks else ""
                sample = f" e.g. {col.sample_values}" if col.sample_values else ""
                desc = f" -- {col.description}" if col.description else ""
                lines.append(f"  {col.name} {col.data_type}{mark_str}{sample}{desc}")
            lines.append("")
        if self.foreign_keys:
            lines.append("FOREIGN KEYS:")
            for fk in self.foreign_keys:
                lines.append(
                    f"  {fk.from_table}({', '.join(fk.from_columns)}) -> "
                    f"{fk.to_table}({', '.join(fk.to_columns)})"
                )
        return "\n".join(lines)


class Metric(BaseModel):
    name: str
    description: str
    calculation: str
    table: str | None = None
    tables: list[str] = Field(default_factory=list)
    join: str | None = None


class QueryResult(BaseModel):
    sql: str
    results: list[dict]
    row_count: int
    columns: list[str]
    dialect: str
    confidence: int = 100
    explanation: str | None = None
    warnings: list[str] = Field(default_factory=list)
    insight: str | None = None
    key_findings: list[str] = Field(default_factory=list)
    chart: dict | None = None        # {"type": ..., "image_base64": ..., "title": ...}
    chart_skipped_reason: str | None = None
