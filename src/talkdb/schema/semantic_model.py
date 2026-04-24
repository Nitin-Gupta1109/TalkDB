"""YAML-based semantic model: business metric definitions, column descriptions, joins, and examples."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class SemanticColumn(BaseModel):
    name: str
    description: str | None = None
    aliases: list[str] = Field(default_factory=list)
    valid_values: list[str] = Field(default_factory=list)


class SemanticTable(BaseModel):
    name: str
    description: str | None = None
    columns: list[SemanticColumn] = Field(default_factory=list)

    def column(self, name: str) -> SemanticColumn | None:
        for c in self.columns:
            if c.name == name:
                return c
        return None


class SemanticMetric(BaseModel):
    name: str
    description: str
    calculation: str
    table: str | None = None
    tables: list[str] = Field(default_factory=list)
    join: str | None = None


class SemanticJoin(BaseModel):
    left: str
    right: str
    on: str
    type: str = "INNER JOIN"
    description: str | None = None


class SemanticExample(BaseModel):
    question: str
    sql: str


class InsightHint(BaseModel):
    metric: str
    normal_range: str | None = None
    seasonality: str | None = None
    alert_threshold: str | None = None
    trend: str | None = None


class SemanticModel(BaseModel):
    """Parsed semantic model YAML — metrics, tables, joins, examples, insight hints."""

    version: str = "1.0"
    database: str | None = None
    metrics: list[SemanticMetric] = Field(default_factory=list)
    tables: list[SemanticTable] = Field(default_factory=list)
    joins: list[SemanticJoin] = Field(default_factory=list)
    examples: list[SemanticExample] = Field(default_factory=list)
    insight_hints: list[InsightHint] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> SemanticModel:
        p = Path(path)
        with p.open() as f:
            data = yaml.safe_load(f) or {}
        # YAML 1.1 treats 'on' as True. Fix up join dicts so users can write `on:` naturally.
        for join in data.get("joins") or []:
            if True in join and "on" not in join:
                join["on"] = join.pop(True)
        return cls.model_validate(data)

    @classmethod
    def load_directory(cls, directory: str | Path) -> list[SemanticModel]:
        """Load every *.yaml / *.yml file in a directory."""
        d = Path(directory)
        if not d.exists():
            return []
        models: list[SemanticModel] = []
        for path in sorted([*d.glob("*.yaml"), *d.glob("*.yml")]):
            models.append(cls.load(path))
        return models

    def metric(self, name: str) -> SemanticMetric | None:
        for m in self.metrics:
            if m.name == name:
                return m
        return None

    def table(self, name: str) -> SemanticTable | None:
        for t in self.tables:
            if t.name == name:
                return t
        return None
