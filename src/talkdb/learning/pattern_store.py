"""
Proven-pattern store. SQLite-backed. Grows as users submit corrections via `correct_query`
or when the watchdog validates a query on a live schedule (Phase 7).

Separate from the YAML semantic model (which is static, versioned, and human-authored).
Patterns from here are loaded into the retriever alongside semantic examples so the LLM
sees what actually worked in the past.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Session

PatternSource = Literal["user_correction", "user_approval", "community", "semantic_model"]


class _Base(DeclarativeBase):
    pass


class _PatternRow(_Base):
    __tablename__ = "patterns"

    id = Column(Integer, primary_key=True)
    question = Column(String(2000), nullable=False)
    sql = Column(Text, nullable=False)
    database = Column(String(200), nullable=True)
    source = Column(String(50), nullable=False)
    score = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, nullable=False, default=func.now())


@dataclass
class ProvenPattern:
    id: int
    question: str
    sql: str
    database: str | None
    source: PatternSource
    score: int
    created_at: datetime


class PatternStore:
    """Thin SQLAlchemy-backed pattern store. One row per (question, sql) pair."""

    def __init__(self, path: str = "./data/patterns.sqlite"):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(f"sqlite:///{p}", future=True)
        _Base.metadata.create_all(self._engine)

    def add(
        self,
        question: str,
        sql: str,
        *,
        database: str | None = None,
        source: PatternSource = "user_correction",
        score: int = 1,
    ) -> ProvenPattern:
        with Session(self._engine) as session:
            row = _PatternRow(
                question=question.strip(),
                sql=sql.strip(),
                database=database,
                source=source,
                score=score,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return _to_dc(row)

    def list(self, database: str | None = None, limit: int = 500) -> list[ProvenPattern]:
        with Session(self._engine) as session:
            query = session.query(_PatternRow)
            if database is not None:
                query = query.filter(
                    (_PatternRow.database == database) | (_PatternRow.database.is_(None))
                )
            rows = query.order_by(_PatternRow.score.desc(), _PatternRow.id.desc()).limit(limit).all()
            return [_to_dc(r) for r in rows]

    def bump_score(self, pattern_id: int, delta: int = 1) -> None:
        with Session(self._engine) as session:
            row = session.get(_PatternRow, pattern_id)
            if row is not None:
                row.score = (row.score or 0) + delta
                session.commit()

    def count(self) -> int:
        with Session(self._engine) as session:
            return session.query(_PatternRow).count()


def _to_dc(row: _PatternRow) -> ProvenPattern:
    return ProvenPattern(
        id=row.id,
        question=row.question,
        sql=row.sql,
        database=row.database,
        source=row.source,  # type: ignore[arg-type]
        score=row.score,
        created_at=row.created_at,
    )
