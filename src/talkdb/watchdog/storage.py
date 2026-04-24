"""
SQLite-backed storage for watch definitions and historical values.

Per CLAUDE.md: watchdog stores historical values in SQLite (a local file),
NOT in the target database. This keeps the watchdog decoupled from the user's
data and works even across DB restarts.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Session

from talkdb.watchdog.watch import AlertCondition, Watch


class _Base(DeclarativeBase):
    pass


class _WatchRow(_Base):
    __tablename__ = "watches"

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False, unique=True)
    question = Column(String(2000), nullable=False)
    sql = Column(Text, nullable=False)
    database = Column(String(200), nullable=True)
    schedule = Column(String(200), nullable=False)
    alert_condition_json = Column(Text, nullable=False)
    delivery_channels = Column(String(200), nullable=False, default="stdout")
    webhook_url = Column(String(1000), nullable=True)
    slack_webhook_url = Column(String(1000), nullable=True)
    created_at = Column(DateTime, nullable=False, default=func.now())
    is_active = Column(Integer, nullable=False, default=1)
    last_run = Column(DateTime, nullable=True)
    last_value = Column(Float, nullable=True)
    last_status = Column(String(50), nullable=True)
    last_message = Column(String(2000), nullable=True)


class _HistoryRow(_Base):
    __tablename__ = "watch_history"

    id = Column(Integer, primary_key=True)
    watch_id = Column(Integer, ForeignKey("watches.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=func.now())
    value = Column(Float, nullable=False)


class WatchdogStorage:
    """Thin persistence layer for watches and their run history."""

    def __init__(self, path: str = "./data/watchdog.sqlite"):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(f"sqlite:///{p}", future=True)
        _Base.metadata.create_all(self._engine)

    def upsert(self, watch: Watch) -> Watch:
        with Session(self._engine) as session:
            row = session.query(_WatchRow).filter_by(name=watch.name).one_or_none()
            condition_json = json.dumps(asdict(watch.alert_condition))
            if row is None:
                row = _WatchRow(name=watch.name)
                session.add(row)
            row.question = watch.question
            row.sql = watch.sql
            row.database = watch.database
            row.schedule = watch.schedule
            row.alert_condition_json = condition_json
            row.delivery_channels = ",".join(watch.delivery_channels)
            row.webhook_url = watch.webhook_url
            row.slack_webhook_url = watch.slack_webhook_url
            row.is_active = 1 if watch.is_active else 0
            session.commit()
            session.refresh(row)
            watch.created_at = row.created_at
            return watch

    def mark_run(
        self,
        name: str,
        *,
        ran_at: datetime,
        value: float | None,
        status: str,
        message: str,
    ) -> None:
        with Session(self._engine) as session:
            row = session.query(_WatchRow).filter_by(name=name).one_or_none()
            if row is None:
                return
            row.last_run = ran_at
            row.last_value = value
            row.last_status = status
            row.last_message = message[:2000]
            session.commit()

    def record_history(self, name: str, value: float, *, timestamp: datetime | None = None) -> None:
        with Session(self._engine) as session:
            w = session.query(_WatchRow).filter_by(name=name).one_or_none()
            if w is None:
                return
            row = _HistoryRow(
                watch_id=w.id,
                timestamp=timestamp or datetime.utcnow(),
                value=value,
            )
            session.add(row)
            session.commit()

    def history(self, name: str, limit: int = 500) -> list[tuple[datetime, float]]:
        with Session(self._engine) as session:
            w = session.query(_WatchRow).filter_by(name=name).one_or_none()
            if w is None:
                return []
            rows = (
                session.query(_HistoryRow)
                .filter_by(watch_id=w.id)
                .order_by(_HistoryRow.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [(r.timestamp, r.value) for r in rows]

    def all(self, active_only: bool = False) -> list[Watch]:
        with Session(self._engine) as session:
            query = session.query(_WatchRow)
            if active_only:
                query = query.filter_by(is_active=1)
            return [_row_to_watch(r) for r in query.order_by(_WatchRow.name).all()]

    def get(self, name: str) -> Watch | None:
        with Session(self._engine) as session:
            row = session.query(_WatchRow).filter_by(name=name).one_or_none()
            return _row_to_watch(row) if row else None

    def delete(self, name: str) -> bool:
        with Session(self._engine) as session:
            row = session.query(_WatchRow).filter_by(name=name).one_or_none()
            if row is None:
                return False
            # Cascade: clear history first to avoid FK lock issues on SQLite.
            session.query(_HistoryRow).filter_by(watch_id=row.id).delete()
            session.delete(row)
            session.commit()
            return True


def _row_to_watch(row: _WatchRow) -> Watch:
    cond_data: dict[str, Any] = json.loads(row.alert_condition_json)
    condition = AlertCondition(**cond_data)
    channels = row.delivery_channels.split(",") if row.delivery_channels else ["stdout"]
    return Watch(
        name=row.name,
        question=row.question,
        sql=row.sql,
        database=row.database,
        schedule=row.schedule,
        alert_condition=condition,
        webhook_url=row.webhook_url,
        delivery_channels=channels,
        slack_webhook_url=row.slack_webhook_url,
        created_at=row.created_at,
        is_active=bool(row.is_active),
        last_run=row.last_run,
        last_value=row.last_value,
        last_status=row.last_status,
        last_message=row.last_message,
    )
