"""
Conversation session state and storage.

A Session groups a sequence of ConversationTurns under a single session_id.
The in-memory store is the default (fine for single-user MCP). A Redis or
SQLite-backed store can plug in behind the SessionStore ABC without engine changes.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


@dataclass
class ConversationTurn:
    turn_number: int
    question: str                    # Original user question (possibly a follow-up fragment)
    rewritten_question: str          # Standalone question fed to the SQL generator
    sql: str
    results_summary: str             # Short natural-language summary of rows for the next turn's rewriter
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    sample_rows: list[dict[str, Any]] = field(default_factory=list)  # First 3 rows, for context
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Session:
    session_id: str
    database: str | None
    turns: list[ConversationTurn] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)
    ttl_minutes: int = 60

    def add_turn(self, turn: ConversationTurn) -> None:
        self.turns.append(turn)
        self.last_active = datetime.utcnow()

    def has_turns(self) -> bool:
        return len(self.turns) > 0

    def is_expired(self) -> bool:
        return datetime.utcnow() - self.last_active > timedelta(minutes=self.ttl_minutes)

    def recent_turns(self, n: int = 3) -> list[ConversationTurn]:
        return self.turns[-n:]

    def next_turn_number(self) -> int:
        return len(self.turns) + 1


class SessionStore(ABC):
    @abstractmethod
    def get(self, session_id: str) -> Session | None:
        ...

    @abstractmethod
    def save(self, session: Session) -> None:
        ...

    @abstractmethod
    def delete(self, session_id: str) -> None:
        ...

    @abstractmethod
    def get_or_create(self, session_id: str | None, database: str | None, ttl_minutes: int) -> Session:
        ...


class InMemorySessionStore(SessionStore):
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def get(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session and session.is_expired():
            self.delete(session_id)
            return None
        return session

    def save(self, session: Session) -> None:
        self._sessions[session.session_id] = session

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def get_or_create(self, session_id: str | None, database: str | None, ttl_minutes: int) -> Session:
        if session_id:
            existing = self.get(session_id)
            if existing is not None:
                return existing
        new_id = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        session = Session(session_id=new_id, database=database, ttl_minutes=ttl_minutes)
        self.save(session)
        return session


def summarize_result(columns: list[str], rows: list[dict[str, Any]]) -> str:
    """Short natural-language summary of a result set, for injection into the next turn's rewriter prompt."""
    n = len(rows)
    if n == 0:
        return "Returned 0 rows."
    col_list = ", ".join(columns)
    if n == 1 and len(columns) <= 2:
        # Single-value result — include the value so the next turn can reason about it.
        row = rows[0]
        parts = [f"{k}={v}" for k, v in row.items()]
        return f"Returned a single row with {', '.join(parts)}."
    return f"Returned {n} row(s) with columns [{col_list}]."
