"""Phase 4 unit tests: session store, follow-up detection, rewriter mocking, multi-turn flow."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine, text

from talkdb.config.settings import Settings
from talkdb.conversation.resolver import ReferenceResolver
from talkdb.conversation.session import (
    ConversationTurn,
    InMemorySessionStore,
    Session,
    summarize_result,
)
from talkdb.core.engine import Engine


@pytest.fixture
def sqlite_db(tmp_path: Path) -> str:
    db_path = tmp_path / "conv.db"
    conn_str = f"sqlite:///{db_path}"
    eng = create_engine(conn_str)
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE orders (id INTEGER PRIMARY KEY, amount REAL, status TEXT, region TEXT, created_at TEXT)"))
        conn.execute(text(
            "INSERT INTO orders (id, amount, status, region, created_at) VALUES "
            "(1, 100.0, 'completed', 'NE', '2026-01-15'),"
            "(2, 200.0, 'completed', 'W', '2026-02-20'),"
            "(3, 50.0, 'refunded', 'NE', '2026-03-10'),"
            "(4, 300.0, 'completed', 'W', '2026-04-05')"
        ))
    eng.dispose()
    return conn_str


@pytest.fixture
def settings(sqlite_db: str) -> Settings:
    return Settings(default_db=sqlite_db, llm_model="mock", confidence_threshold=0)


class TestSessionStore:
    def test_get_or_create_new(self):
        store = InMemorySessionStore()
        s = store.get_or_create(None, database="db1", ttl_minutes=60)
        assert s.session_id.startswith("sess_")
        assert s.database == "db1"
        assert s.turns == []

    def test_get_or_create_reuses_existing(self):
        store = InMemorySessionStore()
        s1 = store.get_or_create(None, database="db1", ttl_minutes=60)
        s2 = store.get_or_create(s1.session_id, database="db1", ttl_minutes=60)
        assert s1 is s2

    def test_expiry(self):
        store = InMemorySessionStore()
        s = store.get_or_create(None, database="db1", ttl_minutes=60)
        s.last_active = datetime.utcnow() - timedelta(minutes=120)
        assert store.get(s.session_id) is None  # Deleted on access.

    def test_add_turn(self):
        s = Session(session_id="x", database="db1")
        s.add_turn(ConversationTurn(turn_number=1, question="q", rewritten_question="q", sql="sql", results_summary=""))
        assert len(s.turns) == 1
        assert s.next_turn_number() == 2


class TestResolver:
    def setup_method(self):
        self.r = ReferenceResolver()
        self.session = Session(session_id="x", database="db")
        self.session.add_turn(ConversationTurn(turn_number=1, question="revenue?", rewritten_question="revenue?", sql="...", results_summary=""))

    def test_no_session_is_not_followup(self):
        r = self.r.resolve("anything", session=None)
        assert not r.is_follow_up

    def test_pronoun_triggers_followup(self):
        r = self.r.resolve("break that down by region", self.session)
        assert r.is_follow_up
        assert "that" in (r.trigger or "")

    def test_leading_connector_triggers(self):
        r = self.r.resolve("just Q4", self.session)
        assert r.is_follow_up

    def test_long_standalone_not_flagged(self):
        r = self.r.resolve(
            "What is the average order value across all completed orders in the last calendar quarter?",
            self.session,
        )
        assert not r.is_follow_up

    def test_short_question_treated_as_followup(self):
        r = self.r.resolve("by region", self.session)
        assert r.is_follow_up


class TestSummarize:
    def test_empty(self):
        assert "0" in summarize_result([], [])

    def test_single_value(self):
        s = summarize_result(["revenue"], [{"revenue": 100.0}])
        assert "revenue=100.0" in s


class TestEngineSessionIntegration:
    """
    Integration-level: follow_up() resolves an unknown session id to a refusal.
    Full multi-turn LLM chaining is covered by the MCP stdio script in scripts/test_mcp_phase4.py.
    """

    async def test_follow_up_unknown_session_refuses(self, settings: Settings) -> None:
        engine = Engine(settings)
        result = await engine.follow_up("break that down", session_id="sess_does_not_exist")
        assert result.sql == ""
        assert result.confidence == 0
        assert "not found" in (result.explanation or "").lower() or "expired" in (result.explanation or "").lower()

    async def test_get_session_returns_none_for_unknown(self, settings: Settings) -> None:
        engine = Engine(settings)
        assert engine.get_session("sess_does_not_exist") is None
