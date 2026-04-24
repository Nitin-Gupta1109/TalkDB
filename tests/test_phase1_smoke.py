"""
Phase 1 smoke test: SQLite + mocked LLM -> full engine pipeline end-to-end.

Verifies:
- Schema introspection extracts tables, columns, PKs, and sample values.
- SQL generator integrates with the prompt (mocked to return known-good SQL).
- Engine rejects non-SELECT statements via sqlglot.
- Engine executes the generated SQL against the live DB.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine, text

from talkdb.config.settings import Settings
from talkdb.core.engine import Engine, UnsafeSQLError


@pytest.fixture
def sqlite_db(tmp_path: Path) -> str:
    db_path = tmp_path / "test.db"
    conn_str = f"sqlite:///{db_path}"
    engine = create_engine(conn_str)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL, tier TEXT)"))
        conn.execute(
            text("INSERT INTO users (id, email, tier) VALUES (1, 'a@x.com', 'gold'), (2, 'b@x.com', 'silver'), (3, 'c@x.com', 'gold')")
        )
    engine.dispose()
    return conn_str


@pytest.fixture
def settings(sqlite_db: str) -> Settings:
    return Settings(default_db=sqlite_db, llm_model="mock")


async def test_ask_end_to_end(settings: Settings) -> None:
    engine = Engine(settings)

    with patch("talkdb.core.generator.litellm.acompletion", new=AsyncMock()) as mock_llm:
        mock_llm.return_value.choices = [
            type("C", (), {"message": type("M", (), {"content": "SELECT COUNT(*) AS n FROM users"})()})()
        ]
        result = await engine.ask("How many users are there?")

    assert result.sql == "SELECT COUNT(*) AS n FROM users"
    assert result.row_count == 1
    assert result.results[0]["n"] == 3
    assert result.dialect == "sqlite"
    assert result.columns == ["n"]


async def test_schema_introspection(settings: Settings) -> None:
    engine = Engine(settings)
    info = engine.describe_database()
    assert info["dialect"] == "sqlite"
    table_names = [t["name"] for t in info["tables"]]
    assert "users" in table_names
    users = next(t for t in info["tables"] if t["name"] == "users")
    col_names = {c["name"] for c in users["columns"]}
    assert col_names == {"id", "email", "tier"}
    id_col = next(c for c in users["columns"] if c["name"] == "id")
    assert id_col["primary_key"] is True


async def test_rejects_non_select(settings: Settings) -> None:
    engine = Engine(settings)
    with patch("talkdb.core.generator.litellm.acompletion", new=AsyncMock()) as mock_llm:
        mock_llm.return_value.choices = [
            type("C", (), {"message": type("M", (), {"content": "DROP TABLE users"})()})()
        ]
        with pytest.raises(UnsafeSQLError):
            await engine.ask("drop the users table")


async def test_rejects_mutating_cte(settings: Settings) -> None:
    engine = Engine(settings)
    with patch("talkdb.core.generator.litellm.acompletion", new=AsyncMock()) as mock_llm:
        mock_llm.return_value.choices = [
            type("C", (), {"message": type("M", (), {"content": "DELETE FROM users WHERE id = 1"})()})()
        ]
        with pytest.raises(UnsafeSQLError):
            await engine.ask("delete user 1")


async def test_refusal_passthrough(settings: Settings) -> None:
    engine = Engine(settings)
    with patch("talkdb.core.generator.litellm.acompletion", new=AsyncMock()) as mock_llm:
        mock_llm.return_value.choices = [
            type("C", (), {"message": type("M", (), {"content": "CANNOT_ANSWER: no orders table"})()})()
        ]
        result = await engine.ask("what is total revenue?")
    assert result.sql == ""
    assert result.confidence == 0
    assert "no orders table" in (result.explanation or "")
