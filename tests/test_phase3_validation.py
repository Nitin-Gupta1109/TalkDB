"""Phase 3 unit tests: schema validator, execution validator, intent, shape, confidence."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from talkdb.connectors.base import get_connector
from talkdb.core.intent import IntentType, classify_intent
from talkdb.schema.introspector import SchemaIntrospector
from talkdb.validation.confidence import calculate_confidence
from talkdb.validation.execution_validator import ExecutionValidator, _apply_limit
from talkdb.validation.schema_validator import SchemaValidator
from talkdb.validation.shape_validator import validate_shape


@pytest.fixture
def sqlite_db(tmp_path: Path) -> str:
    db_path = tmp_path / "test.db"
    conn_str = f"sqlite:///{db_path}"
    engine = create_engine(conn_str)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE customers (id INTEGER PRIMARY KEY, email TEXT, tier TEXT)"))
        conn.execute(text("CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL, status TEXT)"))
        conn.execute(text("INSERT INTO customers VALUES (1,'a@x.com','gold'),(2,'b@x.com','silver')"))
        conn.execute(text("INSERT INTO orders VALUES (1,1,100.0,'completed'),(2,2,50.0,'refunded')"))
    engine.dispose()
    return conn_str


@pytest.fixture
def schema(sqlite_db: str):
    conn = get_connector(sqlite_db)
    return SchemaIntrospector(conn).introspect(), conn


class TestSchemaValidator:
    def test_valid_sql_passes(self, schema):
        s, _ = schema
        r = SchemaValidator(s).validate("SELECT COUNT(*) FROM customers")
        assert r.valid
        assert r.tables_referenced == ["customers"]

    def test_unknown_table_flagged(self, schema):
        s, _ = schema
        r = SchemaValidator(s).validate("SELECT * FROM customerz")
        assert not r.valid
        assert r.issues[0].kind == "unknown_table"
        assert r.issues[0].suggestion == "customers"

    def test_unknown_column_flagged(self, schema):
        s, _ = schema
        r = SchemaValidator(s).validate("SELECT emial FROM customers")
        assert not r.valid
        assert any(i.kind == "unknown_column" for i in r.issues)
        # Suggestion should surface 'email'.
        assert any(i.suggestion and "email" in i.suggestion for i in r.issues)

    def test_qualified_column(self, schema):
        s, _ = schema
        r = SchemaValidator(s).validate("SELECT c.email FROM customers c WHERE c.tier='gold'")
        assert r.valid

    def test_cte_not_flagged_as_unknown_table(self, schema):
        s, _ = schema
        sql = "WITH recent AS (SELECT * FROM orders) SELECT * FROM recent"
        r = SchemaValidator(s).validate(sql)
        assert r.valid

    def test_parse_error(self, schema):
        s, _ = schema
        r = SchemaValidator(s).validate("SELECT FROM FROM")
        assert not r.valid

    def test_select_list_alias_allowed_in_order_by(self, schema):
        s, _ = schema
        r = SchemaValidator(s).validate(
            "SELECT tier AS t, COUNT(*) AS n FROM customers GROUP BY tier ORDER BY n DESC"
        )
        assert r.valid, f"unexpected issues: {r.issues}"


class TestExecutionValidator:
    def test_valid_query_executes(self, schema):
        _, conn = schema
        v = ExecutionValidator(conn)
        r = v.validate("SELECT COUNT(*) AS n FROM customers")
        assert r.ok
        assert r.rows[0]["n"] == 2

    def test_runtime_error_captured(self, schema):
        _, conn = schema
        v = ExecutionValidator(conn)
        r = v.validate("SELECT notacol FROM customers")
        assert not r.ok
        assert r.error is not None

    def test_apply_limit_adds_when_missing(self):
        out = _apply_limit("SELECT * FROM customers", 10, dialect="sqlite")
        assert "LIMIT 10" in out.upper()

    def test_apply_limit_respects_smaller_existing(self):
        out = _apply_limit("SELECT * FROM customers LIMIT 3", 10, dialect="sqlite")
        assert "LIMIT 3" in out.upper()
        assert "LIMIT 10" not in out.upper()


class TestIntent:
    def test_aggregation(self):
        i = classify_intent("How many customers are there?")
        assert i.type == IntentType.AGGREGATION
        assert i.is_single_value

    def test_ranking(self):
        i = classify_intent("Top 10 customers by revenue")
        assert i.type == IntentType.RANKING

    def test_distribution_beats_aggregation(self):
        i = classify_intent("Total revenue by region")
        assert i.type == IntentType.DISTRIBUTION
        assert not i.is_single_value

    def test_lookup(self):
        i = classify_intent("List all users")
        assert i.type == IntentType.LOOKUP


class TestShape:
    def test_aggregation_single_value_passes(self):
        intent = classify_intent("How many customers?")
        r = validate_shape(intent, columns=["n"], row_count=1)
        assert r.matches

    def test_aggregation_wrong_shape_warns(self):
        intent = classify_intent("How many customers?")
        r = validate_shape(intent, columns=["region", "n"], row_count=5)
        assert not r.matches
        assert r.warnings

    def test_distribution_with_one_row_warns(self):
        intent = classify_intent("Revenue by region")
        r = validate_shape(intent, columns=["region", "rev"], row_count=1)
        assert not r.matches


class TestConfidence:
    def test_refused_on_schema_failure(self, schema):
        s, conn = schema
        sv = SchemaValidator(s).validate("SELECT notacol FROM customers")
        ev = ExecutionValidator(conn).validate("SELECT 1")
        sh = validate_shape(classify_intent("anything"), ["x"], 1)
        score = calculate_confidence(sv, ev, sh)
        assert score.refused
        assert score.value == 0

    def test_passes_when_all_green(self, schema):
        s, conn = schema
        sv = SchemaValidator(s).validate("SELECT COUNT(*) FROM customers")
        ev = ExecutionValidator(conn).validate("SELECT COUNT(*) FROM customers")
        sh = validate_shape(classify_intent("how many customers?"), ["count(*)"], 1)
        score = calculate_confidence(sv, ev, sh, retrieval_similarity=0.9, semantic_coverage=0.6)
        assert not score.refused
        assert score.value >= 50
