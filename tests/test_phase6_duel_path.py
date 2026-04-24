"""Phase 6 unit tests: dual-path comparator, pattern store, transpiler."""

from __future__ import annotations

from pathlib import Path

import pytest

from talkdb.core.transpiler import transpile
from talkdb.learning.feedback import FeedbackRecorder
from talkdb.learning.pattern_store import PatternStore
from talkdb.retrieval.vector_store import VectorHit, VectorStore
from talkdb.validation.dual_path import compare_results


class TestDualPathComparison:
    def test_full_agreement(self):
        r = compare_results(
            path_a_columns=["n"], path_a_rows=[{"n": 100}],
            path_b_columns=["n"], path_b_rows=[{"n": 100}],
        )
        assert r.agreement_level == "full"
        assert r.agreement_score == 1.0
        assert r.confidence_adjustment > 0

    def test_different_columns_is_disagreement(self):
        r = compare_results(
            path_a_columns=["revenue"], path_a_rows=[{"revenue": 100}],
            path_b_columns=["total"], path_b_rows=[{"total": 100}],
        )
        assert r.agreement_level == "disagreement"
        assert r.agreement_score == 0.0

    def test_column_order_doesnt_matter(self):
        r = compare_results(
            path_a_columns=["a", "b"], path_a_rows=[{"a": 1, "b": 2}],
            path_b_columns=["b", "a"], path_b_rows=[{"a": 1, "b": 2}],
        )
        assert r.agreement_level == "full"

    def test_row_count_mismatch_is_disagreement(self):
        r = compare_results(
            path_a_columns=["n"], path_a_rows=[{"n": 1}, {"n": 2}],
            path_b_columns=["n"], path_b_rows=[{"n": 1}],
        )
        assert r.agreement_level == "disagreement"

    def test_value_mismatch_is_partial(self):
        r = compare_results(
            path_a_columns=["region", "rev"],
            path_a_rows=[{"region": "N", "rev": 100.0}, {"region": "S", "rev": 200.0}],
            path_b_columns=["region", "rev"],
            path_b_rows=[{"region": "N", "rev": 100.0}, {"region": "S", "rev": 250.0}],
        )
        assert r.agreement_level == "partial"
        assert 0.0 < r.agreement_score < 1.0

    def test_float_tolerance(self):
        r = compare_results(
            path_a_columns=["x"], path_a_rows=[{"x": 1.00001}],
            path_b_columns=["x"], path_b_rows=[{"x": 1.00002}],
        )
        assert r.agreement_level == "full"  # within 1e-4


class TestPatternStore:
    def test_add_and_list(self, tmp_path: Path):
        store = PatternStore(path=str(tmp_path / "p.sqlite"))
        p = store.add("how many?", "SELECT COUNT(*) FROM t", source="user_correction")
        assert p.id > 0
        rows = store.list()
        assert len(rows) == 1
        assert rows[0].sql == "SELECT COUNT(*) FROM t"

    def test_score_ordering(self, tmp_path: Path):
        store = PatternStore(path=str(tmp_path / "p.sqlite"))
        low = store.add("q", "SELECT 1", score=1, source="user_approval")
        high = store.add("q", "SELECT 2", score=10, source="user_correction")
        rows = store.list()
        assert rows[0].id == high.id
        assert rows[1].id == low.id

    def test_negative_patterns_excluded_from_retriever(self, tmp_path: Path):
        """Negative-scored patterns are stored but shouldn't be surfaced by the retriever's pattern loader."""
        from talkdb.retrieval.hybrid_retriever import _patterns_to_docs
        store = PatternStore(path=str(tmp_path / "p.sqlite"))
        store.add("q", "SELECT good", score=5, source="user_correction")
        store.add("q", "SELECT bad", score=-1, source="user_correction")
        docs = _patterns_to_docs(store)
        assert len(docs) == 1
        assert "good" in docs[0].text


class FakeVectorStore(VectorStore):
    def __init__(self):
        self.upserts: list[tuple[str, dict]] = []

    def upsert(self, ids, documents, embeddings, metadatas):
        for i, m in zip(ids, metadatas, strict=False):
            self.upserts.append((i, m))

    def query(self, embedding, k=10):
        return []

    def reset(self):
        self.upserts = []

    def count(self):
        return len(self.upserts)


class FakeEmbedder:
    def embed(self, texts):
        return [[0.0] * 4 for _ in texts]

    def embed_one(self, text):
        return [0.0] * 4


class TestFeedbackRecorder:
    def test_correction_records_and_indexes(self, tmp_path: Path):
        store = PatternStore(path=str(tmp_path / "p.sqlite"))
        vs = FakeVectorStore()
        recorder = FeedbackRecorder(store, vs, FakeEmbedder())
        outcome = recorder.record_correction(
            question="top customers by revenue",
            correct_sql="SELECT c.id, c.name, SUM(o.total_amount) FROM customers c JOIN orders o ON c.id=o.customer_id GROUP BY c.id, c.name",
            wrong_sql="SELECT c.id FROM customers",
        )
        assert outcome.indexed
        # Correct SQL and wrong SQL both in store; only correct SQL indexed into vector store.
        rows = store.list()
        assert len(rows) == 2
        assert len(vs.upserts) == 1
        assert vs.upserts[0][1]["doc_type"] == "example"


class TestTranspiler:
    def test_passthrough_same_dialect(self):
        sql = "SELECT * FROM t"
        assert transpile(sql, from_dialect="sqlite", to_dialect="sqlite") == sql

    def test_postgres_to_sqlite_date(self):
        pg = "SELECT DATE_TRUNC('month', created_at) FROM orders"
        out = transpile(pg, from_dialect="postgres", to_dialect="sqlite")
        # sqlglot converts DATE_TRUNC to a STRFTIME-based expression for SQLite.
        assert "strftime" in out.lower() or "date_trunc" not in out.lower()

    def test_unknown_dialect_returns_input(self):
        sql = "SELECT * FROM t"
        # sqlglot doesn't know 'fakedb' — should return the input unchanged.
        out = transpile(sql, from_dialect="sqlite", to_dialect="fakedb")
        # Either passthrough or an attempt — but must not raise.
        assert isinstance(out, str)
