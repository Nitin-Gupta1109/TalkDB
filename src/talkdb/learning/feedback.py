"""
User feedback API. Records corrections and approvals into the pattern store and
pushes approved patterns into the vector store so future retrievals pick them up.
"""

from __future__ import annotations

from dataclasses import dataclass

from talkdb.learning.pattern_store import PatternStore, ProvenPattern
from talkdb.retrieval.embeddings import EmbeddingClient
from talkdb.retrieval.vector_store import VectorStore


@dataclass
class FeedbackOutcome:
    pattern_id: int
    indexed: bool


class FeedbackRecorder:
    """
    Thin wrapper: persist the correction, then index the (question, sql) pair into
    the vector store so the next retriever query can surface it. The vector store
    is assumed to already hold schema/semantic docs — we just upsert one more doc.
    """

    def __init__(
        self,
        store: PatternStore,
        vector_store: VectorStore,
        embedder: EmbeddingClient,
    ):
        self.store = store
        self.vector_store = vector_store
        self.embedder = embedder

    def record_correction(
        self,
        question: str,
        correct_sql: str,
        *,
        database: str | None = None,
        wrong_sql: str | None = None,
    ) -> FeedbackOutcome:
        pattern = self.store.add(
            question=question,
            sql=correct_sql,
            database=database,
            source="user_correction",
            score=5,  # Corrections trump plain examples.
        )
        # Optionally record the wrong SQL as a negative example. We store it but never index
        # into the retriever — negative signals are read by future ranking/dispute flows.
        if wrong_sql:
            self.store.add(
                question=question,
                sql=wrong_sql,
                database=database,
                source="user_correction",
                score=-1,
            )
        indexed = self._index(pattern)
        return FeedbackOutcome(pattern_id=pattern.id, indexed=indexed)

    def record_approval(
        self, question: str, sql: str, *, database: str | None = None
    ) -> FeedbackOutcome:
        pattern = self.store.add(
            question=question,
            sql=sql,
            database=database,
            source="user_approval",
            score=3,
        )
        indexed = self._index(pattern)
        return FeedbackOutcome(pattern_id=pattern.id, indexed=indexed)

    def _index(self, pattern: ProvenPattern) -> bool:
        doc = f"Proven pattern. Example question: {pattern.question}\nSQL: {pattern.sql}"
        try:
            embedding = self.embedder.embed_one(doc)
            self.vector_store.upsert(
                ids=[f"pattern:{pattern.id}"],
                documents=[doc],
                embeddings=[embedding],
                metadatas=[
                    {
                        "doc_type": "example",
                        "source": pattern.source,
                        "score": pattern.score,
                        "pattern_id": pattern.id,
                    }
                ],
            )
            return True
        except Exception:  # noqa: BLE001 — indexing failures should not break the user flow
            return False
