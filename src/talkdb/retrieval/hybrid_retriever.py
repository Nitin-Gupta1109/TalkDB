"""
Hybrid retriever: BM25 (keyword) + vector (semantic) + Reciprocal Rank Fusion.

Indexes a schema + semantic model + proven examples as a flat set of documents.
Given a user question, returns the top-k most relevant documents as a
focused context string for the LLM prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from talkdb.learning.pattern_store import PatternStore
from talkdb.retrieval.embeddings import EmbeddingClient
from talkdb.retrieval.vector_store import VectorStore
from talkdb.schema.models import DatabaseSchema
from talkdb.schema.semantic_model import SemanticModel


@dataclass
class RetrievedDoc:
    id: str
    text: str
    doc_type: str  # "table" | "column" | "metric" | "join" | "example" | "insight_hint"
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


class HybridRetriever:
    """
    Owns:
    - A vector store (persisted) for semantic search
    - An in-memory BM25 index for keyword search
    - The raw document catalog for rebuilding BM25 after restart

    Indexing (build_index) is an offline step. Retrieval is online.
    """

    def __init__(self, vector_store: VectorStore, embedder: EmbeddingClient):
        self.vector_store = vector_store
        self.embedder = embedder
        self._docs: list[RetrievedDoc] = []
        self._bm25: BM25Okapi | None = None

    def build_index(
        self,
        schema: DatabaseSchema,
        semantic_models: list[SemanticModel],
        pattern_store: PatternStore | None = None,
    ) -> int:
        """Rebuild the vector and BM25 indexes from scratch. Returns the number of documents indexed."""
        docs = _assemble_documents(schema, semantic_models)
        if pattern_store is not None:
            docs.extend(_patterns_to_docs(pattern_store))
        self._docs = docs

        if docs:
            texts = [d.text for d in docs]
            embeddings = self.embedder.embed(texts)
            self.vector_store.reset()
            self.vector_store.upsert(
                ids=[d.id for d in docs],
                documents=texts,
                embeddings=embeddings,
                metadatas=[{"doc_type": d.doc_type, **d.metadata} for d in docs],
            )
            self._build_bm25(texts)
        return len(docs)

    def load_bm25_from_existing(
        self,
        schema: DatabaseSchema,
        semantic_models: list[SemanticModel],
        pattern_store: PatternStore | None = None,
    ) -> None:
        """Rebuild the BM25 index in memory without re-embedding (used at server startup)."""
        self._docs = _assemble_documents(schema, semantic_models)
        if pattern_store is not None:
            self._docs.extend(_patterns_to_docs(pattern_store))
        self._build_bm25([d.text for d in self._docs])

    def retrieve(self, question: str, k: int = 10) -> list[RetrievedDoc]:
        """Retrieve top-k documents by reciprocal rank fusion over BM25 and vector scores."""
        if not self._docs:
            return []

        bm25_ranking = self._bm25_rank(question)
        vector_ranking = self._vector_rank(question, k=k * 2)

        fused = _reciprocal_rank_fusion([bm25_ranking, vector_ranking])
        top_ids = [doc_id for doc_id, _ in fused[:k]]
        by_id = {d.id: d for d in self._docs}
        return [by_id[i] for i in top_ids if i in by_id]

    def _bm25_rank(self, question: str) -> list[tuple[str, float]]:
        if not self._bm25 or not self._docs:
            return []
        tokens = _tokenize(question)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(self._docs, scores, strict=False), key=lambda x: x[1], reverse=True)
        return [(d.id, float(s)) for d, s in ranked]

    def _vector_rank(self, question: str, k: int) -> list[tuple[str, float]]:
        embedding = self.embedder.embed_one(question)
        hits = self.vector_store.query(embedding, k=k)
        return [(h.id, h.score) for h in hits]

    def _build_bm25(self, texts: list[str]) -> None:
        tokenized = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k_rrf: int = 60,
) -> list[tuple[str, float]]:
    """Standard RRF: score_i = sum over rankers of 1 / (k_rrf + rank_in_ranker)."""
    combined: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking):
            combined[doc_id] = combined.get(doc_id, 0.0) + 1.0 / (k_rrf + rank + 1)
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)


def _assemble_documents(
    schema: DatabaseSchema,
    semantic_models: list[SemanticModel],
) -> list[RetrievedDoc]:
    """Flatten schema + semantic models into indexable documents."""
    docs: list[RetrievedDoc] = []

    # Merge semantic table/column overrides by name.
    sem_tables: dict[str, object] = {}
    for sm in semantic_models:
        for t in sm.tables:
            sem_tables[t.name] = t

    for table in schema.tables:
        sem_t = sem_tables.get(table.name)
        table_desc = (getattr(sem_t, "description", None) or table.description) if sem_t else table.description
        col_summary = ", ".join(c.name for c in table.columns)
        text = f"Table: {table.name}. Columns: {col_summary}."
        if table_desc:
            text += f" {table_desc}"
        if table.row_count is not None:
            text += f" Approx rows: {table.row_count}."
        docs.append(
            RetrievedDoc(
                id=f"table:{table.name}",
                text=text,
                doc_type="table",
                metadata={"table": table.name},
            )
        )

        for col in table.columns:
            sem_c = sem_t.column(col.name) if sem_t else None
            aliases = getattr(sem_c, "aliases", []) if sem_c else []
            valid_values = getattr(sem_c, "valid_values", []) if sem_c else []
            desc = (getattr(sem_c, "description", None) if sem_c else None) or col.description
            parts = [f"Column: {table.name}.{col.name} ({col.data_type})."]
            if desc:
                parts.append(desc)
            if aliases:
                parts.append(f"Aliases: {', '.join(aliases)}.")
            if valid_values:
                parts.append(f"Valid values: {', '.join(valid_values)}.")
            elif col.sample_values:
                parts.append(f"Examples: {', '.join(col.sample_values)}.")
            if col.is_primary_key:
                parts.append("Primary key.")
            if col.is_foreign_key and col.foreign_key_references:
                parts.append(f"Foreign key -> {col.foreign_key_references}.")
            docs.append(
                RetrievedDoc(
                    id=f"column:{table.name}.{col.name}",
                    text=" ".join(parts),
                    doc_type="column",
                    metadata={"table": table.name, "column": col.name},
                )
            )

    for fk in schema.foreign_keys:
        text = (
            f"Join: {fk.from_table}({', '.join(fk.from_columns)}) -> "
            f"{fk.to_table}({', '.join(fk.to_columns)})."
        )
        docs.append(
            RetrievedDoc(
                # Include from_columns so multiple FKs between the same pair of tables
                # (e.g. Students.permanent_address_id + Students.current_address_id both
                # FK'ing Addresses) don't collide on the same doc ID.
                id=f"fk:{fk.from_table}({','.join(fk.from_columns)})->{fk.to_table}",
                text=text,
                doc_type="join",
                metadata={"from": fk.from_table, "to": fk.to_table},
            )
        )

    for sm in semantic_models:
        for metric in sm.metrics:
            tables = metric.tables or ([metric.table] if metric.table else [])
            text = f"Metric: {metric.name}. {metric.description}. Calculation: {metric.calculation}."
            if tables:
                text += f" Tables: {', '.join(t for t in tables if t)}."
            docs.append(
                RetrievedDoc(
                    id=f"metric:{metric.name}",
                    text=text,
                    doc_type="metric",
                    metadata={"metric": metric.name},
                )
            )

        for j in sm.joins:
            text = f"Join rule: {j.left} {j.type} {j.right} ON {j.on}."
            if j.description:
                text += f" {j.description}"
            docs.append(
                RetrievedDoc(
                    id=f"join:{j.left}-{j.right}",
                    text=text,
                    doc_type="join",
                    metadata={"left": j.left, "right": j.right},
                )
            )

        for idx, ex in enumerate(sm.examples):
            docs.append(
                RetrievedDoc(
                    id=f"example:{idx}:{_slug(ex.question)}",
                    text=f"Example question: {ex.question}\nSQL: {ex.sql}",
                    doc_type="example",
                    metadata={"question": ex.question},
                )
            )

        for hint in sm.insight_hints:
            parts = [f"Insight hint for {hint.metric}."]
            for attr in ("normal_range", "seasonality", "alert_threshold", "trend"):
                val = getattr(hint, attr, None)
                if val:
                    parts.append(f"{attr}: {val}.")
            docs.append(
                RetrievedDoc(
                    id=f"hint:{hint.metric}",
                    text=" ".join(parts),
                    doc_type="insight_hint",
                    metadata={"metric": hint.metric},
                )
            )

    return docs


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]


def _patterns_to_docs(store: PatternStore) -> list[RetrievedDoc]:
    """Fold approved/corrected patterns into the retriever's document corpus."""
    docs: list[RetrievedDoc] = []
    for p in store.list(limit=200):
        if p.score < 0:
            continue  # skip explicit negatives
        docs.append(
            RetrievedDoc(
                id=f"pattern:{p.id}",
                text=f"Proven pattern ({p.source}). Example question: {p.question}\nSQL: {p.sql}",
                doc_type="example",
                score=0.0,
                metadata={"source": p.source, "pattern_id": p.id, "score": p.score},
            )
        )
    return docs
