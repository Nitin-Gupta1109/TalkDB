"""Vector store abstraction. ChromaDB for dev; pgvector can plug in behind the same interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VectorHit:
    id: str
    document: str
    metadata: dict
    score: float  # Higher is better (1 - distance for cosine)


class VectorStore(ABC):
    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        ...

    @abstractmethod
    def query(self, embedding: list[float], k: int = 10) -> list[VectorHit]:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def count(self) -> int:
        ...


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_path: str, collection: str = "talkdb"):
        import chromadb

        self._persist_path = persist_path
        self._collection_name = collection
        Path(persist_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        if not ids:
            return
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, embedding: list[float], k: int = 10) -> list[VectorHit]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, self.count()),
        )
        hits: list[VectorHit] = []
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        distances = result["distances"][0]
        for i, d, m, dist in zip(ids, docs, metas, distances, strict=False):
            hits.append(VectorHit(id=i, document=d, metadata=m or {}, score=1.0 - float(dist)))
        return hits

    def reset(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()
