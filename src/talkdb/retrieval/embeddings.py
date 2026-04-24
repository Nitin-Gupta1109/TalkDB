"""Embedding abstraction over LiteLLM. Never imports openai/anthropic directly."""

from __future__ import annotations

import litellm


class EmbeddingClient:
    """Small wrapper so the rest of the system never touches LiteLLM's embedding API directly."""

    def __init__(self, model: str):
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = litellm.embedding(model=self.model, input=texts)
        return [item["embedding"] for item in response["data"]]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
