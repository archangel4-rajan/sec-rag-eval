"""Embedding providers used by ingestion and retrieval."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI

from sec_rag.config import settings

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-']*")


class EmbeddingClient(Protocol):
    """Common interface for embedding providers."""

    provider: str
    model: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed one query."""
        ...


@dataclass
class OpenAIEmbeddingClient:
    """OpenAI embedding provider for production corpus builds."""

    model: str = "text-embedding-3-small"
    api_key: str | None = None
    provider: str = "openai"
    dimension: int = 1536

    def __post_init__(self) -> None:
        key = self.api_key or settings.openai_api_key
        if not key:
            raise ValueError(
                "OPENAI_API_KEY is required for provider='openai'. "
                "Use --provider hash for offline local testing."
            )
        self._client = OpenAI(api_key=key)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model, input=list(texts))
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@dataclass
class HashEmbeddingClient:
    """Deterministic offline embeddings for tests and no-key local smoke runs.

    This is intentionally simple and explicit. It is not a semantic embedding
    model, but it allows the Chroma indexing/retrieval pipeline to be tested
    without network access or API cost.
    """

    model: str = "hash-embedding-v1"
    provider: str = "hash"
    dimension: int = 384

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = TOKEN_PATTERN.findall(text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [v / norm for v in vector]


def create_embedding_client(
    *,
    provider: str | None = None,
    model: str | None = None,
) -> EmbeddingClient:
    """Build an embedding client from explicit args or settings."""
    provider_name = (provider or settings.embedding_provider).lower()
    model_name = model or settings.embedding_model

    if provider_name == "openai":
        return OpenAIEmbeddingClient(model=model_name)
    if provider_name == "hash":
        return HashEmbeddingClient(model=model or "hash-embedding-v1")
    raise ValueError(f"Unsupported embedding provider: {provider_name}")
