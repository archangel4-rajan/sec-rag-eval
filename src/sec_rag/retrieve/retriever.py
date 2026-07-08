"""Chroma-backed retrieval for embedded SEC filing chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import chromadb

from sec_rag.config import settings
from sec_rag.embeddings import EmbeddingClient, create_embedding_client


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieved chunk."""

    id: str
    text: str
    metadata: dict[str, Any]
    distance: float

    @property
    def citation(self) -> str:
        ticker = self.metadata.get("ticker", "?")
        year = self.metadata.get("year", "?")
        section = self.metadata.get("section", "?")
        chunk_index = self.metadata.get("chunk_index", "?")
        return f"{ticker} {year} Item {section} chunk {chunk_index}"


class Retriever:
    """Thin retrieval wrapper around a Chroma collection."""

    def __init__(
        self,
        *,
        chroma_path: Path | None = None,
        collection_name: str | None = None,
        embedding_client: EmbeddingClient | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        self.chroma_path = chroma_path or settings.chroma_path
        self.collection_name = collection_name or settings.embedding_collection
        self.embedding_client = embedding_client or create_embedding_client(
            provider=provider,
            model=model,
        )
        self._client = chromadb.PersistentClient(path=str(self.chroma_path))
        self._collection = self._client.get_collection(self.collection_name)

    def search(
        self,
        question: str,
        *,
        top_k: int = 5,
        ticker: str | None = None,
        year: int | None = None,
        section: str | None = None,
    ) -> list[RetrievalResult]:
        """Return top matching chunks, optionally filtered by metadata."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_embedding = self.embedding_client.embed_query(question)
        where = _build_where_filter(ticker=ticker, year=year, section=section)
        query_args: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            query_args["where"] = where
        raw = self._collection.query(**query_args)
        return _parse_chroma_results(cast(dict[str, Any], raw))


def _build_where_filter(
    *,
    ticker: str | None = None,
    year: int | None = None,
    section: str | None = None,
) -> dict[str, Any] | None:
    clauses: list[dict[str, str | int]] = []
    if ticker:
        clauses.append({"ticker": ticker.upper()})
    if year is not None:
        clauses.append({"year": year})
    if section:
        clauses.append({"section": section.upper()})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _parse_chroma_results(raw: dict[str, Any]) -> list[RetrievalResult]:
    ids = _first(raw.get("ids"))
    documents = _first(raw.get("documents"))
    metadatas = _first(raw.get("metadatas"))
    distances = _first(raw.get("distances"))

    results: list[RetrievalResult] = []
    for item_id, document, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
        strict=False,
    ):
        results.append(
            RetrievalResult(
                id=str(item_id),
                text=str(document),
                metadata=dict(metadata or {}),
                distance=float(distance),
            )
        )
    return results


def _first(value: Any) -> list[Any]:
    if not value:
        return []
    return list(value[0])
