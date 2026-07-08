"""Embed chunk JSONL records and persist them to ChromaDB."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import chromadb
from tqdm import tqdm

from sec_rag.config import settings
from sec_rag.embeddings import EmbeddingClient, create_embedding_client
from sec_rag.ingest.chunk import CHUNKS_JSONL
from sec_rag.logging_setup import get_logger

logger = get_logger(__name__)

EMBED_REPORT_JSON = "embed_report.json"


@dataclass(frozen=True)
class EmbedReport:
    """Summary of a Chroma indexing run."""

    total_chunks: int
    indexed_chunks: int
    collection_name: str
    chroma_path: str
    provider: str
    model: str
    dimension: int
    chunk_source: str
    report_path: str


def embed_all(
    chunks_path: Path | None = None,
    chroma_path: Path | None = None,
    *,
    collection_name: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    batch_size: int | None = None,
    reset_collection: bool = False,
    embedding_client: EmbeddingClient | None = None,
) -> EmbedReport:
    """Embed every chunk in chunks.jsonl and upsert into Chroma."""
    chunks_path = chunks_path or settings.chunks_dir / CHUNKS_JSONL
    chroma_path = chroma_path or settings.chroma_path
    collection_name = collection_name or settings.embedding_collection
    batch_size = batch_size or settings.embedding_batch_size
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    chunks = load_chunk_records(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    client = embedding_client or create_embedding_client(provider=provider, model=model)
    chroma_path.mkdir(parents=True, exist_ok=True)
    chroma = chromadb.PersistentClient(path=str(chroma_path))

    if reset_collection:
        _delete_collection_if_exists(chroma, collection_name)
    collection = chroma.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    indexed = 0
    pbar = tqdm(
        _batched(chunks, batch_size),
        total=_batch_count(len(chunks), batch_size),
        desc="Embedding chunks",
        unit="batch",
    )
    for batch in pbar:
        embeddings = client.embed_documents([_require_str(chunk, "text") for chunk in batch])
        collection.upsert(
            ids=[_require_str(chunk, "id") for chunk in batch],
            documents=[_require_str(chunk, "text") for chunk in batch],
            metadatas=[_metadata_for_chroma(chunk) for chunk in batch],
            embeddings=cast(Any, embeddings),
        )
        indexed += len(batch)
        pbar.set_postfix_str(f"{indexed}/{len(chunks)}")

    report_path = chunks_path.parent / EMBED_REPORT_JSON
    report = EmbedReport(
        total_chunks=len(chunks),
        indexed_chunks=indexed,
        collection_name=collection_name,
        chroma_path=str(chroma_path),
        provider=client.provider,
        model=client.model,
        dimension=client.dimension,
        chunk_source=str(chunks_path),
        report_path=str(report_path),
    )
    report_path.write_text(json.dumps(asdict(report), indent=2))
    logger.info(
        "embedding_complete",
        chunks=indexed,
        collection=collection_name,
        provider=client.provider,
        model=client.model,
    )
    return report


def load_chunk_records(path: Path) -> list[dict[str, object]]:
    """Read chunk JSONL records from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Chunk line {line_number} is not a JSON object")
        _validate_chunk_payload(payload, line_number)
        records.append(payload)
    return records


def _validate_chunk_payload(payload: Mapping[str, object], line_number: int) -> None:
    for key in (
        "id",
        "ticker",
        "year",
        "section",
        "section_title",
        "chunk_index",
        "chunk_count",
        "text",
        "token_count",
        "source_file",
        "source_section_file",
    ):
        if key not in payload:
            raise ValueError(f"Chunk line {line_number} missing required field: {key}")


def _metadata_for_chroma(chunk: Mapping[str, object]) -> dict[str, str | int | float | bool]:
    return {
        "ticker": _require_str(chunk, "ticker"),
        "year": _require_int(chunk, "year"),
        "section": _require_str(chunk, "section"),
        "section_title": _require_str(chunk, "section_title"),
        "chunk_index": _require_int(chunk, "chunk_index"),
        "chunk_count": _require_int(chunk, "chunk_count"),
        "token_count": _require_int(chunk, "token_count"),
        "char_count": _optional_int(chunk, "char_count"),
        "start_token": _optional_int(chunk, "start_token"),
        "end_token": _optional_int(chunk, "end_token"),
        "source_file": _require_str(chunk, "source_file"),
        "source_section_file": _require_str(chunk, "source_section_file"),
        "extraction_score": _optional_float(chunk, "extraction_score"),
    }


def _delete_collection_if_exists(chroma: chromadb.ClientAPI, name: str) -> None:
    try:
        chroma.delete_collection(name)
    except Exception:  # noqa: BLE001 - Chroma raises different errors by version
        return


def _batched(
    records: list[dict[str, object]], batch_size: int
) -> Iterator[list[dict[str, object]]]:
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def _batch_count(total: int, batch_size: int) -> int:
    return (total + batch_size - 1) // batch_size


def _require_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"chunk payload missing string field: {key}")
    return value


def _require_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"chunk payload missing integer field: {key}")
    return value


def _optional_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key, 0)
    if isinstance(value, int):
        return value
    raise ValueError(f"chunk payload field must be integer: {key}")


def _optional_float(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key, 0.0)
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"chunk payload field must be numeric: {key}")
