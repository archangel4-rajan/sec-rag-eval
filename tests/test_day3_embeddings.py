"""Day 3 tests: embedding and retrieval over Chroma."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import chromadb

from sec_rag.embeddings import HashEmbeddingClient
from sec_rag.ingest.chunk import ChunkRecord
from sec_rag.ingest.embed import embed_all, load_chunk_records
from sec_rag.retrieve.retriever import Retriever, _build_where_filter


def test_hash_embedding_is_deterministic_and_normalized() -> None:
    client = HashEmbeddingClient(dimension=32)

    first = client.embed_query("Apple supply chain risk")
    second = client.embed_query("Apple supply chain risk")

    assert first == second
    assert len(first) == 32
    assert abs(sum(v * v for v in first) - 1.0) < 1e-9


def test_load_chunk_records_validates_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    chunk = _chunk("AAPL_2025_item1A_chunk0000", "AAPL", 2025, "1A", "supply chain risk")
    path.write_text(json.dumps(asdict(chunk)) + "\n")

    records = load_chunk_records(path)

    assert len(records) == 1
    assert records[0]["id"] == "AAPL_2025_item1A_chunk0000"


def test_embed_all_indexes_chunks_into_chroma(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    chroma_path = tmp_path / "chroma"
    _write_chunks(
        chunks_path,
        [
            _chunk("AAPL_2025_item1A_chunk0000", "AAPL", 2025, "1A", "supply chain risk"),
            _chunk("MSFT_2025_item7_chunk0000", "MSFT", 2025, "7", "cloud revenue growth"),
        ],
    )

    report = embed_all(
        chunks_path=chunks_path,
        chroma_path=chroma_path,
        collection_name="test_chunks",
        embedding_client=HashEmbeddingClient(dimension=64),
        batch_size=1,
    )

    assert report.total_chunks == 2
    assert report.indexed_chunks == 2
    assert report.collection_name == "test_chunks"
    assert report.provider == "hash"
    assert (tmp_path / "embed_report.json").exists()

    collection = chromadb.PersistentClient(path=str(chroma_path)).get_collection("test_chunks")
    assert collection.count() == 2


def test_retriever_returns_filtered_results(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    chroma_path = tmp_path / "chroma"
    embedding_client = HashEmbeddingClient(dimension=64)
    _write_chunks(
        chunks_path,
        [
            _chunk(
                "AAPL_2025_item1A_chunk0000",
                "AAPL",
                2025,
                "1A",
                "Apple supply chain risk manufacturing",
            ),
            _chunk(
                "MSFT_2025_item7_chunk0000", "MSFT", 2025, "7", "Microsoft cloud revenue operations"
            ),
        ],
    )
    embed_all(
        chunks_path=chunks_path,
        chroma_path=chroma_path,
        collection_name="test_retrieval",
        embedding_client=embedding_client,
        batch_size=2,
    )

    retriever = Retriever(
        chroma_path=chroma_path,
        collection_name="test_retrieval",
        embedding_client=embedding_client,
    )
    results = retriever.search("supply chain risk", top_k=2, ticker="AAPL")

    assert results
    assert results[0].metadata["ticker"] == "AAPL"
    assert "supply chain" in results[0].text
    assert results[0].citation == "AAPL 2025 Item 1A chunk 0"


def test_build_where_filter_composes_metadata_clauses() -> None:
    assert _build_where_filter(ticker="aapl") == {"ticker": "AAPL"}
    assert _build_where_filter(ticker="aapl", year=2025, section="1a") == {
        "$and": [{"ticker": "AAPL"}, {"year": 2025}, {"section": "1A"}]
    }


def _chunk(chunk_id: str, ticker: str, year: int, section: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id,
        ticker=ticker,
        year=year,
        section=section,
        section_title="Risk Factors" if section == "1A" else "MD&A",
        chunk_index=0,
        chunk_count=1,
        text=text,
        token_count=len(text.split()),
        char_count=len(text),
        start_token=0,
        end_token=len(text.split()),
        source_file=f"{ticker}_{year}.html",
        source_section_file=f"{ticker}_{year}_item{section}.json",
        extraction_score=42.0,
    )


def _write_chunks(path: Path, chunks: list[ChunkRecord]) -> None:
    path.write_text("\n".join(json.dumps(asdict(chunk)) for chunk in chunks) + "\n")
