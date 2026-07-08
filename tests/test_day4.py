"""Day 4 tests: query orchestration and API contracts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fastapi.testclient import TestClient

from sec_rag.api.app import create_app
from sec_rag.embeddings import HashEmbeddingClient
from sec_rag.generate import (
    ExtractiveAnswerGenerator,
    QueryEngine,
    RagResponse,
    format_rag_response,
)
from sec_rag.ingest.chunk import ChunkRecord
from sec_rag.ingest.embed import embed_all
from sec_rag.retrieve import Retriever


def test_extractive_generator_returns_cited_answer() -> None:
    result = _fake_result(
        text=(
            "Apple depends on manufacturing outsourcing partners. "
            "Supply chain disruptions can affect product availability."
        )
    )

    response = ExtractiveAnswerGenerator().generate("What supply chain risks exist?", [result])

    assert "Supply chain disruptions" in response.answer
    assert "[1]" in response.answer
    assert response.citations[0].label == "AAPL 2025 Item 1A chunk 0"
    assert response.retrieved_count == 1


def test_query_engine_runs_retrieval_and_generation(tmp_path: Path) -> None:
    chroma_path = tmp_path / "chroma"
    chunks_path = tmp_path / "chunks.jsonl"
    embedding_client = HashEmbeddingClient(dimension=64)
    _write_chunks(
        chunks_path,
        [
            _chunk(
                "AAPL_2025_item1A_chunk0000",
                "AAPL",
                2025,
                "1A",
                "Apple supply chain disruptions can affect manufacturing and availability.",
            ),
            _chunk(
                "MSFT_2025_item7_chunk0000",
                "MSFT",
                2025,
                "7",
                "Microsoft cloud revenue increased due to demand for services.",
            ),
        ],
    )
    embed_all(
        chunks_path=chunks_path,
        chroma_path=chroma_path,
        collection_name="day4",
        embedding_client=embedding_client,
    )
    retriever = Retriever(
        chroma_path=chroma_path,
        collection_name="day4",
        embedding_client=embedding_client,
    )
    engine = QueryEngine(retriever=retriever)

    response = engine.answer("What supply chain risks did Apple describe?", ticker="AAPL")

    assert response.retrieved_count == 1
    assert response.citations[0].ticker == "AAPL"
    assert "supply chain" in response.answer.lower()


def test_format_rag_response_includes_citations() -> None:
    response = ExtractiveAnswerGenerator().generate("Question?", [_fake_result()])

    rendered = format_rag_response(response)

    assert "Citations:" in rendered
    assert "AAPL 2025 Item 1A chunk 0" in rendered


def test_api_health_and_query_contract() -> None:
    class FakeEngine:
        def answer(self, *_args: object, **_kwargs: object) -> RagResponse:
            return ExtractiveAnswerGenerator().generate("Question?", [_fake_result()])

    app = create_app(engine_factory=lambda _request: FakeEngine())  # type: ignore[return-value]
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    response = client.post("/query", json={"question": "What did Apple say?", "top_k": 1})
    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieved_count"] == 1
    assert payload["citations"][0]["ticker"] == "AAPL"
    assert "answer" in payload


def _fake_result(text: str = "Apple described supply chain and manufacturing risks."):
    from sec_rag.retrieve import RetrievalResult

    return RetrievalResult(
        id="AAPL_2025_item1A_chunk0000",
        text=text,
        metadata={
            "ticker": "AAPL",
            "year": 2025,
            "section": "1A",
            "chunk_index": 0,
        },
        distance=0.12,
    )


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
