"""Day 3 tests: token chunking for parsed SEC sections."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import sec_rag.ingest.chunk as chunk_module
from sec_rag.ingest.chunk import chunk_all, chunk_section, chunk_text


class FakeEncoding:
    """Minimal tokenizer for offline tests."""

    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


@pytest.fixture(autouse=True)
def fake_tiktoken_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chunk_module.tiktoken, "get_encoding", lambda _: FakeEncoding())


def _section_payload(
    *,
    ticker: str = "AAPL",
    year: int = 2025,
    section: str = "1A",
    text: str | None = None,
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "year": year,
        "section": section,
        "section_title": "Risk Factors" if section == "1A" else "MD&A",
        "text": text or _long_text(),
        "char_count": len(text or _long_text()),
        "source_file": f"{ticker}_{year}.html",
        "extraction_score": 42.0,
    }


def _long_text(sentence_count: int = 240) -> str:
    return " ".join(
        f"Risk factor sentence {i} describes supply chain, revenue, liquidity, and operations."
        for i in range(sentence_count)
    )


def test_chunk_text_uses_token_windows_with_overlap() -> None:
    windows = chunk_text(_long_text(), chunk_size_tokens=64, overlap_tokens=8)

    assert len(windows) > 1
    assert all(window.token_count <= 64 for window in windows)
    assert windows[0].start_token == 0
    for previous, current in zip(windows, windows[1:], strict=False):
        assert current.start_token == previous.end_token - 8
    assert windows[-1].end_token > windows[-1].start_token


def test_chunk_text_rejects_invalid_window_sizes() -> None:
    with pytest.raises(ValueError, match="positive"):
        chunk_text("text", chunk_size_tokens=0, overlap_tokens=0)
    with pytest.raises(ValueError, match="smaller"):
        chunk_text("text", chunk_size_tokens=10, overlap_tokens=10)


def test_chunk_section_creates_stable_metadata() -> None:
    chunks = chunk_section(
        _section_payload(),
        source_section_file="AAPL_2025_item1A.json",
        chunk_size_tokens=64,
        overlap_tokens=8,
    )

    assert len(chunks) > 1
    first = chunks[0]
    assert first.id == "AAPL_2025_item1A_chunk0000"
    assert first.ticker == "AAPL"
    assert first.year == 2025
    assert first.section == "1A"
    assert first.section_title == "Risk Factors"
    assert first.chunk_index == 0
    assert first.chunk_count == len(chunks)
    assert first.source_file == "AAPL_2025.html"
    assert first.source_section_file == "AAPL_2025_item1A.json"
    assert first.extraction_score == 42.0
    assert first.token_count <= 64


def test_chunk_section_rejects_malformed_payload() -> None:
    payload = _section_payload()
    del payload["ticker"]

    with pytest.raises(ValueError, match="ticker"):
        chunk_section(payload, source_section_file="bad.json")


def test_chunk_all_writes_jsonl_and_report(tmp_path: Path) -> None:
    _write_section(tmp_path / "AAPL_2025_item1A.json", _section_payload())
    _write_section(
        tmp_path / "MSFT_2025_item7.json",
        _section_payload(ticker="MSFT", section="7"),
    )
    (tmp_path / "parse_report.json").write_text("{}")

    output_path = tmp_path / "chunks.jsonl"
    report_path = tmp_path / "chunk_report.json"

    chunks = chunk_all(
        input_dir=tmp_path,
        output_path=output_path,
        report_path=report_path,
        chunk_size_tokens=80,
        overlap_tokens=10,
    )

    assert chunks
    assert output_path.exists()
    assert report_path.exists()

    lines = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(lines) == len(chunks)
    assert {line["ticker"] for line in lines} == {"AAPL", "MSFT"}
    assert all(line["token_count"] <= 80 for line in lines)

    report = json.loads(report_path.read_text())
    assert report["total_sections"] == 2
    assert report["total_chunks"] == len(chunks)
    assert report["chunk_size_tokens"] == 80
    assert report["overlap_tokens"] == 10
    assert report["by_section"]["1A"] > 0
    assert report["by_section"]["7"] > 0
    assert report["by_ticker"]["AAPL"] > 0
    assert report["by_ticker"]["MSFT"] > 0


def _write_section(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload))
