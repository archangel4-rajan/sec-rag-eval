"""Chunk parsed SEC filing sections into embedding-ready records."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import tiktoken
from tqdm import tqdm

from sec_rag.config import settings
from sec_rag.logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_CHUNK_SIZE_TOKENS = 512
DEFAULT_CHUNK_OVERLAP_TOKENS = 50
DEFAULT_ENCODING = "cl100k_base"

CHUNKS_JSONL = "chunks.jsonl"
CHUNK_REPORT_JSON = "chunk_report.json"


@dataclass(frozen=True)
class TokenWindow:
    """A decoded text window plus token offsets in the original section."""

    text: str
    start_token: int
    end_token: int
    token_count: int


@dataclass(frozen=True)
class ChunkRecord:
    """One embedding-ready chunk with retrieval metadata."""

    id: str
    ticker: str
    year: int
    section: str
    section_title: str
    chunk_index: int
    chunk_count: int
    text: str
    token_count: int
    char_count: int
    start_token: int
    end_token: int
    source_file: str
    source_section_file: str
    extraction_score: float


def chunk_text(
    text: str,
    *,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    encoding_name: str = DEFAULT_ENCODING,
) -> list[TokenWindow]:
    """Split text into overlapping token windows."""
    _validate_window(chunk_size_tokens, overlap_tokens)
    if not text:
        return []

    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    if not tokens:
        return []

    step = chunk_size_tokens - overlap_tokens
    windows: list[TokenWindow] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size_tokens, len(tokens))
        window_tokens = tokens[start:end]
        windows.append(
            TokenWindow(
                text=encoding.decode(window_tokens),
                start_token=start,
                end_token=end,
                token_count=len(window_tokens),
            )
        )
        if end == len(tokens):
            break
        start += step
    return windows


def chunk_section(
    section: Mapping[str, object],
    *,
    source_section_file: str,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    encoding_name: str = DEFAULT_ENCODING,
) -> list[ChunkRecord]:
    """Create chunk records for a parsed section JSON payload."""
    ticker = _require_str(section, "ticker")
    year = _require_int(section, "year")
    section_id = _require_str(section, "section")
    section_title = _require_str(section, "section_title")
    text = _require_str(section, "text").strip()
    source_file = _require_str(section, "source_file")
    extraction_score = _optional_float(section, "extraction_score")

    windows = chunk_text(
        text,
        chunk_size_tokens=chunk_size_tokens,
        overlap_tokens=overlap_tokens,
        encoding_name=encoding_name,
    )
    chunk_count = len(windows)

    records: list[ChunkRecord] = []
    for index, window in enumerate(windows):
        records.append(
            ChunkRecord(
                id=f"{ticker}_{year}_item{section_id}_chunk{index:04d}",
                ticker=ticker,
                year=year,
                section=section_id,
                section_title=section_title,
                chunk_index=index,
                chunk_count=chunk_count,
                text=window.text,
                token_count=window.token_count,
                char_count=len(window.text),
                start_token=window.start_token,
                end_token=window.end_token,
                source_file=source_file,
                source_section_file=source_section_file,
                extraction_score=extraction_score,
            )
        )
    return records


def chunk_all(
    input_dir: Path | None = None,
    output_path: Path | None = None,
    report_path: Path | None = None,
    *,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    encoding_name: str = DEFAULT_ENCODING,
) -> list[ChunkRecord]:
    """Chunk every parsed section JSON file and write a JSONL corpus."""
    _validate_window(chunk_size_tokens, overlap_tokens)
    input_dir = input_dir or settings.chunks_dir
    output_path = output_path or input_dir / CHUNKS_JSONL
    report_path = report_path or input_dir / CHUNK_REPORT_JSON

    section_files = sorted(input_dir.glob("*_item*.json"))
    if not section_files:
        logger.warning("no_sections_found", dir=str(input_dir))
        return []

    all_chunks: list[ChunkRecord] = []
    pbar = tqdm(section_files, desc="Chunking sections", unit="section")
    for section_file in pbar:
        pbar.set_postfix_str(section_file.stem)
        payload = json.loads(section_file.read_text())
        all_chunks.extend(
            chunk_section(
                payload,
                source_section_file=section_file.name,
                chunk_size_tokens=chunk_size_tokens,
                overlap_tokens=overlap_tokens,
                encoding_name=encoding_name,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, all_chunks)
    _write_chunk_report(
        report_path,
        section_files=section_files,
        chunks=all_chunks,
        chunk_size_tokens=chunk_size_tokens,
        overlap_tokens=overlap_tokens,
        encoding_name=encoding_name,
    )
    _log_summary(section_files, all_chunks)
    return all_chunks


def _write_jsonl(path: Path, chunks: list[ChunkRecord]) -> None:
    lines = [json.dumps(asdict(chunk), ensure_ascii=False) for chunk in chunks]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _write_chunk_report(
    path: Path,
    *,
    section_files: list[Path],
    chunks: list[ChunkRecord],
    chunk_size_tokens: int,
    overlap_tokens: int,
    encoding_name: str,
) -> None:
    by_section: dict[str, int] = {}
    by_ticker: dict[str, int] = {}
    token_counts = [chunk.token_count for chunk in chunks]

    for chunk in chunks:
        by_section[chunk.section] = by_section.get(chunk.section, 0) + 1
        by_ticker[chunk.ticker] = by_ticker.get(chunk.ticker, 0) + 1

    report = {
        "total_sections": len(section_files),
        "total_chunks": len(chunks),
        "chunk_size_tokens": chunk_size_tokens,
        "overlap_tokens": overlap_tokens,
        "encoding": encoding_name,
        "avg_tokens_per_chunk": round(sum(token_counts) / len(token_counts), 2)
        if token_counts
        else 0.0,
        "min_tokens_per_chunk": min(token_counts) if token_counts else 0,
        "max_tokens_per_chunk": max(token_counts) if token_counts else 0,
        "by_section": dict(sorted(by_section.items())),
        "by_ticker": dict(sorted(by_ticker.items())),
    }
    path.write_text(json.dumps(report, indent=2))


def _log_summary(section_files: list[Path], chunks: list[ChunkRecord]) -> None:
    logger.info(
        "chunking_complete",
        sections=len(section_files),
        chunks=len(chunks),
        output=CHUNKS_JSONL,
    )


def _validate_window(chunk_size_tokens: int, overlap_tokens: int) -> None:
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative")
    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")


def _require_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"section payload missing string field: {key}")
    return value


def _require_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"section payload missing integer field: {key}")
    return value


def _optional_float(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key, 0.0)
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"section payload field must be numeric: {key}")
